// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

//! Camera frames converted to model-input size on arrival, kept by stamp.
//!
//! A `CameraFrame` only references a DMA buffer the camera recycles, so the
//! pixels must be copied out promptly; the radar cube for the same instant
//! arrives 100+ ms later and picks the nearest converted slot.

use crate::sync::stale_cutoff;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex, MutexGuard,
    },
    time::{Duration, Instant},
};
use tokio::sync::Notify;

/// Window used until the camera's buffer pool and frame period are learned:
/// 4 buffers at 30 FPS.
pub const DEFAULT_RECYCLE_WINDOW_NS: u64 = 100_000_000;

/// Frames of history used to count the pool and estimate the period. Covers
/// every buffer of the largest pool the camera service allows (32) twice.
const POOL_HISTORY: usize = 64;

/// Frames observed before the learned window replaces the default.
const MIN_FRAMES: usize = POOL_HISTORY / 2;

/// Consecutive stamps further apart than this are a gap, not a frame period.
const MAX_PERIOD_NS: u64 = 1_000_000_000;

/// Lower bound on the safety margin: the frame is converted after the age
/// check, and the ISP begins rewriting a buffer shortly before the stamp
/// arithmetic says it does.
const MIN_MARGIN_NS: u64 = 5_000_000;

/// How long after its stamp a `CameraFrame`'s DMA buffer stays intact.
///
/// The camera re-queues each capture buffer as soon as the frame is
/// published, behind the buffers already queued, so with `N` buffers at
/// frame period `T` the driver starts overwriting it about `(N - 1) T` after
/// the frame's end-of-frame stamp. `N` is the number of distinct DMA-BUF
/// file descriptors the publishing process cycles through, and `T` the
/// median stamp interval. A camera restart (new pid) starts learning again.
#[derive(Debug, Default)]
pub struct RecycleWindow {
    pid: Option<u32>,
    handles: VecDeque<i64>,
    periods: VecDeque<u64>,
    last_stamp: Option<u64>,
    learned: Option<CameraPool>,
}

/// Capture pool as inferred from the frame stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraPool {
    pub buffers: usize,
    pub period_ns: u64,
}

impl CameraPool {
    /// Age after capture beyond which a frame may be partly overwritten,
    /// less a margin of a quarter period (at least `MIN_MARGIN_NS`).
    pub fn window_ns(&self) -> u64 {
        let span = (self.buffers.saturating_sub(1) as u64).saturating_mul(self.period_ns);
        let margin = (self.period_ns / 4).max(MIN_MARGIN_NS);
        span.saturating_sub(margin)
    }
}

impl RecycleWindow {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a frame from process `pid` in buffer `handle`, stamped
    /// `stamp_ns`. Returns the pool when the learned estimate changes.
    pub fn observe(&mut self, pid: u32, handle: i64, stamp_ns: u64) -> Option<CameraPool> {
        if self.pid != Some(pid) {
            *self = Self {
                pid: Some(pid),
                ..Self::default()
            };
        }
        if let Some(last) = self.last_stamp {
            let period = stamp_ns.saturating_sub(last);
            if period > 0 && period <= MAX_PERIOD_NS {
                push_bounded(&mut self.periods, period);
            }
        }
        self.last_stamp = Some(stamp_ns);
        push_bounded(&mut self.handles, handle);

        let pool = self.estimate()?;
        let changed = self.learned.is_none_or(|old| {
            old.buffers != pool.buffers
                || old.period_ns.abs_diff(pool.period_ns) > pool.period_ns / 10
        });
        if changed {
            self.learned = Some(pool);
            return Some(pool);
        }
        None
    }

    fn estimate(&self) -> Option<CameraPool> {
        if self.handles.len() < MIN_FRAMES || self.periods.len() < MIN_FRAMES / 2 {
            return None;
        }
        let mut handles: Vec<i64> = self.handles.iter().copied().collect();
        handles.sort_unstable();
        handles.dedup();
        let mut periods: Vec<u64> = self.periods.iter().copied().collect();
        periods.sort_unstable();
        let pool = CameraPool {
            buffers: handles.len(),
            period_ns: periods[periods.len() / 2],
        };
        (pool.buffers >= 2).then_some(pool)
    }

    /// Current window: learned, or `DEFAULT_RECYCLE_WINDOW_NS`.
    pub fn window_ns(&self) -> u64 {
        self.learned
            .map_or(DEFAULT_RECYCLE_WINDOW_NS, |p| p.window_ns())
    }

    /// True when a frame stamped `stamp_ns` is too old at `now_ns` to read.
    pub fn is_stale(&self, stamp_ns: u64, now_ns: u64) -> bool {
        now_ns.saturating_sub(stamp_ns) > self.window_ns()
    }
}

fn push_bounded<T>(q: &mut VecDeque<T>, v: T) {
    if q.len() == POOL_HISTORY {
        q.pop_front();
    }
    q.push_back(v);
}

struct Slot<I> {
    image: I,
    /// Frame stamp and monotonic receive time; `None` while empty.
    stamp: Option<(u64, Instant)>,
}

/// Fixed set of slots overwritten round-robin, each tagged with the stamp of
/// the frame it holds. Slots received more than `sync::STALE_AFTER` ago are
/// never selected.
pub struct CameraRing<I> {
    slots: Vec<Slot<I>>,
    next: usize,
}

impl<I> CameraRing<I> {
    /// Ring over `slots`, all initially empty. Panics if `slots` is empty.
    pub fn new(slots: Vec<I>) -> Self {
        assert!(!slots.is_empty(), "camera ring needs at least one slot");
        Self {
            slots: slots
                .into_iter()
                .map(|image| Slot { image, stamp: None })
                .collect(),
            next: 0,
        }
    }

    /// Store `image` (stamped `stamp_ns`, received at `received`) in place of
    /// the oldest slot and return that slot's previous image for reuse.
    pub fn replace_next(&mut self, image: I, stamp_ns: u64, received: Instant) -> I {
        let idx = self.next;
        self.next = (idx + 1) % self.slots.len();
        let slot = &mut self.slots[idx];
        slot.stamp = Some((stamp_ns, received));
        std::mem::replace(&mut slot.image, image)
    }

    /// Slots still fresh at `now`, as (index, stamp).
    fn live(&self, now: Instant) -> impl Iterator<Item = (usize, u64)> + '_ {
        let cutoff = stale_cutoff(now);
        self.slots.iter().enumerate().filter_map(move |(i, s)| {
            let (stamp, received) = s.stamp?;
            cutoff.is_none_or(|c| received >= c).then_some((i, stamp))
        })
    }

    /// Fresh slot nearest `target_ns`, with the signed delta
    /// (slot stamp − target) clamped to `i64`.
    pub fn nearest(&self, target_ns: u64, now: Instant) -> Option<(usize, i64)> {
        self.live(now)
            .min_by_key(|(_, s)| s.abs_diff(target_ns))
            .map(|(i, s)| {
                let delta = (i128::from(s) - i128::from(target_ns))
                    .clamp(i64::MIN.into(), i64::MAX.into()) as i64;
                (i, delta)
            })
    }

    /// Stamp of the newest fresh slot.
    pub fn newest_ns(&self, now: Instant) -> Option<u64> {
        self.live(now).map(|(_, s)| s).max()
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut I {
        &mut self.slots[idx].image
    }

    /// Forget every stamp, e.g. after a clock step.
    pub fn clear(&mut self) {
        for s in &mut self.slots {
            s.stamp = None;
        }
    }
}

/// Camera ring shared between the converter thread (writer) and the fusion
/// model thread (reader), with the converter's stale-conversion counter.
pub struct SharedCameraRing<I> {
    ring: Mutex<CameraRing<I>>,
    notify: Notify,
    stale: AtomicU64,
    warned_stale: AtomicBool,
}

impl<I> SharedCameraRing<I> {
    pub fn new(ring: CameraRing<I>) -> Self {
        Self {
            ring: Mutex::new(ring),
            notify: Notify::new(),
            stale: AtomicU64::new(0),
            warned_stale: AtomicBool::new(false),
        }
    }

    /// Never hold the guard across `.await`.
    pub fn lock(&self) -> MutexGuard<'_, CameraRing<I>> {
        self.ring.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Store a converted frame, wake waiters, and return the replaced image.
    pub fn publish(&self, image: I, stamp_ns: u64, received: Instant) -> I {
        let old = self.lock().replace_next(image, stamp_ns, received);
        self.notify.notify_waiters();
        old
    }

    /// Wait at most `wait` for a fresh slot stamped at or after `target_ns`.
    pub async fn wait_for_stamp(&self, target_ns: u64, wait: Duration) {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let notified = self.notify.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            let covered = self
                .lock()
                .newest_ns(Instant::now())
                .is_some_and(|n| n >= target_ns);
            if covered || tokio::time::timeout_at(deadline, notified).await.is_err() {
                return;
            }
        }
    }

    /// Count a stale conversion; true if the caller should log the warning
    /// (the first one since the warning was last re-armed).
    pub fn record_stale(&self) -> bool {
        self.stale.fetch_add(1, Ordering::Relaxed);
        !self.warned_stale.swap(true, Ordering::Relaxed)
    }

    /// Stale conversions since the last call. Re-arms the warning if any.
    pub fn take_stale(&self) -> u64 {
        let n = self.stale.swap(0, Ordering::Relaxed);
        if n > 0 {
            self.warned_stale.store(false, Ordering::Relaxed);
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::STALE_AFTER;
    use std::sync::Arc;

    const MS: u64 = 1_000_000;

    const PERIOD: u64 = 33_333_333;

    fn feed(w: &mut RecycleWindow, pid: u32, buffers: i64, frames: u64) -> Option<CameraPool> {
        let mut last = None;
        for k in 0..frames {
            if let Some(p) = w.observe(pid, 40 + (k as i64 % buffers), 1_000 * MS + k * PERIOD) {
                last = Some(p);
            }
        }
        last
    }

    #[test]
    fn default_window_until_learned() {
        let mut w = RecycleWindow::new();
        assert!(feed(&mut w, 1, 4, (MIN_FRAMES - 1) as u64).is_none());
        assert_eq!(w.window_ns(), DEFAULT_RECYCLE_WINDOW_NS);
        assert!(w.is_stale(0, DEFAULT_RECYCLE_WINDOW_NS + 1));
        assert!(!w.is_stale(0, DEFAULT_RECYCLE_WINDOW_NS));
    }

    #[test]
    fn learns_pool_depth_and_period() {
        let mut w = RecycleWindow::new();
        let pool = feed(&mut w, 1, 6, 64).expect("learned");
        assert_eq!(pool.buffers, 6);
        assert_eq!(pool.period_ns, PERIOD);
        // 5 periods less a quarter period.
        assert_eq!(w.window_ns(), 5 * PERIOD - PERIOD / 4);
    }

    #[test]
    fn four_buffers_at_sixty_fps_is_under_fifty_ms() {
        let pool = CameraPool {
            buffers: 4,
            period_ns: 16_666_667,
        };
        assert_eq!(pool.window_ns(), 3 * 16_666_667 - 5_000_000);
    }

    #[test]
    fn reports_a_change_once() {
        let mut w = RecycleWindow::new();
        assert!(feed(&mut w, 1, 4, 64).is_some());
        assert!(feed(&mut w, 1, 4, 64).is_none());
    }

    #[test]
    fn camera_restart_relearns() {
        let mut w = RecycleWindow::new();
        feed(&mut w, 1, 4, 64);
        assert!(w.observe(2, 99, 5_000 * MS).is_none());
        assert_eq!(w.learned, None);
        assert_eq!(w.window_ns(), DEFAULT_RECYCLE_WINDOW_NS);
        assert_eq!(feed(&mut w, 2, 8, 64).map(|p| p.buffers), Some(8));
    }

    #[test]
    fn dropped_frames_do_not_move_the_median_period() {
        let mut w = RecycleWindow::new();
        let mut stamp = 1_000 * MS;
        for k in 0..64u64 {
            // Every fifth frame missing: a double interval.
            stamp += if k % 5 == 4 { 2 * PERIOD } else { PERIOD };
            w.observe(1, 40 + (k as i64 % 4), stamp);
        }
        assert_eq!(w.learned.map(|p| p.period_ns), Some(PERIOD));
    }

    #[test]
    fn gaps_and_steps_are_not_periods() {
        let mut w = RecycleWindow::new();
        feed(&mut w, 1, 4, 64);
        w.observe(1, 40, 1);
        w.observe(1, 41, 10_000_000 * MS);
        assert_eq!(w.learned.map(|p| p.period_ns), Some(PERIOD));
    }

    #[test]
    fn single_buffer_keeps_default() {
        let mut w = RecycleWindow::new();
        assert!(feed(&mut w, 1, 1, 64).is_none());
        assert_eq!(w.window_ns(), DEFAULT_RECYCLE_WINDOW_NS);
    }

    fn ring(stamps_ms: &[u64]) -> CameraRing<u32> {
        let mut r = CameraRing::new(vec![0u32; 4]);
        for ms in stamps_ms {
            r.replace_next(*ms as u32, ms * MS, Instant::now());
        }
        r
    }

    #[test]
    fn nearest_slot_by_stamp() {
        let mut r = ring(&[0, 33, 66, 100]);
        let (idx, delta) = r.nearest(40 * MS, Instant::now()).unwrap();
        assert_eq!(*r.get_mut(idx), 33);
        assert_eq!(delta, -7 * MS as i64);
    }

    #[test]
    fn overwrite_oldest_slot() {
        let mut r = ring(&[0, 33, 66, 100, 133]);
        let (idx, _) = r.nearest(0, Instant::now()).unwrap();
        assert_eq!(*r.get_mut(idx), 33);
    }

    #[test]
    fn replace_returns_previous_image() {
        let mut r = CameraRing::new(vec![7u32]);
        assert_eq!(r.replace_next(1, 0, Instant::now()), 7);
        assert_eq!(r.replace_next(2, MS, Instant::now()), 1);
    }

    #[test]
    fn empty_ring_has_no_nearest() {
        let r: CameraRing<u32> = CameraRing::new(vec![0; 2]);
        assert!(r.nearest(0, Instant::now()).is_none());
    }

    #[test]
    fn clear_forgets_all_slots() {
        let mut r = ring(&[0, 33]);
        r.clear();
        assert!(r.nearest(0, Instant::now()).is_none());
        assert!(r.newest_ns(Instant::now()).is_none());
    }

    #[test]
    fn slots_received_long_ago_are_never_selected() {
        let mut r = CameraRing::new(vec![0u32; 4]);
        let old = Instant::now()
            .checked_sub(STALE_AFTER + Duration::from_secs(1))
            .expect("host uptime > 3 s");
        r.replace_next(1, 100 * MS, old);
        r.replace_next(2, 0, Instant::now());
        let now = Instant::now();
        assert_eq!(r.newest_ns(now), Some(0));
        let (idx, _) = r.nearest(100 * MS, now).unwrap();
        assert_eq!(*r.get_mut(idx), 2);

        let later = now + STALE_AFTER + Duration::from_secs(1);
        assert!(r.nearest(0, later).is_none());
        assert!(r.newest_ns(later).is_none());
    }

    #[test]
    fn nearest_delta_does_not_wrap_for_extreme_target() {
        let r = ring(&[0]);
        let (_, delta) = r.nearest(u64::MAX, Instant::now()).unwrap();
        assert_eq!(delta, i64::MIN);
        let r = ring(&[0]);
        let (_, delta) = r.nearest(0, Instant::now()).unwrap();
        assert_eq!(delta, 0);
    }

    #[test]
    fn stale_warning_rearms_only_after_take() {
        let shared = SharedCameraRing::new(CameraRing::new(vec![0u32]));
        assert!(shared.record_stale());
        assert!(!shared.record_stale());
        assert_eq!(shared.take_stale(), 2);
        assert_eq!(shared.take_stale(), 0);
        assert!(shared.record_stale());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_returns_when_covering_frame_is_published() {
        let shared = Arc::new(SharedCameraRing::new(CameraRing::new(vec![0u32; 4])));
        shared.publish(1, 0, Instant::now());
        let writer = shared.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            writer.publish(2, 33 * MS, Instant::now());
        });
        let start = Instant::now();
        shared.wait_for_stamp(30 * MS, Duration::from_secs(2)).await;
        assert!(start.elapsed() < Duration::from_secs(1));
        let (idx, _) = shared.lock().nearest(30 * MS, Instant::now()).unwrap();
        assert_eq!(*shared.lock().get_mut(idx), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wait_times_out_without_covering_frame() {
        let shared = SharedCameraRing::new(CameraRing::new(vec![0u32; 4]));
        shared.publish(1, 0, Instant::now());
        let start = Instant::now();
        shared
            .wait_for_stamp(30 * MS, Duration::from_millis(30))
            .await;
        assert!(start.elapsed() >= Duration::from_millis(30));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wait_returns_immediately_when_already_covered() {
        let shared = SharedCameraRing::new(CameraRing::new(vec![0u32; 4]));
        shared.publish(1, 40 * MS, Instant::now());
        let start = Instant::now();
        shared.wait_for_stamp(30 * MS, Duration::from_secs(2)).await;
        assert!(start.elapsed() < Duration::from_millis(500));
    }
}
