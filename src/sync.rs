// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

//! Stamp-keyed buffers used to pair inputs by acquisition time.

use crate::stamp::{time_to_ns, Continuity, StampTimeline, STEP_THRESHOLD_NS};
use edgefirst_schemas::builtin_interfaces::Time;
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

/// Entries received longer ago than this are never paired.
pub const STALE_AFTER: Duration = Duration::from_secs(2);

#[derive(Debug)]
pub struct Stamped<T> {
    pub stamp: Time,
    pub stamp_ns: u64,
    pub received: Instant,
    pub data: Arc<T>,
}

impl<T> Clone for Stamped<T> {
    fn clone(&self) -> Self {
        Self {
            stamp: self.stamp,
            stamp_ns: self.stamp_ns,
            received: self.received,
            data: Arc::clone(&self.data),
        }
    }
}

#[derive(Debug)]
pub enum Selection<T> {
    Empty,
    /// Every buffered stamp is before the target; a later sample may be closer.
    Pending,
    Match {
        item: Stamped<T>,
        delta_ns: i64,
    },
    TooFar {
        delta_ns: i64,
    },
}

/// Bounded buffer ordered by arrival. Stamps within one topic are monotone
/// except across a clock step, which clears the buffer.
#[derive(Debug)]
pub struct StampedBuffer<T> {
    buf: VecDeque<Stamped<T>>,
    capacity: usize,
    timeline: StampTimeline,
    last_received: Option<Instant>,
}

impl<T> StampedBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
            timeline: StampTimeline::new(STEP_THRESHOLD_NS),
            last_received: None,
        }
    }

    pub fn push(&mut self, stamp: Time, received: Instant, data: Arc<T>) -> Continuity {
        let stamp_ns = time_to_ns(stamp);
        let continuity = self.timeline.observe(stamp_ns);
        self.last_received = Some(received);
        if matches!(continuity, Continuity::SteppedBack(_) | Continuity::Gap(_)) {
            self.buf.clear();
        }
        if self.buf.len() >= self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(Stamped {
            stamp,
            stamp_ns,
            received,
            data,
        });
        continuity
    }

    pub fn evict_received_before(&mut self, cutoff: Instant) {
        self.buf.retain(|s| s.received >= cutoff);
    }

    pub fn select(&self, target_ns: u64, max_delta_ns: u64, allow_pending: bool) -> Selection<T> {
        let Some(nearest) = self
            .buf
            .iter()
            .min_by_key(|s| s.stamp_ns.abs_diff(target_ns))
        else {
            // Entries are kept for `STALE_AFTER`, so an empty buffer means the
            // producer has been silent at least that long (or a clock step
            // just cleared it). Waiting would delay every output by the wait
            // while the producer is down.
            return Selection::Empty;
        };
        let newest_ns = self.buf.iter().map(|s| s.stamp_ns).max().unwrap_or(0);
        if allow_pending && newest_ns < target_ns {
            return Selection::Pending;
        }
        let delta_ns = (nearest.stamp_ns as i128 - target_ns as i128)
            .clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        if max_delta_ns == 0 || delta_ns.unsigned_abs() <= max_delta_ns {
            Selection::Match {
                item: nearest.clone(),
                delta_ns,
            }
        } else {
            Selection::TooFar { delta_ns }
        }
    }

    /// Receive time of the most recent push. Unaffected by eviction or a
    /// clock-step clear, so it keeps reporting how long the producer has
    /// been silent.
    pub fn newest_received(&self) -> Option<Instant> {
        self.last_received
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

/// Compute the cutoff instant for stale entry eviction.
/// Returns None if the monotonic clock is below STALE_AFTER.
pub fn stale_cutoff(now: Instant) -> Option<Instant> {
    now.checked_sub(STALE_AFTER)
}

/// Receive timeout for one topic that warns on silence, doubling from 2 s up
/// to 1 h and resetting when data arrives.
#[derive(Debug)]
pub struct SilenceWatch {
    topic: String,
    timeout: Duration,
    since: Instant,
}

impl SilenceWatch {
    const INITIAL: Duration = Duration::from_secs(2);
    const MAX: Duration = Duration::from_secs(3600);

    pub fn new(topic: impl Into<String>) -> Self {
        Self {
            topic: topic.into(),
            timeout: Self::INITIAL,
            since: Instant::now(),
        }
    }

    /// How long to wait for the next sample before calling [`Self::timed_out`].
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn heard(&mut self) {
        self.timeout = Self::INITIAL;
        self.since = Instant::now();
    }

    pub fn timed_out(&mut self) {
        log::warn!(
            "no {} for {:.0} s",
            self.topic,
            self.since.elapsed().as_secs_f32()
        );
        self.timeout = (self.timeout * 2).min(Self::MAX);
    }
}

/// A buffered input topic shared between a Zenoh callback (producer) and
/// fusion threads (consumers).
#[derive(Debug)]
pub struct SyncedTopic<T> {
    name: &'static str,
    buffer: std::sync::Mutex<StampedBuffer<T>>,
    notify: tokio::sync::Notify,
}

impl<T> SyncedTopic<T> {
    pub fn new(name: &'static str, capacity: usize) -> Self {
        Self {
            name,
            buffer: std::sync::Mutex::new(StampedBuffer::new(capacity)),
            notify: tokio::sync::Notify::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StampedBuffer<T>> {
        self.buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn push(&self, stamp: Time, data: T) {
        let continuity = self.lock().push(stamp, Instant::now(), Arc::new(data));
        match continuity {
            Continuity::SteppedBack(ns) => log::warn!(
                "{}: stamp stepped back by {:.3} s (clock step); pairing buffer cleared",
                self.name,
                ns as f64 * 1e-9
            ),
            Continuity::Gap(ns) => log::debug!(
                "{}: stamp jumped forward by {:.3} s; pairing buffer cleared",
                self.name,
                ns as f64 * 1e-9
            ),
            Continuity::First | Continuity::Continuous => {}
        }
        self.notify.notify_waiters();
    }

    /// Monotonic receive time of the most recently pushed sample, or `None`
    /// if nothing has been pushed.
    pub fn newest_received(&self) -> Option<Instant> {
        self.lock().newest_received()
    }

    /// Nearest buffered sample to `target_ns`, waiting up to `wait` for a
    /// sample stamped at or after the target so the nearest one is final.
    pub async fn select_nearest(
        &self,
        target_ns: u64,
        max_delta_ns: u64,
        wait: Duration,
    ) -> Selection<T> {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let notified = self.notify.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            let can_wait = tokio::time::Instant::now() < deadline;
            let selection = {
                let mut buf = self.lock();
                if let Some(cutoff) = stale_cutoff(Instant::now()) {
                    buf.evict_received_before(cutoff);
                }
                buf.select(target_ns, max_delta_ns, can_wait)
            };
            match selection {
                Selection::Pending => {
                    let _ = tokio::time::timeout_at(deadline, notified).await;
                }
                other => return other,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stamp::ns_to_time;

    const MS: u64 = 1_000_000;
    const BASE: u64 = 1_790_000_000_000_000_000;

    fn buf_with(stamps_ms: &[u64]) -> StampedBuffer<u32> {
        let mut b = StampedBuffer::new(8);
        for (i, ms) in stamps_ms.iter().enumerate() {
            b.push(
                ns_to_time(BASE + ms * MS),
                Instant::now(),
                Arc::new(i as u32),
            );
        }
        b
    }

    #[test]
    fn empty_buffer_selects_empty() {
        let b: StampedBuffer<u32> = StampedBuffer::new(4);
        assert!(matches!(b.select(BASE, 0, true), Selection::Empty));
    }

    #[test]
    fn selects_nearest_when_bracketed() {
        let b = buf_with(&[0, 33, 66, 100]);
        match b.select(BASE + 70 * MS, 50 * MS, true) {
            Selection::Match { item, delta_ns } => {
                assert_eq!(*item.data, 2);
                assert_eq!(delta_ns, -4 * MS as i64);
            }
            _ => panic!("expected match"),
        }
    }

    #[test]
    fn pending_until_a_later_sample_exists() {
        let b = buf_with(&[0, 33]);
        assert!(matches!(
            b.select(BASE + 50 * MS, 50 * MS, true),
            Selection::Pending
        ));
        assert!(matches!(
            b.select(BASE + 50 * MS, 50 * MS, false),
            Selection::Match { .. }
        ));
    }

    #[test]
    fn select_reports_too_far_and_zero_disables_gate() {
        let b = buf_with(&[0]);
        let target = BASE + 500 * MS;
        assert!(matches!(
            b.select(target, 100 * MS, false),
            Selection::TooFar { .. }
        ));
        assert!(matches!(
            b.select(target, 0, false),
            Selection::Match { .. }
        ));
    }

    #[test]
    fn capacity_drops_oldest() {
        let mut b = StampedBuffer::new(2);
        for ms in [0u64, 10, 20] {
            b.push(ns_to_time(BASE + ms * MS), Instant::now(), Arc::new(ms));
        }
        assert_eq!(b.len(), 2);
        match b.select(BASE, 0, false) {
            Selection::Match { item, .. } => assert_eq!(*item.data, 10),
            _ => panic!(),
        }
    }

    #[test]
    fn backward_step_clears_buffer() {
        let mut b = buf_with(&[0, 33, 66]);
        let c = b.push(
            ns_to_time(BASE - 3_600_000 * MS),
            Instant::now(),
            Arc::new(9),
        );
        assert!(matches!(c, Continuity::SteppedBack(_)));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn forward_gap_clears_buffer() {
        let mut b = buf_with(&[0, 33]);
        let c = b.push(ns_to_time(BASE + 5_000 * MS), Instant::now(), Arc::new(9));
        assert!(matches!(c, Continuity::Gap(_)));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn evict_drops_entries_received_long_ago() {
        let mut b = StampedBuffer::new(4);
        let old = Instant::now()
            .checked_sub(Duration::from_secs(5))
            .expect("host uptime > 2 s");
        b.push(ns_to_time(BASE), old, Arc::new(1u32));
        b.push(ns_to_time(BASE + 10 * MS), Instant::now(), Arc::new(2u32));
        b.evict_received_before(
            Instant::now()
                .checked_sub(STALE_AFTER)
                .expect("host uptime > 2 s"),
        );
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn newest_received_survives_eviction() {
        let mut b = StampedBuffer::new(4);
        let old = Instant::now()
            .checked_sub(Duration::from_secs(5))
            .expect("host uptime > 2 s");
        b.push(ns_to_time(BASE), old, Arc::new(1u32));
        b.evict_received_before(stale_cutoff(Instant::now()).expect("host uptime > 2 s"));
        assert!(b.is_empty());
        assert_eq!(b.newest_received(), Some(old));
    }

    #[test]
    fn stale_cutoff_is_before_now() {
        let now = Instant::now();
        let c = stale_cutoff(now).expect("host uptime > 2 s");
        assert_eq!(now.duration_since(c), STALE_AFTER);
    }

    #[test]
    fn select_delta_does_not_wrap_for_extreme_target() {
        let b = buf_with(&[0]);
        match b.select(u64::MAX, 0, false) {
            Selection::Match { delta_ns, .. } => assert_eq!(delta_ns, i64::MIN),
            other => panic!("expected match, got {other:?}"),
        }
    }

    #[test]
    fn silence_watch_backs_off_and_resets() {
        let mut w = SilenceWatch::new("radar/cube");
        assert_eq!(w.timeout(), Duration::from_secs(2));
        w.timed_out();
        assert_eq!(w.timeout(), Duration::from_secs(4));
        for _ in 0..20 {
            w.timed_out();
        }
        assert_eq!(w.timeout(), Duration::from_secs(3600));
        w.heard();
        assert_eq!(w.timeout(), Duration::from_secs(2));
    }

    #[test]
    fn newest_received_tracks_latest_push() {
        let topic = SyncedTopic::new("model", 4);
        assert!(topic.newest_received().is_none());
        let before = Instant::now();
        topic.push(ns_to_time(BASE), 1u32);
        topic.push(ns_to_time(BASE + 33 * MS), 2u32);
        let received = topic.newest_received().expect("buffered");
        assert!(received >= before && received <= Instant::now());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_returns_match_when_bracketing_sample_arrives() {
        let topic = Arc::new(SyncedTopic::new("model", 8));
        topic.push(ns_to_time(BASE), 1u32);
        let t2 = topic.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            t2.push(ns_to_time(BASE + 33 * MS), 2u32);
        });
        let sel = topic
            .select_nearest(BASE + 30 * MS, 50 * MS, Duration::from_millis(200))
            .await;
        match sel {
            Selection::Match { item, .. } => assert_eq!(*item.data, 2),
            other => panic!("expected match, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wait_times_out_to_nearest() {
        let topic = SyncedTopic::new("model", 8);
        topic.push(ns_to_time(BASE), 1u32);
        let start = Instant::now();
        let sel = topic
            .select_nearest(BASE + 20 * MS, 50 * MS, Duration::from_millis(30))
            .await;
        assert!(start.elapsed() >= Duration::from_millis(30));
        assert!(matches!(sel, Selection::Match { .. }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn two_waiters_both_wake() {
        let topic = Arc::new(SyncedTopic::new("model", 8));
        topic.push(ns_to_time(BASE), 0u32);
        let (a, b) = (topic.clone(), topic.clone());
        let wa = tokio::spawn(async move {
            a.select_nearest(BASE + 30 * MS, 50 * MS, Duration::from_secs(1))
                .await
        });
        let wb = tokio::spawn(async move {
            b.select_nearest(BASE + 30 * MS, 50 * MS, Duration::from_secs(1))
                .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        let start = Instant::now();
        topic.push(ns_to_time(BASE + 33 * MS), 1u32);
        for w in [wa, wb] {
            assert!(matches!(w.await.unwrap(), Selection::Match { .. }));
        }
        assert!(start.elapsed() < Duration::from_millis(500));
    }
}
