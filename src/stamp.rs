// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

//! Time conversions shared by every publisher and pairing stage.
//!
//! Published instants are CLOCK_REALTIME acquisition stamps; durations,
//! tracker lifetimes and expiry use CLOCK_MONOTONIC so a wall-clock step
//! cannot stall or expire them.

use edgefirst_schemas::builtin_interfaces::Time;
use std::{
    sync::OnceLock,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use zenoh::{
    time::{Timestamp, TimestampId, NTP64},
    Session,
};

const NSEC_PER_SEC: u64 = 1_000_000_000;

/// Stamp discontinuity treated as a clock step rather than jitter or reordering.
pub const STEP_THRESHOLD_NS: u64 = NSEC_PER_SEC;

/// Nanoseconds since the Unix epoch; pre-epoch stamps saturate to 0.
pub fn time_to_ns(t: Time) -> u64 {
    t.to_nanos().unwrap_or(0)
}

/// Inverse of [`time_to_ns`], saturating at `i32::MAX` seconds.
pub fn ns_to_time(ns: u64) -> Time {
    let sec = ns / NSEC_PER_SEC;
    if sec > i32::MAX as u64 {
        return Time {
            sec: i32::MAX,
            nanosec: 999_999_999,
        };
    }
    Time {
        sec: sec as i32,
        nanosec: (ns % NSEC_PER_SEC) as u32,
    }
}

/// Apply a signed offset in seconds, saturating at 0 and `u64::MAX`.
pub fn offset_ns(ns: u64, offset_s: f32) -> u64 {
    let off = (offset_s as f64 * NSEC_PER_SEC as f64) as i64;
    if off >= 0 {
        ns.saturating_add(off as u64)
    } else {
        ns.saturating_sub(off.unsigned_abs())
    }
}

/// Identifier attached to every Zenoh timestamp this session publishes.
pub fn timestamp_id(session: &Session) -> TimestampId {
    *session.new_timestamp().get_id()
}

/// Zenoh sample timestamp carrying `stamp`, so storage and alignment see the
/// acquisition instant rather than the publish instant.
pub fn zenoh_timestamp(id: TimestampId, stamp: Time) -> Timestamp {
    Timestamp::new(NTP64::from(Duration::from_nanos(time_to_ns(stamp))), id)
}

/// Current CLOCK_REALTIME as a stamp; pre-epoch clocks saturate to 0.
pub fn now_stamp() -> Time {
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos().min(u64::MAX as u128) as u64)
        .unwrap_or(0);
    ns_to_time(ns)
}

/// CLOCK_MONOTONIC nanoseconds since the first call in this process.
pub fn mono_ns() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// How a stamp relates to the previous one on the same stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Continuity {
    First,
    Continuous,
    /// Moved backwards by more than the threshold (ns): a clock step.
    SteppedBack(u64),
    /// Moved forward by more than the threshold (ns): a forward step or a
    /// sensor dropout; both leave buffered data unusable for pairing.
    Gap(u64),
}

/// Classifies successive stamps on one stream.
#[derive(Debug)]
pub struct StampTimeline {
    last: Option<u64>,
    threshold_ns: u64,
}

impl StampTimeline {
    pub fn new(threshold_ns: u64) -> Self {
        Self {
            last: None,
            threshold_ns,
        }
    }

    pub fn observe(&mut self, ns: u64) -> Continuity {
        let Some(last) = self.last else {
            self.last = Some(ns);
            return Continuity::First;
        };
        if ns.saturating_add(self.threshold_ns) < last {
            self.last = Some(ns);
            return Continuity::SteppedBack(last - ns);
        }
        if ns > last.saturating_add(self.threshold_ns) {
            self.last = Some(ns);
            return Continuity::Gap(ns - last);
        }
        self.last = Some(last.max(ns));
        Continuity::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_ns_round_trip() {
        let t = Time {
            sec: 1_790_000_000,
            nanosec: 123_456_789,
        };
        assert_eq!(ns_to_time(time_to_ns(t)), t);
    }

    #[test]
    fn pre_epoch_saturates_to_zero() {
        assert_eq!(
            time_to_ns(Time {
                sec: -5,
                nanosec: 7
            }),
            0
        );
    }

    #[test]
    fn ns_to_time_saturates() {
        let t = ns_to_time(u64::MAX);
        assert_eq!(t.sec, i32::MAX);
    }

    #[test]
    fn offset_is_signed_and_saturating() {
        assert_eq!(offset_ns(1_000, 0.0), 1_000);
        assert_eq!(offset_ns(1_000_000_000, -0.5), 500_000_000);
        assert_eq!(offset_ns(1_000_000_000, 0.25), 1_250_000_000);
        assert_eq!(offset_ns(10, -1.0), 0);
    }

    #[test]
    fn zenoh_timestamp_matches_stamp_within_2ns() {
        let id = TimestampId::try_from([1u8; 16]).unwrap();
        for nanosec in [0u32, 1, 999_999_999, 123_456_789] {
            let stamp = Time {
                sec: 1_790_000_000,
                nanosec,
            };
            let ts = zenoh_timestamp(id, stamp);
            let got = ts.get_time().to_duration().as_nanos() as u64;
            assert!(
                got.abs_diff(time_to_ns(stamp)) <= 2,
                "nanosec={nanosec} got={got}"
            );
            assert_eq!(*ts.get_id(), id);
        }
    }

    #[test]
    fn timeline_classifies_steps_and_gaps() {
        let mut tl = StampTimeline::new(STEP_THRESHOLD_NS);
        assert_eq!(tl.observe(10_000_000_000), Continuity::First);
        assert_eq!(tl.observe(10_050_000_000), Continuity::Continuous);
        // Small reordering below the threshold is not a step.
        assert_eq!(tl.observe(10_040_000_000), Continuity::Continuous);
        assert_eq!(
            tl.observe(5_000_000_000),
            Continuity::SteppedBack(5_050_000_000)
        );
        assert_eq!(tl.observe(9_000_000_000), Continuity::Gap(4_000_000_000));
    }

    #[test]
    fn mono_ns_is_nondecreasing() {
        let a = mono_ns();
        let b = mono_ns();
        assert!(b >= a);
    }

    #[test]
    fn timeline_does_not_overflow_near_u64_max() {
        let mut tl = StampTimeline::new(STEP_THRESHOLD_NS);
        assert_eq!(tl.observe(10_000_000_000), Continuity::First);
        assert_eq!(
            tl.observe(u64::MAX),
            Continuity::Gap(u64::MAX - 10_000_000_000)
        );
        assert_eq!(tl.observe(u64::MAX), Continuity::Continuous);
        assert_eq!(
            tl.observe(10_000_000_000),
            Continuity::SteppedBack(u64::MAX - 10_000_000_000)
        );
    }
}
