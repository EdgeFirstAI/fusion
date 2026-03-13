// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use edgefirst_schemas::builtin_interfaces::Time;
use std::collections::VecDeque;

/// Converts a ROS2 Time to nanoseconds since epoch.
pub fn time_to_nanos(t: &Time) -> i64 {
    (t.sec as i64) * 1_000_000_000 + (t.nanosec as i64)
}

/// Absolute difference between two timestamps in nanoseconds.
pub fn time_delta_nanos(a: &Time, b: &Time) -> u64 {
    let a_ns = time_to_nanos(a);
    let b_ns = time_to_nanos(b);
    a_ns.abs_diff(b_ns)
}

/// Converts a nanosecond delta to seconds (f64).
pub fn nanos_to_secs(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000_000.0
}

/// Absolute difference between two timestamps in seconds.
pub fn time_delta_secs(a: &Time, b: &Time) -> f64 {
    nanos_to_secs(time_delta_nanos(a, b))
}

/// A timestamped item in the ring buffer.
#[derive(Debug)]
pub struct Stamped<T> {
    pub stamp: Time,
    pub data: T,
}

/// A bounded ring buffer of timestamped messages.
#[derive(Debug)]
pub struct TimestampedBuffer<T> {
    buf: VecDeque<Stamped<T>>,
    capacity: usize,
}

impl<T> TimestampedBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "buffer capacity must be > 0");
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a new item. Drops the oldest if at capacity.
    pub fn push(&mut self, stamp: Time, data: T) {
        if self.buf.len() >= self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(Stamped { stamp, data });
    }

    /// Find the item with the closest timestamp to `target`.
    /// Returns None if the buffer is empty.
    pub fn closest(&self, target: &Time) -> Option<&Stamped<T>> {
        self.buf
            .iter()
            .min_by_key(|item| time_delta_nanos(&item.stamp, target))
    }

    /// Number of items currently buffered.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Most recent item (by insertion order).
    pub fn latest(&self) -> Option<&Stamped<T>> {
        self.buf.back()
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

/// Rolling statistics for a single input topic.
#[derive(Debug)]
pub struct TopicStats {
    /// Topic name for logging.
    name: String,
    /// Timestamp of the last received message.
    last_stamp: Option<Time>,
    /// Inter-message intervals in nanoseconds (rolling window).
    intervals: VecDeque<u64>,
    /// Maximum window size for rolling stats.
    window: usize,
    /// Count of messages received.
    pub count: u64,
}

impl TopicStats {
    pub fn new(name: impl Into<String>, window: usize) -> Self {
        Self {
            name: name.into(),
            last_stamp: None,
            intervals: VecDeque::with_capacity(window),
            window,
            count: 0,
        }
    }

    /// Record a new message arrival. Call with the message's header.stamp.
    pub fn record(&mut self, stamp: &Time) {
        if let Some(prev) = &self.last_stamp {
            let delta = time_delta_nanos(prev, stamp);
            if self.intervals.len() >= self.window {
                self.intervals.pop_front();
            }
            self.intervals.push_back(delta);
        }
        self.last_stamp = Some(stamp.clone());
        self.count += 1;
    }

    /// Minimum inter-message interval in seconds, or None if < 2 messages.
    pub fn min_interval_secs(&self) -> Option<f64> {
        self.intervals.iter().min().map(|&v| nanos_to_secs(v))
    }

    /// Maximum inter-message interval in seconds, or None if < 2 messages.
    pub fn max_interval_secs(&self) -> Option<f64> {
        self.intervals.iter().max().map(|&v| nanos_to_secs(v))
    }

    /// Average inter-message interval in seconds, or None if < 2 messages.
    pub fn avg_interval_secs(&self) -> Option<f64> {
        if self.intervals.is_empty() {
            return None;
        }
        let sum: u64 = self.intervals.iter().sum();
        Some(nanos_to_secs(sum) / self.intervals.len() as f64)
    }

    /// Log a summary of the statistics.
    pub fn log_summary(&self) {
        if let (Some(min), Some(max), Some(avg)) = (
            self.min_interval_secs(),
            self.max_interval_secs(),
            self.avg_interval_secs(),
        ) {
            log::info!(
                "[{}] count={}, interval min={:.1}ms avg={:.1}ms max={:.1}ms",
                self.name,
                self.count,
                min * 1000.0,
                avg * 1000.0,
                max * 1000.0,
            );
        } else {
            log::info!(
                "[{}] count={}, insufficient data for interval stats",
                self.name,
                self.count,
            );
        }
    }

    /// Topic name accessor.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_time(sec: i32, nanosec: u32) -> Time {
        Time { sec, nanosec }
    }

    #[test]
    fn test_time_to_nanos() {
        let t = make_time(1, 500_000_000);
        assert_eq!(time_to_nanos(&t), 1_500_000_000);
    }

    #[test]
    fn test_time_delta_nanos() {
        let a = make_time(1, 0);
        let b = make_time(1, 100_000_000); // 100ms later
        assert_eq!(time_delta_nanos(&a, &b), 100_000_000);
        // Order-independent
        assert_eq!(time_delta_nanos(&b, &a), 100_000_000);
    }

    #[test]
    fn test_buffer_push_and_capacity() {
        let mut buf: TimestampedBuffer<i32> = TimestampedBuffer::new(3);
        buf.push(make_time(1, 0), 10);
        buf.push(make_time(2, 0), 20);
        buf.push(make_time(3, 0), 30);
        assert_eq!(buf.len(), 3);

        // Push one more, oldest should be evicted
        buf.push(make_time(4, 0), 40);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.buf.front().unwrap().data, 20);
    }

    #[test]
    fn test_closest_exact_match() {
        let mut buf: TimestampedBuffer<&str> = TimestampedBuffer::new(5);
        buf.push(make_time(1, 0), "a");
        buf.push(make_time(2, 0), "b");
        buf.push(make_time(3, 0), "c");

        let closest = buf.closest(&make_time(2, 0)).unwrap();
        assert_eq!(closest.data, "b");
    }

    #[test]
    fn test_closest_nearest() {
        let mut buf: TimestampedBuffer<&str> = TimestampedBuffer::new(5);
        buf.push(make_time(1, 0), "a"); // t=1.0s
        buf.push(make_time(2, 0), "b"); // t=2.0s
        buf.push(make_time(3, 0), "c"); // t=3.0s

        // t=2.3s is closest to "b" (t=2.0s, delta=0.3s) vs "c" (t=3.0s, delta=0.7s)
        let closest = buf.closest(&make_time(2, 300_000_000)).unwrap();
        assert_eq!(closest.data, "b");

        // t=2.8s is closest to "c" (t=3.0s, delta=0.2s)
        let closest = buf.closest(&make_time(2, 800_000_000)).unwrap();
        assert_eq!(closest.data, "c");
    }

    #[test]
    fn test_closest_empty_buffer() {
        let buf: TimestampedBuffer<i32> = TimestampedBuffer::new(3);
        assert!(buf.closest(&make_time(1, 0)).is_none());
    }

    #[test]
    fn test_latest() {
        let mut buf: TimestampedBuffer<i32> = TimestampedBuffer::new(3);
        buf.push(make_time(1, 0), 10);
        buf.push(make_time(2, 0), 20);
        assert_eq!(buf.latest().unwrap().data, 20);
    }

    #[test]
    fn test_topic_stats_intervals() {
        let mut stats = TopicStats::new("test_topic", 10);
        stats.record(&make_time(1, 0));
        assert!(stats.avg_interval_secs().is_none()); // Need at least 2

        stats.record(&make_time(1, 100_000_000)); // +100ms
        stats.record(&make_time(1, 250_000_000)); // +150ms
        stats.record(&make_time(1, 350_000_000)); // +100ms

        assert_eq!(stats.count, 4);
        assert!((stats.min_interval_secs().unwrap() - 0.1).abs() < 0.001);
        assert!((stats.max_interval_secs().unwrap() - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_time_delta_secs() {
        let a = make_time(1, 0);
        let b = make_time(1, 100_000_000);
        assert!((time_delta_secs(&a, &b) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_closest_single_element() {
        let mut buf: TimestampedBuffer<&str> = TimestampedBuffer::new(5);
        buf.push(make_time(5, 0), "only");
        let closest = buf.closest(&make_time(1, 0)).unwrap();
        assert_eq!(closest.data, "only");
    }

    #[test]
    fn test_closest_out_of_order_insertion() {
        let mut buf: TimestampedBuffer<&str> = TimestampedBuffer::new(5);
        buf.push(make_time(3, 0), "c");
        buf.push(make_time(1, 0), "a"); // inserted out of chronological order
        buf.push(make_time(2, 0), "b");

        // Should still find "a" as closest to t=1.0s
        let closest = buf.closest(&make_time(1, 0)).unwrap();
        assert_eq!(closest.data, "a");

        // Should find "b" as closest to t=2.3s
        let closest = buf.closest(&make_time(2, 300_000_000)).unwrap();
        assert_eq!(closest.data, "b");
    }

    #[test]
    fn test_topic_stats_rolling_window() {
        let mut stats = TopicStats::new("test", 3);
        stats.record(&make_time(0, 0));
        stats.record(&make_time(1, 0)); // interval: 1s
        stats.record(&make_time(2, 0)); // interval: 1s
        stats.record(&make_time(3, 0)); // interval: 1s
        stats.record(&make_time(3, 100_000_000)); // interval: 0.1s — pushes out oldest

        // Window should contain [1s, 1s, 0.1s]
        assert_eq!(stats.intervals.len(), 3);
        assert!((stats.min_interval_secs().unwrap() - 0.1).abs() < 0.001);
    }
}
