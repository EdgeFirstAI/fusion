// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

//! Rolling pairing and latency statistics, logged at `--stats-interval`.

use std::collections::VecDeque;

const WINDOW: usize = 256;

/// Rolling window of signed nanosecond samples summarised as median and
/// largest magnitude.
#[derive(Debug)]
pub struct DelayStats {
    name: &'static str,
    values: VecDeque<i64>,
}

impl DelayStats {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            values: VecDeque::with_capacity(WINDOW),
        }
    }

    pub fn record_ns(&mut self, v: i64) {
        if self.values.len() == WINDOW {
            self.values.pop_front();
        }
        self.values.push_back(v);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// `name p50=…ms max=…ms`, where max is the largest absolute value.
    pub fn summary(&self) -> Option<String> {
        if self.values.is_empty() {
            return None;
        }
        let mut sorted: Vec<i64> = self.values.iter().copied().collect();
        sorted.sort_unstable();
        let p50 = sorted[sorted.len() / 2];
        let max = sorted.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
        Some(format!(
            "{} p50={:.1}ms max={:.1}ms",
            self.name,
            p50 as f64 * 1e-6,
            max as f64 * 1e-6
        ))
    }
}

/// Per-thread pairing counters. A point-cloud thread counts `model/output`
/// selections and grid selections separately; the fusion-model thread counts
/// radar-cube and camera-frame pairing and stale conversions.
#[derive(Debug)]
pub struct PairStats {
    cube_camera: bool,
    pub paired: u64,
    pub too_far: u64,
    pub missing: u64,
    /// Camera frames converted later than the camera buffer pool guarantees.
    pub stale: u64,
    pub grid_too_far: u64,
    pub grid_missing: u64,
    /// Too-far warning already logged; re-armed after each stats interval
    /// that counted a too-far pairing.
    pub warned_too_far: bool,
    pub warned_grid_too_far: bool,
    /// Pairing stamp difference (input − target).
    pub delta_ns: DelayStats,
    /// Grid stamp − point-cloud stamp.
    pub grid_delta_ns: DelayStats,
    /// Time spent waiting for a bracketing sample.
    pub waited_ns: DelayStats,
    /// Receive wall time − point-cloud stamp.
    pub latency_ns: DelayStats,
}

impl PairStats {
    fn new(cube_camera: bool) -> Self {
        Self {
            cube_camera,
            paired: 0,
            too_far: 0,
            missing: 0,
            stale: 0,
            grid_too_far: 0,
            grid_missing: 0,
            warned_too_far: false,
            warned_grid_too_far: false,
            delta_ns: DelayStats::new("delta"),
            grid_delta_ns: DelayStats::new("grid_delta"),
            waited_ns: DelayStats::new("wait"),
            latency_ns: DelayStats::new("latency"),
        }
    }

    /// Point cloud paired with `model/output` and the fusion grid.
    pub fn point_cloud() -> Self {
        Self::new(false)
    }

    /// Radar cube paired with a converted camera frame.
    pub fn cube_camera() -> Self {
        Self::new(true)
    }

    fn line(&self, label: &str) -> String {
        let mut line = format!(
            "[{label}] paired={} too_far={} missing={}",
            self.paired, self.too_far, self.missing
        );
        if self.cube_camera {
            line += &format!(" stale={}", self.stale);
        } else {
            line += &format!(
                " grid_too_far={} grid_missing={}",
                self.grid_too_far, self.grid_missing
            );
        }
        let delays = [
            &self.delta_ns,
            &self.grid_delta_ns,
            &self.waited_ns,
            &self.latency_ns,
        ];
        for summary in delays.iter().filter_map(|s| s.summary()) {
            line.push(' ');
            line += &summary;
        }
        line
    }

    /// Log counters and delay summaries, then start a new window.
    pub fn log_and_reset(&mut self, label: &str) {
        log::info!("{}", self.line(label));
        if self.too_far > 0 {
            self.warned_too_far = false;
        }
        if self.grid_too_far > 0 {
            self.warned_grid_too_far = false;
        }
        self.paired = 0;
        self.too_far = 0;
        self.missing = 0;
        self.stale = 0;
        self.grid_too_far = 0;
        self.grid_missing = 0;
        self.delta_ns.clear();
        self.grid_delta_ns.clear();
        self.waited_ns.clear();
        self.latency_ns.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_stats_median_and_max() {
        let mut s = DelayStats::new("delta");
        for v in [5_000_000i64, -1_000_000, 3_000_000] {
            s.record_ns(v);
        }
        let line = s.summary().unwrap();
        assert!(line.contains("delta"));
        assert!(line.contains("p50=3.0ms"), "{line}");
        assert!(line.contains("max=5.0ms"), "{line}");
    }

    #[test]
    fn empty_stats_have_no_summary() {
        assert!(DelayStats::new("x").summary().is_none());
    }

    #[test]
    fn window_is_bounded() {
        let mut s = DelayStats::new("x");
        for i in 0..1000 {
            s.record_ns(i);
        }
        assert_eq!(s.len(), 256);
    }

    #[test]
    fn log_and_reset_clears_interval_counters() {
        let mut s = PairStats {
            paired: 3,
            too_far: 2,
            missing: 1,
            grid_too_far: 4,
            grid_missing: 5,
            warned_too_far: true,
            warned_grid_too_far: true,
            ..PairStats::point_cloud()
        };
        s.delta_ns.record_ns(1);
        s.grid_delta_ns.record_ns(1);
        s.log_and_reset("test");
        assert_eq!((s.paired, s.too_far, s.missing), (0, 0, 0));
        assert_eq!((s.grid_too_far, s.grid_missing), (0, 0));
        assert!(!s.warned_too_far && !s.warned_grid_too_far);
        assert!(s.delta_ns.summary().is_none());
        assert!(s.grid_delta_ns.summary().is_none());
    }

    #[test]
    fn warnings_rearm_only_after_an_interval_with_skips() {
        let mut s = PairStats::point_cloud();
        s.too_far = 1;
        s.warned_too_far = true;
        s.warned_grid_too_far = true;
        s.log_and_reset("test");
        assert!(!s.warned_too_far);
        assert!(s.warned_grid_too_far, "no grid skip in the interval");
    }

    #[test]
    fn point_cloud_line_has_grid_counters() {
        let mut s = PairStats::point_cloud();
        s.grid_too_far = 2;
        s.grid_missing = 3;
        s.grid_delta_ns.record_ns(-144_000_000);
        let line = s.line("fusion/radar");
        assert!(
            line.starts_with(
                "[fusion/radar] paired=0 too_far=0 missing=0 grid_too_far=2 grid_missing=3"
            ),
            "{line}"
        );
        assert!(line.contains("grid_delta p50=-144.0ms"), "{line}");
        assert!(!line.contains("stale"), "{line}");
    }

    #[test]
    fn cube_camera_line_has_stale() {
        let mut s = PairStats::cube_camera();
        s.stale = 1;
        let line = s.line("fusion/model cube↔camera");
        assert!(
            line.ends_with("paired=0 too_far=0 missing=0 stale=1"),
            "{line}"
        );
    }
}
