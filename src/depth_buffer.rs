// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

/// A 2D depth buffer at reduced resolution relative to the camera image.
///
/// For each cell, stores the minimum camera-space depth (cam_z) of any
/// projected point. Points that project to a cell but have significantly
/// greater depth than the stored minimum are considered occluded.
#[derive(Debug)]
pub struct DepthBuffer {
    /// Buffer storing minimum depth per cell (f32::MAX = empty).
    buf: Vec<f32>,
    width: usize,
    height: usize,
    /// Depth difference threshold: a point is occluded if its cam_z exceeds
    /// the cell minimum by more than this value (meters).
    occlusion_threshold: f32,
}

impl DepthBuffer {
    /// Create a new depth buffer.
    ///
    /// - `image_width`, `image_height`: full camera image dimensions
    /// - `divisor`: resolution reduction factor (e.g., 4 = 1/4 resolution)
    /// - `occlusion_threshold`: minimum depth difference (meters) to consider
    ///   a point occluded. A value of 1.0 means a point must be >1m behind
    ///   the nearest geometry to be rejected.
    pub fn new(
        image_width: u32,
        image_height: u32,
        divisor: u32,
        occlusion_threshold: f32,
    ) -> Self {
        let divisor = divisor.max(1);
        let width = (image_width / divisor).max(1) as usize;
        let height = (image_height / divisor).max(1) as usize;
        Self {
            buf: vec![f32::MAX; width * height],
            width,
            height,
            occlusion_threshold,
        }
    }

    /// Clear the buffer for a new frame. Must be called before each
    /// projection pass.
    pub fn clear(&mut self) {
        self.buf.fill(f32::MAX);
    }

    /// Record a point's depth at the given normalized image coordinates.
    /// `u`, `v` are in [0, 1]. `depth` is the camera-space z value (positive
    /// = in front of camera).
    ///
    /// Updates the cell minimum if this point is closer than any previously
    /// recorded point in the same cell.
    pub fn record(&mut self, u: f32, v: f32, depth: f32) {
        if let Some(idx) = self.cell_index(u, v) {
            if depth < self.buf[idx] {
                self.buf[idx] = depth;
            }
        }
    }

    /// Check whether a point at the given coordinates and depth is occluded.
    /// Returns `true` if the point is behind closer geometry by more than
    /// `occlusion_threshold`.
    ///
    /// Points in empty cells (no closer geometry recorded) are never occluded.
    pub fn is_occluded(&self, u: f32, v: f32, depth: f32) -> bool {
        if let Some(idx) = self.cell_index(u, v) {
            let min_depth = self.buf[idx];
            if min_depth < f32::MAX {
                return depth - min_depth > self.occlusion_threshold;
            }
        }
        false
    }

    /// Convert normalized [0,1] coordinates to a buffer cell index.
    fn cell_index(&self, u: f32, v: f32) -> Option<usize> {
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return None;
        }
        let col = (u * self.width as f32) as usize;
        let row = (v * self.height as f32) as usize;
        let col = col.min(self.width - 1);
        let row = row.min(self.height - 1);
        Some(row * self.width + col)
    }

    /// Buffer dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

/// Compute a classification confidence multiplier based on distance.
///
/// Returns 1.0 for distances <= `onset`, linearly decays to 0.0 at `max_dist`,
/// and returns 0.0 beyond `max_dist`. Returns 1.0 if decay is disabled
/// (onset <= 0 or onset >= max_dist).
///
/// Currently unused — the fusion pipeline applies a hard cutoff at `max_dist`
/// instead of soft decay. Retained for future soft-decay implementation.
#[allow(dead_code)]
pub fn distance_confidence(distance: f32, onset: f32, max_dist: f32) -> f32 {
    if onset <= 0.0 || onset >= max_dist || distance <= onset {
        return 1.0;
    }
    if distance >= max_dist {
        return 0.0;
    }
    1.0 - (distance - onset) / (max_dist - onset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_dimensions() {
        let db = DepthBuffer::new(640, 480, 4, 1.0);
        assert_eq!(db.dimensions(), (160, 120));
    }

    #[test]
    fn test_record_and_not_occluded() {
        let mut db = DepthBuffer::new(100, 100, 1, 1.0);
        db.record(0.5, 0.5, 10.0);
        // Same depth: not occluded
        assert!(!db.is_occluded(0.5, 0.5, 10.0));
        // Slightly behind (within threshold): not occluded
        assert!(!db.is_occluded(0.5, 0.5, 10.5));
    }

    #[test]
    fn test_occluded() {
        let mut db = DepthBuffer::new(100, 100, 1, 1.0);
        db.record(0.5, 0.5, 5.0); // foreground at 5m
                                  // 7m behind -> delta = 2m > threshold of 1m: occluded
        assert!(db.is_occluded(0.5, 0.5, 7.0));
    }

    #[test]
    fn test_closer_point_updates_minimum() {
        let mut db = DepthBuffer::new(100, 100, 1, 1.0);
        db.record(0.5, 0.5, 10.0);
        db.record(0.5, 0.5, 3.0); // closer point

        // 5m: delta from min(3.0) = 2.0 > 1.0 threshold: occluded
        assert!(db.is_occluded(0.5, 0.5, 5.0));
        // 3.5m: delta = 0.5 < 1.0 threshold: not occluded
        assert!(!db.is_occluded(0.5, 0.5, 3.5));
    }

    #[test]
    fn test_out_of_bounds() {
        let mut db = DepthBuffer::new(100, 100, 1, 1.0);
        db.record(2.0, 2.0, 5.0); // out of bounds, should be ignored
        assert!(!db.is_occluded(2.0, 2.0, 10.0)); // out of bounds, never occluded
    }

    #[test]
    fn test_clear() {
        let mut db = DepthBuffer::new(100, 100, 1, 1.0);
        db.record(0.5, 0.5, 5.0);
        db.clear();
        // After clear, no geometry recorded, so nothing is occluded
        assert!(!db.is_occluded(0.5, 0.5, 100.0));
    }

    #[test]
    fn test_empty_cell_not_occluded() {
        let db = DepthBuffer::new(100, 100, 1, 1.0);
        assert!(!db.is_occluded(0.5, 0.5, 100.0));
    }

    #[test]
    fn test_distance_confidence_within_onset() {
        assert_eq!(distance_confidence(10.0, 30.0, 50.0), 1.0);
        assert_eq!(distance_confidence(30.0, 30.0, 50.0), 1.0);
    }

    #[test]
    fn test_distance_confidence_decay() {
        let c = distance_confidence(40.0, 30.0, 50.0);
        assert!((c - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_distance_confidence_beyond_max() {
        assert_eq!(distance_confidence(60.0, 30.0, 50.0), 0.0);
    }

    #[test]
    fn test_distance_confidence_disabled() {
        assert_eq!(distance_confidence(100.0, 0.0, 50.0), 1.0);
    }

    #[test]
    fn test_distance_confidence_onset_equals_max() {
        // When onset == max_dist, decay is disabled (degenerate range)
        assert_eq!(distance_confidence(35.0, 50.0, 50.0), 1.0);
    }

    #[test]
    fn test_distance_confidence_onset_exceeds_max() {
        // Misconfiguration: onset > max_dist should disable decay
        assert_eq!(distance_confidence(35.0, 60.0, 50.0), 1.0);
    }
}
