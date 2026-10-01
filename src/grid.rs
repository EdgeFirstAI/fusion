// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

//! Occupancy-grid predictions from the fusion model, optionally tracked.

use crate::{
    args::Args,
    mask::Box2D,
    stamp::{mono_ns, time_to_ns, Continuity, StampTimeline, STEP_THRESHOLD_NS},
    sync::SyncedTopic,
    tracker::{ByteTrack, ByteTrackSettings, TrackerBox},
};
use edgefirst_schemas::builtin_interfaces::Time;
use std::sync::Arc;

/// Predictions derived from one fusion-model grid, as cell centres in metres.
#[derive(Debug, Clone, Default)]
pub struct GridFrame {
    pub predictions: Vec<Box2D>,
}

/// Grid predictions keyed by the stamp of the model input they came from.
pub type SharedGrid = Arc<SyncedTopic<GridFrame>>;

/// Centre of grid cell `(i, j)` in metres. Half of the grid width offsets
/// the `j` axis so the sensor sits on the centre column.
pub fn grid_to_xy(i: f32, j: f32, width: usize, args: &Args) -> (f32, f32) {
    let i_width = args.model_grid_size[0];
    let j_width = args.model_grid_size[1];

    if args.model_polar {
        let angle = -(width as f32) / 2.0 + j_width * (j + 0.5);
        let range = i_width * (i + 0.5);
        let x = (-angle).to_radians().cos() * range;
        let y = (-angle).to_radians().sin() * range;
        (x, y)
    } else {
        let x = i_width * (i + 0.5);
        let y = -(width as f32) / 2.0 + j_width * (j + 0.5);
        (x, y)
    }
}

/// One prediction per cell at or above `--model-threshold`.
pub fn raw_predictions(cells: &[Vec<f32>], args: &Args) -> Vec<Box2D> {
    let width = cells.first().map_or(0, Vec::len);
    let mut class = Vec::new();
    for (i, cells_i) in cells.iter().enumerate() {
        for (j, cell) in cells_i.iter().enumerate() {
            if *cell < args.model_threshold {
                continue;
            }
            let (x, y) = grid_to_xy(i as f32, j as f32, width, args);
            class.push(Box2D {
                center_x: x,
                center_y: y,
                width: args.model_grid_size[0],
                height: args.model_grid_size[1],
                label: 1,
            });
        }
    }
    class
}

/// ByteTrack over grid cells. Owned by the fusion-model thread so each grid
/// is tracked exactly once, in grid-index coordinates only.
pub struct GridTracker {
    tracker: ByteTrack,
    timeline: StampTimeline,
    track: bool,
    args: Args,
}

impl GridTracker {
    pub fn new(args: &Args) -> Self {
        Self {
            tracker: new_grid_bytetrack(args),
            timeline: StampTimeline::new(STEP_THRESHOLD_NS),
            track: args.track,
            args: args.clone(),
        }
    }

    /// Returns predictions for `cells` and, when tracking, the tracked-grid
    /// mask bytes (`[threshold, value]` pairs) for `…/tracked`.
    pub fn update(&mut self, cells: &[Vec<f32>], stamp: Time) -> (Vec<Box2D>, Option<Vec<u8>>) {
        if !self.track {
            return (raw_predictions(cells, &self.args), None);
        }
        if let Continuity::SteppedBack(ns) = self.timeline.observe(time_to_ns(stamp)) {
            log::warn!(
                "fusion grid stamp stepped back by {:.3} s (clock step); resetting grid tracks",
                ns as f64 * 1e-9
            );
            self.tracker = new_grid_bytetrack(&self.args);
        }
        let height = cells.len();
        let width = cells.first().map_or(0, Vec::len);

        let mut boxes = Vec::new();
        for (i, cells_i) in cells.iter().enumerate() {
            for (j, cell) in cells_i.iter().enumerate() {
                if *cell < self.args.model_threshold {
                    continue;
                }
                boxes.push(TrackerBox {
                    xmin: j as f32 - 1.0,
                    ymin: i as f32 - 1.0,
                    xmax: j as f32 + 1.0,
                    ymax: i as f32 + 1.0,
                    score: 1.0,
                    vision_class: 1,
                    fusion_class: 1,
                });
            }
        }
        self.tracker.update(&mut boxes, mono_ns());

        let mut tracked = vec![vec![0.0f64; width]; height];
        for tracklet in self.tracker.get_tracklets() {
            if tracklet.count < 3 {
                continue;
            }
            let pred = tracklet.get_predicted_location();
            let i = ((pred.ymin + pred.ymax) / 2.0).round() as i32;
            let j = ((pred.xmin + pred.xmax) / 2.0).round() as i32;
            if i < 0 || i >= height as i32 || j < 0 || j >= width as i32 {
                continue;
            }
            tracked[i as usize][j as usize] = 1.0;
        }
        let tracked_mask: Vec<u8> = tracked
            .iter()
            .flatten()
            .flat_map(|v| [128, (*v * 255.0).min(255.0) as u8])
            .collect();

        let mut predictions = Vec::new();
        for tracklet in self.tracker.get_tracklets() {
            if tracklet.count < 2 {
                continue;
            }
            let pred = tracklet.get_predicted_location();
            let i = (pred.ymin + pred.ymax) / 2.0;
            let j = (pred.xmin + pred.xmax) / 2.0;
            let (x, y) = grid_to_xy(i, j, width, &self.args);
            predictions.push(Box2D {
                center_x: x,
                center_y: y,
                width: self.args.model_grid_size[0],
                height: self.args.model_grid_size[1],
                label: 1,
            });
        }
        (predictions, Some(tracked_mask))
    }
}

fn new_grid_bytetrack(args: &Args) -> ByteTrack {
    ByteTrack::new_with_settings(ByteTrackSettings {
        track_high_conf: 0.5,
        track_extra_lifespan: args.track_extra_lifespan,
        track_iou: args.track_iou,
        track_update: args.track_update,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stamp::ns_to_time;
    use clap::Parser;

    const S: u64 = 1_790_000_000_000_000_000;

    fn args() -> Args {
        let mut a = Args::parse_from(["edgefirst-fusion", "--track", "--model-threshold", "0.5"]);
        a.normalize();
        a
    }

    fn cells_with(i: usize, j: usize) -> Vec<Vec<f32>> {
        let mut g = vec![vec![0.0; 8]; 8];
        g[i][j] = 1.0;
        g
    }

    #[test]
    fn raw_predictions_one_per_occupied_cell() {
        let p = raw_predictions(&cells_with(2, 3), &args());
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn tracked_predictions_need_two_updates() {
        let mut t = GridTracker::new(&args());
        let (p1, _) = t.update(&cells_with(2, 3), ns_to_time(S));
        assert!(p1.is_empty());
        let (p2, mask) = t.update(&cells_with(2, 3), ns_to_time(S + 55_000_000));
        assert_eq!(p2.len(), 1);
        assert!(mask.is_some());
    }

    #[test]
    fn grid_tracker_resets_on_backward_stamp() {
        let mut t = GridTracker::new(&args());
        t.update(&cells_with(2, 3), ns_to_time(S));
        t.update(&cells_with(2, 3), ns_to_time(S + 55_000_000));
        // Clock stepped back one hour: the tracker starts over instead of stalling.
        let (p, _) = t.update(&cells_with(2, 3), ns_to_time(S - 3_600_000_000_000));
        assert!(p.is_empty());
        let (p, _) = t.update(
            &cells_with(2, 3),
            ns_to_time(S - 3_600_000_000_000 + 55_000_000),
        );
        assert_eq!(p.len(), 1);
    }
}
