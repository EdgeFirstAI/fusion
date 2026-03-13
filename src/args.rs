// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use clap::{Parser, ValueEnum};
use serde_json::json;
use std::path::PathBuf;
use zenoh::config::{Config, WhatAmI};

/// Parse a path that may be empty. An empty string is accepted and stored as an
/// empty PathBuf, which `Args::normalize()` later converts to `None`.
fn parse_optional_path(s: &str) -> Result<PathBuf, std::convert::Infallible> {
    Ok(PathBuf::from(s))
}

#[derive(Debug, Clone, ValueEnum, Copy, Eq, PartialEq)]
pub enum PCDSource {
    Disabled,
    Radar,
    Lidar,
}
#[derive(Debug, Clone, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// radar input topic. leave empty to disable radar fusion
    #[arg(long, env, default_value = "")]
    pub radar_pcd_topic: String,

    /// lidar input topic. leave empty to disable lidar fusion
    #[arg(long, env, default_value = "")]
    pub lidar_pcd_topic: String,

    /// lidar fusion output topic. leave empty to disable
    #[arg(long, env, default_value = "rt/fusion/lidar")]
    pub lidar_output_topic: String,

    /// radar fusion output topic. leave empty to disable
    #[arg(long, env, default_value = "rt/fusion/radar")]
    pub radar_output_topic: String,

    /// camera info input topic
    #[arg(long, env, default_value = "rt/camera/info")]
    pub info_topic: String,

    /// unified vision model output topic. leave empty to disable
    #[arg(long, env, default_value = "rt/model/output")]
    pub vision_model_topic: String,

    /// model info topic for label resolution. leave empty to disable
    #[arg(long, env, default_value = "rt/model/info")]
    pub model_info_topic: String,

    /// Maximum age in seconds for model output data. At 50% of this threshold
    /// an info-level warning is logged. At 100% the model output is dropped
    /// and a warning is logged. 0 = disabled.
    #[arg(long, env, default_value = "0.5")]
    pub max_model_age: f32,

    /// Buffer size for model output messages (ring buffer depth)
    #[arg(long, env, default_value = "5")]
    pub model_buffer_size: usize,

    /// Maximum temporal delta in seconds between paired PCD and model inputs.
    /// Pairs exceeding this threshold are skipped with a warning. This is a
    /// stricter per-frame pairing gate than max_model_age; when both are
    /// enabled the lower of the two is the effective limit. 0 = disabled.
    #[arg(long, env, default_value = "0.1")]
    pub max_temporal_delta: f32,

    /// Interval in seconds for logging per-topic latency statistics. 0 = disabled.
    #[arg(long, env, default_value = "10.0")]
    pub stats_interval: f32,

    /// Enable depth-buffer occlusion rejection during projection.
    /// Points occluded by closer geometry (from the camera's perspective) will
    /// not receive a camera-derived class label.
    #[arg(long, env, default_value_t = true)]
    pub occlusion_filter: bool,

    /// Depth difference threshold in meters for occlusion rejection.
    /// A point is occluded if its depth exceeds the nearest geometry by more
    /// than this value. Higher values are more permissive.
    #[arg(long, env, default_value = "1.0")]
    pub occlusion_threshold: f32,

    /// Resolution divisor for the occlusion depth buffer relative to the camera
    /// image. 4 means the depth buffer is 1/4 the camera resolution in each
    /// dimension. Higher values use less memory but reduce occlusion precision.
    #[arg(long, env, default_value = "4")]
    pub occlusion_resolution_divisor: u32,

    /// Distance in meters beyond which classification confidence begins to decay.
    /// Currently used only as a precursor to the hard cutoff at
    /// confidence_decay_max. Reserved for future soft-decay implementation.
    /// 0 = distance filtering disabled entirely.
    #[arg(long, env, default_value = "30.0")]
    pub confidence_decay_onset: f32,

    /// Maximum classification distance in meters. Points beyond this distance
    /// are excluded from camera-based classification (proj coords set to
    /// PROJ_SKIP). Must be greater than confidence_decay_onset.
    #[arg(long, env, default_value = "50.0")]
    pub confidence_decay_max: f32,

    /// bbox3d output topic
    #[arg(long, env, default_value = "rt/fusion/boxes3d")]
    pub bbox3d_topic: String,

    /// bbox3d source
    #[arg(long, env, default_value = "lidar")]
    pub bbox3d_src: PCDSource,

    /// camera DMA buffers input topic
    #[arg(long, env, default_value = "rt/camera/dma")]
    pub camera_topic: String,

    /// radarcube input topic
    #[arg(long, env, default_value = "rt/radar/cube")]
    pub radarcube_topic: String,

    /// radar model output
    #[arg(long, env, default_value = "rt/fusion/model_output")]
    pub model_output_topic: String,

    /// model, leave empty to disable
    #[arg(short, long, env, value_parser = parse_optional_path)]
    pub model: Option<PathBuf>,

    /// model decoder
    #[arg(long, env, value_parser = parse_optional_path)]
    pub model_decoder: Option<PathBuf>,

    /// set the model to be polar
    #[arg(long, env, action)]
    pub model_polar: bool,

    /// Model threshold for the model output topic
    #[arg(long, env, default_value = "0.5")]
    pub model_threshold: f32,

    /// model grid size. This is the real world size (length, width) in meters
    /// of a 1x1 model output grid cell. If the model is polar, the width is
    /// in degrees.
    #[arg(long, env, value_delimiter = ' ', default_value = "1 1")]
    pub model_grid_size: Vec<f32>,

    /// engine for model context
    #[arg(long, env, default_value = "npu")]
    pub engine: String,

    /// apply sigmoid the model output
    #[arg(long, env, default_value = "true")]
    pub logits: bool,

    /// enable tracking to reduce flickering in model output
    #[arg(long, env, action)]
    pub track: bool,

    /// The number of seconds the tracked object can be missing for before being
    /// removed.
    #[arg(long, env, default_value = "0.5")]
    pub track_extra_lifespan: f32,

    /// Tracking iou threshold for box association. Higher values will require
    /// boxes to have higher IOU to the predicted track to be associated.
    #[arg(long, env, default_value = "0.1")]
    pub track_iou: f32,

    /// Higher update factor will also mean
    /// less smoothing but more rapid response to change (0.0 to 1.0)
    #[arg(long, env, default_value = "0.4")]
    pub track_update: f32,

    /// Occupancy grid output topic
    #[arg(long, env, default_value = "rt/fusion/occupancy")]
    pub grid_topic: String,

    /// Occupancy grid source
    #[arg(long, env, default_value = "radar")]
    pub grid_src: PCDSource,

    /// Used for model grid. Used for occupancy if input PCD
    /// does not have cluster_id field.
    #[arg(long, env, num_args = 2, value_delimiter = ' ', default_value = "0 16")]
    pub range_bin_limit: Vec<f32>,

    /// Used for model grid. Used for occupancy if input PCD
    /// does not have cluster_id field.
    #[arg(long, env, default_value = "1.0")]
    pub range_bin_width: f32,

    /// angle_bin_limit, 0 degrees is forwards. Used for occupancy if input PCD
    /// does not have cluster_id field. Used for model grid if model_polar.
    /// is true
    #[arg(
        long,
        env,
        num_args = 2,
        value_delimiter = ' ',
        default_value = "-55 55"
    )]
    pub angle_bin_limit: Vec<f32>,

    /// angle_bin_width in degrees. Used for occupancy if input PCD
    /// does not have cluster_id field. Used for model grid if model_polar
    /// is true.
    #[arg(long, env, default_value = "6.875")]
    pub angle_bin_width: f32,

    /// occupancy threshold. Only used if input PCD does not have cluster_id
    /// field
    #[arg(long, env, default_value = "1")]
    pub threshold: u32,

    /// bin delay in radar message count (grid needs to be valid for `BIN_DELAY`
    /// frames before it is drawn. grid stops being drawn after being invalid
    /// for `BIN_DELAY` frames). Only used if input PCD does not have
    /// cluster_id field
    #[arg(long, env, default_value = "3")]
    pub bin_delay: u128,

    /// Background class index for semantic segmentation instance detection.
    /// -1 = last channel (default), 0..N = explicit index.
    #[arg(long, env, default_value = "-1", allow_hyphen_values = true)]
    pub background_index: i16,

    /// Enable Tracy profiler broadcast
    #[arg(long, env)]
    pub tracy: bool,

    /// zenoh connection mode
    #[arg(long, env, default_value = "peer")]
    mode: WhatAmI,

    /// connect to zenoh endpoints
    #[arg(long, env)]
    connect: Vec<String>,

    /// listen to zenoh endpoints
    #[arg(long, env)]
    listen: Vec<String>,

    /// disable zenoh multicast scouting
    #[arg(long, env)]
    no_multicast_scouting: bool,
}

impl Args {
    /// Returns true when a fusion model is configured (early/mid fusion mode).
    pub fn has_fusion_model(&self) -> bool {
        self.model.is_some()
    }

    /// Normalize parsed arguments: convert empty strings to None for optional
    /// path parameters, and filter empty strings from endpoint lists. This
    /// allows setting `MODEL = ""` or `CONNECT = ""` in environment files to
    /// represent the disabled/unset state.
    pub fn normalize(&mut self) {
        if self
            .model
            .as_ref()
            .is_some_and(|p| p.as_os_str().is_empty())
        {
            self.model = None;
        }
        if self
            .model_decoder
            .as_ref()
            .is_some_and(|p| p.as_os_str().is_empty())
        {
            self.model_decoder = None;
        }
        self.connect.retain(|s| !s.is_empty());
        self.listen.retain(|s| !s.is_empty());
    }
}

impl From<Args> for Config {
    fn from(args: Args) -> Self {
        let mut config = Config::default();

        config
            .insert_json5("mode", &json!(args.mode).to_string())
            .unwrap();

        if !args.connect.is_empty() {
            config
                .insert_json5("connect/endpoints", &json!(args.connect).to_string())
                .unwrap();
        }

        if !args.listen.is_empty() {
            config
                .insert_json5("listen/endpoints", &json!(args.listen).to_string())
                .unwrap();
        }

        if args.no_multicast_scouting {
            config
                .insert_json5("scouting/multicast/enabled", &json!(false).to_string())
                .unwrap();
        }

        config
            .insert_json5("scouting/multicast/interface", &json!("lo").to_string())
            .unwrap();

        config
    }
}
