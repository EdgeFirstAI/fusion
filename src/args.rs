use clap::{Parser, ValueEnum};
use serde_json::json;
use std::path::PathBuf;
use zenoh::config::{Config, WhatAmI};

type BoolDefaultTrue = bool;

#[derive(Debug, Clone, ValueEnum, Copy, Eq, PartialEq)]
pub enum PCDSource {
    Disabled,
    Radar,
    Lidar,
}
#[derive(Debug, Clone, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// radar input topic. leave empty to disable
    #[arg(long, env, default_value = "rt/radar/clusters")]
    pub radar_pcd_topic: String,

    /// lidar input topic. leave empty to disable
    #[arg(long, env, default_value = "rt/lidar/clusters")]
    pub lidar_pcd_topic: String,

    /// mask topic
    #[arg(long, env, default_value = "rt/model/mask")]
    pub mask_topic: String,

    /// zenoh key expression for camera DMA buffers
    #[arg(long, env, default_value = "rt/camera/dma")]
    pub camera_topic: String,

    /// camera info topic
    #[arg(long, env, default_value = "rt/camera/info")]
    pub info_topic: String,

    /// radar output topic. leave empty to disable
    #[arg(long, env, default_value = "rt/fusion/radar")]
    pub radar_output_topic: String,

    /// lidar output topic. leave empty to disable
    #[arg(long, env, default_value = "")]
    pub lidar_output_topic: String,

    /// occupancy grid output topic
    #[arg(long, env, default_value = "rt/fusion/occupancy")]
    pub occ_topic: String,

    /// occupancy grid source
    #[arg(long, env, default_value = "radar")]
    pub occ_src: PCDSource,

    /// bbox3d output topic
    #[arg(long, env, default_value = "rt/fusion/boxes3d")]
    pub bbox3d_topic: String,

    /// bbox3d source
    #[arg(long, env, default_value = "lidar")]
    pub bbox3d_src: PCDSource,

    /// model
    #[arg(short, long, env)]
    pub model: Option<PathBuf>,

    /// model decoder
    #[arg(long, env)]
    pub model_decoder: Option<PathBuf>,

    /// set the model to be polar
    #[arg(long, env, action)]
    pub model_polar: bool,

    /// model threshold.
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

    /// radarcube input topic
    #[arg(long, env, default_value = "rt/radar/cube")]
    pub radarcube_topic: String,

    /// radar model output
    #[arg(long, env, default_value = "rt/fusion/model_output")]
    pub model_output_topic: String,

    /// apply sigmoid the model output
    #[arg(long, env, default_value = "true")]
    pub logits: BoolDefaultTrue,

    /// enable tracking to reduce flickering in model output
    #[arg(long, env, action)]
    pub track: bool,

    // currently unused
    #[arg(long, env, default_value = "0.5")]
    pub track_high_conf: f32,

    /// number of seconds the tracked object can be missing for before being
    /// removed.
    #[arg(long, env, default_value = "0.5")]
    pub track_extra_lifespan: f32,

    /// tracking iou threshold for box association. Higher values will require
    /// boxes to have higher IOU to the predicted track to be associated.
    #[arg(long, env, default_value = "0.1")]
    pub track_iou: f32,

    /// tracking update factor. Higher update factor will also mean
    /// less smoothing but more rapid response to change (0.0 to 1.0)
    #[arg(long, env, default_value = "0.4")]
    pub track_update: f32,

    /// range_bin_limit. Used for model grid. Used for occupancy if input PCD
    /// does not have cluster_id field.
    #[arg(long, env, num_args = 2, value_delimiter = ' ', default_value = "0 16")]
    pub range_bin_limit: Vec<f32>,

    /// range_bin_width. Used for model grid. Used for occupancy if input PCD
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

    /// occlusion angle limit in degrees.
    #[arg(long, env, default_value = "20")]
    pub occ_angle_limit: f32,

    /// occlusion range limit.
    #[arg(long, env, default_value = "1.0")]
    pub occ_range_limit: f32,

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
