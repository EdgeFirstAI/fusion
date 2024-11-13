use async_std::path::PathBuf;
use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// connect to zenoh endpoints
    #[arg(long, env, default_value = "tcp/127.0.0.1:7447")]
    pub connect: Vec<String>,

    /// listen to zenoh endpoints
    #[arg(long, env)]
    pub listen: Vec<String>,

    /// zenoh connection mode
    #[arg(long, env, default_value = "client")]
    pub mode: String,

    /// radar input topic
    #[arg(long, env, default_value = "rt/radar/targets")]
    pub radar_input_topic: String,

    /// mask topic
    #[arg(long, env, default_value = "rt/model/mask")]
    pub mask_topic: String,

    /// camera info topic
    #[arg(long, env, default_value = "rt/camera/info")]
    pub info_topic: String,

    /// radar output topic
    #[arg(long, env, default_value = "rt/fusion/targets")]
    pub radar_output_topic: String,

    /// occupancy output topic
    #[arg(long, env, default_value = "rt/fusion/occupancy")]
    pub occ_topic: String,

    /// model
    #[arg(short, long, env)]
    pub model: Option<PathBuf>,

    /// engine for model context
    #[arg(long, env, default_value = "npu")]
    pub engine: String,

    /// radarcube input topic
    #[arg(long, env, default_value = "rt/radar/cube")]
    pub radarcube_topic: String,

    /// occupancy threshold. Only used if input PCD does not have cluster_id
    /// field
    #[arg(long, env, default_value = "1")]
    pub threshold: u32,

    /// occlusion angle limit in degrees. Only used if input PCD does not have
    /// cluster_id field
    #[arg(long, env, default_value = "20")]
    pub occ_angle_limit: f64,

    /// occlusion range limit. Only used if input PCD does not have cluster_id
    /// field
    #[arg(long, env, default_value = "1.0")]
    pub occ_range_limit: f64,

    /// range_bin_limit. Only used if input PCD does not have cluster_id field
    #[arg(long, env, num_args = 2, value_delimiter = ' ', default_value = "0 16")]
    pub range_bin_limit: Vec<f64>,

    /// range_bin_width. Only used if input PCD does not have cluster_id field
    #[arg(long, env, default_value = "1.0")]
    pub range_bin_width: f64,

    /// angle_bin_limit, 0 degrees is forwards. Only used if input PCD does not
    /// have cluster_id field
    #[arg(
        long,
        env,
        num_args = 2,
        value_delimiter = ' ',
        default_value = "-55 55"
    )]
    pub angle_bin_limit: Vec<f64>,

    /// angle_bin_width in degrees. Only used if input PCD does not have
    /// cluster_id field
    #[arg(long, env, default_value = "6.875")]
    pub angle_bin_width: f64,

    /// bin delay in radar message count (grid needs to be valid for `BIN_DELAY`
    /// frames before it is drawn. grid stops being drawn after being invalid
    /// for `BIN_DELAY` frames). Only used if input PCD does not have
    /// cluster_id field
    #[arg(long, env, default_value = "3")]
    pub bin_delay: u128,

    /// enable tracking objects. Must be enabled for other --track_[...] flags
    /// to work
    #[arg(long, env, action)]
    pub track: bool,

    /// number of seconds the tracked object can be missing for before being
    /// removed.
    #[arg(long, env, default_value = "0.5")]
    pub track_extra_lifespan: f32,

    /// high score threshold for ByteTrack algorithm.
    #[arg(long, env, default_value = "0.7")]
    pub track_high_conf: f32,

    /// tracking iou threshold for box association. Higher values will require
    /// boxes to have higher IOU to the predicted track to be associated.
    #[arg(long, env, default_value = "0.01")]
    pub track_iou: f32,

    /// tracking update factor. Higher update factor will also mean
    /// less smoothing but more rapid response to change (0.0 to 1.0)
    #[arg(long, env, default_value = "0.4")]
    pub track_update: f32,
}
