use clap::Parser;

#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Copy)]
pub enum LabelSetting {
    Index,
    Label,
    Score,
    LabelScore,
    Track,
}

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
    #[arg(long, env, default_value = "rt/detect/mask")]
    pub mask_topic: String,

    /// camera info topic
    #[arg(long, env, default_value = "rt/camera/info")]
    pub info_topic: String,

    /// radar output topic
    #[arg(long, env, default_value = "rt/fusion/targets")]
    pub radar_output_topic: String,

    /// occlusion angle limit in degrees
    #[arg(long, env, default_value = "20")]
    pub occ_angle_limit: f64,

    /// occlusion angle limit
    #[arg(long, env, default_value = "1.0")]
    pub occ_range_limit: f64,

    /// occupancy output topic
    #[arg(long, env, default_value = "rt/fusion/occupancy")]
    pub occupancy_topic: String,

    /// occupancy threshold
    #[arg(long, env, default_value = "1")]
    pub occupancy_threshold: u32,

    /// range_bin_limit
    #[arg(long, env, num_args = 2, value_delimiter = ' ', default_value = "0 20")]
    pub range_bin_limit: Vec<f64>,

    /// range_bin_width
    #[arg(long, env, default_value = "0.5")]
    pub range_bin_width: f64,

    /// angle_bin_limit, 0 degrees is forwards
    #[arg(
        long,
        env,
        num_args = 2,
        value_delimiter = ' ',
        default_value = "-70 70"
    )]
    pub angle_bin_limit: Vec<f64>,

    /// angle_bin_width in degrees
    #[arg(long, env, default_value = "10")]
    pub angle_bin_width: f64,
}
