use clap::Parser;
use std::path::PathBuf;

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
    #[arg(long, default_value = "tcp/127.0.0.1:7447")]
    pub connect: Vec<String>,

    /// listen to zenoh endpoints
    #[arg(long)]
    pub listen: Vec<String>,

    /// zenoh connection mode
    #[arg(long, default_value = "client")]
    pub mode: String,

    /// radar input topic
    #[arg(long, default_value = "rt/radar/targets")]
    pub radar_input_topic: String,

    /// mask topic
    #[arg(long, default_value = "rt/detect/mask")]
    pub mask_topic: String,

    /// camera info topic
    #[arg(long, default_value = "rt/camera/info")]
    pub info_topic: String,

    /// radar output topic
    #[arg(long, default_value = "rt/fusion/targets")]
    pub radar_output_topic: String,
}
