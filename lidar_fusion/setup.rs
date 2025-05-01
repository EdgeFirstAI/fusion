use clap::Parser;
use serde_json::json;
use zenoh::{config::WhatAmI, Config};

#[derive(Debug, Clone, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// radar input topic
    #[arg(long, env, default_value = "rt/lidar/clusters")]
    pub lidar_input_topic: String,

    /// mask topic
    #[arg(long, env, default_value = "rt/model/mask")]
    pub mask_topic: String,

    /// camera info topic
    #[arg(long, env, default_value = "rt/camera/info")]
    pub info_topic: String,

    /// output topic
    #[arg(long, env, default_value = "rt/fusion/boxes3d")]
    pub output_topic: String,

    /// Enable Tracy profiler broadcast
    #[arg(long, env)]
    pub tracy: bool,

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
    #[arg(long, env, default_value = "0.25")]
    pub track_iou: f32,

    /// tracking update factor. Higher update factor will also mean
    /// less smoothing but more rapid response to change (0.0 to 1.0)
    #[arg(long, env, default_value = "0.25")]
    pub track_update: f32,

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
