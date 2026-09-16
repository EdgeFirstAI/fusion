// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use clap::{CommandFactory, Parser, ValueEnum};
use serde_json::json;
use std::path::PathBuf;
use zenoh::config::{Config, WhatAmI};

/// Parse a path that may be empty. An empty string is accepted and stored as an
/// empty PathBuf, which `Args::normalize()` later converts to `None`.
fn parse_optional_path(s: &str) -> Result<PathBuf, std::convert::Infallible> {
    Ok(PathBuf::from(s))
}

/// Environment variables where an empty value is meaningful and must be
/// preserved: each has a non-empty default but `""` is the documented
/// "leave empty to disable" sentinel, so scrubbing it would silently
/// re-enable an output the operator turned off.
pub const KEEP: &[&str] = &[
    "LIDAR_OUTPUT_TOPIC",
    "RADAR_OUTPUT_TOPIC",
    "VISION_MODEL_TOPIC",
    "MODEL_INFO_TOPIC",
    // Non-empty default, but `""` disables the radar late-fusion pipeline.
    "RADAR_PCD_TOPIC",
];

/// Names of this program's env-bound arguments whose value, as reported by
/// `var`, is present but empty and not listed in `keep`.
///
/// Pure: the environment is only read through `var`, so this can be unit
/// tested with a fake lookup and no process-wide mutation.
pub fn empty_env_vars<C: CommandFactory>(
    keep: &[&str],
    var: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    C::command()
        .get_arguments()
        .filter_map(|arg| arg.get_env().map(|e| e.to_string_lossy().into_owned()))
        .filter(|name| !keep.contains(&name.as_str()))
        .filter(|name| var(name).is_some_and(|v| v.is_empty()))
        .collect()
}

/// Treat an empty environment variable as unset, so clap's declared
/// `default_value` applies instead of failing to parse.
///
/// Only variables bound to this program's own arguments are considered;
/// unrelated process environment is left alone. `keep` names variables
/// where an empty value is meaningful and must be preserved.
///
/// # Safety
/// Must be called before any thread is spawned — that is, before the tokio
/// runtime is built. Mutating the process environment is not thread-safe.
pub unsafe fn scrub_empty_env<C: CommandFactory>(keep: &[&str]) {
    for name in empty_env_vars::<C>(keep, |name| std::env::var(name).ok()) {
        std::env::remove_var(&name);
    }
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
    /// Radar point cloud input topic. Default `radar/clusters`. Set to empty
    /// (`""`) to disable radar late-fusion. Unset uses the default; empty is
    /// not the same as unset.
    #[arg(long, env, default_value = "radar/clusters")]
    pub radar_pcd_topic: String,

    /// LiDAR point cloud input topic. Default empty (disabled). Set a topic
    /// such as `lidar/clusters` to enable LiDAR late-fusion.
    #[arg(long, env, default_value = "")]
    pub lidar_pcd_topic: String,

    /// LiDAR fusion output topic. Default `fusion/lidar`. Set to empty (`""`)
    /// to disable publishing. Unset uses the default; empty is not the same
    /// as unset.
    #[arg(long, env, default_value = "fusion/lidar")]
    pub lidar_output_topic: String,

    /// Radar fusion output topic. Default `fusion/radar`. Set to empty (`""`)
    /// to disable publishing. Unset uses the default; empty is not the same
    /// as unset.
    #[arg(long, env, default_value = "fusion/radar")]
    pub radar_output_topic: String,

    /// camera info input topic
    #[arg(long, env, default_value = "camera/info")]
    pub info_topic: String,

    /// unified vision model output topic. leave empty to disable
    #[arg(long, env, default_value = "model/output")]
    pub vision_model_topic: String,

    /// model info topic for label resolution. leave empty to disable
    #[arg(long, env, default_value = "model/info")]
    pub model_info_topic: String,

    /// Maximum age in seconds for model output data before warning. 0 = disabled
    #[arg(long, env, default_value = "0.5")]
    pub max_model_age: f32,

    /// bbox3d output topic
    #[arg(long, env, default_value = "fusion/boxes3d")]
    pub bbox3d_topic: String,

    /// bbox3d source
    #[arg(long, env, default_value = "radar")]
    pub bbox3d_src: PCDSource,

    /// camera frame input topic (`CameraFrame`)
    #[arg(long, env, default_value = "camera/frame")]
    pub camera_topic: String,

    /// radarcube input topic
    #[arg(long, env, default_value = "radar/cube")]
    pub radarcube_topic: String,

    /// radar model output
    #[arg(long, env, default_value = "fusion/model_output")]
    pub model_output_topic: String,

    /// TFLite fusion model path (.tflite). Leave empty to disable.
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
    #[arg(long, env, default_value = "fusion/occupancy")]
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
    /// allows setting `MODEL=""` or `CONNECT=""` in environment files to
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

    /// True when at least one late-fusion PCD input topic is set.
    pub fn has_pcd_input(&self) -> bool {
        !self.radar_pcd_topic.is_empty() || !self.lidar_pcd_topic.is_empty()
    }

    /// PCD topic for a sensor source. Empty when the source is disabled.
    pub fn pcd_topic_for(&self, src: PCDSource) -> &str {
        match src {
            PCDSource::Radar => self.radar_pcd_topic.as_str(),
            PCDSource::Lidar => self.lidar_pcd_topic.as_str(),
            PCDSource::Disabled => "",
        }
    }

    /// True when 3D boxes should be published: `bbox3d_src` names a sensor
    /// whose PCD topic is non-empty.
    pub fn bbox3d_enabled(&self) -> bool {
        !matches!(self.bbox3d_src, PCDSource::Disabled)
            && !self.pcd_topic_for(self.bbox3d_src).is_empty()
    }

    /// True when the occupancy-grid publisher should be declared.
    pub fn grid_enabled(&self) -> bool {
        self.has_fusion_model()
            && !self.grid_topic.is_empty()
            && !matches!(self.grid_src, PCDSource::Disabled)
            && !self.pcd_topic_for(self.grid_src).is_empty()
    }

    /// Reject a config that would start no pipeline (no model and no PCD).
    pub fn validate_pipeline(&self) -> Result<(), String> {
        if self.has_fusion_model() || self.has_pcd_input() {
            return Ok(());
        }
        Err(
            "no fusion pipeline is configured: set MODEL and/or RADAR_PCD_TOPIC / LIDAR_PCD_TOPIC"
                .into(),
        )
    }

    /// Log when bbox/grid sources point at a disabled PCD topic.
    pub fn warn_disabled_outputs(&self) {
        if !matches!(self.bbox3d_src, PCDSource::Disabled)
            && self.pcd_topic_for(self.bbox3d_src).is_empty()
        {
            tracing::warn!(
                "BBOX3D_SRC={:?} names a sensor with an empty PCD topic; fusion/boxes3d will not be declared",
                self.bbox3d_src
            );
        }
        if self.has_fusion_model()
            && !matches!(self.grid_src, PCDSource::Disabled)
            && self.pcd_topic_for(self.grid_src).is_empty()
        {
            tracing::warn!(
                "GRID_SRC={:?} names a sensor with an empty PCD topic; occupancy grid will not be declared",
                self.grid_src
            );
        }
    }
}

/// System hostname used as the Zenoh session namespace.
///
/// Empty or `/`-containing hostnames would create unintended sub-keys, so we
/// fall back to `"localhost"` and warn. Two devices both falling back would
/// silently share a namespace; that is a deployment defect.
fn zenoh_namespace() -> String {
    let raw = gethostname::gethostname().to_string_lossy().into_owned();
    if raw.is_empty() || raw.contains('/') {
        tracing::warn!(
            hostname = %raw,
            "system hostname is empty or contains '/' — falling back to \"localhost\""
        );
        "localhost".into()
    } else {
        raw
    }
}

impl From<Args> for Config {
    fn from(args: Args) -> Self {
        let mut config = Config::default();

        // Session namespace = hostname: application keys are bare
        // (`fusion/radar`) and the wire form is `{hostname}/fusion/radar`.
        config
            .insert_json5("namespace", &json!(zenoh_namespace()).to_string())
            .unwrap();

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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::collections::HashMap;

    /// Env-bound arguments with a non-empty default where we have consciously
    /// decided that an empty value is NOT meaningful (so scrubbing to the
    /// default is correct).
    const SCRUB_REVIEWED: &[&str] = &[
        "INFO_TOPIC",
        "MAX_MODEL_AGE",
        "BBOX3D_TOPIC",
        "BBOX3D_SRC",
        "CAMERA_TOPIC",
        "RADARCUBE_TOPIC",
        "MODEL_OUTPUT_TOPIC",
        "MODEL_THRESHOLD",
        "MODEL_GRID_SIZE",
        "ENGINE",
        "LOGITS",
        "TRACK_EXTRA_LIFESPAN",
        "TRACK_IOU",
        "TRACK_UPDATE",
        "GRID_TOPIC",
        "GRID_SRC",
        "RANGE_BIN_LIMIT",
        "RANGE_BIN_WIDTH",
        "ANGLE_BIN_LIMIT",
        "ANGLE_BIN_WIDTH",
        "THRESHOLD",
        "BIN_DELAY",
        "BACKGROUND_INDEX",
        "MODE",
    ];

    #[test]
    fn every_env_arg_is_either_scrubbable_or_explicitly_kept() {
        for arg in Args::command().get_arguments() {
            let Some(env) = arg.get_env() else { continue };
            let name = env.to_string_lossy().into_owned();
            let has_nonempty_default = arg
                .get_default_values()
                .first()
                .is_some_and(|d| !d.is_empty());
            if has_nonempty_default && !KEEP.contains(&name.as_str()) {
                assert!(
                    SCRUB_REVIEWED.contains(&name.as_str()),
                    "{name} has a non-empty default; decide whether empty is meaningful \
                     and add it to KEEP or SCRUB_REVIEWED"
                );
            }
        }
    }

    /// Fake environment lookup over a fixed table; never touches the process
    /// environment.
    fn lookup(env: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let table: HashMap<String, String> = env
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name| table.get(name).cloned()
    }

    #[test]
    fn empty_env_vars_lists_only_empty_bound_vars() {
        let env = [
            ("MAX_MODEL_AGE", ""),
            ("THRESHOLD", ""),
            ("LOGITS", ""),
            ("MODEL_POLAR", ""),
            ("TRACK_IOU", "0.3"),
            ("NOT_A_FUSION_ARG", ""),
        ];
        let mut found = empty_env_vars::<Args>(&[], lookup(&env));
        found.sort();
        assert_eq!(
            found,
            ["LOGITS", "MAX_MODEL_AGE", "MODEL_POLAR", "THRESHOLD"]
        );
    }

    #[test]
    fn empty_env_vars_ignores_unset_and_nonempty() {
        let env = [("MAX_MODEL_AGE", "0.5"), ("ENGINE", "cpu")];
        assert!(empty_env_vars::<Args>(&[], lookup(&env)).is_empty());
        assert!(empty_env_vars::<Args>(&[], lookup(&[])).is_empty());
    }

    #[test]
    fn empty_env_vars_never_lists_unbound_names() {
        let env = [("NOT_A_FUSION_ARG", ""), ("PATH", "")];
        assert!(empty_env_vars::<Args>(&[], lookup(&env)).is_empty());
    }

    #[test]
    fn keep_excludes_empty_disable_sentinels() {
        let env = [
            ("LIDAR_OUTPUT_TOPIC", ""),
            ("RADAR_OUTPUT_TOPIC", ""),
            ("VISION_MODEL_TOPIC", ""),
            ("MODEL_INFO_TOPIC", ""),
            ("RADAR_PCD_TOPIC", ""),
            ("BBOX3D_TOPIC", ""),
            ("MAX_MODEL_AGE", ""),
        ];
        let mut found = empty_env_vars::<Args>(KEEP, lookup(&env));
        found.sort();
        for kept in KEEP {
            assert!(!found.contains(&(*kept).to_owned()), "{kept} must be kept");
        }
        assert_eq!(found, ["BBOX3D_TOPIC", "MAX_MODEL_AGE"]);

        // Without the allowlist the same sentinels would be scrubbed.
        let unkept = empty_env_vars::<Args>(&[], lookup(&env));
        for kept in KEEP {
            assert!(unkept.contains(&(*kept).to_owned()), "{kept} is env-bound");
        }
    }

    fn parse_cli() -> Args {
        Args::parse_from([
            "edgefirst-fusion",
            "--mode",
            "peer",
            "--lidar-output-topic",
            "fusion/lidar",
            "--radar-output-topic",
            "fusion/radar",
            "--info-topic",
            "camera/info",
            "--vision-model-topic",
            "model/output",
            "--model-info-topic",
            "model/info",
            "--bbox3d-topic",
            "fusion/boxes3d",
            "--radarcube-topic",
            "radar/cube",
            "--model-output-topic",
            "fusion/model_output",
            "--grid-topic",
            "fusion/occupancy",
            "--camera-topic",
            "camera/frame",
        ])
    }

    #[test]
    fn zenoh_config_sets_namespace() {
        let ns = zenoh_namespace();
        assert!(!ns.is_empty(), "namespace should be non-empty");
        assert!(!ns.contains('/'), "namespace must not contain '/'");
        let rendered = Config::from(parse_cli()).to_string();
        assert!(
            rendered.contains(&ns),
            "config should include namespace {ns}: {rendered}"
        );
    }

    #[test]
    fn cli_topics_have_no_rt_prefix() {
        let args = parse_cli();
        assert_eq!(args.lidar_output_topic, "fusion/lidar");
        assert_eq!(args.radar_output_topic, "fusion/radar");
        assert_eq!(args.info_topic, "camera/info");
        assert_eq!(args.vision_model_topic, "model/output");
        assert_eq!(args.model_info_topic, "model/info");
        assert_eq!(args.bbox3d_topic, "fusion/boxes3d");
        assert_eq!(args.radarcube_topic, "radar/cube");
        assert_eq!(args.model_output_topic, "fusion/model_output");
        assert_eq!(args.grid_topic, "fusion/occupancy");
        assert_eq!(args.camera_topic, "camera/frame");
        assert_eq!(args.radar_pcd_topic, "radar/clusters");
        assert_eq!(args.bbox3d_src, PCDSource::Radar);
    }

    fn args_from(extra: &[&str]) -> Args {
        let mut argv = vec!["edgefirst-fusion"];
        argv.extend(extra);
        let mut args = Args::parse_from(argv);
        args.normalize();
        args
    }

    #[test]
    fn validate_rejects_idle_pipeline() {
        let args = args_from(&["--radar-pcd-topic", ""]);
        assert!(args.validate_pipeline().is_err());
        assert!(!args.bbox3d_enabled());
    }

    #[test]
    fn validate_accepts_stock_radar_defaults() {
        let args = args_from(&[]);
        assert!(args.validate_pipeline().is_ok());
        assert_eq!(args.radar_pcd_topic, "radar/clusters");
        assert!(args.bbox3d_enabled());
        assert!(!args.grid_enabled());
    }

    #[test]
    fn bbox3d_omitted_when_src_topic_disagree() {
        let args = args_from(&["--bbox3d-src", "lidar", "--lidar-pcd-topic", ""]);
        assert!(!args.bbox3d_enabled());
        let args = args_from(&["--bbox3d-src", "disabled"]);
        assert!(!args.bbox3d_enabled());
    }

    #[test]
    fn fusion_default_assignments_have_no_spaces_around_eq() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fusion.default");
        let text = std::fs::read_to_string(path).expect("fusion.default");
        let assignment = regex_lite_or_manual_assignment_lines(&text);
        assert!(
            !assignment.is_empty(),
            "fusion.default should contain KEY=\"value\" assignments"
        );
        for (line_no, line) in assignment {
            assert!(
                line.contains("=\""),
                "line {line_no}: expected KEY=\"value\", got {line}"
            );
            assert!(
                !line.contains(" =") && !line.contains("= "),
                "line {line_no}: spaces around '=': {line}"
            );
        }
    }

    fn regex_lite_or_manual_assignment_lines(text: &str) -> Vec<(usize, &str)> {
        text.lines()
            .enumerate()
            .filter_map(|(i, line)| {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    return None;
                }
                Some((i + 1, trimmed))
            })
            .collect()
    }
}
