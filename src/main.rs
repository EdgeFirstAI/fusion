// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use args::{Args, PCDSource};
use clap::Parser;
use edgefirst_schemas::{
    builtin_interfaces::Time,
    edgefirst_msgs::{Box as DetectBox, Detect, Mask, Model, ModelInfo, Track},
    geometry_msgs::{Quaternion, Transform, TransformStamped, Vector3},
    sensor_msgs::{CameraInfo, PointCloud2},
    serde_cdr,
    std_msgs::Header,
};
use fusion_model::spawn_fusion_model_thread;
use log::{error, trace, warn};
use mask::{mask_instance, process_mask, resolve_box_label, Box2D};
use pcd::{parse_pcd, serialize_classes, serialize_grid, serialize_late_fusion, FusionFrame};
use std::{
    collections::HashMap,
    hash::Hash,
    sync::Arc,
    thread::{self, JoinHandle},
    time::Duration,
};
use tokio::{join, sync::Mutex};
use tracing::{instrument, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt as _, Layer as _, Registry};
use tracker::{ByteTrack, ByteTrackSettings, TrackerBox};
use tracy_client::frame_mark;
use transform::transform_and_project_points;
use zenoh::{
    bytes::{Encoding, ZBytes},
    handlers::FifoChannelHandler,
    pubsub::{Publisher, Subscriber},
    sample::Sample,
    Session,
};

mod args;
mod fusion_model;
mod image;
mod kalman;
mod mask;
mod pcd;
#[cfg(feature = "deepviewrt")]
mod rtm_model;
mod simd;
mod tflite_model;
mod tracker;
mod transform;

const BASE_LINK_FRAME_ID: &str = "base_link";

type Grid = (Vec<Vec<f32>>, u64);

/// Data loaded from a single point cloud frame: the cloud header, parsed frame,
/// sensor-to-base transform, camera-to-base transform, and camera intrinsics.
type LoadedFrame = (Header, FusionFrame, Transform, Transform, CameraInfo);

const FUSION_CLASS: &str = "fusion_class";
const INSTANCE_ID: &str = "instance_id";
const TRACK_ID: &str = "track_id";
const VISION_CLASS: &str = "vision_class";
const MAX_CLASSIFICATION_DISTANCE: f32 = 2.0;
const UNINITIALIZED_COORD: f32 = 99999.0;

/// Pre-computed (sin, cos) for 8 evenly spaced angles (0, 45, 90, ..., 315°).
/// Used by edge-tolerance loops to avoid per-point trig calls.
const EDGE_OFFSETS: [(f32, f32); 8] = {
    const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
    [
        (0.0, 1.0),  // 0°
        (S, S),      // 45°
        (1.0, 0.0),  // 90°
        (S, -S),     // 135°
        (0.0, -1.0), // 180°
        (-S, -S),    // 225°
        (-1.0, 0.0), // 270°
        (-S, S),     // 315°
    ]
};

#[tokio::main]
async fn main() {
    let mut args = Args::parse();
    args.normalize();

    args.tracy.then(tracy_client::Client::start);

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let stdout_log = tracing_subscriber::fmt::layer()
        .pretty()
        .with_filter(env_filter);

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let journald = match tracing_journald::layer() {
        Ok(journald) => Some(journald.with_filter(env_filter)),
        Err(_) => None,
    };

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let tracy = match args.tracy {
        true => Some(tracing_tracy::TracyLayer::default().with_filter(env_filter)),
        false => None,
    };

    let subscriber = Registry::default()
        .with(stdout_log)
        .with(journald)
        .with(tracy);
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    tracing_log::LogTracer::init().unwrap();

    let session = zenoh::open(args.clone()).await.unwrap();

    let info = Arc::new(Mutex::new(None));
    let model_info_cb = model_info_callback(info.clone());
    let _info_sub = session
        .declare_subscriber(args.info_topic.clone())
        .callback_mut(model_info_cb)
        .await
        .expect("Failed to declare Zenoh subscriber");

    let model_output: Arc<Mutex<Option<(Model, std::time::Instant)>>> = Arc::new(Mutex::new(None));
    let _model_output_sub = if !args.vision_model_topic.is_empty() {
        let cb = model_output_callback(model_output.clone());
        Some(
            session
                .declare_subscriber(args.vision_model_topic.clone())
                .callback_mut(cb)
                .await
                .expect("Failed to declare Zenoh subscriber"),
        )
    } else {
        None
    };

    let model_labels: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
    let _model_info_sub = if !args.model_info_topic.is_empty() {
        let cb = model_labels_callback(model_labels.clone());
        Some(
            session
                .declare_subscriber(args.model_info_topic.clone())
                .callback_mut(cb)
                .await
                .expect("Failed to declare Zenoh subscriber"),
        )
    } else {
        None
    };

    let transform = Arc::new(Mutex::new(HashMap::<(String, String), Transform>::new()));
    let tf_static_cb = tf_static_callback(transform.clone());
    let _transform_sub = session
        .declare_subscriber("rt/tf_static")
        .callback_mut(tf_static_cb)
        .await
        .expect("Failed to declare Zenoh subscriber");

    let grid: Arc<Mutex<Option<Grid>>> = Arc::new(Mutex::new(None));
    let fusion_model_handle =
        spawn_fusion_model_thread(session.clone(), args.clone(), grid.clone());

    let (radar_sub, lidar_sub, grid_publ, bbox_publ) = declare_sub_pub(&session, &args).await;

    // Declare per-context publishers for the per-sensor output topics.
    // Each fusion thread needs its own publisher instance.
    let radar_output_publ = if !args.radar_output_topic.is_empty() {
        Some(
            session
                .declare_publisher(args.radar_output_topic.clone())
                .await
                .expect("Failed to declare Zenoh publisher"),
        )
    } else {
        None
    };
    let lidar_output_publ = if !args.lidar_output_topic.is_empty() {
        Some(
            session
                .declare_publisher(args.lidar_output_topic.clone())
                .await
                .expect("Failed to declare Zenoh publisher"),
        )
    } else {
        None
    };

    // wait 2s for the tf_static to get transforms
    tokio::time::sleep(Duration::from_secs(2)).await;

    let tf_session = session.clone();
    let tf_msg = build_tf_msg();
    let tf_msg = ZBytes::from(serde_cdr::serialize(&tf_msg).unwrap());
    let tf_enc = Encoding::APPLICATION_CDR.with_schema("geometry_msgs/msg/TransformStamped");
    tokio::spawn(async move {
        if let Err(e) = tf_static(tf_session, tf_msg, tf_enc).await {
            log::error!("TF static publisher failed: {e}");
        }
    });

    let mut zenoh_radar = ZenohCtx {
        session: session.clone(),
        pcd_sub: radar_sub,
        output_publ: radar_output_publ,
        grid_publ: None,
        bbox_publ: None,
    };

    let mut zenoh_lidar = ZenohCtx {
        session,
        pcd_sub: lidar_sub,
        output_publ: lidar_output_publ,
        grid_publ: None,
        bbox_publ: None,
    };
    let data_radar = Mutexes {
        model_output: model_output.clone(),
        info: info.clone(),
        tf_static: transform.clone(),
        grid: grid.clone(),
        model_info: model_labels.clone(),
    };

    let data_lidar = Mutexes {
        model_output,
        info,
        tf_static: transform,
        grid,
        model_info: model_labels,
    };

    if let Some(gp) = grid_publ {
        match args.grid_src {
            PCDSource::Radar => zenoh_radar.grid_publ = Some(gp),
            PCDSource::Lidar => zenoh_lidar.grid_publ = Some(gp),
            _ => {}
        }
    }

    match args.bbox3d_src {
        PCDSource::Radar => zenoh_radar.bbox_publ = Some(bbox_publ),
        PCDSource::Lidar => zenoh_lidar.bbox_publ = Some(bbox_publ),
        _ => {}
    }
    let lidar_handle = spawn_fusion_thread(data_lidar, zenoh_lidar, args.clone());
    let radar_handle = spawn_fusion_thread(data_radar, zenoh_radar, args.clone());

    let _ = lidar_handle.join();
    let _ = radar_handle.join();
    let _ = fusion_model_handle.join();
}

fn model_info_callback(info: Arc<Mutex<Option<CameraInfo>>>) -> impl FnMut(zenoh::sample::Sample) {
    move |s: Sample| {
        let new_info: CameraInfo = match serde_cdr::deserialize(&s.payload().to_bytes()) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize message: {e:?}");
                return;
            }
        };
        let mut guard = info.try_lock();
        if let Ok(ref mut guard) = guard {
            **guard = Some(new_info);
        }
    }
}

fn model_output_callback(
    model: Arc<Mutex<Option<(Model, std::time::Instant)>>>,
) -> impl FnMut(Sample) {
    move |s: Sample| {
        let new_model: Model = match serde_cdr::deserialize(&s.payload().to_bytes()) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize Model: {e:?}");
                return;
            }
        };
        if let Ok(mut guard) = model.try_lock() {
            *guard = Some((new_model, std::time::Instant::now()));
        }
    }
}

fn model_labels_callback(labels: Arc<Mutex<Option<Vec<String>>>>) -> impl FnMut(Sample) {
    move |s: Sample| {
        let info: ModelInfo = match serde_cdr::deserialize(&s.payload().to_bytes()) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize ModelInfo: {e:?}");
                return;
            }
        };
        if let Ok(mut guard) = labels.try_lock() {
            *guard = Some(info.labels);
        }
    }
}

fn tf_static_callback(
    transform: Arc<Mutex<HashMap<(String, String), Transform>>>,
) -> impl FnMut(zenoh::sample::Sample) {
    move |s: Sample| {
        let new_transform: TransformStamped = match serde_cdr::deserialize(&s.payload().to_bytes())
        {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize message: {e:?}");
                return;
            }
        };

        if let Ok(mut guard) = transform.try_lock() {
            guard.insert(
                (new_transform.header.frame_id, new_transform.child_frame_id),
                new_transform.transform,
            );
        }
    }
}

async fn declare_sub_pub(
    session: &Session,
    args: &Args,
) -> (
    Option<Subscriber<FifoChannelHandler<Sample>>>,
    Option<Subscriber<FifoChannelHandler<Sample>>>,
    Option<Publisher<'static>>,
    Publisher<'static>,
) {
    let radar_sub = if !args.radar_pcd_topic.is_empty() {
        Some(
            session
                .declare_subscriber(args.radar_pcd_topic.clone())
                .await
                .expect("Failed to declare Zenoh subscriber"),
        )
    } else {
        None
    };

    let lidar_sub = if !args.lidar_pcd_topic.is_empty() {
        Some(
            session
                .declare_subscriber(args.lidar_pcd_topic.clone())
                .await
                .expect("Failed to declare Zenoh subscriber"),
        )
    } else {
        None
    };

    let grid_publ = if args.has_fusion_model() && !args.grid_topic.is_empty() {
        Some(
            session
                .declare_publisher(args.grid_topic.clone())
                .await
                .expect("Failed to declare Zenoh publisher"),
        )
    } else {
        None
    };

    let bbox_publ = session
        .declare_publisher(args.bbox3d_topic.clone())
        .await
        .expect("Failed to declare Zenoh publisher");

    (radar_sub, lidar_sub, grid_publ, bbox_publ)
}

pub fn spawn_fusion_thread(data: Mutexes, zenoh: ZenohCtx, args: Args) -> JoinHandle<()> {
    thread::Builder::new()
        .name("fusion_thread".to_string())
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(fusion_loop(data, zenoh, &args));
        })
        .unwrap()
}

#[derive(Debug)]
pub struct ZenohCtx {
    session: Session,
    pcd_sub: Option<Subscriber<FifoChannelHandler<Sample>>>,
    output_publ: Option<Publisher<'static>>,
    grid_publ: Option<Publisher<'static>>,
    bbox_publ: Option<Publisher<'static>>,
}

#[derive(Debug, Clone)]
pub struct Mutexes {
    model_output: Arc<Mutex<Option<(Model, std::time::Instant)>>>,
    info: Arc<Mutex<Option<CameraInfo>>>,
    tf_static: Arc<Mutex<HashMap<(String, String), Transform>>>,
    grid: Arc<Mutex<Option<Grid>>>,
    model_info: Arc<Mutex<Option<Vec<String>>>>,
}

fn setup_bins(bins: &mut Vec<Vec<Bin>>, args: &Args) {
    let mut i = args.angle_bin_limit[0];
    while i <= args.angle_bin_limit[1] {
        let mut range_bins = Vec::new();
        let mut j = args.range_bin_limit[0];
        while j <= args.range_bin_limit[1] {
            range_bins.push(Bin {
                vision_classes: Vec::new(),
                fusion_classes: Vec::new(),
                last_masked: 0,
                first_marked: u128::MAX,
            });
            j += args.range_bin_width
        }
        bins.push(range_bins);
        i += args.angle_bin_width;
    }
}

#[instrument(skip_all)]
async fn load_data(msg: &Sample, data: &Mutexes) -> Result<LoadedFrame, String> {
    let (header, frame) = {
        let pcd: PointCloud2 = match serde_cdr::deserialize(&msg.payload().to_bytes()) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to deserialize PCD: {e:?}")),
        };
        let frame = parse_pcd(&pcd);
        let header = pcd.header;
        (header, frame)
    };

    let pcd_frame_id = header.frame_id.clone();

    let transform = data
        .tf_static
        .lock()
        .await
        .get(&(BASE_LINK_FRAME_ID.to_owned(), pcd_frame_id.clone()))
        .map_or_else(
            || {
                warn!("Did not find transform from base_link to {}", pcd_frame_id);
                Transform {
                    translation: Vector3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    rotation: Quaternion {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        w: 1.0,
                    },
                }
            },
            |v| *v,
        );
    let header = Header {
        stamp: header.stamp,
        frame_id: pcd_frame_id.clone(),
    };

    let cam_info = match data.info.lock().await {
        v if v.is_some() => v.as_ref().unwrap().clone(),
        _ => return Err("No Camera Info".to_string()),
    };

    let cam_transform = data
        .tf_static
        .lock()
        .await
        .get(&(
            BASE_LINK_FRAME_ID.to_owned(),
            cam_info.header.frame_id.clone(),
        ))
        .map_or_else(
            || {
                warn!(
                    "Did not find transform from base_link to {}",
                    cam_info.header.frame_id
                );
                Transform {
                    translation: Vector3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    rotation: Quaternion {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        w: 1.0,
                    },
                }
            },
            |v| *v,
        );

    Ok((header, frame, transform, cam_transform, cam_info))
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
async fn fusion(
    frame: &mut FusionFrame,
    transform: Transform,
    cam_transform: Transform,
    cam_info: &CameraInfo,
    track: bool,
    tracker: &mut ByteTrack,
    grid: &Arc<Mutex<Option<Grid>>>,
    args: &Args,
    session: &Session,
    data: &Mutexes,
) -> HashMap<u32, Vec<usize>> {
    let cam_mtx = cam_info.k.map(|v| v as f32);
    transform_and_project_points(
        frame,
        &transform,
        &cam_transform,
        &cam_mtx,
        (cam_info.width as f32, cam_info.height as f32),
    );

    let ids = get_cluster_ids(frame);

    // Lock model output and check age
    let model_guard = data.model_output.lock().await;
    if args.max_model_age > 0.0 {
        if let Some((_, received)) = model_guard.as_ref() {
            let age = received.elapsed();
            if age.as_secs_f32() > args.max_model_age {
                warn!(
                    "Model output is {:.0}ms old (limit: {:.0}ms)",
                    age.as_millis(),
                    args.max_model_age * 1000.0
                );
            }
        }
    }
    let model_ref = model_guard.as_ref().map(|(m, _)| m);

    let model_labels_guard = data.model_info.lock().await;
    let labels = model_labels_guard.as_deref();

    get_vision_class_and_instance(frame, model_ref, &ids, labels);
    drop(model_labels_guard);
    drop(model_guard);

    if args.has_fusion_model() {
        let fusion_predictions = get_fusion_predictions(track, tracker, grid, args, session).await;
        get_fusion_class(frame, &fusion_predictions, &ids);
    }

    ids
}

async fn fusion_loop(data: Mutexes, zenoh: ZenohCtx, args: &Args) {
    if zenoh.pcd_sub.is_none() {
        return;
    }

    let mut bins = Vec::new();
    let mut frame_index = 0;
    let mut timeout = DrainRecvTimeoutSettings::default();
    let mut tracking_detected = false;

    let mut tracker = ByteTrack::new_with_settings(ByteTrackSettings {
        track_high_conf: 0.5,
        track_extra_lifespan: args.track_extra_lifespan,
        track_iou: args.track_iou,
        track_update: args.track_update,
    });

    setup_bins(&mut bins, args);

    loop {
        let msg = match drain_recv(zenoh.pcd_sub.as_ref().unwrap(), &mut timeout).await {
            Some(v) => v,
            None => {
                continue;
            }
        };

        let (header, mut frame, transform, cam_transform, cam_info) =
            match load_data(&msg, &data).await {
                Ok(v) => v,
                Err(e) => {
                    warn!("{e}");
                    continue;
                }
            };

        let ids = fusion(
            &mut frame,
            transform,
            cam_transform,
            &cam_info,
            args.track,
            &mut tracker,
            &data.grid,
            args,
            &zenoh.session,
            &data,
        )
        .await;

        // Detect tracking from model output: once any track_id is non-zero,
        // enable tracked output layout for the rest of the session.
        if !tracking_detected && frame.track_id.iter().any(|&t| t != 0) {
            tracking_detected = true;
            log::info!("Tracking detected in model output, enabling track_id in output topic");
        }

        publish(
            &zenoh,
            args,
            &header,
            &frame,
            &ids,
            &mut tracker,
            &mut bins,
            frame_index,
            tracking_detected,
        )
        .await;

        clear_bins(&mut bins, frame_index, args);
        frame_index += 1;

        args.tracy.then(frame_mark);
    }
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
async fn publish(
    zenoh: &ZenohCtx,
    args: &Args,
    header: &Header,
    frame: &FusionFrame,
    ids: &HashMap<u32, Vec<usize>>,
    point_tracker: &mut ByteTrack,
    bins: &mut [Vec<Bin>],
    frame_index: u128,
    tracking: bool,
) {
    let publ_bbox = publish_bbox3d(zenoh.bbox_publ.as_ref(), header, frame, ids);

    let publ_output = publish_output(
        zenoh.output_publ.as_ref(),
        frame,
        header,
        args.has_fusion_model(),
        tracking,
    );

    let publ_grid = publish_grid(
        zenoh.grid_publ.as_ref(),
        header,
        frame,
        point_tracker,
        (bins, frame_index),
        args,
        ids,
    );

    join!(publ_bbox, publ_output, publ_grid);
}

/// Hash a track ID string to a u32 using FNV-1a.
///
/// Empty string returns 0 (reserved for no-track). Non-empty string hashes
/// to a u32 (never 0) that is deterministic and stable across processes,
/// platforms, and language implementations.
///
/// Algorithm: FNV-1a 32-bit (offset basis `0x811c9dc5`, prime `0x01000193`)
/// operating on the raw UTF-8 bytes of the input string.
///
/// Reference: <https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function>
pub fn hash_track_id(track_id: &str) -> u32 {
    if track_id.is_empty() {
        return 0;
    }
    let mut h: u32 = 0x811c_9dc5;
    for &byte in track_id.as_bytes() {
        h ^= byte as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    if h == 0 {
        1
    } else {
        h
    }
}

/// Parse a detection box label string to u8 class index.
///
/// When a `labels` list is provided (from `ModelInfo`), the label is looked up
/// by string and its index is returned (0 = background, matching the mask
/// argmax convention). Falls back to numeric parsing, then to 0.
fn parse_box_label(label: &str, labels: Option<&[String]>) -> u8 {
    if let Some(labels) = labels {
        if let Some(idx) = labels.iter().position(|l| l == label) {
            return idx.min(255) as u8;
        }
    }
    label.parse::<u8>().unwrap_or_else(|_| {
        warn!("Unknown box label '{}', defaulting to 0", label);
        0
    })
}

/// Assign each projected point to the detection box whose centre is nearest.
/// When a point falls inside multiple overlapping boxes, the box with the
/// smallest squared distance to the point wins.  Equal distances are broken
/// by iteration order (first box in `detect.boxes` wins), which is
/// deterministic for a given detection frame.
fn box_fusion_no_cluster(frame: &mut FusionFrame, boxes: &[DetectBox], labels: Option<&[String]>) {
    let n = frame.len;
    for i in 0..n {
        let px = frame.proj_u[i];
        let py = frame.proj_v[i];
        if !check_in_bounds(&px, &py) {
            continue;
        }
        let mut best_dist2 = f32::MAX;
        let mut best_class = 0u8;
        let mut best_instance = 0u16;
        let mut best_track = 0u32;

        for (box_idx, b) in boxes.iter().enumerate() {
            let half_w = b.width / 2.0;
            let half_h = b.height / 2.0;
            if (px - b.center_x).abs() <= half_w && (py - b.center_y).abs() <= half_h {
                let dist2 = (px - b.center_x).powi(2) + (py - b.center_y).powi(2);
                if dist2 < best_dist2 {
                    best_dist2 = dist2;
                    best_class = parse_box_label(&b.label, labels);
                    best_instance = (box_idx + 1) as u16;
                    best_track = hash_track_id(&b.track.id);
                }
            }
        }
        frame.vision_class[i] = best_class;
        frame.instance_id[i] = best_instance;
        frame.track_id[i] = best_track;
    }
}

fn box_fusion_clustered(
    frame: &mut FusionFrame,
    boxes: &[DetectBox],
    clusters: &HashMap<u32, Vec<usize>>,
    labels: Option<&[String]>,
) {
    // For each cluster, track best box assignment: (class, instance_id, track_id, point_count)
    let mut cluster_assignment: HashMap<u32, (u8, u16, u32, usize)> = HashMap::new();

    for (box_idx, b) in boxes.iter().enumerate() {
        let half_w = b.width / 2.0;
        let half_h = b.height / 2.0;
        let label = parse_box_label(&b.label, labels);
        let instance = (box_idx + 1) as u16;
        let track = hash_track_id(&b.track.id);

        let mut cluster_counts: HashMap<u32, usize> = HashMap::new();
        for i in 0..frame.len {
            let px = frame.proj_u[i];
            let py = frame.proj_v[i];
            if !check_in_bounds(&px, &py) {
                continue;
            }
            if (px - b.center_x).abs() <= half_w && (py - b.center_y).abs() <= half_h {
                let id = frame.cluster_id[i];
                if id > 0 {
                    *cluster_counts.entry(id).or_default() += 1;
                }
            }
        }

        if let Some((&best_cluster, &best_count)) =
            cluster_counts.iter().max_by_key(|(_, count)| *count)
        {
            let entry = cluster_assignment
                .entry(best_cluster)
                .or_insert((0, 0, 0, 0));
            if best_count > entry.3 {
                *entry = (label, instance, track, best_count);
            }
        }
    }

    for (cluster_id, (label, instance, track, _)) in &cluster_assignment {
        if let Some(point_indices) = clusters.get(cluster_id) {
            for &idx in point_indices {
                frame.vision_class[idx] = *label;
                frame.instance_id[idx] = *instance;
                frame.track_id[idx] = *track;
            }
        }
    }
}

fn get_vision_class_and_instance(
    frame: &mut FusionFrame,
    model: Option<&Model>,
    ids: &HashMap<u32, Vec<usize>>,
    labels: Option<&[String]>,
) {
    let model = match model {
        Some(m) => m,
        None => return,
    };

    // PREFER masks when available (more precise)
    if let Some(first_mask) = model.masks.first() {
        if first_mask.boxed && !model.boxes.is_empty() {
            // Instance segmentation: per-box masks placed at box coordinates
            boxed_mask_fusion(frame, &model.masks, &model.boxes, labels, ids);
            return;
        } else {
            // Semantic segmentation: process full-frame mask
            let mut mask = first_mask.clone();
            process_mask(&mut mask);
            if frame.has_clusters() {
                late_fusion_clustered(frame, &mask, ids);
            } else {
                late_fusion_no_cluster(frame, &mask, 0.02);
            }
            return;
        }
    }

    // Fallback: detection only (boxes, no masks)
    if !model.boxes.is_empty() {
        if frame.has_clusters() {
            box_fusion_clustered(frame, &model.boxes, ids, labels);
        } else {
            box_fusion_no_cluster(frame, &model.boxes, labels);
        }
    }
}

/// Per-box instance mask data after preprocessing.
struct PreparedBoxMask {
    mask: Vec<u8>,
    width: usize,
    height: usize,
    class_idx: u8,
    instance_id: u16,
    track_id: u32,
    /// True if the mask went through argmax (values are class indices, 0=bg)
    argmaxed: bool,
    /// Box bounds in normalized [0,1] image coordinates
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

/// Prepare per-box masks: decompress/argmax, resolve labels, compute box bounds.
fn prepare_box_masks(
    masks: &[Mask],
    boxes: &[DetectBox],
    labels: Option<&[String]>,
) -> Vec<PreparedBoxMask> {
    masks
        .iter()
        .zip(boxes.iter())
        .enumerate()
        .filter_map(|(i, (mask, det_box))| {
            let mut m = mask.clone();
            let channels = process_mask(&mut m);
            let w = m.width as usize;
            let h = m.height as usize;
            if w == 0 || h == 0 || m.mask.len() != w * h {
                return None;
            }
            let class_idx = resolve_box_label(&det_box.label, labels);
            if class_idx == 0 {
                return None;
            }
            let argmaxed = channels > 1;
            let half_w = det_box.width / 2.0;
            let half_h = det_box.height / 2.0;
            Some(PreparedBoxMask {
                mask: m.mask,
                width: w,
                height: h,
                class_idx,
                instance_id: (i + 1) as u16,
                track_id: hash_track_id(&det_box.track.id),
                argmaxed,
                x0: det_box.center_x - half_w,
                y0: det_box.center_y - half_h,
                x1: det_box.center_x + half_w,
                y1: det_box.center_y + half_h,
            })
        })
        .collect()
}

/// Mask confidence threshold: pixels below this in single-channel masks are
/// treated as background. For multi-channel (argmax) masks, background is 0.
const MASK_THRESHOLD: u8 = 128;

/// Maximum depth spread (in meters) within an instance. Points farther than
/// this from the median depth of the instance are removed.
const MAX_INSTANCE_DEPTH_SPREAD: f32 = 1.0;

/// Look up the best (highest-confidence) box mask hit for a point at (u,v)
/// in normalized image coordinates. Returns (class, instance, track).
fn lookup_point_in_masks(u: f32, v: f32, prepared: &[PreparedBoxMask]) -> (u8, u16, u32) {
    let mut best_score = 0u8;
    let mut best = (0u8, 0u16, 0u32);
    for bm in prepared {
        if u < bm.x0 || u > bm.x1 || v < bm.y0 || v > bm.y1 {
            continue;
        }
        // Relative position within box [0, 1)
        let rel_x = ((u - bm.x0) / (bm.x1 - bm.x0)).clamp(0.0, 0.9999);
        let rel_y = ((v - bm.y0) / (bm.y1 - bm.y0)).clamp(0.0, 0.9999);
        let mx = (rel_x * bm.width as f32) as usize;
        let my = (rel_y * bm.height as f32) as usize;
        let val = bm.mask[my * bm.width + mx];
        if bm.argmaxed {
            // Argmaxed: val is a class index (0=background). First hit wins.
            if val > 0 {
                return (bm.class_idx, bm.instance_id, bm.track_id);
            }
        } else {
            // Single-channel confidence: pick highest above threshold.
            if val >= MASK_THRESHOLD && val > best_score {
                best_score = val;
                best = (bm.class_idx, bm.instance_id, bm.track_id);
            }
        }
    }
    best
}

/// Remove outlier points from each instance based on depth. Points farther
/// than MAX_INSTANCE_DEPTH_SPREAD from the median depth are reset to 0.
fn filter_depth_outliers(frame: &mut FusionFrame) {
    let n = frame.len;

    // Pre-compute all depths in a single SIMD-accelerated pass
    let mut all_depths = vec![0.0f32; n];
    simd::magnitude3_f32(&frame.x[..n], &frame.y[..n], &frame.z[..n], &mut all_depths);

    // Group point indices by instance_id
    let mut instance_points: HashMap<u16, Vec<usize>> = HashMap::new();
    for (i, &inst) in frame.instance_id[..n].iter().enumerate() {
        if inst > 0 {
            instance_points.entry(inst).or_default().push(i);
        }
    }

    for indices in instance_points.values() {
        if indices.len() < 2 {
            continue;
        }
        // Gather pre-computed depths and find median
        let mut depths: Vec<f32> = indices.iter().map(|&i| all_depths[i]).collect();
        depths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = depths[depths.len() / 2];

        // Remove points too far from the median
        for &idx in indices {
            if (all_depths[idx] - median).abs() > MAX_INSTANCE_DEPTH_SPREAD {
                frame.vision_class[idx] = 0;
                frame.instance_id[idx] = 0;
                frame.track_id[idx] = 0;
            }
        }
    }
}

/// Fuse per-box instance masks with projected lidar points.
/// Each per-box mask is placed at its detection box coordinates in image space.
fn boxed_mask_fusion(
    frame: &mut FusionFrame,
    masks: &[Mask],
    boxes: &[DetectBox],
    labels: Option<&[String]>,
    ids: &HashMap<u32, Vec<usize>>,
) {
    let prepared = prepare_box_masks(masks, boxes, labels);

    if frame.has_clusters() {
        boxed_mask_fusion_clustered(frame, &prepared, ids);
    } else {
        boxed_mask_fusion_no_cluster(frame, &prepared);
    }
}

fn boxed_mask_fusion_no_cluster(frame: &mut FusionFrame, prepared: &[PreparedBoxMask]) {
    let n = frame.len;

    for i in 0..n {
        let u = frame.proj_u[i];
        let v = frame.proj_v[i];
        if !check_in_bounds(&u, &v) {
            continue;
        }
        let (cls, inst, trk) = lookup_point_in_masks(u, v, prepared);
        frame.vision_class[i] = cls;
        frame.instance_id[i] = inst;
        frame.track_id[i] = trk;
    }

    // Edge tolerance: check 8 surrounding points for unclassified pixels
    let point_radius = 0.02f32;
    for i in 0..n {
        if frame.vision_class[i] != 0 {
            continue;
        }
        let u = frame.proj_u[i];
        let v = frame.proj_v[i];
        for &(sin_a, cos_a) in &EDGE_OFFSETS {
            let nu = u + point_radius * sin_a;
            let nv = v + point_radius * cos_a;
            if !check_in_bounds(&nu, &nv) {
                continue;
            }
            let (cls, inst, trk) = lookup_point_in_masks(nu, nv, prepared);
            if cls != 0 {
                frame.vision_class[i] = cls;
                frame.instance_id[i] = inst;
                frame.track_id[i] = trk;
                break;
            }
        }
    }

    filter_depth_outliers(frame);
}

fn boxed_mask_fusion_clustered(
    frame: &mut FusionFrame,
    prepared: &[PreparedBoxMask],
    clusters: &HashMap<u32, Vec<usize>>,
) {
    for point_indices in clusters.values() {
        // Majority vote on instance_id among projected pixels in this cluster
        let mut votes: HashMap<u16, (u8, u32, usize)> = HashMap::new();
        for &idx in point_indices {
            let u = frame.proj_u[idx];
            let v = frame.proj_v[idx];
            if !check_in_bounds(&u, &v) {
                continue;
            }
            let (cls, inst, trk) = lookup_point_in_masks(u, v, prepared);
            if cls > 0 {
                let entry = votes.entry(inst).or_insert((cls, trk, 0));
                entry.2 += 1;
            }
        }
        // Pick the instance with the most votes
        if let Some((&best_inst, &(best_class, best_track, _))) =
            votes.iter().max_by_key(|(_, (_, _, count))| *count)
        {
            for &idx in point_indices {
                frame.vision_class[idx] = best_class;
                frame.instance_id[idx] = best_inst;
                frame.track_id[idx] = best_track;
            }
        }
    }

    filter_depth_outliers(frame);
}

fn get_fusion_class(
    frame: &mut FusionFrame,
    fusion_predictions: &[Box2D],
    ids: &HashMap<u32, Vec<usize>>,
) {
    if frame.has_clusters() {
        grid_nearest_cluster(fusion_predictions, frame, ids);
    } else {
        grid_nearest_point_no_cluster(fusion_predictions, frame);
    }
}

async fn get_fusion_predictions(
    track: bool,
    tracker: &mut ByteTrack,
    grid: &Arc<Mutex<Option<Grid>>>,
    args: &Args,
    session: &Session,
) -> Vec<Box2D> {
    if track {
        grid_radar_tracked(grid, tracker, args, session).await
    } else {
        grid_radar(grid, args).await
    }
}

#[instrument(skip_all)]
async fn publish_bbox3d(
    bbox_publ: Option<&Publisher<'_>>,
    header: &Header,
    frame: &FusionFrame,
    ids: &HashMap<u32, Vec<usize>>,
) {
    if bbox_publ.is_none() {
        return;
    }
    // Only create 3d bbox message when there are cluster IDs and bbox publishing is
    // enabled
    if !frame.has_clusters() {
        return;
    }

    let bbox_publ = bbox_publ.unwrap();
    let (buf_bbox, enc_bbox) = get_3d_bbox(header, frame, ids);

    match bbox_publ.put(buf_bbox).encoding(enc_bbox).await {
        Ok(_) => trace!("Message Sent on {:?}", bbox_publ.key_expr()),
        Err(e) => error!("Message Error on {:?}: {:?}", bbox_publ.key_expr(), e),
    }
}

#[instrument(skip_all)]
async fn publish_output(
    publ: Option<&Publisher<'_>>,
    frame: &FusionFrame,
    header: &Header,
    has_fusion_model: bool,
    tracking: bool,
) {
    let publ = match publ {
        Some(p) => p,
        None => return,
    };
    let pcd = if has_fusion_model {
        serialize_classes(frame, header)
    } else {
        serialize_late_fusion(frame, header, tracking)
    };
    let buf = ZBytes::from(serde_cdr::serialize(&pcd).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("sensor_msgs/msg/PointCloud2");
    match publ.put(buf).encoding(enc).await {
        Ok(_) => trace!("Message Sent on {:?}", publ.key_expr()),
        Err(e) => error!("Message Error on {:?}: {:?}", publ.key_expr(), e),
    }
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
async fn publish_grid(
    grid_publ: Option<&Publisher<'_>>,
    header: &Header,
    frame: &FusionFrame,
    point_tracker: &mut ByteTrack,
    bins_data: (&mut [Vec<Bin>], u128),
    args: &Args,
    ids: &HashMap<u32, Vec<usize>>,
) {
    if grid_publ.is_none() {
        return;
    };
    let grid_publ = grid_publ.unwrap();
    let (buf_grid, enc_grid) = if frame.has_clusters() {
        get_occupied_cluster(header, frame, ids, point_tracker)
    } else {
        let (bins, frame_index) = bins_data;
        get_occupied_no_cluster(header, frame, bins, frame_index, args)
    };
    match grid_publ.put(buf_grid).encoding(enc_grid).await {
        Ok(_) => trace!("Message Sent on {:?}", grid_publ.key_expr()),
        Err(e) => error!("Message Error on {:?}: {:?}", grid_publ.key_expr(), e),
    }
}

// Gets 3D bounding boxes from the PCD points. Any clusters that have class > 0
// will get a 3D bounding box generated. It is assumed that all points in the
// same cluster ID will have the same class.
fn get_3d_bbox(
    header: &Header,
    frame: &FusionFrame,
    cluster_ids: &HashMap<u32, Vec<usize>>,
) -> (ZBytes, Encoding) {
    let mut bbox_3d = Vec::new();
    for inds in cluster_ids.values() {
        if inds.is_empty() {
            continue;
        }

        // assumes that all points with the same class ID has the same class
        let class = frame.vision_class[inds[0]];
        if class == 0 {
            continue;
        }

        // get the max and min x,y,z values of the cluster
        let (mut x_max, mut y_max, mut z_max) = (
            -UNINITIALIZED_COORD,
            -UNINITIALIZED_COORD,
            -UNINITIALIZED_COORD,
        );
        let (mut x_min, mut y_min, mut z_min) = (
            UNINITIALIZED_COORD,
            UNINITIALIZED_COORD,
            UNINITIALIZED_COORD,
        );

        for ind in inds.iter() {
            debug_assert_eq!(class, frame.vision_class[*ind]);
            x_max = x_max.max(frame.x[*ind]);
            x_min = x_min.min(frame.x[*ind]);

            y_max = y_max.max(frame.y[*ind]);
            y_min = y_min.min(frame.y[*ind]);

            z_max = z_max.max(frame.z[*ind]);
            z_min = z_min.min(frame.z[*ind]);
        }

        // Add a 3D box using the max and min x,y,z values
        // TODO: Add 3D tracking to improve smoothness
        let inst = frame.instance_id[inds[0]];
        bbox_3d.push(DetectBox {
            center_x: -(y_max + y_min) / 2.0, // we use an optical frame, so positive X is right
            center_y: -(z_max + z_min) / 2.0, // we use an optical frame, so positive Y is down
            width: (y_max - y_min),
            height: (z_max - z_min),
            distance: (x_max + x_min) / 2.0,
            label: class.to_string(),
            score: 1.0,
            speed: 0.0,
            track: Track {
                id: if inst > 0 {
                    inst.to_string()
                } else {
                    "".to_string()
                },
                lifetime: 0,
                created: header.stamp.clone(),
            },
        });
    }
    let new_msg = Detect {
        input_timestamp: header.stamp.clone(),
        model_time: Time { sec: 0, nanosec: 0 },
        output_time: header.stamp.clone(),
        boxes: bbox_3d,
        header: Header {
            stamp: header.stamp.clone(),
            frame_id: header.frame_id.clone(),
        },
    };
    let msg = ZBytes::from(serde_cdr::serialize(&new_msg).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Detect");
    (msg, enc)
}

async fn tf_static(
    session: Session,
    msg: ZBytes,
    enc: Encoding,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let topic = "rt/tf_static".to_string();
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    loop {
        interval.tick().await;
        session
            .put(&topic, msg.clone())
            .encoding(enc.clone())
            .await?;
    }
}

fn build_tf_msg() -> TransformStamped {
    TransformStamped {
        header: Header {
            frame_id: BASE_LINK_FRAME_ID.to_string(),
            stamp: Time { sec: 0, nanosec: 0 },
        },
        child_frame_id: format!("{BASE_LINK_FRAME_ID}_optical"),
        transform: Transform {
            translation: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            // Un-normalized quaternion; will be normalized by UnitQuaternion::new_normalize
            // in transform.rs when applied
            rotation: Quaternion {
                x: -1.0,
                y: 1.0,
                z: -1.0,
                w: 1.0,
            },
        },
    }
}

#[instrument(skip_all)]
// For each predicted grid box, find the nearest point in the PCD. If the
// nearest point is within 2m, set the class of the point to the class of the
// grid box
fn grid_nearest_point_no_cluster(fusion_predictions: &[Box2D], frame: &mut FusionFrame) {
    for b in fusion_predictions {
        let mut min_dist2 = f32::MAX;
        let mut min_point_ind = 0;
        for ind in 0..frame.len {
            let dist2 = (frame.x[ind] - b.center_x).powi(2) + (frame.y[ind] - b.center_y).powi(2);
            if dist2 < min_dist2 {
                min_dist2 = dist2;
                min_point_ind = ind;
            }
        }
        if min_dist2 < MAX_CLASSIFICATION_DISTANCE * MAX_CLASSIFICATION_DISTANCE {
            frame.fusion_class[min_point_ind] = b.label;
        }
    }
}

// For each predicted grid box, find the nearest cluster in the PCD. If the
// nearest cluster is within 2m, set the class of all points in the cluster to
// the class of the grid box
fn grid_nearest_cluster(
    fusion_predictions: &[Box2D],
    frame: &mut FusionFrame,
    clusters: &HashMap<u32, Vec<usize>>,
) {
    for b in fusion_predictions {
        let mut min_dist2 = f32::MAX;
        let mut min_point_ind = 0;
        for ind in 0..frame.len {
            if frame.cluster_id[ind] == 0 {
                continue;
            }

            let dist2 = (frame.x[ind] - b.center_x).powi(2) + (frame.y[ind] - b.center_y).powi(2);
            if dist2 < min_dist2 {
                min_dist2 = dist2;
                min_point_ind = ind;
            }
        }
        if min_dist2 > MAX_CLASSIFICATION_DISTANCE * MAX_CLASSIFICATION_DISTANCE {
            continue;
        }
        let cluster_id = frame.cluster_id[min_point_ind];
        if cluster_id == 0 {
            continue;
        }
        if let Some(indices) = clusters.get(&cluster_id) {
            for ind in indices {
                frame.fusion_class[*ind] = b.label;
            }
        }
    }
}

/// For each point, get the class of the point using the projection onto the
/// mask. If point_radius > 0, then also checks 8 points in a circle around the
/// projection, and uses the first non-zero class found as the class of
/// the point.
///
/// Assumes that the mask is already argmax'd
#[instrument(skip_all)]
fn late_fusion_no_cluster(frame: &mut FusionFrame, mask: &Mask, point_radius: f32) {
    let mask_height = mask.height as usize;
    let mask_width = mask.width as usize;
    let index_mask = |x: f32, y: f32| -> u8 {
        let x = (x * mask_width as f32) as usize;
        let y = (y * mask_height as f32) as usize;
        mask.mask[y * mask_width + x]
    };

    for i in 0..frame.len {
        let x = frame.proj_u[i];
        let y = frame.proj_v[i];
        if !check_in_bounds(&x, &y) {
            continue;
        }
        frame.vision_class[i] = index_mask(x, y);
    }

    if point_radius <= 0.0 {
        return;
    }

    for i in 0..frame.len {
        if frame.vision_class[i] != 0 {
            continue;
        }
        let x = frame.proj_u[i];
        let y = frame.proj_v[i];
        for &(sin_a, cos_a) in &EDGE_OFFSETS {
            let new_x = x + point_radius * sin_a;
            let new_y = y + point_radius * cos_a;
            if !check_in_bounds(&new_x, &new_y) {
                continue;
            }
            let argmax = index_mask(new_x, new_y);
            if argmax != 0 {
                frame.vision_class[i] = argmax;
                break;
            }
        }
    }
}

fn check_in_bounds(x: &f32, y: &f32) -> bool {
    (0.0..1.0).contains(x) && (0.0..1.0).contains(y)
}

/// For each mask instance, (found using flood fill of the mask), find the most
/// common cluster ID that projects only the mask. All points with that cluster
/// ID will become the class of that mask instance.
/// Assumes that the mask is already argmax'd
#[instrument(skip_all)]
fn late_fusion_clustered(
    frame: &mut FusionFrame,
    mask: &Mask,
    clusters: &HashMap<u32, Vec<usize>>,
) {
    let mask_height = mask.height as usize;
    let mask_width = mask.width as usize;
    let index_mask = |x: f32, y: f32| -> u8 {
        let x = (x * mask_width as f32) as usize;
        let y = (y * mask_height as f32) as usize;
        mask.mask[y * mask_width + x]
    };

    let bbox_2d = mask_instance(&mask.mask, mask_width);
    let mut bbox_id = Vec::new();
    for b in &bbox_2d {
        let mut bbox_cluster_ids = Vec::new();
        for i in 0..frame.len {
            let x = frame.proj_u[i];
            let y = frame.proj_v[i];
            if (0.0..1.0).contains(&y)
                && (0.0..1.0).contains(&x)
                && (b.center_x - x).abs() <= b.width / 2.0
                && (b.center_y - y).abs() <= b.height / 2.0
                && frame.cluster_id[i] > 0
                && index_mask(x, y) == b.label
            {
                bbox_cluster_ids.push(frame.cluster_id[i]);
            }
        }
        if let Some(cluster_id) = mode_slice(&bbox_cluster_ids) {
            bbox_id.push(*cluster_id);
        }
    }
    for (box2d, cluster_id) in bbox_2d.into_iter().zip(bbox_id) {
        if let Some(indices) = clusters.get(&cluster_id) {
            for i in indices {
                frame.vision_class[*i] = box2d.label;
            }
        }
    }
}

/// Checks if there are any cluster IDs in the PCD and builds the cluster map.
/// Noise points (id == 0) are not included in the HashMap.
fn get_cluster_ids(frame: &FusionFrame) -> HashMap<u32, Vec<usize>> {
    let mut cluster_ids: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &id) in frame.cluster_id.iter().enumerate() {
        if id == 0 {
            // we ignore noise points
            continue;
        }
        cluster_ids.entry(id).or_default().push(i);
    }
    cluster_ids
}

async fn grid_radar_tracked(
    grid: &Arc<Mutex<Option<Grid>>>,
    grid_tracker: &mut ByteTrack,
    args: &Args,
    session: &Session,
) -> Vec<Box2D> {
    let mut class = Vec::new();

    let guard = grid.lock().await;
    if guard.is_none() {
        return class;
    }
    let (g, timestamp) = guard.as_ref().unwrap();

    if *timestamp > grid_tracker.timestamp {
        grid_radar_update_tracker(g, timestamp, grid_tracker, args, session).await;
    }

    for tracklet in grid_tracker.get_tracklets() {
        if tracklet.count < 2 {
            continue;
        }
        let pred = tracklet.get_predicted_location();
        let i = (pred.ymin + pred.ymax) / 2.0;
        let j = (pred.xmin + pred.xmax) / 2.0;

        // center of grid
        let width = g[0].len();
        let (x, y) = grid_to_xy(i, j, width, args);

        class.push(Box2D {
            center_x: x,
            center_y: y,
            width: args.model_grid_size[0],
            height: args.model_grid_size[1],
            label: 1,
        });
    }

    class
}

async fn grid_radar_update_tracker(
    g: &[Vec<f32>],
    timestamp: &u64,
    grid_tracker: &mut ByteTrack,
    args: &Args,
    session: &Session,
) {
    let height = g.len();
    let width = g[0].len();

    let mut boxes = Vec::new();
    for (i, g_i) in g.iter().enumerate() {
        for (j, g_ij) in g_i.iter().enumerate() {
            if *g_ij < args.model_threshold {
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

    grid_tracker.update(&mut boxes, *timestamp);

    {
        let mut tracked_g = vec![vec![0.0; width]; height];
        for tracklet in grid_tracker.get_tracklets() {
            if tracklet.count < 3 {
                continue;
            }
            let pred = tracklet.get_predicted_location();
            let i = ((pred.ymin + pred.ymax) / 2.0).round() as i32;
            let j = ((pred.xmin + pred.xmax) / 2.0).round() as i32;
            if i < 0 || i >= height as i32 || j < 0 || j >= width as i32 {
                continue;
            }

            tracked_g[i as usize][j as usize] = 1.0
        }

        let mask = tracked_g
            .iter()
            .flatten()
            .flat_map(|v| [128, (*v * 255.0f64).min(255.0) as u8])
            .collect();
        let msg = Mask {
            height: height as u32,
            width: width as u32,
            length: 1,
            encoding: "".to_string(),
            mask,
            boxed: false,
        };

        let buf = ZBytes::from(serde_cdr::serialize(&msg).unwrap());
        let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

        session
            .put(format!("{}/tracked", args.model_output_topic), buf)
            .encoding(enc)
            .await
            .unwrap();
    }
}

#[instrument(skip_all)]
async fn grid_radar(grid: &Arc<Mutex<Option<Grid>>>, args: &Args) -> Vec<Box2D> {
    let mut class = Vec::new();

    let guard = grid.lock().await;
    if guard.is_none() {
        return class;
    }

    let (g, _) = guard.as_ref().unwrap();

    let width = g[0].len();
    for (i, g_i) in g.iter().enumerate() {
        for (j, g_ij) in g_i.iter().enumerate() {
            if *g_ij < args.model_threshold {
                continue;
            }
            // center of grid
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

// Half of the width of the model is used to offset the j value
fn grid_to_xy(i: f32, j: f32, width: usize, args: &Args) -> (f32, f32) {
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

fn centroids_get_class(cluster_ids: &HashMap<u32, Vec<usize>>, frame: &FusionFrame) -> FusionFrame {
    let capacity = cluster_ids.len();
    let mut centroid = FusionFrame::new(capacity);

    for id in cluster_ids {
        // sanity check, should not have cluster_ids with no points
        if id.1.is_empty() {
            continue;
        }
        let vision_class = frame.vision_class[id.1[0]];
        let fusion_class = frame.fusion_class[id.1[0]];
        let instance_id = frame.instance_id[id.1[0]];
        let mut xyzv = id.1.iter().fold([0.0, 0.0, 0.0, 0.0], |mut xyzv, ind| {
            xyzv[0] += frame.x[*ind];
            xyzv[1] += frame.y[*ind];
            xyzv[2] += frame.z[*ind];
            xyzv
        });
        for v in xyzv.iter_mut() {
            *v /= id.1.len() as f32
        }

        centroid.x.push(xyzv[0]);
        centroid.y.push(xyzv[1]);
        centroid.z.push(xyzv[2]);
        centroid.cluster_id.push(*id.0);
        centroid.vision_class[centroid.len] = vision_class;
        centroid.fusion_class[centroid.len] = fusion_class;
        centroid.instance_id[centroid.len] = instance_id;
        centroid.len += 1;
    }
    centroid
}

/// Update centroid classes using the ByteTrack multi-object tracker.
fn centroids_update_tracker_classes(
    centroid: &mut FusionFrame,
    point_tracker: &mut ByteTrack,
    timestamp: u64,
) {
    let mut boxes: Vec<TrackerBox> = (0..centroid.len)
        .map(|ind| TrackerBox {
            xmin: centroid.x[ind] - 0.5,
            xmax: centroid.x[ind] + 0.5,
            ymin: centroid.y[ind] - 0.5,
            ymax: centroid.y[ind] + 0.5,
            score: if centroid.vision_class[ind] > 0 || centroid.fusion_class[ind] > 0 {
                1.0
            } else {
                0.3
            },
            vision_class: centroid.vision_class[ind],
            fusion_class: centroid.fusion_class[ind],
        })
        .collect();
    let track_info = point_tracker.update(&mut boxes, timestamp);
    for (i, inf) in track_info.into_iter().enumerate() {
        if inf.is_none() {
            continue;
        }
        let uuid = inf.unwrap().uuid;
        let tracklet = point_tracker
            .get_tracklet_from_uuid(&uuid)
            .expect("Got invalid UUID for track match");
        if tracklet.last_updated_high_conf
            + ((point_tracker.settings.track_extra_lifespan * 1e9) as u64)
            < timestamp
        {
            continue;
        }
        if boxes[i].vision_class == 0 {
            centroid.vision_class[i] = point_tracker.uuid_map_vision_class[&uuid];
        }
        if boxes[i].fusion_class == 0 {
            centroid.fusion_class[i] = point_tracker.uuid_map_fusion_class[&uuid];
        }
    }
}

fn centroids_add_tracks(centroid: &mut FusionFrame, point_tracker: &mut ByteTrack, timestamp: u64) {
    for i in point_tracker.get_tracklets() {
        if i.last_updated == timestamp {
            continue;
        }
        if i.last_updated_high_conf + ((point_tracker.settings.track_extra_lifespan * 1e9) as u64)
            < timestamp
        {
            continue;
        }
        if i.count < 5 {
            continue;
        }
        // this track wasn't updated

        let predicted = i.get_predicted_location();
        let px = (predicted.xmin + predicted.xmax) / 2.0;
        let py = (predicted.ymin + predicted.ymax) / 2.0;

        centroid.x.push(px);
        centroid.y.push(py);
        centroid.z.push(0.0);
        centroid.cluster_id.push(0);
        // Extend annotation arrays
        if centroid.len >= centroid.vision_class.len() {
            centroid
                .vision_class
                .push(point_tracker.uuid_map_vision_class[&i.id]);
            centroid
                .fusion_class
                .push(point_tracker.uuid_map_fusion_class[&i.id]);
            centroid.instance_id.push(0);
            centroid.track_id.push(0);
        } else {
            centroid.vision_class[centroid.len] = point_tracker.uuid_map_vision_class[&i.id];
            centroid.fusion_class[centroid.len] = point_tracker.uuid_map_fusion_class[&i.id];
            centroid.instance_id[centroid.len] = 0;
            centroid.track_id[centroid.len] = 0;
        }
        centroid.len += 1;
        trace!("added extra point");
    }
}

/// Returns the centroid of clusters that have non-zero class_id. All points in
/// a class should have the same class_id
fn get_occupied_cluster(
    header: &Header,
    frame: &FusionFrame,
    cluster_ids: &HashMap<u32, Vec<usize>>,
    point_tracker: &mut ByteTrack,
) -> (ZBytes, Encoding) {
    let mut centroid = centroids_get_class(cluster_ids, frame);
    // want to track points that have class != 0
    let timestamp = header.stamp.to_nanos();
    centroids_update_tracker_classes(&mut centroid, point_tracker, timestamp);
    centroids_add_tracks(&mut centroid, point_tracker, timestamp);

    let pcd = serialize_grid(&centroid, header);
    let buf = ZBytes::from(serde_cdr::serialize(&pcd).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("sensor_msgs/msg/PointCloud2");

    (buf, enc)
}

fn update_bins(frame: &FusionFrame, bins: &mut [Vec<Bin>], args: &Args) {
    let n = frame.len;

    // Batch-compute range and angle using SIMD-accelerated functions
    let mut ranges = vec![0.0f32; n];
    let mut angles = vec![0.0f32; n];
    simd::magnitude3_f32(&frame.x[..n], &frame.y[..n], &frame.z[..n], &mut ranges);
    simd::atan2_f32(&frame.y[..n], &frame.x[..n], &mut angles);

    // Convert radians to degrees and assign bins (scalar — involves Vec pushes)
    for ind in 0..n {
        let mut angle = angles[ind].to_degrees();
        let mut range = ranges[ind];

        angle = angle.clamp(args.angle_bin_limit[0], args.angle_bin_limit[1] - 0.001);
        range = range.clamp(args.range_bin_limit[0], args.range_bin_limit[1] - 0.001);

        let i = ((angle - args.angle_bin_limit[0]) / args.angle_bin_width).floor() as usize;
        let j = ((range - args.range_bin_limit[0]) / args.range_bin_width).floor() as usize;
        let vision_class = frame.vision_class[ind];
        if vision_class > 0 {
            bins[i][j].vision_classes.push(vision_class);
        }

        let fusion_class = frame.fusion_class[ind];
        if fusion_class > 0 {
            bins[i][j].fusion_classes.push(fusion_class);
        }
    }
}

fn mark_grid_one_column(
    bins: &mut [Vec<Bin>],
    frame_index: u128,
    angle_found_occupied: &mut [bool],
    args: &Args,
) {
    for i in 0..bins.len() {
        for j in 0..bins[i].len() {
            let sum0 = get_val_in_bin(bins, i as i32, j as i32, 0, 0);
            let sum1 = get_val_in_bin(bins, i as i32, j as i32, 0, -1);
            let sum2 = get_val_in_bin(bins, i as i32, j as i32, 0, -2);
            if sum0 >= args.threshold {
                mark_grid(&mut bins[i][j], frame_index);
                angle_found_occupied[i] = true;
                // don't check more ranges
                break;
            }
            if j >= 1 && sum0 + sum1 >= args.threshold {
                mark_grid(&mut bins[i][j - 1], frame_index);
                angle_found_occupied[i] = true;
                // don't check more ranges
                break;
            }
            if j >= 2 && sum0 + sum1 + sum2 >= args.threshold {
                mark_grid(&mut bins[i][j - 2], frame_index);
                angle_found_occupied[i] = true;
                // don't check more ranges
                break;
            }
        }
    }
}

fn mark_cell_three_column(
    i: usize,
    j: usize,
    bins: &mut [Vec<Bin>],
    frame_index: u128,
    angle_found_occupied: &mut [bool],
    args: &Args,
) -> bool {
    let mut sum0 = get_val_in_bin(bins, i as i32, j as i32, 0, 0);
    let mut sum1 = get_val_in_bin(bins, i as i32, j as i32, 0, -1);
    let mut sum2 = get_val_in_bin(bins, i as i32, j as i32, 0, -2);
    if 0 < i && !angle_found_occupied[i - 1] {
        sum0 += get_val_in_bin(bins, i as i32, j as i32, -1, 0);
        sum1 += get_val_in_bin(bins, i as i32, j as i32, -1, -1);
        sum2 += get_val_in_bin(bins, i as i32, j as i32, -1, -2);
    }
    if i + 1 < bins.len() && !angle_found_occupied[i + 1] {
        sum0 += get_val_in_bin(bins, i as i32, j as i32, 1, 0);
        sum1 += get_val_in_bin(bins, i as i32, j as i32, 1, -1);
        sum2 += get_val_in_bin(bins, i as i32, j as i32, 1, -2);
    }

    let mut set_bins_occupied = |i: usize, bins: &mut [Vec<Bin>]| {
        angle_found_occupied[i] = true;
        if 0 < i {
            angle_found_occupied[i - 1] = true;
        }
        if i + 1 < bins.len() {
            angle_found_occupied[i + 1] = true;
        }
    };

    if sum0 >= args.threshold {
        mark_grid(&mut bins[i][j], frame_index);
        set_bins_occupied(i, bins);
        // don't check more ranges
        return true;
    }
    if j >= 1 && sum0 + sum1 >= args.threshold {
        mark_grid(&mut bins[i][j - 1], frame_index);
        set_bins_occupied(i, bins);
        // don't check more ranges
        return true;
    }
    if j >= 2 && sum0 + sum1 + sum2 >= args.threshold {
        mark_grid(&mut bins[i][j - 2], frame_index);
        set_bins_occupied(i, bins);
        // don't check more ranges
        return true;
    }
    false
}

fn mark_grid_three_column(
    bins: &mut [Vec<Bin>],
    frame_index: u128,
    angle_found_occupied: &mut [bool],
    args: &Args,
) {
    for i in 0..bins.len() {
        for j in 0..bins[i].len() {
            if mark_cell_three_column(i, j, bins, frame_index, angle_found_occupied, args) {
                break;
            }
        }
    }
}

fn find_marked_bins(bins: &[Vec<Bin>], frame_index: u128, args: &Args) -> FusionFrame {
    let mut grid_frame = FusionFrame::new(0);

    let mut angle_found_marked = vec![false; bins.len()];
    for thresh in 0..=args.bin_delay {
        for i in 0..bins.len() {
            if angle_found_marked[i] {
                continue;
            }
            for j in 0..bins[i].len() {
                if bins[i][j].first_marked != u128::MAX
                    && frame_index - bins[i][j].first_marked >= args.bin_delay
                    && frame_index - bins[i][j].last_masked <= thresh
                {
                    let (x, y, vision, fusion) = draw_point(bins, i, j, args);
                    grid_frame.x.push(x);
                    grid_frame.y.push(y);
                    grid_frame.z.push(0.0);
                    grid_frame.vision_class.push(vision);
                    grid_frame.fusion_class.push(fusion);
                    grid_frame.instance_id.push(0);
                    grid_frame.track_id.push(0);
                    grid_frame.len += 1;
                    angle_found_marked[i] = true;
                    // don't check more ranges
                    break;
                }
            }
        }
    }
    grid_frame
}

/// Do a grid and highlight the grid based on point classes
fn get_occupied_no_cluster(
    header: &Header,
    frame: &FusionFrame,
    bins: &mut [Vec<Bin>],
    frame_index: u128,
    args: &Args,
) -> (ZBytes, Encoding) {
    update_bins(frame, bins, args);

    let mut angle_found_occupied = vec![false; bins.len()];
    mark_grid_one_column(bins, frame_index, &mut angle_found_occupied, args);
    mark_grid_three_column(bins, frame_index, &mut angle_found_occupied, args);

    let grid_frame = find_marked_bins(bins, frame_index, args);
    let pcd = serialize_grid(&grid_frame, header);
    let buf = ZBytes::from(serde_cdr::serialize(&pcd).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("sensor_msgs/msg/PointCloud2");

    (buf, enc)
}

/// Returns the mode of the slice. Returns None if the slice is empty.
/// From https://stackoverflow.com/a/50000027
fn mode_slice<T: Ord + Hash>(numbers: &[T]) -> Option<&T> {
    let mut counts = HashMap::new();

    numbers.iter().max_by_key(|&n| {
        let count = counts.entry(n).or_insert(0);
        *count += 1;
        *count
    })
}

fn clear_bins(bins: &mut Vec<Vec<Bin>>, curr: u128, args: &Args) {
    for i in bins {
        for j in i {
            j.vision_classes.clear();
            j.fusion_classes.clear();
            if j.last_masked + args.bin_delay < curr {
                j.first_marked = u128::MAX;
            }
        }
    }
}

fn get_val_in_bin(bins: &[Vec<Bin>], i: i32, j: i32, offset_i: i32, offset_j: i32) -> u32 {
    if i + offset_i < 0 {
        return 0;
    }
    if i + offset_i >= bins.len() as i32 {
        return 0;
    }

    if j + offset_j < 0 {
        return 0;
    }
    if j + offset_j > bins[0].len() as i32 {
        return 0;
    }

    let radar = bins[(i + offset_i) as usize][(j + offset_j) as usize]
        .fusion_classes
        .len() as u32;
    let vision = bins[(i + offset_i) as usize][(j + offset_j) as usize]
        .vision_classes
        .len() as u32;

    radar + vision
}

fn mark_grid(bin: &mut Bin, curr: u128) {
    bin.last_masked = curr;
    if bin.first_marked == u128::MAX {
        bin.first_marked = curr;
    }
}

fn draw_point(bins: &[Vec<Bin>], i: usize, j: usize, args: &Args) -> (f32, f32, u8, u8) {
    let vision_class = *mode_slice(bins[i][j].vision_classes.as_slice()).unwrap_or(&0);
    let fusion_class = *mode_slice(bins[i][j].fusion_classes.as_slice()).unwrap_or(&0);

    let angle = args.angle_bin_width * (i as f32 + 0.5) + args.angle_bin_limit[0];
    let range = args.range_bin_width * (j as f32 + 0.5) + args.range_bin_limit[0];
    let x = angle.to_radians().cos() * range;
    let y = angle.to_radians().sin() * range;
    (x, y, vision_class, fusion_class)
}

struct Bin {
    vision_classes: Vec<u8>,
    fusion_classes: Vec<u8>,
    last_masked: u128,
    first_marked: u128,
}

struct DrainRecvTimeoutSettings {
    initial_timeout: std::time::Duration,
    max_timeout: std::time::Duration,
    curr_timeout: std::time::Duration,
    multiplier: u32,
}

impl Default for DrainRecvTimeoutSettings {
    fn default() -> Self {
        Self {
            initial_timeout: Duration::from_secs(2),
            curr_timeout: Duration::from_secs(2),
            max_timeout: Duration::from_secs(3600),
            multiplier: 2,
        }
    }
}

/// If the receiver is empty, waits for the next message, otherwise returns the
/// most recent message on this receiver. If the receiver times out or is
/// closed, returns None
async fn drain_recv(
    sub: &Subscriber<zenoh::handlers::FifoChannelHandler<Sample>>,
    timeout: &mut DrainRecvTimeoutSettings,
) -> Option<Sample> {
    if let Some(v) = sub.drain().last() {
        Some(v)
    } else {
        match sub.recv_timeout(timeout.curr_timeout) {
            Ok(v) => match v {
                Some(v) => {
                    timeout.curr_timeout = timeout.initial_timeout;
                    Some(v)
                }
                None => {
                    warn!(
                        "Timeout on {} after waiting for {:?}",
                        sub.key_expr(),
                        timeout.curr_timeout
                    );
                    timeout.curr_timeout =
                        (timeout.curr_timeout * timeout.multiplier).min(timeout.max_timeout);
                    None
                }
            },
            Err(e) => {
                error!("error receiving radar cube on {}: {:?}", sub.key_expr(), e);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefirst_schemas::builtin_interfaces::Duration;

    fn make_box(cx: f32, cy: f32, w: f32, h: f32, label: &str, track_id: &str) -> DetectBox {
        DetectBox {
            center_x: cx,
            center_y: cy,
            width: w,
            height: h,
            label: label.to_string(),
            score: 1.0,
            distance: 0.0,
            speed: 0.0,
            track: Track {
                id: track_id.to_string(),
                lifetime: 0,
                created: Time { sec: 0, nanosec: 0 },
            },
        }
    }

    fn make_model(boxes: Vec<DetectBox>, masks: Vec<Mask>) -> Model {
        Model {
            header: Header {
                stamp: Time { sec: 0, nanosec: 0 },
                frame_id: String::new(),
            },
            input_time: Duration { sec: 0, nanosec: 0 },
            model_time: Duration { sec: 0, nanosec: 0 },
            output_time: Duration { sec: 0, nanosec: 0 },
            decode_time: Duration { sec: 0, nanosec: 0 },
            boxes,
            masks,
        }
    }

    /// Helper to build a FusionFrame with projection data set.
    fn make_frame_with_proj(
        points: &[(f32, f32, f32)],
        cluster_ids: Option<&[u32]>,
        proj: &[[f32; 2]],
    ) -> FusionFrame {
        let n = points.len();
        let mut frame = FusionFrame::new(n);
        for (x, y, z) in points {
            frame.x.push(*x);
            frame.y.push(*y);
            frame.z.push(*z);
        }
        if let Some(cids) = cluster_ids {
            frame.cluster_id = cids.to_vec();
        }
        frame.proj_u = proj.iter().map(|p| p[0]).collect();
        frame.proj_v = proj.iter().map(|p| p[1]).collect();
        frame.len = n;
        frame
    }

    #[test]
    fn test_box_fusion_no_cluster_single_box() {
        let boxes = vec![make_box(0.5, 0.5, 0.4, 0.4, "3", "track-abc")];
        let proj = vec![[0.5, 0.5], [0.1, 0.1]];
        let mut frame = make_frame_with_proj(&[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], None, &proj);
        box_fusion_no_cluster(&mut frame, &boxes, None);

        assert_eq!(
            frame.vision_class[0], 3,
            "point inside box should get class 3"
        );
        assert_eq!(frame.instance_id[0], 1, "serial instance_id should be 1");
        assert_ne!(
            frame.track_id[0], 0,
            "tracked box should have non-zero track_id"
        );
        assert_eq!(
            frame.vision_class[1], 0,
            "point outside box should get class 0"
        );
        assert_eq!(
            frame.instance_id[1], 0,
            "point outside box should get instance 0"
        );
        assert_eq!(
            frame.track_id[1], 0,
            "point outside box should get track_id 0"
        );
    }

    #[test]
    fn test_box_fusion_no_cluster_no_track_id() {
        let boxes = vec![make_box(0.5, 0.5, 0.4, 0.4, "2", "")];
        let proj = vec![[0.5, 0.5]];
        let mut frame = make_frame_with_proj(&[(1.0, 0.0, 0.0)], None, &proj);
        box_fusion_no_cluster(&mut frame, &boxes, None);

        assert_eq!(frame.vision_class[0], 2);
        assert_eq!(frame.instance_id[0], 1, "serial instance_id should be 1");
        assert_eq!(
            frame.track_id[0], 0,
            "empty track_id should produce track_id 0"
        );
    }

    #[test]
    fn test_box_fusion_clustered_assigns_whole_cluster() {
        let proj = vec![[0.5, 0.5], [0.55, 0.5], [0.6, 0.5]];
        let mut frame = make_frame_with_proj(
            &[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
            Some(&[10, 10, 20]),
            &proj,
        );
        let boxes = vec![make_box(0.55, 0.5, 0.2, 0.2, "5", "uuid-xyz")];
        let mut clusters: HashMap<u32, Vec<usize>> = HashMap::new();
        clusters.insert(10, vec![0, 1]);
        clusters.insert(20, vec![2]);

        box_fusion_clustered(&mut frame, &boxes, &clusters, None);

        assert_eq!(
            frame.vision_class[0], 5,
            "first point of winning cluster should get class"
        );
        assert_eq!(
            frame.vision_class[1], 5,
            "second point of winning cluster should get class"
        );
        assert_eq!(
            frame.instance_id[0], frame.instance_id[1],
            "all points in cluster get same instance"
        );
        assert_ne!(frame.instance_id[0], 0, "instance should be non-zero");
        assert_ne!(
            frame.track_id[0], 0,
            "track_id should be non-zero for tracked box"
        );
        assert_eq!(
            frame.track_id[0], frame.track_id[1],
            "all points in cluster get same track_id"
        );
        assert_eq!(
            frame.vision_class[2], 0,
            "losing cluster should not get class"
        );
        assert_eq!(
            frame.instance_id[2], 0,
            "losing cluster should not get instance"
        );
    }

    #[test]
    fn test_box_fusion_empty_boxes_returns_zeros() {
        let boxes: Vec<DetectBox> = vec![];
        let proj = vec![[0.5, 0.5], [0.3, 0.7]];
        let mut frame = make_frame_with_proj(&[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], None, &proj);
        box_fusion_no_cluster(&mut frame, &boxes, None);

        assert!(
            frame.vision_class.iter().all(|&c| c == 0),
            "all classes should be 0"
        );
        assert!(
            frame.instance_id.iter().all(|&i| i == 0),
            "all instances should be 0"
        );
        assert!(
            frame.track_id.iter().all(|&t| t == 0),
            "all track_ids should be 0"
        );
    }

    #[test]
    fn test_hash_track_id_deterministic() {
        // FNV-1a 32-bit of b"some-uuid-123" — verifiable in any language.
        assert_eq!(hash_track_id("some-uuid-123"), 0x781c_f9b2);
        assert_eq!(
            hash_track_id("some-uuid-123"),
            hash_track_id("some-uuid-123"),
            "same input must always produce the same hash"
        );
        assert_ne!(hash_track_id("some-uuid-123"), 0);
    }

    #[test]
    fn test_hash_track_id_empty_returns_zero() {
        assert_eq!(hash_track_id(""), 0, "empty track_id should return 0");
    }

    #[test]
    fn test_parse_box_label_valid() {
        assert_eq!(parse_box_label("0", None), 0);
        assert_eq!(parse_box_label("1", None), 1);
        assert_eq!(parse_box_label("255", None), 255);
    }

    #[test]
    fn test_parse_box_label_empty_defaults_to_zero() {
        assert_eq!(parse_box_label("", None), 0);
    }

    #[test]
    fn test_parse_box_label_non_numeric_defaults_to_zero() {
        assert_eq!(parse_box_label("abc", None), 0);
    }

    #[test]
    fn test_parse_box_label_out_of_range_defaults_to_zero() {
        assert_eq!(parse_box_label("300", None), 0);
    }

    #[test]
    fn test_parse_box_label_with_labels_list() {
        let labels = vec![
            "background".to_string(),
            "person".to_string(),
            "car".to_string(),
        ];
        assert_eq!(parse_box_label("person", Some(&labels)), 1);
        assert_eq!(parse_box_label("car", Some(&labels)), 2);
        assert_eq!(parse_box_label("background", Some(&labels)), 0);
    }

    #[test]
    fn test_parse_box_label_unknown_label_with_list() {
        let labels = vec!["person".to_string(), "car".to_string()];
        assert_eq!(parse_box_label("truck", Some(&labels)), 0);
        assert_eq!(parse_box_label("5", Some(&labels)), 5);
    }

    #[test]
    fn test_box_fusion_clustered_overlapping_boxes() {
        let proj = vec![[0.5, 0.5], [0.55, 0.5], [0.6, 0.5]];
        let mut frame = make_frame_with_proj(
            &[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
            Some(&[10, 10, 10]),
            &proj,
        );
        let box_a = make_box(0.55, 0.5, 0.3, 0.3, "7", "track-a");
        let box_b = make_box(0.5, 0.5, 0.02, 0.02, "9", "track-b");
        let boxes = vec![box_a, box_b];

        let mut clusters: HashMap<u32, Vec<usize>> = HashMap::new();
        clusters.insert(10, vec![0, 1, 2]);

        box_fusion_clustered(&mut frame, &boxes, &clusters, None);

        assert_eq!(frame.vision_class[0], 7, "cluster should get Box A's class");
        assert_eq!(frame.vision_class[1], 7);
        assert_eq!(frame.vision_class[2], 7);
        assert_ne!(frame.instance_id[0], 0);
        assert_eq!(frame.instance_id[0], frame.instance_id[1]);
        assert_eq!(frame.instance_id[1], frame.instance_id[2]);
        assert_ne!(frame.track_id[0], 0);
        assert_eq!(frame.track_id[0], frame.track_id[1]);
    }

    #[test]
    fn test_process_mask_argmax() {
        // 2x2 mask with 3 channels: pixel scores determine class via argmax
        let mut mask = Mask {
            width: 2,
            height: 2,
            length: 1,
            encoding: String::new(),
            mask: vec![
                10, 20, 5, // pixel (0,0): channel 1 wins -> class 1
                30, 5, 10, // pixel (1,0): channel 0 wins -> class 0
                5, 5, 40, // pixel (0,1): channel 2 wins -> class 2
                2, 1, 1, // pixel (1,1): channel 0 wins -> class 0
            ],
            boxed: false,
        };
        process_mask(&mut mask);
        assert_eq!(mask.mask, vec![1, 0, 2, 0]);
    }

    #[test]
    fn test_process_mask_single_channel() {
        // Single channel: mask passes through unchanged
        let mut mask = Mask {
            width: 2,
            height: 2,
            length: 1,
            encoding: String::new(),
            mask: vec![0, 1, 2, 3],
            boxed: false,
        };
        process_mask(&mut mask);
        assert_eq!(mask.mask, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_process_mask_zstd() {
        // Compress a known mask and verify decompression works
        let original = vec![0u8, 1, 2, 3];
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();
        let mut mask = Mask {
            width: 2,
            height: 2,
            length: 1,
            encoding: "zstd".to_string(),
            mask: compressed,
            boxed: false,
        };
        process_mask(&mut mask);
        assert_eq!(mask.encoding, "");
        assert_eq!(mask.mask, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_get_vision_class_and_instance_detection_only() {
        // Model with boxes only, no masks
        let boxes = vec![make_box(0.5, 0.5, 0.4, 0.4, "2", "det-track")];
        let model = make_model(boxes, vec![]);
        let proj = vec![[0.5, 0.5]];
        let mut frame = make_frame_with_proj(&[(1.0, 0.0, 0.0)], None, &proj);
        let ids = HashMap::new();

        get_vision_class_and_instance(&mut frame, Some(&model), &ids, None);
        assert_eq!(frame.vision_class[0], 2);
        assert_ne!(frame.instance_id[0], 0);
        assert_ne!(
            frame.track_id[0], 0,
            "tracked detection should have non-zero track_id"
        );
    }

    #[test]
    fn test_get_vision_class_and_instance_semantic() {
        // Model with semantic mask (not boxed), no boxes
        let mask = Mask {
            width: 2,
            height: 2,
            length: 1,
            encoding: String::new(),
            mask: vec![0, 1, 2, 3],
            boxed: false,
        };
        let model = make_model(vec![], vec![mask]);
        // Project to pixel (1,1) in normalized coords -> (0.75, 0.75) -> mask[1*2+1] = 3
        let proj = vec![[0.75, 0.75]];
        let mut frame = make_frame_with_proj(&[(1.0, 0.0, 0.0)], None, &proj);
        let ids = HashMap::new();

        get_vision_class_and_instance(&mut frame, Some(&model), &ids, None);
        assert_eq!(frame.vision_class[0], 3);
        assert_eq!(
            frame.instance_id[0], 0,
            "semantic segmentation has no instance IDs"
        );
        assert_eq!(
            frame.track_id[0], 0,
            "semantic segmentation has no track IDs"
        );
    }

    #[test]
    fn test_get_vision_class_and_instance_instance_seg() {
        // Model with boxed instance masks
        let labels = vec!["bg".to_string(), "person".to_string()];
        let mask = Mask {
            width: 4,
            height: 4,
            length: 1,
            encoding: String::new(),
            // Single-channel confidence mask: high confidence in upper-left quadrant
            mask: vec![255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            boxed: true,
        };
        let det_box = make_box(0.25, 0.25, 0.5, 0.5, "person", "track-1");
        let model = make_model(vec![det_box], vec![mask]);
        // First point projects into the mask, second projects outside
        let proj = vec![[0.1, 0.1], [0.9, 0.9]];
        let mut frame = make_frame_with_proj(&[(1.0, 0.0, 0.0), (5.0, 0.0, 0.0)], None, &proj);
        let ids = HashMap::new();

        get_vision_class_and_instance(&mut frame, Some(&model), &ids, Some(&labels));
        assert_ne!(
            frame.vision_class[0], 0,
            "point inside instance mask should get class"
        );
        assert_ne!(
            frame.instance_id[0], 0,
            "point inside instance mask should get instance"
        );
        assert_ne!(
            frame.track_id[0], 0,
            "point inside instance mask should get track_id"
        );
        assert_eq!(
            frame.vision_class[1], 0,
            "point outside instance mask should be background"
        );
    }

    #[test]
    fn test_instance_mask_fusion_no_cluster() {
        // Box 1: centered at (0.25, 0.25), size 0.5x0.5 -> covers [0.0, 0.0] to [0.5, 0.5]
        // Box 2: centered at (0.75, 0.75), size 0.5x0.5 -> covers [0.5, 0.5] to [1.0, 1.0]
        let masks = vec![
            Mask {
                width: 2,
                height: 2,
                length: 1,
                encoding: String::new(),
                mask: vec![255, 255, 255, 255], // all foreground
                boxed: true,
            },
            Mask {
                width: 2,
                height: 2,
                length: 1,
                encoding: String::new(),
                mask: vec![255, 255, 255, 255],
                boxed: true,
            },
        ];
        let boxes = vec![
            DetectBox {
                center_x: 0.25,
                center_y: 0.25,
                width: 0.5,
                height: 0.5,
                label: "1".to_string(),
                score: 0.9,
                distance: 0.0,
                speed: 0.0,
                track: Track {
                    id: "track-a".to_string(),
                    lifetime: 0,
                    created: Time { sec: 0, nanosec: 0 },
                },
            },
            DetectBox {
                center_x: 0.75,
                center_y: 0.75,
                width: 0.5,
                height: 0.5,
                label: "2".to_string(),
                score: 0.9,
                distance: 0.0,
                speed: 0.0,
                track: Track {
                    id: "track-b".to_string(),
                    lifetime: 0,
                    created: Time { sec: 0, nanosec: 0 },
                },
            },
        ];
        let ids: HashMap<u32, Vec<usize>> = HashMap::new();
        // Point in box 1, point in box 2, point well outside both boxes (beyond edge tolerance)
        let proj = vec![[0.1, 0.1], [0.75, 0.75], [0.55, 1.1]];
        let mut frame = make_frame_with_proj(
            &[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            None,
            &proj,
        );

        boxed_mask_fusion(&mut frame, &masks, &boxes, None, &ids);
        assert_eq!(frame.vision_class[0], 1, "point in box 1 should be class 1");
        assert_eq!(
            frame.instance_id[0], 1,
            "point in box 1 should be instance 1"
        );
        assert_ne!(frame.track_id[0], 0, "point in box 1 should have track_id");
        assert_eq!(frame.vision_class[1], 2, "point in box 2 should be class 2");
        assert_eq!(
            frame.instance_id[1], 2,
            "point in box 2 should be instance 2"
        );
        assert_ne!(frame.track_id[1], 0, "point in box 2 should have track_id");
        assert_eq!(
            frame.vision_class[2], 0,
            "point outside boxes should be background"
        );
    }

    #[test]
    fn test_instance_mask_fusion_clustered() {
        // Box centered at (0.25, 0.25), covering [0.0, 0.0] to [0.5, 0.5]
        let masks = vec![Mask {
            width: 2,
            height: 2,
            length: 1,
            encoding: String::new(),
            mask: vec![255, 255, 255, 255],
            boxed: true,
        }];
        let boxes = vec![DetectBox {
            center_x: 0.25,
            center_y: 0.25,
            width: 0.5,
            height: 0.5,
            label: "1".to_string(),
            score: 0.9,
            distance: 0.0,
            speed: 0.0,
            track: Track {
                id: "track-a".to_string(),
                lifetime: 0,
                created: Time { sec: 0, nanosec: 0 },
            },
        }];
        // Both points in cluster 5 project into box region
        let proj = vec![[0.1, 0.1], [0.2, 0.2]];
        let mut frame =
            make_frame_with_proj(&[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], Some(&[5, 5]), &proj);
        let mut clusters: HashMap<u32, Vec<usize>> = HashMap::new();
        clusters.insert(5, vec![0, 1]);

        boxed_mask_fusion(&mut frame, &masks, &boxes, None, &clusters);
        assert_eq!(frame.vision_class[0], 1);
        assert_eq!(frame.vision_class[1], 1);
        assert_eq!(frame.instance_id[0], 1);
        assert_eq!(frame.instance_id[1], 1);
        assert_ne!(frame.track_id[0], 0);
        assert_ne!(frame.track_id[1], 0);
    }
}
