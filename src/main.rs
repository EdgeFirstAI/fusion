// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use args::{Args, PCDSource};
use clap::Parser;
use edgefirst_schemas::{
    builtin_interfaces::Time,
    edgefirst_msgs::{Box as DetectBox, Detect, Mask, Track},
    geometry_msgs::{Quaternion, Transform, TransformStamped, Vector3},
    sensor_msgs::{point_field, CameraInfo, PointCloud2, PointField},
    serde_cdr,
    std_msgs::Header,
};
use fusion_model::spawn_fusion_model_thread;
use log::{error, trace, warn};
use mask::{mask_handler, mask_instance, Box2D};
use pcd::{insert_field, parse_pcd, serialize_pcd, ParsedPoint};
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
mod tflite_model;
mod tracker;
mod transform;

const BASE_LINK_FRAME_ID: &str = "base_link";

type Grid = (Vec<Vec<f32>>, u64);
const FUSION_CLASS: &str = "fusion_class";
const VISION_CLASS: &str = "vision_class";
const MAX_CLASSIFICATION_DISTANCE: f32 = 2.0;
const UNINITIALIZED_COORD: f32 = 99999.0;

#[tokio::main]
async fn main() {
    let args = Args::parse();

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

    let mask = Arc::new(Mutex::new(None));
    let mask_clone = mask.clone();
    let session_clone = session.clone();
    let args_clone = args.clone();
    thread::Builder::new()
        .name("mask".to_string())
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(mask_handler(session_clone, args_clone, mask_clone));
        })
        .unwrap();

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

    let (radar_sub, radar_publ, lidar_sub, lidar_publ, grid_publ, bbox_publ) =
        declare_sub_pub(&session, &args).await;
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
        pcd_publ: radar_publ,
        grid_publ: None,
        bbox_publ: None,
    };

    let mut zenoh_lidar = ZenohCtx {
        session,
        pcd_sub: lidar_sub,
        pcd_publ: lidar_publ,
        grid_publ: None,
        bbox_publ: None,
    };
    let data_radar = Mutexes {
        mask: mask.clone(),
        info: info.clone(),
        tf_static: transform.clone(),
        grid: grid.clone(),
    };

    let data_lidar = Mutexes {
        mask,
        info,
        tf_static: transform,
        grid,
    };

    match args.grid_src {
        PCDSource::Radar => zenoh_radar.grid_publ = Some(grid_publ),
        PCDSource::Lidar => zenoh_lidar.grid_publ = Some(grid_publ),
        _ => {}
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
    Option<Publisher<'static>>,
    Option<Subscriber<FifoChannelHandler<Sample>>>,
    Option<Publisher<'static>>,
    Publisher<'static>,
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

    let radar_publ = if !args.radar_output_topic.is_empty() {
        Some(
            session
                .declare_publisher(args.radar_output_topic.clone())
                .await
                .expect("Failed to declare Zenoh publisher"),
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

    let lidar_publ = if !args.lidar_output_topic.is_empty() {
        Some(
            session
                .declare_publisher(args.lidar_output_topic.clone())
                .await
                .expect("Failed to declare Zenoh publisher"),
        )
    } else {
        None
    };
    let grid_publ = session
        .declare_publisher(args.grid_topic.clone())
        .await
        .expect("Failed to declare Zenoh publisher");

    let bbox_publ = session
        .declare_publisher(args.bbox3d_topic.clone())
        .await
        .expect("Failed to declare Zenoh publisher");

    (
        radar_sub, radar_publ, lidar_sub, lidar_publ, grid_publ, bbox_publ,
    )
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
    pcd_publ: Option<Publisher<'static>>,
    grid_publ: Option<Publisher<'static>>,
    bbox_publ: Option<Publisher<'static>>,
}

#[derive(Debug, Clone)]
pub struct Mutexes {
    mask: Arc<Mutex<Option<Mask>>>,
    info: Arc<Mutex<Option<CameraInfo>>>,
    tf_static: Arc<Mutex<HashMap<(String, String), Transform>>>,
    grid: Arc<Mutex<Option<Grid>>>,
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
async fn load_data(
    msg: &Sample,
    data: &Mutexes,
) -> Result<(PointCloud2, Vec<ParsedPoint>, Transform, CameraInfo, Mask), String> {
    let (mut pcd, points) = {
        let pcd: PointCloud2 = match serde_cdr::deserialize(&msg.payload().to_bytes()) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to deserialize PCD: {e:?}")),
        };
        let points = parse_pcd(&pcd);
        (pcd, points)
    };

    let transform = data
        .tf_static
        .lock()
        .await
        .get(&(BASE_LINK_FRAME_ID.to_owned(), pcd.header.frame_id.clone()))
        .map_or_else(
            || {
                warn!(
                    "Did not find transform from base_link to {}",
                    pcd.header.frame_id
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
    pcd.header.frame_id = BASE_LINK_FRAME_ID.to_string(); // frame_id is base link because the tf transform was applied

    let cam_info = match data.info.lock().await {
        v if v.is_some() => v.as_ref().unwrap().clone(),
        _ => return Err("No Camera Info".to_string()),
    };

    let mask = match data.mask.lock().await {
        v if v.is_some() => v.as_ref().unwrap().clone(),
        _ => return Err("No Mask".to_string()),
    };

    Ok((pcd, points, transform, cam_info, mask))
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
async fn fusion(
    points: &mut [ParsedPoint],
    transform: Transform,
    cam_info: &CameraInfo,
    mask: &Mask,
    track: bool,
    tracker: &mut ByteTrack,
    grid: &Arc<Mutex<Option<Grid>>>,
    args: &Args,
    session: &Session,
) -> (Vec<u8>, Vec<u8>, bool, HashMap<u32, Vec<usize>>) {
    let cam_mtx = cam_info.k.map(|v| v as f32);
    let proj = transform_and_project_points(
        points,
        &[transform],
        &cam_mtx,
        (cam_info.width as f32, cam_info.height as f32),
    );

    let (has_cluster_ids, ids) = get_cluster_ids(points);
    let vision_class = get_vision_class(points, &proj, mask, has_cluster_ids, &ids);

    let fusion_predictions = get_fusion_predictions(track, tracker, grid, args, session).await;
    let fusion_class = get_fusion_class(points, &fusion_predictions, has_cluster_ids, &ids);

    (vision_class, fusion_class, has_cluster_ids, ids)
}

async fn fusion_loop(data: Mutexes, zenoh: ZenohCtx, args: &Args) {
    if zenoh.pcd_sub.is_none() {
        return;
    }

    let mut bins = Vec::new();
    let mut frame_index = 0;
    let mut timeout = DrainRecvTimeoutSettings::default();

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

        let (mut pcd, mut points, transform, cam_info, mask) = match load_data(&msg, &data).await {
            Ok(v) => v,
            Err(e) => {
                warn!("{e}");
                continue;
            }
        };

        let (vision_class, fusion_class, has_cluster_ids, ids) = fusion(
            &mut points,
            transform,
            &cam_info,
            &mask,
            args.track,
            &mut tracker,
            &data.grid,
            args,
            &zenoh.session,
        )
        .await;

        publish(
            &zenoh,
            args,
            &mut pcd,
            &points,
            &vision_class,
            &fusion_class,
            has_cluster_ids,
            &ids,
            &mut tracker,
            &mut bins,
            frame_index,
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
    pcd: &mut PointCloud2,
    points: &[ParsedPoint],
    vision_class: &[u8],
    fusion_class: &[u8],
    has_cluster_ids: bool,
    ids: &HashMap<u32, Vec<usize>>,
    point_tracker: &mut ByteTrack,
    bins: &mut [Vec<Bin>],
    frame_index: u128,
) {
    let pcd_header = pcd.header.clone();

    let publ_bbox = publish_bbox3d(
        zenoh.bbox_publ.as_ref(),
        pcd_header.clone(),
        (points, vision_class, fusion_class),
        (has_cluster_ids, ids),
    );

    let publ_pcd = publish_pcd(
        zenoh.pcd_publ.as_ref(),
        pcd,
        (points, vision_class, fusion_class),
    );

    let publ_grid = publish_grid(
        zenoh.grid_publ.as_ref(),
        pcd_header.clone(),
        (points, vision_class, fusion_class),
        point_tracker,
        (bins, frame_index),
        args,
        (has_cluster_ids, ids),
    );

    join!(publ_bbox, publ_pcd, publ_grid);
}

fn get_vision_class(
    points: &[ParsedPoint],
    proj: &[[f32; 2]],
    mask: &Mask,
    has_cluster_ids: bool,
    ids: &HashMap<u32, Vec<usize>>,
) -> Vec<u8> {
    if has_cluster_ids {
        late_fusion_clustered(points, proj, mask, ids)
    } else {
        late_fusion_no_cluster(proj, mask, 0.02)
    }
}

fn get_fusion_class(
    points: &[ParsedPoint],
    fusion_predictions: &[Box2D],
    has_cluster_ids: bool,
    ids: &HashMap<u32, Vec<usize>>,
) -> Vec<u8> {
    if has_cluster_ids {
        grid_nearest_cluster(fusion_predictions, points, ids)
    } else {
        grid_nearest_point_no_cluster(fusion_predictions, points)
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
    header: Header,
    points_data: (&[ParsedPoint], &[u8], &[u8]),
    cluster_id_data: (bool, &HashMap<u32, Vec<usize>>),
) {
    if bbox_publ.is_none() {
        return;
    }
    let (has_cluster_ids, ids) = cluster_id_data;
    // Only create 3d bbox message when there are cluster IDs and bbox publishing is
    // enabled
    if !has_cluster_ids {
        return;
    }

    let bbox_publ = bbox_publ.unwrap();
    let (points, vision_class, _) = points_data;
    let (buf_bbox, enc_bbox) = get_3d_bbox(header, points, vision_class, ids);

    match bbox_publ.put(buf_bbox).encoding(enc_bbox).await {
        Ok(_) => trace!("Message Sent on {:?}", bbox_publ.key_expr()),
        Err(e) => error!("Message Error on {:?}: {:?}", bbox_publ.key_expr(), e),
    }
}

#[instrument(skip_all)]
async fn publish_pcd(
    publ: Option<&Publisher<'_>>,
    pcd: &mut PointCloud2,
    points_data: (&[ParsedPoint], &[u8], &[u8]),
) {
    if publ.is_none() {
        return;
    }
    let (points, vision_class, fusion_class) = points_data;
    let publ = publ.unwrap();
    pcd.fields = Vec::new();
    for (field_name, datatype) in [
        ("x", point_field::FLOAT32),
        ("y", point_field::FLOAT32),
        ("z", point_field::FLOAT32),
        ("cluster_id", point_field::UINT32),
        (FUSION_CLASS, point_field::UINT8),
        (VISION_CLASS, point_field::UINT8),
    ] {
        insert_field(
            pcd,
            PointField {
                name: field_name.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype,
                count: 1,
            },
        );
    }
    let data = serialize_pcd(points, &pcd.fields, vision_class, fusion_class);
    pcd.row_step = data.len() as u32;
    pcd.data = data;
    pcd.is_bigendian = cfg!(target_endian = "big");
    let buf = serde_cdr::serialize(&pcd).unwrap();
    let buf_pcd = ZBytes::from(buf);
    let enc_pcd = Encoding::APPLICATION_CDR.with_schema("sensor_msgs/msg/PointCloud2");
    match publ.put(buf_pcd).encoding(enc_pcd).await {
        Ok(_) => trace!("Message Sent on {:?}", publ.key_expr()),
        Err(e) => error!("Message Error on {:?}: {:?}", publ.key_expr(), e),
    }
}

#[instrument(skip_all)]
async fn publish_grid(
    grid_publ: Option<&Publisher<'_>>,
    header: Header,
    points_data: (&[ParsedPoint], &[u8], &[u8]),
    point_tracker: &mut ByteTrack,
    bins_data: (&mut [Vec<Bin>], u128),
    args: &Args,
    cluster_id_data: (bool, &HashMap<u32, Vec<usize>>),
) {
    if grid_publ.is_none() {
        return;
    };
    let grid_publ = grid_publ.unwrap();
    let (has_cluster_ids, ids) = cluster_id_data;
    let (points, vision_class, fusion_class) = points_data;
    let (buf_grid, enc_grid) = if has_cluster_ids {
        get_occupied_cluster(
            header,
            points,
            vision_class,
            fusion_class,
            ids,
            point_tracker,
        )
    } else {
        let (bins, frame_index) = bins_data;
        get_occupied_no_cluster(
            header,
            points,
            vision_class,
            fusion_class,
            bins,
            frame_index,
            args,
        )
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
    header: Header,
    points: &[ParsedPoint],
    classes: &[u8],
    cluster_ids: &HashMap<u32, Vec<usize>>,
) -> (ZBytes, Encoding) {
    let mut bbox_3d = Vec::new();
    for inds in cluster_ids.values() {
        if inds.is_empty() {
            continue;
        }

        // assumes that all points with the same class ID has the same class
        let class = classes[inds[0]];
        if class == 0 {
            continue;
        }

        // get the max and min x,y,z values of the cluster
        let (mut x_max, mut y_max, mut z_max) = (-UNINITIALIZED_COORD, -UNINITIALIZED_COORD, -UNINITIALIZED_COORD);
        let (mut x_min, mut y_min, mut z_min) = (UNINITIALIZED_COORD, UNINITIALIZED_COORD, UNINITIALIZED_COORD);

        for ind in inds.iter() {
            let p = &points[*ind];
            debug_assert_eq!(class, classes[*ind]);
            x_max = x_max.max(p.x);
            x_min = x_min.min(p.x);

            y_max = y_max.max(p.y);
            y_min = y_min.min(p.y);

            z_max = z_max.max(p.z);
            z_min = z_min.min(p.z);
        }

        // Add a 3D box using the max and min x,y,z values
        // TODO: Add 3D tracking to improve smoothness
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
                id: "".to_string(),
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
            stamp: header.stamp,
            frame_id: format!("{BASE_LINK_FRAME_ID}_optical"),
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
fn grid_nearest_point_no_cluster(fusion_predictions: &[Box2D], points: &[ParsedPoint]) -> Vec<u8> {
    let mut class = vec![0; points.len()];
    for b in fusion_predictions {
        let mut min_dist2 = f32::MAX;
        let mut min_point_ind = 0;
        for (ind, p) in points.iter().enumerate() {
            let dist2 = (p.x - b.center_x).powi(2) + (p.y - b.center_y).powi(2);
            if dist2 < min_dist2 {
                min_dist2 = dist2;
                min_point_ind = ind;
            }
        }
        if min_dist2 < MAX_CLASSIFICATION_DISTANCE * MAX_CLASSIFICATION_DISTANCE {
            class[min_point_ind] = b.label;
        }
    }

    class
}

// For each predicted grid box, find the nearest cluster in the PCD. If the
// nearest cluster is within 2m, set the class of all points in the cluster to
// the class of the grid box
fn grid_nearest_cluster(
    fusion_predictions: &[Box2D],
    points: &[ParsedPoint],
    clusters: &HashMap<u32, Vec<usize>>,
) -> Vec<u8> {
    let mut class = vec![0; points.len()];
    for b in fusion_predictions {
        let mut min_dist2 = f32::MAX;
        let mut min_point_ind = 0;
        for (ind, p) in points.iter().enumerate() {
            if p.id.is_none_or(|id| id == 0) {
                continue;
            }

            let dist2 = (p.x - b.center_x).powi(2) + (p.y - b.center_y).powi(2);
            if dist2 < min_dist2 {
                min_dist2 = dist2;
                min_point_ind = ind;
            }
        }
        if min_dist2 > MAX_CLASSIFICATION_DISTANCE * MAX_CLASSIFICATION_DISTANCE {
            continue;
        }
        let cluster_id = points[min_point_ind].id.unwrap();
        for ind in clusters.get(&cluster_id).unwrap() {
            class[*ind] = b.label;
        }
    }

    class
}

/// For each point, get the class of the point using the projection onto the
/// mask. If point_radius > 0, then also checks 8 points in a circle around the
/// projection, and uses the first non-zero class found as the class of
/// the point.
///
/// Assumes that the mask is already argmax'd
#[instrument(skip_all)]
fn late_fusion_no_cluster(projection: &[[f32; 2]], mask: &Mask, point_radius: f32) -> Vec<u8> {
    let mask_height = mask.height as usize;
    let mask_width = mask.width as usize;
    let index_mask = |x: f32, y: f32| -> u8 {
        let x = (x * mask_width as f32) as usize;
        let y = (y * mask_height as f32) as usize;
        mask.mask[y * mask_width + x]
    };

    let mut class = vec![0; projection.len()];
    for ([x, y], class) in projection.iter().zip(class.iter_mut()) {
        if !check_in_bounds(x, y) {
            continue;
        }
        *class = index_mask(*x, *y);
    }

    if point_radius <= 0.0 {
        return class;
    }

    for ([x, y], class) in projection.iter().zip(class.iter_mut()) {
        if *class != 0 {
            continue;
        }
        // first do the center of the point, then 8 points around circumference
        // negative -45 represents the center
        for angle in (0..360).step_by(45) {
            let new_x = *x + (point_radius * (angle as f32).to_radians().sin());
            let new_y = *y + (point_radius * (angle as f32).to_radians().cos());
            if !check_in_bounds(&new_x, &new_y) {
                continue;
            }
            let argmax = index_mask(new_x, new_y);
            if argmax != 0 {
                *class = argmax;
                break;
            }
        }
    }
    class
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
    points: &[ParsedPoint],
    proj: &[[f32; 2]],
    mask: &Mask,
    clusters: &HashMap<u32, Vec<usize>>,
) -> Vec<u8> {
    let mask_height = mask.height as usize;
    let mask_width = mask.width as usize;
    let index_mask = |x: f32, y: f32| -> u8 {
        let x = (x * mask_width as f32) as usize;
        let y = (y * mask_height as f32) as usize;
        mask.mask[y * mask_width + x]
    };

    let mut class = vec![0; points.len()];

    let bbox_2d = mask_instance(&mask.mask, mask_width);
    let mut bbox_id = Vec::new();
    for b in &bbox_2d {
        let mut bbox_cluster_ids = Vec::new();
        for i in 0..points.len() {
            let [x, y] = proj[i];
            if (0.0..1.0).contains(&y)
                && (0.0..1.0).contains(&x)
                && (b.center_x - x).abs() <= b.width / 2.0
                && (b.center_y - y).abs() <= b.height / 2.0
                && points[i].id.unwrap_or_default() > 0
                && index_mask(x, y) == b.label
            {
                bbox_cluster_ids.push(points[i].id.unwrap_or_default());
            }
        }
        if let Some(cluster_id) = mode_slice(&bbox_cluster_ids) {
            bbox_id.push(*cluster_id);
        }
    }
    for (box2d, cluster_id) in bbox_2d.into_iter().zip(bbox_id) {
        for i in clusters.get(&cluster_id).unwrap() {
            class[*i] = box2d.label;
        }
    }

    class
}

/// Checks if there are any cluster IDs in the PCD. Each cluster IDs and a
/// vector of the indices of all points with that cluster ID are placed into a
/// HashMap. Noise points are not included in the HashMap.
fn get_cluster_ids(points: &[ParsedPoint]) -> (bool, HashMap<u32, Vec<usize>>) {
    let mut has_cluster_id = false;
    let mut cluster_ids: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, point) in points.iter().enumerate() {
        if let Some(id) = point.id {
            has_cluster_id = true;
            if id == 0 {
                // we ignore noise points
                continue;
            }
            cluster_ids.entry(id).or_default().push(i);
        }
    }
    (has_cluster_id, cluster_ids)
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

fn centroids_get_class(
    cluster_ids: &HashMap<u32, Vec<usize>>,
    points: &[ParsedPoint],
    vision_class: &[u8],
    fusion_class: &[u8],
) -> (Vec<ParsedPoint>, Vec<u8>, Vec<u8>) {
    let capacity = cluster_ids.len();
    let mut centroid_points = Vec::with_capacity(capacity);
    let mut centroid_vision_class = Vec::with_capacity(capacity);
    let mut centroid_fusion_class = Vec::with_capacity(capacity);
    for id in cluster_ids {
        // sanity check, should not have cluster_ids with no points
        if id.1.is_empty() {
            continue;
        }
        let vision_class = vision_class[id.1[0]];
        let fusion_class = fusion_class[id.1[0]];
        let mut xyzv = id.1.iter().fold([0.0, 0.0, 0.0, 0.0], |mut xyzv, ind| {
            xyzv[0] += points[*ind].x;
            xyzv[1] += points[*ind].y;
            xyzv[2] += points[*ind].z;
            xyzv
        });
        for v in xyzv.iter_mut() {
            *v /= id.1.len() as f32
        }
        let p = ParsedPoint {
            x: xyzv[0],
            y: xyzv[1],
            z: xyzv[2],
            id: Some(*id.0),
        };

        centroid_points.push(p);
        centroid_vision_class.push(vision_class);
        centroid_fusion_class.push(fusion_class);
    }
    (
        centroid_points,
        centroid_vision_class,
        centroid_fusion_class,
    )
}

fn centroids_update_tracker_classes(
    centroid_points: &[ParsedPoint],
    centroid_vision_class: &mut [u8],
    centroid_fusion_class: &mut [u8],
    point_tracker: &mut ByteTrack,
    timestamp: u64,
) {
    let mut boxes: Vec<TrackerBox> = centroid_points
        .iter()
        .enumerate()
        .map(|(ind, p)| TrackerBox {
            xmin: p.x - 0.5,
            xmax: p.x + 0.5,
            ymin: p.y - 0.5,
            ymax: p.y + 0.5,
            score: if centroid_vision_class[ind] > 0 || centroid_fusion_class[ind] > 0 {
                1.0
            } else {
                0.3
            },
            vision_class: centroid_vision_class[ind],
            fusion_class: centroid_fusion_class[ind],
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
            centroid_vision_class[i] = point_tracker.uuid_map_vision_class[&uuid];
        }
        if boxes[i].fusion_class == 0 {
            centroid_fusion_class[i] = point_tracker.uuid_map_fusion_class[&uuid];
        }
    }
}

fn centroids_add_tracks(
    centroid_points: &mut Vec<ParsedPoint>,
    centroid_vision_class: &mut Vec<u8>,
    centroid_fusion_class: &mut Vec<u8>,
    point_tracker: &mut ByteTrack,
    timestamp: u64,
) {
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

        // should I check if there is still a point nearby?
        // nearby points still should've been updated

        let mut p = ParsedPoint {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            id: Some(0),
        };

        let predicted = i.get_predicted_location();
        p.x = (predicted.xmin + predicted.xmax) / 2.0;
        p.y = (predicted.ymin + predicted.ymax) / 2.0;
        centroid_points.push(p);
        centroid_vision_class.push(point_tracker.uuid_map_vision_class[&i.id]);
        centroid_fusion_class.push(point_tracker.uuid_map_fusion_class[&i.id]);
        trace!("added extra point");
    }
}
/// Returns the centroid of clusters that have non-zero class_id. All points in
/// a class should have the same class_id
fn get_occupied_cluster(
    header: Header,
    points: &[ParsedPoint],
    vision_class: &[u8],
    fusion_class: &[u8],
    cluster_ids: &HashMap<u32, Vec<usize>>,
    point_tracker: &mut ByteTrack,
) -> (ZBytes, Encoding) {
    let (mut centroid_points, mut centroid_vision_class, mut centroid_fusion_class) =
        centroids_get_class(cluster_ids, points, vision_class, fusion_class);
    // want to track points that have class != 0
    let timestamp = header.stamp.to_nanos();
    centroids_update_tracker_classes(
        &centroid_points,
        &mut centroid_vision_class,
        &mut centroid_fusion_class,
        point_tracker,
        timestamp,
    );

    centroids_add_tracks(
        &mut centroid_points,
        &mut centroid_vision_class,
        &mut centroid_fusion_class,
        point_tracker,
        timestamp,
    );

    let mut centroid_pcd = PointCloud2 {
        header,
        height: 1,
        width: centroid_points.len() as u32,
        is_bigendian: cfg!(target_endian = "big"),
        is_dense: true,
        fields: Vec::new(), // will be set by insert_field
        point_step: 0,      // will be set by insert_field
        data: Vec::new(),
        row_step: 0,
    };
    for (field_name, datatype) in [
        ("x", point_field::FLOAT32),
        ("y", point_field::FLOAT32),
        ("z", point_field::FLOAT32),
        ("cluster_id", point_field::UINT32),
        (FUSION_CLASS, point_field::UINT8),
        (VISION_CLASS, point_field::UINT8),
    ] {
        insert_field(
            &mut centroid_pcd,
            PointField {
                name: field_name.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype,
                count: 1,
            },
        );
    }

    let data = serialize_pcd(
        &centroid_points,
        &centroid_pcd.fields,
        &centroid_vision_class,
        &centroid_fusion_class,
    );
    centroid_pcd.row_step = data.len() as u32;
    centroid_pcd.data = data;

    let buf = ZBytes::from(serde_cdr::serialize(&centroid_pcd).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("sensor_msgs/msg/PointCloud2");

    (buf, enc)
}

fn update_bins(
    points: &[ParsedPoint],
    vision_class: &[u8],
    fusion_class: &[u8],
    bins: &mut [Vec<Bin>],
    args: &Args,
) {
    for (ind, p) in points.iter().enumerate() {
        let mut range = (p.x.powi(2) + p.y.powi(2) + p.z.powi(2)).sqrt();
        let mut angle = p.y.atan2(p.x).to_degrees();

        angle = angle.clamp(args.angle_bin_limit[0], args.angle_bin_limit[1] - 0.001);
        range = range.clamp(args.range_bin_limit[0], args.range_bin_limit[1] - 0.001);

        let i = ((angle - args.angle_bin_limit[0]) / args.angle_bin_width).floor() as usize;
        let j = ((range - args.range_bin_limit[0]) / args.range_bin_width).floor() as usize;
        let vision_class = vision_class[ind];
        if vision_class > 0 {
            bins[i][j].vision_classes.push(vision_class);
        }

        let fusion_class = fusion_class[ind];
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

fn find_marked_bins(
    bins: &[Vec<Bin>],
    frame_index: u128,
    args: &Args,
) -> (Vec<ParsedPoint>, Vec<u8>, Vec<u8>) {
    let mut grid_points = Vec::new();
    let mut vision_class = Vec::new();
    let mut fusion_class = Vec::new();

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
                    let (p, vision, fusion) = draw_point(bins, i, j, args);
                    grid_points.push(p);
                    vision_class.push(vision);
                    fusion_class.push(fusion);
                    angle_found_marked[i] = true;
                    // don't check more ranges
                    break;
                }
            }
        }
    }
    (grid_points, vision_class, fusion_class)
}

/// Do a grid and highlight the grid based on point classes
fn get_occupied_no_cluster(
    header: Header,
    points: &[ParsedPoint],
    vision_class: &[u8],
    fusion_class: &[u8],
    bins: &mut [Vec<Bin>],
    frame_index: u128,
    args: &Args,
) -> (ZBytes, Encoding) {
    update_bins(points, vision_class, fusion_class, bins, args);

    let mut angle_found_occupied = vec![false; bins.len()];
    mark_grid_one_column(bins, frame_index, &mut angle_found_occupied, args);
    mark_grid_three_column(bins, frame_index, &mut angle_found_occupied, args);

    let (grid_points, vision_class, fusion_class) = find_marked_bins(bins, frame_index, args);
    let mut grid_pcd = PointCloud2 {
        header,
        height: 1,
        width: grid_points.len() as u32,
        is_bigendian: cfg!(target_endian = "big"),
        is_dense: true,
        fields: Vec::new(), // will be set by insert_field
        point_step: 0,      // will be set by insert_field
        data: Vec::new(),
        row_step: 0,
    };
    for (field_name, datatype) in [
        ("x", point_field::FLOAT32),
        ("y", point_field::FLOAT32),
        ("z", point_field::FLOAT32),
        ("cluster_id", point_field::UINT32),
        (FUSION_CLASS, point_field::UINT8),
        (VISION_CLASS, point_field::UINT8),
    ] {
        insert_field(
            &mut grid_pcd,
            PointField {
                name: field_name.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype,
                count: 1,
            },
        );
    }

    let data = serialize_pcd(&grid_points, &grid_pcd.fields, &vision_class, &fusion_class);
    grid_pcd.row_step = data.len() as u32;
    grid_pcd.data = data;

    let buf = ZBytes::from(serde_cdr::serialize(&grid_pcd).unwrap());
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

fn draw_point(bins: &[Vec<Bin>], i: usize, j: usize, args: &Args) -> (ParsedPoint, u8, u8) {
    let mut grid_point = ParsedPoint {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        id: None,
    };
    let vision_class = *mode_slice(bins[i][j].vision_classes.as_slice()).unwrap_or(&0);
    let fusion_class = *mode_slice(bins[i][j].fusion_classes.as_slice()).unwrap_or(&0);

    let angle = args.angle_bin_width * (i as f32 + 0.5) + args.angle_bin_limit[0];
    let range = args.range_bin_width * (j as f32 + 0.5) + args.range_bin_limit[0];
    grid_point.x = angle.to_radians().cos() * range;
    grid_point.y = angle.to_radians().sin() * range;
    (grid_point, vision_class, fusion_class)
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
