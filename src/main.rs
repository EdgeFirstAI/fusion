use args::{Args, PCDSource};
use cdr::{CdrLe, Infinite, LittleEndian};
use clap::Parser;
use edgefirst_schemas::{
    builtin_interfaces::Time,
    edgefirst_msgs::{Detect, DetectBox2D, DetectTrack, Mask},
    geometry_msgs::{Quaternion, Transform, TransformStamped, Vector3},
    sensor_msgs::{point_field, CameraInfo, PointCloud2, PointField},
    std_msgs::Header,
};
use fusion_model::spawn_fusion_model_thread;
use log::{error, trace, warn};
use mask::{mask_handler, mask_instance, Box2D};
use pcd::{insert_field, parse_pcd, serialize_pcd, ParsedPoint};
use std::{
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tokio::{join, sync::Mutex};
use tracing::instrument;
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
mod rtm_model;
mod tflite_model;
mod tracker;
mod transform;

const BASE_LINK_FRAME_ID: &str = "base_link";

fn clear_bins(bins: &mut Vec<Vec<Bin>>, curr: u128, args: &Args) {
    for i in bins {
        for j in i {
            j.vision_classes.clear();
            j.fusion_classes.clear();
            if j.last_masked + args.bin_delay < curr {
                j.first_marked = 0;
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
    grid_point.x = 0.0;
    (grid_point, vision_class, fusion_class)
}

struct Bin {
    vision_classes: Vec<u8>,
    fusion_classes: Vec<u8>,
    last_masked: u128,
    first_marked: u128,
}

type Grid = (Vec<Vec<f32>>, u64);
const FUSION_CLASS: &str = "fusion_class";
const VISION_CLASS: &str = "vision_class";

// If the receiver is empty, waits for the next message, otherwise returns the
// most recent message on this receiver. If the receiver is closed, returns None
async fn drain_recv(
    sub: &Subscriber<zenoh::handlers::FifoChannelHandler<Sample>>,
    timeout: std::time::Duration,
) -> Option<Sample> {
    if let Some(v) = sub.drain().last() {
        Some(v)
    } else {
        match sub.recv_timeout(timeout) {
            Ok(v) => match v {
                Some(v) => Some(v),
                None => {
                    warn!("Timeout on {}", sub.key_expr());
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

#[tokio::main]
async fn main() {
    let args = Args::parse();

    args.tracy.then(tracy_client::Client::start);

    let stdout_log = tracing_subscriber::fmt::layer()
        .pretty()
        .with_filter(args.rust_log);

    let journald = match tracing_journald::layer() {
        Ok(journald) => Some(journald.with_filter(args.rust_log)),
        Err(_) => None,
    };

    let tracy = match args.tracy {
        true => Some(tracing_tracy::TracyLayer::default().with_filter(args.rust_log)),
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
    let info_clone = info.clone();
    let _info_sub = session
        .declare_subscriber(args.info_topic.clone())
        .callback_mut(move |s| {
            let new_info: CameraInfo = match cdr::deserialize(&s.payload().to_bytes()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    return;
                }
            };
            let mut guard = info_clone.try_lock();
            if let Ok(ref mut guard) = guard {
                **guard = Some(new_info);
            }
        })
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
    let transform_clone = transform.clone();
    let _transform_sub = session
        .declare_subscriber("rt/tf_static")
        .callback_mut(move |s| {
            let new_transform: TransformStamped = match cdr::deserialize(&s.payload().to_bytes()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    return;
                }
            };

            if let Ok(mut guard) = transform_clone.try_lock() {
                guard.insert(
                    (new_transform.header.frame_id, new_transform.child_frame_id),
                    new_transform.transform,
                );
            }
        })
        .await
        .expect("Failed to declare Zenoh subscriber");

    let radar_sub = if args.radar_pcd_topic != "" {
        Some(
            session
                .declare_subscriber(args.radar_pcd_topic.clone())
                .await
                .expect("Failed to declare Zenoh subscriber"),
        )
    } else {
        None
    };

    let radar_publ = if args.radar_output_topic != "" {
        Some(
            session
                .declare_publisher(args.radar_output_topic.clone())
                .await
                .expect("Failed to declare Zenoh publisher"),
        )
    } else {
        None
    };

    let lidar_sub = if args.lidar_pcd_topic != "" {
        Some(
            session
                .declare_subscriber(args.lidar_pcd_topic.clone())
                .await
                .expect("Failed to declare Zenoh subscriber"),
        )
    } else {
        None
    };

    let lidar_publ = if args.lidar_output_topic != "" {
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
        .declare_publisher(args.occ_topic.clone())
        .await
        .expect("Failed to declare Zenoh publisher");

    let bbox_publ = session
        .declare_publisher(args.bbox3d_topic.clone())
        .await
        .expect("Failed to declare Zenoh publisher");

    let grid: Arc<Mutex<Option<Grid>>> = Arc::new(Mutex::new(None));
    spawn_fusion_model_thread(session.clone(), args.clone(), grid.clone());

    // wait 2s for the tf_static to get transforms
    thread::sleep(Duration::from_secs(2));

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

    match args.occ_src {
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

    // let radar_handle = spawn_fusion_thread(data_radar, zenoh_radar,
    // args.clone());

    let _ = lidar_handle.join();
    // let _ = radar_handle.join();
}

pub fn spawn_fusion_thread(data: Mutexes, zenoh: ZenohCtx, args: Args) -> JoinHandle<()> {
    thread::Builder::new()
        .name("fusion_thread".to_string())
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(fusion(data, zenoh, &args));
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

async fn fusion(data: Mutexes, zenoh: ZenohCtx, args: &Args) {
    let mut tracker = ByteTrack::new_with_settings(ByteTrackSettings {
        track_high_conf: args.track_high_conf,
        track_extra_lifespan: args.track_extra_lifespan,
        track_iou: args.track_iou,
        track_update: args.track_update,
    });
    let mut point_tracker = ByteTrack::new_with_settings(ByteTrackSettings {
        track_high_conf: 0.5,
        track_extra_lifespan: 0.5,
        track_iou: 0.01,
        track_update: 0.5,
    });
    if zenoh.pcd_sub.is_none() {
        return;
    }
    let mut bins = Vec::new();
    let mut frame_index = 0;
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

    loop {
        let msg = match drain_recv(zenoh.pcd_sub.as_ref().unwrap(), Duration::from_secs(2)).await {
            Some(v) => v,
            None => continue,
        };
        let start = Instant::now();
        let mut pcd: PointCloud2 = cdr::deserialize(&msg.payload().to_bytes()).unwrap();

        println!("cdr::deserialize {:?}", start.elapsed());
        let mut points = parse_pcd(&pcd);
        println!("parse_pcd {:?}", start.elapsed());
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
                |v| v.clone(),
            );
        let cam_info = match data.info.lock().await {
            v if v.is_some() => v.as_ref().unwrap().clone(),
            _ => continue,
        };
        let cam_mtx = cam_info.k.map(|v| v as f32);
        let proj = transform_and_project_points(
            &mut points,
            &[transform],
            &cam_mtx,
            (cam_info.width as f32, cam_info.height as f32),
        );
        println!("transform_and_project_points {:?}", start.elapsed());
        pcd.header.frame_id = BASE_LINK_FRAME_ID.to_string(); // frame_id is base link because the tf transform was applied

        let mask = match data.mask.lock().await {
            v if v.is_some() => v.as_ref().unwrap().clone(),
            _ => continue,
        };

        let (has_cluster_ids, ids) = get_cluster_ids(&points);
        let vision_class = if has_cluster_ids {
            late_fusion_clustered(&points, &proj, &mask, &ids, args)
        } else {
            late_fusion_no_cluster(&points, &proj, &mask, 0.02, args)
        };
        println!("late_fusion {:?}", start.elapsed());
        let fusion_predictions = if args.track {
            grid_radar_tracked(&data.grid, &mut tracker, args, zenoh.session.clone()).await
        } else {
            grid_radar(&data.grid, args).await
        };
        println!("grid predictions {:?}", start.elapsed());
        let fusion_class = if has_cluster_ids {
            grid_nearest_cluster(&fusion_predictions, &points, &ids)
        } else {
            grid_nearest_point_no_cluster(&fusion_predictions, &points)
        };
        println!("grid occupancy {:?}", start.elapsed());

        let bbox_msg = if has_cluster_ids && zenoh.bbox_publ.is_some() {
            let bbox_msg = Some(get_3d_bbox(
                pcd.header.clone(),
                &points,
                &vision_class,
                &ids,
            ));
            println!("boxes {:?}", start.elapsed());
            bbox_msg
        } else {
            None
        };

        let publ_bbox = async || {
            if let Some((msg, enc)) = bbox_msg {
                match zenoh
                    .bbox_publ
                    .as_ref()
                    .unwrap()
                    .put(msg)
                    .encoding(enc)
                    .await
                {
                    Ok(_) => trace!("BBox3D Message Sent"),
                    Err(e) => error!("BBox3D Message Error: {:?}", e),
                }
            }
        };

        let pcd_msg = if zenoh.pcd_publ.is_some() {
            pcd.fields = Vec::new();
            for (char, datatype) in [
                ("x", point_field::FLOAT32),
                ("y", point_field::FLOAT32),
                ("z", point_field::FLOAT32),
                ("cluster_id", point_field::UINT32),
                (FUSION_CLASS, point_field::UINT8),
                (VISION_CLASS, point_field::UINT8),
            ] {
                insert_field(
                    &mut pcd,
                    PointField {
                        name: char.to_string(),
                        offset: 0, // offset is calculated by the insert field function
                        datatype,
                        count: 1,
                    },
                );
            }
            let data = serialize_pcd(&points, &pcd.fields, &vision_class, &fusion_class);
            println!("serialize_pcd {:?}", start.elapsed());
            pcd.row_step = data.len() as u32;
            pcd.data = data;
            pcd.is_bigendian = cfg!(target_endian = "big");

            let buf_pcd = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&pcd, Infinite).unwrap());
            let enc_pcd = Encoding::APPLICATION_CDR.with_schema("sensor_msgs/msg/PointCloud2");
            println!("cdr::serialize {:?}", start.elapsed());
            Some((buf_pcd, enc_pcd))
        } else {
            None
        };

        let publ_pcd = async || {
            if let Some((msg, enc)) = pcd_msg {
                match zenoh
                    .pcd_publ
                    .as_ref()
                    .unwrap()
                    .put(msg)
                    .encoding(enc)
                    .await
                {
                    Ok(_) => trace!("PointCloud2 Message Sent"),
                    Err(e) => error!("PointCloud2 Message Error: {:?}", e),
                }
            }
        };

        let grid_msg = if zenoh.grid_publ.is_none() {
            None
        } else if has_cluster_ids {
            Some(get_occupied_cluster(
                pcd.header.clone(),
                &points,
                &vision_class,
                &fusion_class,
                &ids,
                &mut point_tracker,
            ))
        } else {
            Some(get_occupied_no_cluster(
                pcd.header.clone(),
                &points,
                &vision_class,
                &fusion_class,
                &mut bins,
                frame_index,
                args,
            ))
        };

        let publ_grid = async || {
            if let Some((msg, enc)) = grid_msg {
                match zenoh
                    .grid_publ
                    .as_ref()
                    .unwrap()
                    .put(msg)
                    .encoding(enc)
                    .await
                {
                    Ok(_) => trace!("PointCloud2 Grid Message Sent"),
                    Err(e) => error!("PointCloud2 Message Error: {:?}", e),
                }
            }
        };
        join!(publ_bbox(), publ_pcd(), publ_grid());
        println!("publ {:?}", start.elapsed());
        clear_bins(&mut bins, frame_index, args);
        frame_index += 1;

        args.tracy.then(frame_mark);
    }
}

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
        let (mut x_max, mut y_max, mut z_max) = (-99999f32, -99999f32, -99999f32);
        let (mut x_min, mut y_min, mut z_min) = (99999f32, 99999f32, 99999f32);

        for ind in inds.iter() {
            let p = &points[*ind];
            assert_eq!(class, classes[*ind]);
            x_max = x_max.max(p.x);
            x_min = x_min.min(p.x);

            y_max = y_max.max(p.y);
            y_min = y_min.min(p.y);

            z_max = z_max.max(p.z);
            z_min = z_min.min(p.z);
        }
        bbox_3d.push(DetectBox2D {
            center_x: (y_max + y_min) / 2.0,
            center_y: (z_max + z_min) / 2.0,
            width: (y_max - y_min),
            height: (z_max - z_min),
            distance: (x_max + x_min) / 2.0,
            label: class.to_string(),
            score: 1.0,
            speed: 0.0,
            track: DetectTrack {
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
        header,
    };
    let msg = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&new_msg, Infinite).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Detect");
    (msg, enc)
}

fn grid_nearest_point_no_cluster(fusion_predictions: &[Box2D], points: &[ParsedPoint]) -> Vec<u8> {
    let mut class = vec![0; points.len()];
    let mut min_dist = 9999999.9;
    let mut min_point_ind = 0;
    for b in fusion_predictions {
        for (ind, p) in points.iter().enumerate() {
            let dist = (p.x - b.center_x).powi(2) + (p.y - b.center_y).powi(2);
            if dist < min_dist {
                min_dist = dist;
                min_point_ind = ind;
            }
        }
        if min_dist < 2.0 {
            class[min_point_ind] = b.label;
        }
    }

    class
}

fn grid_nearest_cluster(
    fusion_predictions: &[Box2D],
    points: &[ParsedPoint],
    clusters: &HashMap<u32, Vec<usize>>,
) -> Vec<u8> {
    let mut class = vec![0; points.len()];
    let mut min_dist = 9999999.9;
    let mut min_point_ind = 0;
    for b in fusion_predictions {
        for (ind, p) in points.iter().enumerate() {
            if p.id.is_none() {
                continue;
            }
            if p.id.unwrap() == 0 {
                continue;
            }
            let dist = (p.x - b.center_x).powi(2) + (p.y - b.center_y).powi(2);
            if dist < min_dist {
                min_dist = dist;
                min_point_ind = ind;
            }
        }
        if min_dist > 2.0 {
            continue;
        }
        let cluster_id = points[min_point_ind].id.unwrap();
        for ind in clusters.get(&cluster_id).unwrap() {
            class[*ind] = b.label;
        }
    }

    class
}

fn late_fusion_no_cluster(
    points: &[ParsedPoint],
    proj: &[[f32; 2]],
    mask: &Mask,
    point_size: f32,
    args: &Args,
) -> Vec<u8> {
    let mask_height = mask.height as usize;
    let mask_width = mask.width as usize;
    let index_mask = |x: f32, y: f32| -> u8 {
        let x = (x * mask_width as f32) as usize;
        let y = (y * mask_height as f32) as usize;
        mask.mask[y * mask_width + x]
    };

    let mut class = vec![0; points.len()];
    if point_size <= 0.0 {
        for i in 0..points.len() {
            let [x, y] = proj[i];
            if !(0.0..1.0).contains(&y) || !(0.0..1.0).contains(&x) {
                continue;
            }
            class[i] = index_mask(x, y);
        }
        return class;
    }
    for i in 0..points.len() {
        // first do the center of the point, then 8 points around circumference
        // negative -45 represetns the center
        for angle in (-45..360).step_by(45) {
            let range = if angle < 0 { 0.0 } else { 0.02 };
            let x = proj[i][0] + (range * (angle as f32).to_radians().sin());
            let y = proj[i][1] + (range * (angle as f32).to_radians().cos());
            if !(0.0..1.0).contains(&y) || !(0.0..1.0).contains(&x) {
                continue;
            }
            let argmax = index_mask(x, y);
            if argmax != 0 {
                class[i] = argmax;
                break;
            }
        }
    }
    class
}

fn late_fusion_clustered(
    points: &[ParsedPoint],
    proj: &[[f32; 2]],
    mask: &Mask,
    clusters: &HashMap<u32, Vec<usize>>,
    args: &Args,
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

fn get_cluster_ids(points: &[ParsedPoint]) -> (bool, HashMap<u32, Vec<usize>>) {
    let mut has_cluster_id = false;
    let mut cluster_ids = HashMap::new();
    for (i, point) in points.iter().enumerate() {
        if let Some(id) = point.id {
            has_cluster_id = true;
            if id == 0 {
                // we ignore noise points
                continue;
            }
            if let Entry::Vacant(v) = cluster_ids.entry(id) {
                v.insert(Vec::new());
            }
            cluster_ids.get_mut(&id).unwrap().push(i);
        }
    }
    (has_cluster_id, cluster_ids)
}

#[instrument(skip_all)]
async fn grid_radar_tracked(
    grid: &Arc<Mutex<Option<Grid>>>,
    grid_tracker: &mut ByteTrack,
    args: &Args,
    session: Session,
) -> Vec<Box2D> {
    let mut class = Vec::new();

    let guard = grid.lock().await;
    if guard.is_none() {
        return class;
    }
    let (g, timestamp) = guard.as_ref().unwrap();
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
    if *timestamp > grid_tracker.timestamp {
        grid_tracker.update(&mut boxes, *timestamp);
    }

    let height = g.len();
    let width = g[0].len();

    {
        let mut tracked_g = vec![vec![0.0; width]; height];
        for tracklet in grid_tracker.get_tracklets() {
            if tracklet.count < 3 {
                continue;
            }
            let pred = tracklet.get_predicted_location();
            let i = ((pred.ymin + pred.ymax) / 2.0).round() as i32;
            let j = ((pred.xmin + pred.xmax) / 2.0).round() as i32;
            if i < 0 || i >= height as i32 {
                continue;
            }
            if j < 0 || j >= width as i32 {
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
        };

        let buf = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap());
        let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

        session
            .put("rt/fusion/mask_output_tracked", buf)
            .encoding(enc)
            .await
            .unwrap();
    }

    for tracklet in grid_tracker.get_tracklets() {
        if tracklet.count < 2 {
            continue;
        }
        let pred = tracklet.get_predicted_location();
        let i = (pred.ymin + pred.ymax) / 2.0;
        let j = (pred.xmin + pred.xmax) / 2.0;

        // center of grid
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

// Return the centroid of clusters that have class_id. All points in a class
// should have the same class_id
#[instrument(skip_all)]
fn get_occupied_cluster(
    header: Header,
    points: &[ParsedPoint],
    vision_class: &[u8],
    fusion_class: &[u8],
    cluster_ids: &HashMap<u32, Vec<usize>>,
    point_tracker: &mut ByteTrack,
) -> (ZBytes, Encoding) {
    let capacity = cluster_ids.len();
    let mut centroid_points = Vec::with_capacity(capacity);
    let mut centriod_vision_class = Vec::with_capacity(capacity);
    let mut centriod_fusion_class = Vec::with_capacity(capacity);
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
        centriod_vision_class.push(vision_class);
        centriod_fusion_class.push(fusion_class);
    }

    // want to track points that have class != 0
    let mut boxes: Vec<TrackerBox> = centroid_points
        .iter()
        .enumerate()
        .map(|(ind, p)| TrackerBox {
            xmin: p.x - 0.5,
            xmax: p.x + 0.5,
            ymin: p.y - 0.5,
            ymax: p.y + 0.5,
            score: if centriod_vision_class[ind] > 0 || centriod_fusion_class[ind] > 0 {
                1.0
            } else {
                0.3
            },
            vision_class: centriod_vision_class[ind],
            fusion_class: centriod_fusion_class[ind],
        })
        .collect();
    let timestamp = header.stamp.to_nanos();
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
            centriod_vision_class[i] = point_tracker.uuid_map_vision_class[&uuid];
        }
        if boxes[i].fusion_class == 0 {
            centriod_fusion_class[i] = point_tracker.uuid_map_fusion_class[&uuid];
        }
    }
    for i in point_tracker.get_tracklets() {
        if i.last_updated == timestamp {
            continue;
        }
        // println!("Found tracklet that was not updated this frame");
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
        centriod_vision_class.push(point_tracker.uuid_map_vision_class[&i.id]);
        centriod_fusion_class.push(point_tracker.uuid_map_fusion_class[&i.id]);
        trace!("added extra point");
    }

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
    for (char, datatype) in [
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
                name: char.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype,
                count: 1,
            },
        );
    }

    let data = serialize_pcd(
        &centroid_points,
        &centroid_pcd.fields,
        &centriod_vision_class,
        &centriod_fusion_class,
    );
    centroid_pcd.row_step = data.len() as u32;
    centroid_pcd.data = data;

    let buf = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&centroid_pcd, Infinite).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("sensor_msgs/msg/PointCloud2");

    (buf, enc)
}

// Do a grid and highlight the grid based on point classes
#[instrument(skip_all)]
fn get_occupied_no_cluster(
    header: Header,
    points: &[ParsedPoint],
    vision_class: &[u8],
    fusion_class: &[u8],
    bins: &mut [Vec<Bin>],
    frame_index: u128,
    args: &Args,
) -> (ZBytes, Encoding) {
    for (ind, p) in points.iter().enumerate() {
        let mut range = (p.x.powi(2) + p.y.powi(2) + p.z.powi(2)).sqrt();
        let mut angle = p.y.atan2(p.x).to_degrees();

        if angle < args.angle_bin_limit[0] {
            angle = args.angle_bin_limit[0]
        }
        if angle > args.angle_bin_limit[1] {
            angle = args.angle_bin_limit[1] - 0.001;
        }
        if range < args.range_bin_limit[0] {
            range = args.range_bin_limit[0];
        }
        if range > args.range_bin_limit[1] {
            range = args.range_bin_limit[1] - 0.001;
        }
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
    let mut grid_points = Vec::new();
    let mut vision_class = Vec::new();
    let mut fusion_class = Vec::new();
    let mut angle_found_occupied = vec![false; bins.len()];
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
            if sum0 + sum1 >= args.threshold {
                mark_grid(&mut bins[i][j - 1], frame_index);
                angle_found_occupied[i] = true;
                // don't check more ranges
                break;
            }
            if sum0 + sum1 + sum2 >= args.threshold {
                mark_grid(&mut bins[i][j - 2], frame_index);
                angle_found_occupied[i] = true;
                // don't check more ranges
                break;
            }
        }
    }

    for i in 0..bins.len() {
        for j in 0..bins[i].len() {
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

            if sum0 >= args.threshold {
                mark_grid(&mut bins[i][j], frame_index);
                angle_found_occupied[i] = true;
                if 0 < i {
                    angle_found_occupied[i - 1] = true;
                }
                if i + 1 < bins.len() {
                    angle_found_occupied[i + 1] = true;
                }
                // don't check more ranges
                break;
            }
            if sum0 + sum1 >= args.threshold {
                mark_grid(&mut bins[i][j - 1], frame_index);
                angle_found_occupied[i] = true;
                if 0 < i {
                    angle_found_occupied[i - 1] = true;
                }
                if i + 1 < bins.len() {
                    angle_found_occupied[i + 1] = true;
                }
                // don't check more ranges
                break;
            }
            if sum0 + sum1 + sum2 >= args.threshold {
                mark_grid(&mut bins[i][j - 2], frame_index);
                angle_found_occupied[i] = true;
                if 0 < i {
                    angle_found_occupied[i - 1] = true;
                }
                if i + 1 < bins.len() {
                    angle_found_occupied[i + 1] = true;
                }
                // don't check more ranges
                break;
            }
        }
    }
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
    for (char, datatype) in [
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
                name: char.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype,
                count: 1,
            },
        );
    }

    let data = serialize_pcd(&grid_points, &grid_pcd.fields, &vision_class, &fusion_class);
    grid_pcd.row_step = data.len() as u32;
    grid_pcd.data = data;

    let buf = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&grid_pcd, Infinite).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("sensor_msgs/msg/PointCloud2");

    (buf, enc)
}

/* Returns the mode of the slice. Returns None if the slice is empty.
 * https://stackoverflow.com/a/50000027
 */
fn mode_slice<T: Ord + Hash>(numbers: &[T]) -> Option<&T> {
    let mut counts = HashMap::new();

    numbers.iter().max_by_key(|&n| {
        let count = counts.entry(n).or_insert(0);
        *count += 1;
        *count
    })
}
