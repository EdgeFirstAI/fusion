use args::Args;
use cdr::{CdrLe, Infinite};
use clap::Parser;
use edgefirst_schemas::{
    edgefirst_msgs::Mask,
    geometry_msgs::{Quaternion, Transform, TransformStamped, Vector3},
    sensor_msgs::{point_field, CameraInfo, PointCloud2, PointField},
    std_msgs::Header,
};
use fusion_model::spawn_fusion_model_thread;
use log::{error, trace, warn};
use mask::{argmax_slice, mask_handler, mask_instance, Box2D};
use ndarray::{self, Array2};
use pcd::{insert_field, parse_pcd, serialize_pcd, ParsedPoint};
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    hash::Hash,
    panic,
    sync::Arc,
    thread,
    time::Duration,
};
use tokio::sync::Mutex;
use tracing::{info_span, instrument};
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
        fields: HashMap::new(),
        x: 0.0,
        y: 0.0,
        z: 0.0,
        id: None,
        angle: args.angle_bin_width * (i as f32 + 0.5) + args.angle_bin_limit[0],
        range: args.range_bin_width * (j as f32 + 0.5) + args.range_bin_limit[0],
    };
    let vision_class = *mode_slice(bins[i][j].vision_classes.as_slice()).unwrap_or(&0);
    let fusion_class = *mode_slice(bins[i][j].fusion_classes.as_slice()).unwrap_or(&0);

    grid_point.x = grid_point.angle.to_radians().cos() * grid_point.range;
    grid_point.y = grid_point.angle.to_radians().sin() * grid_point.range;
    grid_point.x = 0.0;
    grid_point.fields.insert(
        "vision_count".to_string(),
        bins[i][j].vision_classes.len() as f32,
    );
    grid_point.fields.insert(
        "radar_count".to_string(),
        bins[i][j].fusion_classes.len() as f32,
    );
    let speed = if !bins[i][j].speeds.is_empty() {
        bins[i][j].speeds.iter().fold(0.0, |a, b| a + b) / bins[i][j].speeds.len() as f32
    } else {
        0.0
    };
    grid_point.fields.insert("speed".to_string(), speed);
    (grid_point, vision_class, fusion_class)
}

struct Bin {
    vision_classes: Vec<u8>,
    fusion_classes: Vec<u8>,
    speeds: Vec<f32>,
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

    let radar_sub = session
        .declare_subscriber(args.radar_pcd_topic.clone())
        .await
        .expect("Failed to declare Zenoh subscriber");

    let radar_publ = session
        .declare_publisher(args.radar_output_topic.clone())
        .await
        .expect("Failed to declare Zenoh publisher");

    let lidar_sub = session
        .declare_subscriber(args.radar_pcd_topic.clone())
        .await
        .expect("Failed to declare Zenoh subscriber");

    let lidar_publ = session
        .declare_publisher(args.radar_output_topic.clone())
        .await
        .expect("Failed to declare Zenoh publisher");

    let grid_publ = session
        .declare_publisher(args.occ_topic.clone())
        .await
        .expect("Failed to declare Zenoh publisher");

    let grid: Arc<Mutex<Option<Grid>>> = Arc::new(Mutex::new(None));
    spawn_fusion_model_thread(session.clone(), args.clone(), grid.clone());

    let zenoh = ZenohCtx {
        session,
        pcd_publ: radar_publ,
        grid_publ,
    };
    let data = Mutexes {
        mask: mask.clone(),
        info: info.clone(),
        tf_static: transform.clone(),
        grid: grid.clone(),
    };
    spawn_fusion_thread(radar_sub, data, zenoh, args.clone());
}

pub fn spawn_fusion_thread(
    pcd_sub: Subscriber<FifoChannelHandler<Sample>>,
    data: Mutexes,
    zenoh: ZenohCtx,
    args: Args,
) {
    thread::Builder::new()
        .name("model".to_string())
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(fusion(pcd_sub, data, zenoh, &args));
        })
        .unwrap();
}

#[derive(Debug)]
pub struct ZenohCtx {
    session: Session,
    pcd_publ: Publisher<'static>,
    grid_publ: Publisher<'static>,
}

#[derive(Debug, Clone)]
pub struct Mutexes {
    mask: Arc<Mutex<Option<Mask>>>,
    info: Arc<Mutex<Option<CameraInfo>>>,
    tf_static: Arc<Mutex<HashMap<(String, String), Transform>>>,
    grid: Arc<Mutex<Option<Grid>>>,
}

async fn fusion(
    pcd_sub: Subscriber<FifoChannelHandler<Sample>>,
    data: Mutexes,
    zenoh: ZenohCtx,
    args: &Args,
) {
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
                speeds: Vec::new(),
                last_masked: 0,
                first_marked: u128::MAX,
            });
            j += args.range_bin_width
        }
        bins.push(range_bins);
        i += args.angle_bin_width;
    }

    loop {
        let msg = match drain_recv(&pcd_sub, Duration::from_secs(2)).await {
            Some(v) => v,
            None => continue,
        };

        let mut pcd: PointCloud2 = cdr::deserialize(&msg.payload().to_bytes()).unwrap();
        let mut points = parse_pcd(&pcd);

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

        let mask = match data.mask.lock().await {
            v if v.is_some() => v.as_ref().unwrap().clone(),
            _ => continue,
        };

        let (has_cluster_ids, ids) = get_cluster_ids(&points);
        let vision_class = if has_cluster_ids {
            late_fusion_clustered(&points, &proj, &mask, &ids, &args)
        } else {
            late_fusion_no_cluster(&points, &proj, &mask, 0.02, &args)
        };

        let fusion_predictions = if args.track {
            grid_radar_tracked(&data.grid, &mut tracker, &args, zenoh.session.clone()).await
        } else {
            grid_radar(&data.grid, &args).await
        };

        let fusion_class = if has_cluster_ids {
            grid_nearest_cluster(&fusion_predictions, &points, &ids)
        } else {
            grid_nearest_point_no_cluster(&fusion_predictions, &points)
        };

        insert_field(
            &mut pcd,
            PointField {
                name: FUSION_CLASS.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype: point_field::UINT8,
                count: 1,
            },
        );

        insert_field(
            &mut pcd,
            PointField {
                name: VISION_CLASS.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype: point_field::UINT8,
                count: 1,
            },
        );
        let data = serialize_pcd(&points, &pcd.fields, &vision_class, &fusion_class);
        pcd.row_step = data.len() as u32;
        pcd.data = data;
        pcd.is_bigendian = cfg!(target_endian = "big");
        pcd.header.frame_id = BASE_LINK_FRAME_ID.to_string(); // frame_id is base link because the tf transform was applied

        let buf = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&pcd, Infinite).unwrap());
        let enc = Encoding::APPLICATION_CDR.with_schema("sensor_msgs/msg/PointCloud2");

        match zenoh.pcd_publ.put(buf).encoding(enc).await {
            Ok(_) => trace!("PointCloud2 Message Sent"),
            Err(e) => error!("PointCloud2 Message Error: {:?}", e),
        }

        let (buf, enc) = if has_cluster_ids {
            get_occupied_cluster(
                pcd.header.clone(),
                &points,
                &vision_class,
                &fusion_class,
                &ids,
                &mut point_tracker,
            )
        } else {
            get_occupied_no_cluster(
                pcd.header.clone(),
                &points,
                &vision_class,
                &fusion_class,
                &mut bins,
                frame_index,
                &args,
            )
        };

        match zenoh.grid_publ.put(buf).encoding(enc).await {
            Ok(_) => trace!("PointCloud2 Grid Message Sent"),
            Err(e) => error!("PointCloud2 Message Error: {:?}", e),
        }
        clear_bins(&mut bins, frame_index, &args);
        frame_index += 1;

        args.tracy.then(frame_mark);
    }
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
    clusters: &HashMap<usize, Vec<usize>>,
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
    clusters: &HashMap<usize, Vec<usize>>,
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

fn get_cluster_ids(points: &[ParsedPoint]) -> (bool, HashMap<usize, Vec<usize>>) {
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
    cluster_ids: &HashMap<usize, Vec<usize>>,
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
            if let Some(v) = points[*ind].fields.get("speed") {
                xyzv[3] += v;
            }
            xyzv
        });
        for v in xyzv.iter_mut() {
            *v /= id.1.len() as f32
        }
        let mut p = ParsedPoint {
            fields: HashMap::new(),
            angle: 0.0,
            x: xyzv[0],
            y: xyzv[1],
            z: xyzv[2],
            id: Some(*id.0),
            range: 0.0,
        };
        p.fields.insert("speed".to_string(), xyzv[3]);
        p.fields.insert("count".to_string(), id.1.len() as f32);

        centroid_points.push(p);
        centriod_vision_class.push(vision_class);
        centriod_fusion_class.push(fusion_class);
    }

    // want to track points that have class != 0
    let mut boxes: Vec<TrackerBox> = centroid_points
        .iter()
        .enumerate()
        .map(|(ind, p)| TrackerBox {
            xmin: p.x as f32 - 0.5,
            xmax: p.x as f32 + 0.5,
            ymin: p.y as f32 - 0.5,
            ymax: p.y as f32 + 0.5,
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
            fields: HashMap::new(),
            angle: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            id: Some(0),
            range: 0.0,
        };

        let predicted = i.get_predicted_location();
        p.x = (predicted.xmin + predicted.xmax) / 2.0;
        p.y = (predicted.ymin + predicted.ymax) / 2.0;
        p.fields.insert("speed".to_string(), 0.0);
        p.fields.insert("count".to_string(), 0.0);
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
    for char in [
        "x",
        "y",
        "z",
        "speed",
        "count",
        "cluster_id",
        FUSION_CLASS,
        VISION_CLASS,
    ] {
        insert_field(
            &mut centroid_pcd,
            PointField {
                name: char.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype: point_field::FLOAT32,
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
    for (i, p) in points.iter().enumerate() {
        let mut angle = p.angle;
        let mut range = p.range;
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
        let vision_class = vision_class[i];
        if vision_class > 0 {
            bins[i][j].vision_classes.push(vision_class);
        }

        let fusion_class = fusion_class[i];
        if fusion_class > 0 {
            bins[i][j].fusion_classes.push(fusion_class);
        }

        if let Some(speed) = p.fields.get("speed") {
            bins[i][j].speeds.push(*speed);
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
    for char in ["x", "y", "z", "speed", "count", FUSION_CLASS, VISION_CLASS] {
        insert_field(
            &mut grid_pcd,
            PointField {
                name: char.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype: point_field::FLOAT32,
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
