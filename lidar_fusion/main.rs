mod mask;
mod setup;
mod transform;

use cdr::{CdrLe, Infinite};
use clap::Parser;
use edgefirst_schemas::{
    builtin_interfaces::Time,
    edgefirst_msgs::{Detect, DetectBox2D, DetectTrack, Mask},
    geometry_msgs::TransformStamped,
    sensor_msgs::{point_field, CameraInfo, PointCloud2, PointField},
    std_msgs::Header,
};
use log::{error, info, trace};
use mask::{argmax_slice, mask_instance};
use setup::Args;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    panic,
    sync::{Arc, Mutex},
    time::Instant,
};
use transform::transform_and_project_points;
use zenoh::bytes::{Encoding, ZBytes};

struct ParsedPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub cluster_id: u32,
}

const SIZE_OF_DATATYPE: [usize; 9] = [
    0, 1, // pub const INT8: u8 = 1;
    1, // pub const UINT8: u8 = 2;
    2, // pub const INT16: u8 = 3;
    2, // pub const UINT16: u8 = 4;
    4, // pub const INT32: u8 = 5;
    4, // pub const UINT32: u8 = 6;
    4, // pub const FLOAT32: u8 = 7;
    8, //pub const FLOAT64: u8 = 8;
];

fn parse_point_be(fields: &Vec<PointField>, data: &[u8]) -> ParsedPoint {
    let mut p = ParsedPoint {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        cluster_id: 0,
    };
    for f in fields {
        let start = f.offset as usize;

        let val = match f.datatype {
            point_field::INT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i8::from_be_bytes(bytes) as f32
            }
            point_field::UINT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u8::from_be_bytes(bytes) as f32
            }
            point_field::INT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i16::from_be_bytes(bytes) as f32
            }
            point_field::UINT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u16::from_be_bytes(bytes) as f32
            }
            point_field::INT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i32::from_be_bytes(bytes) as f32
            }
            point_field::UINT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u32::from_be_bytes(bytes) as f32
            }
            point_field::FLOAT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                f32::from_be_bytes(bytes)
            }
            point_field::FLOAT64 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT64 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                f64::from_be_bytes(bytes) as f32
            }
            d => {
                error!("Unknown datatype in PointField: {}", d);
                continue;
            }
        };
        match f.name.as_str() {
            "x" => p.x = val,
            "y" => p.y = val,
            "z" => p.z = val,
            "cluster_id" => p.cluster_id = val as u32,
            _ => {}
        }
    }
    p
}

fn parse_point_le(fields: &Vec<PointField>, data: &[u8]) -> ParsedPoint {
    let mut p = ParsedPoint {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        cluster_id: 0,
    };
    for f in fields {
        let start = f.offset as usize;
        let val = match f.datatype {
            point_field::INT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i8::from_le_bytes(bytes) as f32
            }
            point_field::UINT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u8::from_le_bytes(bytes) as f32
            }
            point_field::INT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i16::from_le_bytes(bytes) as f32
            }
            point_field::UINT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u16::from_le_bytes(bytes) as f32
            }
            point_field::INT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i32::from_le_bytes(bytes) as f32
            }
            point_field::UINT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u32::from_le_bytes(bytes) as f32
            }
            point_field::FLOAT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                f32::from_le_bytes(bytes)
            }
            point_field::FLOAT64 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT64 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                f64::from_le_bytes(bytes) as f32
            }
            d => {
                error!("Unknown datatype in PointField: {}", d);
                continue;
            }
        };
        match f.name.as_str() {
            "x" => p.x = val,
            "y" => p.y = val,
            "z" => p.z = val,
            "cluster_id" => p.cluster_id = val as u32,
            _ => {}
        }
    }
    p
}

fn parse_pcd(pcd: &PointCloud2) -> Vec<ParsedPoint> {
    let mut points = Vec::new();
    for i in 0..pcd.height {
        for j in 0..pcd.width {
            let start = (i * pcd.row_step + j * pcd.point_step) as usize;
            let end = start + pcd.point_step as usize;
            let p = if pcd.is_bigendian {
                parse_point_be(&pcd.fields, &pcd.data[start..end])
            } else {
                parse_point_le(&pcd.fields, &pcd.data[start..end])
            };
            points.push(p);
        }
    }
    points
}

const BASE_LINK_FRAME_ID: &str = "base_link";

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let session = zenoh::open(args.clone()).await.unwrap();
    info!("Opened Zenoh session");

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
            let mut guard = match info_clone.try_lock() {
                Ok(v) => v,
                Err(_) => return,
            };
            *guard = Some(new_info);
        })
        .await
        .expect("Failed to declare Zenoh subscriber");

    let mask = Arc::new(Mutex::new(None));
    let mask_clone = mask.clone();
    let _mask_sub = session
        .declare_subscriber(args.mask_topic.clone())
        .callback_mut(move |s| {
            let new_mask: Mask = match cdr::deserialize(&s.payload().to_bytes()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    return;
                }
            };
            if let Ok(mut guard) = mask_clone.lock() {
                *guard = Some(new_mask);
            }
        })
        .await
        .expect("Failed to declare Zenoh subscriber");

    let transform = Arc::new(Mutex::new(None));
    let transform_clone = transform.clone();
    let lidar_frame_id = Arc::new(Mutex::new(None));
    let lidar_frame_id_clone = lidar_frame_id.clone();
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
            if new_transform.header.frame_id != BASE_LINK_FRAME_ID {
                return;
            }

            if !lidar_frame_id_clone.lock().is_ok_and(|x| {
                x.as_ref()
                    .is_some_and(|y| *y == new_transform.child_frame_id)
            }) {
                return;
            }

            if let Ok(mut guard) = transform_clone.try_lock() {
                *guard = Some(new_transform);
            }
        })
        .await
        .expect("Failed to declare Zenoh subscriber");

    let lidar_sub = session
        .declare_subscriber(args.lidar_input_topic.clone())
        .await
        .expect("Failed to declare Zenoh subscriber");

    let boxes3d_publ = session
        .declare_publisher(args.output_topic.clone())
        .await
        .expect("Failed to declare Zenoh publisher");

    loop {
        let msgs = lidar_sub.drain();
        let s = match msgs.last() {
            Some(v) => v,
            None => match lidar_sub.recv() {
                Ok(v) => v,
                Err(e) => {
                    error!(
                        "error receiving radar points on {}: {:?}",
                        lidar_sub.key_expr(),
                        e
                    );
                    continue;
                }
            },
        };
        let start = Instant::now();
        let pcd: PointCloud2 = match cdr::deserialize(&s.payload().to_bytes()) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize message: {:?}", e);
                return;
            }
        };
        println!("cdr::deserialize {:?}", start.elapsed());
        if let Ok(mut id) = lidar_frame_id.try_lock() {
            *id = Some(pcd.header.frame_id.clone());
        }

        let mut points = parse_pcd(&pcd);
        println!("parse_pcd {:?}", start.elapsed());
        let transform = match transform.lock() {
            Ok(v) if v.is_some() => v.as_ref().unwrap().transform.clone(),
            _ => continue,
        };
        let info = match info.lock() {
            Ok(v) if v.is_some() => v.as_ref().unwrap().clone(),
            _ => continue,
        };
        let im_shape = (info.width as f32, info.height as f32);
        let cam_mtx = info.k.map(|x| x as f32);
        let proj = transform_and_project_points(&mut points, &[transform], &cam_mtx, im_shape);
        println!("transform_and_project_points {:?}", start.elapsed());
        let mask = match mask.lock() {
            Ok(v) if v.is_some() => v.as_ref().unwrap().clone(),
            _ => continue,
        };
        if mask.width * mask.height == 0 {
            continue;
        }
        let mask_height = mask.height as usize;
        let mask_width = mask.width as usize;
        let mask_classes = mask.mask.len() / mask_width / mask_height;
        let mask_argmax: Vec<usize> = mask
            .mask
            .chunks_exact(mask_classes)
            .map(argmax_slice)
            .collect();
        let index_mask = |x: f32, y: f32| -> usize {
            let x = (x * mask_width as f32) as usize;
            let y = (y * mask_height as f32) as usize;
            mask_argmax[y * mask_width + x]
        };

        let bbox_2d = mask_instance(&mask_argmax, mask_width);
        let mut bbox_id = HashSet::new();

        let mut bbox_3d = Vec::new();
        for b in bbox_2d {
            let mut bbox_cluster_ids = Vec::new();
            for i in 0..points.len() {
                let [x, y] = proj[i];
                if (0.0..1.0).contains(&y)
                    && (0.0..1.0).contains(&x)
                    && (b.center_x - x).abs() <= b.width / 2.0
                    && (b.center_y - y).abs() <= b.height / 2.0
                    && points[i].cluster_id > 0
                    && index_mask(x, y) > 0
                {
                    bbox_cluster_ids.push(points[i].cluster_id);
                }
            }
            if let Some(class) = stats_mode(&bbox_cluster_ids) {
                bbox_id.insert(class);
            }
        }
        println!("bbox_id {:?}", start.elapsed());
        for i in bbox_id {
            let (mut x_max, mut y_max, mut z_max) = (-99999f32, -99999f32, -99999f32);
            let (mut x_min, mut y_min, mut z_min) = (99999f32, 99999f32, 99999f32);

            for p in points.iter() {
                if p.cluster_id == i {
                    x_max = x_max.max(p.x);
                    x_min = x_min.min(p.x);

                    y_max = y_max.max(p.y);
                    y_min = y_min.min(p.y);

                    z_max = z_max.max(p.z);
                    z_min = z_min.min(p.z);
                }
            }
            bbox_3d.push(DetectBox2D {
                center_x: (y_max + y_min) / 2.0,
                center_y: (z_max + z_min) / 2.0,
                width: (y_max - y_min),
                height: (z_max - z_min),
                distance: (x_max + x_min) / 2.0,
                label: "".to_string(),
                score: 1.0,
                speed: 0.0,
                track: DetectTrack {
                    id: "".to_string(),
                    lifetime: 0,
                    created: pcd.header.stamp.clone(),
                },
            });
        }
        let new_msg = Detect {
            header: Header {
                stamp: pcd.header.stamp.clone(),
                frame_id: BASE_LINK_FRAME_ID.to_owned(),
            },
            input_timestamp: pcd.header.stamp.clone(),
            model_time: Time { sec: 0, nanosec: 0 },
            output_time: pcd.header.stamp.clone(),
            boxes: bbox_3d,
        };
        println!("new_msg {:?}", start.elapsed());
        let msg = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&new_msg, Infinite).unwrap());
        let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Detect");
        match boxes3d_publ.put(msg).encoding(enc).await {
            Ok(_) => trace!("BBox3D Grid Message Sent"),
            Err(e) => error!("BBox3D Message Error: {:?}", e),
        }
    }
}

pub fn stats_mode<T>(v: &[T]) -> Option<T>
where
    T: Hash + Copy + Eq,
{
    match v.len() {
        0 => None,
        1 => Some(v[0]),
        _ => {
            let mut counter = HashMap::new();
            for x in v.iter() {
                let count = counter.entry(x).or_insert(0);
                *count += 1;
            }
            let mut max = -1;
            let mut mode = None;

            for (val, count) in counter.iter() {
                if *count > max {
                    max = *count;
                    mode = Some(**val);
                }
            }
            mode
        }
    }
}
