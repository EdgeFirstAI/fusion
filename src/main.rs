use async_std::{sync::Mutex, task::block_on};
use cdr::{CdrLe, Infinite};
use clap::Parser;
use edgefirst_schemas::{
    edgefirst_msgs::Mask,
    sensor_msgs::{point_field, CameraInfo, PointCloud2, PointField},
    std_msgs::Header,
};
use fusion_model::spawn_fusion_model_thread;
use log::{error, info, trace};
use setup::Args;
use std::{
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
    panic,
    str::FromStr,
    sync::Arc,
};
use tracker::ByteTrack;
use vaal::VAALBox;
#[cfg(feature = "model_output")]
use zenoh::prelude::sync::*;
use zenoh::{config::Config, prelude::r#async::*};

use ndarray::{self, Array2};
mod fusion_model;
mod kalman;
mod setup;
mod tracker;

struct ParsedPoint {
    fields: HashMap<String, f64>,
    angle: f64,
    range: f64,
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
const DEFAULT_PCD_RANGE: f64 = 100000.0;

fn parse_point_be(fields: &Vec<PointField>, data: &[u8]) -> ParsedPoint {
    let mut p = ParsedPoint {
        fields: HashMap::new(),
        angle: 0.0,
        range: DEFAULT_PCD_RANGE,
    };
    for f in fields {
        let start = f.offset as usize;

        let val = match f.datatype {
            point_field::INT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i8::from_be_bytes(bytes) as f64
            }
            point_field::UINT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u8::from_be_bytes(bytes) as f64
            }
            point_field::INT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i16::from_be_bytes(bytes) as f64
            }
            point_field::UINT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u16::from_be_bytes(bytes) as f64
            }
            point_field::INT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i32::from_be_bytes(bytes) as f64
            }
            point_field::UINT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u32::from_be_bytes(bytes) as f64
            }
            point_field::FLOAT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                f32::from_be_bytes(bytes) as f64
            }
            point_field::FLOAT64 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT64 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                f64::from_be_bytes(bytes)
            }
            d => {
                error!("Unknown datatype in PointField: {}", d);
                continue;
            }
        };
        p.fields.insert(f.name.clone(), val);
    }
    if p.fields.contains_key("x") && p.fields.contains_key("y") && p.fields.contains_key("z") {
        p.range = (p.fields["x"].powi(2) + p.fields["y"].powi(2) + p.fields["z"].powi(2)).sqrt()
    }
    if p.fields.contains_key("x") && p.fields.contains_key("y") {
        p.angle = p.fields["y"].atan2(p.fields["x"]).to_degrees();
    }
    p
}

fn parse_point_le(fields: &Vec<PointField>, data: &[u8]) -> ParsedPoint {
    let mut p = ParsedPoint {
        fields: HashMap::new(),
        angle: 0.0,
        range: DEFAULT_PCD_RANGE,
    };
    for f in fields {
        let start = f.offset as usize;
        let val = match f.datatype {
            point_field::INT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i8::from_le_bytes(bytes) as f64
            }
            point_field::UINT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u8::from_le_bytes(bytes) as f64
            }
            point_field::INT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i16::from_le_bytes(bytes) as f64
            }
            point_field::UINT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u16::from_le_bytes(bytes) as f64
            }
            point_field::INT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                i32::from_le_bytes(bytes) as f64
            }
            point_field::UINT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                u32::from_le_bytes(bytes) as f64
            }
            point_field::FLOAT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                f32::from_le_bytes(bytes) as f64
            }
            point_field::FLOAT64 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT64 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {:?}", e));
                f64::from_le_bytes(bytes)
            }
            d => {
                error!("Unknown datatype in PointField: {}", d);
                continue;
            }
        };
        p.fields.insert(f.name.clone(), val);
    }
    if p.fields.contains_key("x") && p.fields.contains_key("y") && p.fields.contains_key("z") {
        p.range = (p.fields["x"].powi(2) + p.fields["y"].powi(2) + p.fields["z"].powi(2)).sqrt()
    }
    if p.fields.contains_key("x") && p.fields.contains_key("y") {
        p.angle = p.fields["y"].atan2(p.fields["x"]).to_degrees();
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

fn serialize_pcd(points: &Vec<ParsedPoint>, fields: &Vec<PointField>) -> Vec<u8> {
    let mut name_to_field = HashMap::new();
    let mut point_step = 0;
    for f in fields {
        point_step = point_step.max(f.offset as usize + SIZE_OF_DATATYPE[f.datatype as usize]);
        name_to_field.insert(f.name.clone(), f);
    }
    let row_step = point_step * points.len();
    let mut buf = vec![0u8; row_step];

    let mut point_offset = 0usize;
    for p in points {
        for (name, val) in &p.fields {
            let field = match name_to_field.get(name) {
                Some(v) => v,
                None => {
                    continue;
                }
            };
            let start = point_offset + field.offset as usize;
            let end = start + SIZE_OF_DATATYPE[field.datatype as usize];
            match field.datatype {
                point_field::INT8 => {
                    let d = (*val as i8).to_ne_bytes();
                    buf[start..end].clone_from_slice(&d);
                }
                point_field::UINT8 => {
                    let d = (*val as u8).to_ne_bytes();
                    buf[start..end].clone_from_slice(&d);
                }
                point_field::INT16 => {
                    let d = (*val as i16).to_ne_bytes();
                    buf[start..end].clone_from_slice(&d);
                }
                point_field::UINT16 => {
                    let d = (*val as u16).to_ne_bytes();
                    buf[start..end].clone_from_slice(&d);
                }
                point_field::INT32 => {
                    let d = (*val as i32).to_ne_bytes();
                    buf[start..end].clone_from_slice(&d);
                }
                point_field::UINT32 => {
                    let d = (*val as u32).to_ne_bytes();
                    buf[start..end].clone_from_slice(&d);
                }
                point_field::FLOAT32 => {
                    let d = (*val as f32).to_ne_bytes();
                    buf[start..end].clone_from_slice(&d);
                }
                point_field::FLOAT64 => {
                    let d = (*val).to_ne_bytes();
                    buf[start..end].clone_from_slice(&d);
                }
                d => {
                    error!("Unknown datatype in PointField: {}", d);
                    continue;
                }
            }
        }
        point_offset += point_step;
    }
    buf
}

fn project_point(points: &[ParsedPoint], cam_mtx: &[f64; 9], offset: &[f64; 3]) -> Vec<(f64, f64)> {
    let mtx = Array2::from_shape_fn([3, 3], |(i, j)| cam_mtx[i * 3 + j]);
    // println!("mtx={:?}", mtx);
    // convert from normal ROS conventions to optical conventions
    let coords = Array2::from_shape_fn([3, points.len()], |(i, j)| match i {
        0 => points[j].fields.get("y").unwrap_or(&0.0).to_owned() + offset[1],
        1 => points[j].fields.get("z").unwrap_or(&0.0).to_owned() + offset[2],
        2 => points[j].fields.get("x").unwrap_or(&0.0).to_owned() + offset[0],
        _ => 0.0,
    });
    // println!("coords={:?}", coords);
    let proj = &mtx.dot(&coords);
    // println!("proj={:?}", proj);
    let mut projected = Vec::new();
    for i in 0..points.len() {
        projected.push((
            proj.get((0, i)).unwrap() / proj.get((2, i)).unwrap(),
            proj.get((1, i)).unwrap() / proj.get((2, i)).unwrap(),
        ));
    }
    projected
}

#[cfg(test)]
mod projection_test {
    use super::*;

    #[test]
    fn test_basic() {
        let mut points = Vec::new();
        points.push(ParsedPoint {
            fields: HashMap::new(),
            angle: 0.0,
            range: DEFAULT_PCD_RANGE,
        });
        points[0].fields.insert("x".to_string(), 10.0);
        points[0].fields.insert("y".to_string(), 20.0);
        points[0].fields.insert("z".to_string(), 30.0);

        points.push(ParsedPoint {
            fields: HashMap::new(),
            angle: 0.0,
            range: DEFAULT_PCD_RANGE,
        });
        points[1].fields.insert("x".to_string(), 1.0);
        points[1].fields.insert("y".to_string(), 2.0);
        points[1].fields.insert("z".to_string(), 3.0);
        let cam_mtx = [
            1260.0 / 1920.0,
            0.0,
            960.0 / 1920.0,
            0.0,
            1260.0 / 1080.0,
            540.0 / 1080.0,
            0.0,
            0.0,
            1.0,
        ];
        let proj = project_point(&points, &cam_mtx, &[0.0, 0.0, 0.0]);
        println!("Projected values: {:?}", proj);
    }
}

fn clear_bins(bins: &mut Vec<Vec<Bin>>, curr: u128, args: &Args) {
    for i in bins {
        for j in i {
            j.classes.clear();
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
    bins[(i + offset_i) as usize][(j + offset_j) as usize]
        .classes
        .len() as u32
}

fn mark_grid(bin: &mut Bin, curr: u128) {
    bin.last_masked = curr;
    if bin.first_marked == u128::MAX {
        bin.first_marked = curr;
    }
}

fn draw_point(bins: &[Vec<Bin>], i: usize, j: usize, args: &Args) -> ParsedPoint {
    let mut grid_point = ParsedPoint {
        fields: HashMap::new(),
        angle: args.angle_bin_width * (i as f64 + 0.5) + args.angle_bin_limit[0],
        range: args.range_bin_width * (j as f64 + 0.5) + args.range_bin_limit[0],
    };
    let class = if let Some(mode) = mode_slice(bins[i][j].classes.as_slice()) {
        *mode
    } else {
        0
    };

    grid_point.fields.insert(
        "x".to_string(),
        grid_point.angle.to_radians().cos() * grid_point.range,
    );
    grid_point.fields.insert(
        "y".to_string(),
        grid_point.angle.to_radians().sin() * grid_point.range,
    );
    grid_point.fields.insert("z".to_string(), 0.0);
    grid_point.fields.insert("class".to_string(), class as f64);
    grid_point
        .fields
        .insert("count".to_string(), bins[i][j].classes.len() as f64);
    let speed = if !bins[i][j].speeds.is_empty() {
        bins[i][j].speeds.iter().fold(0.0, |a, b| a + b) / bins[i][j].speeds.len() as f64
    } else {
        0.0
    };
    grid_point.fields.insert("speed".to_string(), speed);
    grid_point
}

fn insert_field(pcd: &mut PointCloud2, new_field: PointField) {
    pcd.fields.push(PointField {
        name: new_field.name,
        offset: pcd.point_step,
        datatype: new_field.datatype,
        count: new_field.count,
    });
    pcd.point_step += SIZE_OF_DATATYPE[new_field.datatype as usize] as u32 * new_field.count;
}

struct Bin {
    classes: Vec<usize>,
    speeds: Vec<f64>,
    last_masked: u128,
    first_marked: u128,
}
type Grid = (Vec<Vec<bool>>, u64);
const CLASS_FIELD: &str = "class";
#[async_std::main]
async fn main() {
    env_logger::init();

    let args = Args::parse();

    let mut config = Config::default();
    let mode = WhatAmI::from_str(&args.mode).unwrap();
    config.set_mode(Some(mode)).unwrap();
    config.connect.endpoints = args.connect.iter().map(|v| v.parse().unwrap()).collect();
    config.listen.endpoints = args.listen.iter().map(|v| v.parse().unwrap()).collect();
    let _ = config.scouting.multicast.set_enabled(Some(true));
    let _ = config
        .scouting
        .multicast
        .set_interface(Some("lo".to_string()));
    let _ = config.scouting.gossip.set_enabled(Some(true));
    let session = match zenoh::open(config.clone()).res_async().await {
        Ok(v) => v,
        Err(e) => {
            error!("Error while opening Zenoh session: {:?}", e);
            return;
        }
    }
    .into_arc();
    info!("Opened Zenoh session");

    let info = Arc::new(Mutex::new(None));
    let info_clone = info.clone();
    let _info_sub = session
        .declare_subscriber(args.info_topic.clone())
        .callback_mut(move |s| {
            let new_info: CameraInfo = match cdr::deserialize(&s.payload.contiguous()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    return;
                }
            };
            let mut guard = match info_clone.try_lock() {
                Some(v) => v,
                None => return,
            };
            *guard = Some(new_info);
        })
        .res_async()
        .await
        .expect("Failed to declare Zenoh subscriber");

    let mask = Arc::new(Mutex::new(None));
    let mask_clone = mask.clone();
    let _mask_sub = session
        .declare_subscriber(args.mask_topic.clone())
        .callback_mut(move |s| {
            let new_mask: Mask = match cdr::deserialize(&s.payload.contiguous()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    return;
                }
            };
            let mut guard = block_on(mask_clone.lock());
            *guard = Some(new_mask);
        })
        .res_async()
        .await
        .expect("Failed to declare Zenoh subscriber");

    let radar_sub = session
        .declare_subscriber(args.radar_input_topic.clone())
        .res_async()
        .await
        .expect("Failed to declare Zenoh subscriber");

    let radar_publ = session
        .declare_publisher(args.radar_output_topic.clone())
        .res_async()
        .await
        .expect("Failed to declare Zenoh publisher");

    let grid_publ = session
        .declare_publisher(args.occ_topic.clone())
        .res_async()
        .await
        .expect("Failed to declare Zenoh publisher");

    let grid: Arc<Mutex<Option<Grid>>> = Arc::new(Mutex::new(None));
    spawn_fusion_model_thread(session.clone(), args.clone(), grid.clone());

    let mut zstd_decomp = zstd::bulk::Decompressor::new().unwrap();
    let mut bins = Vec::new();
    let mut frame_index = 0;
    let mut i = args.angle_bin_limit[0];
    while i <= args.angle_bin_limit[1] {
        let mut range_bins = Vec::new();
        let mut j = args.range_bin_limit[0];
        while j <= args.range_bin_limit[1] {
            range_bins.push(Bin {
                classes: Vec::new(),
                speeds: Vec::new(),
                last_masked: 0,
                first_marked: u128::MAX,
            });
            j += args.range_bin_width
        }
        bins.push(range_bins);
        i += args.angle_bin_width;
    }

    let mut tracker = ByteTrack::new();
    loop {
        let msgs = radar_sub.drain();
        let s = match msgs.last() {
            Some(v) => v,
            None => match radar_sub.recv() {
                Ok(v) => v,
                Err(e) => {
                    error!(
                        "error receiving radar points on {}: {:?}",
                        radar_sub.key_expr(),
                        e
                    );
                    continue;
                }
            },
        };

        let mut pcd: PointCloud2 = match cdr::deserialize(&s.payload.contiguous()) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize message: {:?}", e);
                return;
            }
        };

        // get point data
        let mut points = parse_pcd(&pcd);

        // Add a field to the end of the point fields
        insert_field(
            &mut pcd,
            PointField {
                name: CLASS_FIELD.to_string(),
                offset: 0, // offset is calculated by the insert field function
                datatype: point_field::FLOAT32,
                count: 1,
            },
        );
        points.sort_by(|p, q| p.range.total_cmp(&q.range));

        let mut mask_only = classify_points_mask_proj(&mask, &info, &points, &mut zstd_decomp);

        // filter occlusions
        for i in 0..points.len() {
            if mask_only[i] == 0 {
                continue;
            }
            for j in (i + 1)..points.len() {
                if (points[j].angle - points[i].angle).abs() < args.occ_angle_limit
                    && points[j].range - points[i].range > args.occ_range_limit
                {
                    mask_only[j] = 0;
                }
            }
        }

        let radar_only = if args.track {
            grid_points_radar_tracked(
                &grid,
                &points,
                &mut tracker,
                &args,
                #[cfg(feature = "model_output")]
                session.clone(),
            )
        } else {
            grid_points_radar(&grid, &points, &args)
        };

        for i in 0..points.len() {
            let new_class = if radar_only[i] > 0 || mask_only[i] > 0 {
                radar_only[i] + mask_only[i]
            } else {
                0
            };
            points[i]
                .fields
                .insert("class".to_string(), new_class as f64);
        }

        #[cfg(feature = "model_output")]
        {
            for i in 0..points.len() {
                let new_class = if radar_only[i] > 0 { radar_only[i] } else { 0 };
                points[i]
                    .fields
                    .insert("radar_class".to_string(), new_class as f64);
            }
            // Add a field to the end of the point fields
            insert_field(
                &mut pcd,
                PointField {
                    name: "radar_class".to_string(),
                    offset: 0, // offset is calculated by the insert field function
                    datatype: point_field::FLOAT32,
                    count: 1,
                },
            );
        }
        // add_grid_as_points(&grid, &mut points, &args);
        let mut has_cluster_id = false;
        // points with the same cluster_id get the same class
        let mut cluster_ids = HashMap::new();
        for (i, point) in points.iter_mut().enumerate() {
            if point.fields.contains_key("cluster_id") {
                has_cluster_id = true;
                let id = point.fields["cluster_id"].round() as i32;
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
        for id in cluster_ids.iter() {
            let mut classes = Vec::new();
            for index in id.1 {
                let class = points[*index].fields["class"].round() as i32;
                if class <= 0 {
                    continue;
                }
                classes.push(class)
            }

            // let class = if let Some(mode) = mode_slice(classes.as_slice()) {
            //     *mode
            // } else {
            //     0
            // };

            let class = if let Some(max) = max_slice(classes.as_slice()) {
                *max
            } else {
                0
            };

            for index in id.1 {
                points[*index]
                    .fields
                    .insert("class".to_string(), class as f64);
            }
        }

        // add_grid_as_points(&grid, &mut points, &args);
        let data = serialize_pcd(&points, &pcd.fields);
        pcd.row_step = data.len() as u32;
        pcd.data = data;
        pcd.is_bigendian = cfg!(target_endian = "big");
        pcd.header.frame_id = "radar".to_string(); // to fix the data recorded
        let encoded = Value::from(cdr::serialize::<_, _, CdrLe>(&pcd, Infinite).unwrap()).encoding(
            Encoding::WithSuffix(
                KnownEncoding::AppOctetStream,
                "sensor_msgs/msg/PointCloud2".into(),
            ),
        );

        match radar_publ.put(encoded).res_async().await {
            Ok(_) => trace!("PointCloud2 Message Sent"),
            Err(e) => error!("PointCloud2 Message Error: {:?}", e),
        }

        let encoded = if has_cluster_id {
            get_occupied_cluster(pcd.header.clone(), &points, &cluster_ids)
        } else {
            get_occupied_no_cluster(pcd.header.clone(), &points, &mut bins, frame_index, &args)
        };

        match grid_publ.put(encoded).res_async().await {
            Ok(_) => trace!("PointCloud2 Grid Message Sent"),
            Err(e) => error!("PointCloud2 Message Error: {:?}", e),
        }
        clear_bins(&mut bins, frame_index, &args);
        frame_index += 1;
    }
}

fn classify_points_mask_proj(
    mask: &Arc<Mutex<Option<Mask>>>,
    info: &Arc<Mutex<Option<CameraInfo>>>,
    points: &[ParsedPoint],
    zstd_decomp: &mut zstd::bulk::Decompressor<'_>,
) -> Vec<usize> {
    let mut class = vec![0; points.len()];
    // get mask data
    let guard = block_on(mask.lock());
    let mask_msg = match guard.as_ref() {
        Some(v) => v,
        None => {
            return class;
        }
    };

    // this holds the memory if we need to decompress the mask
    let mut mask_mem = None;

    let mask_width = mask_msg.width as usize;
    let mask_height = mask_msg.height as usize;
    // let mask_length = mask_msg.length as usize;
    let mask = if mask_msg.encoding == "zstd" {
        // accepts a maximum of 32 classes
        let m = mask_mem.insert(
            zstd_decomp
                .decompress(&mask_msg.mask, mask_width * mask_height * 32)
                .unwrap(),
        );
        drop(guard);
        m
    } else {
        &mask_msg.mask
    };
    let mask_classes = mask.len() as f32 / mask_width as f32 / mask_height as f32;
    let mask_classes = mask_classes.round() as usize;
    let (mut cam_mtx, cam_width, cam_height) = match block_on(info.lock()).as_ref() {
        Some(v) => (v.k, v.width as f64, v.height as f64),
        None => {
            return class;
        }
    };

    // create normalized camera matrix
    cam_mtx[0] /= cam_width;
    cam_mtx[1] /= cam_width;
    cam_mtx[2] /= cam_width;
    cam_mtx[3] /= cam_height;
    cam_mtx[4] /= cam_height;
    cam_mtx[5] /= cam_height;

    // project points onto mask
    let point_proj = project_point(points, &cam_mtx, &[0.0, 0.0, 0.0]);
    for i in 0..points.len() {
        let mask_coord = (
            ((1.0 - point_proj[i].0) * mask_width as f64).round() as i64,
            ((1.0 - point_proj[i].1) * mask_height as f64).round() as i64,
        );

        // first do the center of the point, then 8 points around circumference
        // negative -45 represetns the center
        for angle in (-45..360).step_by(45) {
            let range = if angle < 0 {
                0.0
            } else {
                mask_width.max(mask_height) as f64 * 0.02
            };
            let x = mask_coord.0 + (range * (angle as f64).to_radians().sin()) as i64;
            let y = mask_coord.1 + (range * (angle as f64).to_radians().cos()) as i64;
            if x < 0 || x >= mask_width as i64 {
                continue;
            }
            if y < 0 || y >= mask_height as i64 {
                continue;
            }
            let start = y as usize * mask_width * mask_classes + x as usize * mask_classes;
            let end = start + mask_classes;
            let scores = &mask[start..end];
            let mut max = 0;
            let mut argmax = 0;
            for (j, val) in scores.iter().enumerate() {
                if *val > max {
                    max = *val;
                    argmax = j;
                }
            }
            if argmax != 0 {
                class[i] = argmax;
                break;
            }
        }
    }
    class
}

fn grid_points_radar_tracked(
    grid: &Arc<Mutex<Option<Grid>>>,
    points: &[ParsedPoint],
    tracker: &mut ByteTrack,
    args: &Args,
    #[cfg(feature = "model_output")] session: Arc<Session>,
) -> Vec<usize> {
    let mut class = vec![0; points.len()];
    if points.is_empty() {
        return class;
    }
    let guard = block_on(grid.lock());
    if guard.is_none() {
        return class;
    }
    let (g, timestamp) = guard.as_ref().unwrap();
    let mut boxes = Vec::new();
    for (i, g_i) in g.iter().enumerate() {
        for (j, g_ij) in g_i.iter().enumerate() {
            if !g_ij {
                continue;
            }
            boxes.push(VAALBox {
                xmin: j as f32 - 1.0,
                ymin: i as f32 - 1.0,
                xmax: j as f32 + 1.0,
                ymax: i as f32 + 1.0,
                score: 1.0,
                label: 1,
            });
        }
    }
    if *timestamp > tracker.timestamp {
        tracker.update(args, &mut boxes, *timestamp);
    }

    #[cfg(feature = "model_output")]
    {
        let height = g.len();
        let width = g[0].len();
        let mut tracked_g = vec![vec![0.0; width]; g.len()];
        for tracklet in tracker.get_tracklets() {
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
        let val = Value::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap()).encoding(
            Encoding::WithSuffix(
                KnownEncoding::AppOctetStream,
                "edgefirst_msgs/msg/Mask".into(),
            ),
        );
        let _ = session.put("rt/fusion/mask_test_tracked", val).res_sync();
    }

    for tracklet in tracker.get_tracklets() {
        if tracklet.count < 2 {
            continue;
        }
        let pred = tracklet.get_predicted_location();
        let i = (pred.ymin + pred.ymax) / 2.0;
        let j = (pred.xmin + pred.xmax) / 2.0;

        // center of grid
        let (x, y) = if args.model_polar {
            let angle = args.angle_bin_limit[0] + args.angle_bin_width * (j as f64 + 0.5);
            let range = args.range_bin_limit[0] + args.range_bin_width * (i as f64 + 0.5);
            let x = (-angle).to_radians().cos() * range;
            let y = (-angle).to_radians().sin() * range;
            (x, y)
        } else {
            let x = args.range_bin_limit[0] + args.range_bin_width * (i as f64 + 0.5);
            let y = -(args.range_bin_limit[0] - args.range_bin_limit[1] / 2.0
                + args.range_bin_width * (j as f64 + 0.5));
            (x, y)
        };

        // find closest point
        let mut min_dist = 9999999.9;
        let mut min_point_ind = 0;
        for (ind, p) in points.iter().enumerate() {
            let dist = ((p.fields["x"] - x).powi(2) + (p.fields["y"] - y).powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                min_point_ind = ind;
            }
        }
        class[min_point_ind] = 1;
    }

    class
}

fn grid_points_radar(
    grid: &Arc<Mutex<Option<Grid>>>,
    points: &[ParsedPoint],
    args: &Args,
) -> Vec<usize> {
    let mut class = vec![0; points.len()];
    if points.is_empty() {
        return class;
    }
    let guard = block_on(grid.lock());
    if guard.is_none() {
        return class;
    }
    let (g, _) = guard.as_ref().unwrap();
    for (i, g_i) in g.iter().enumerate() {
        for (j, g_ij) in g_i.iter().enumerate() {
            if !g_ij {
                continue;
            }
            // center of grid
            let (x, y) = if args.model_polar {
                let angle = args.angle_bin_limit[0] + args.angle_bin_width * (j as f64 + 0.5);
                let range = args.range_bin_limit[0] + args.range_bin_width * (i as f64 + 0.5);
                let x = (-angle).to_radians().cos() * range;
                let y = (-angle).to_radians().sin() * range;
                (x, y)
            } else {
                let x = args.range_bin_limit[0] + args.range_bin_width * (i as f64 + 0.5);
                let y = -(args.range_bin_limit[0] - args.range_bin_limit[1] / 2.0
                    + args.range_bin_width * (j as f64 + 0.5));
                (x, y)
            };

            // find closest point
            let mut min_dist = 9999999.9;
            let mut min_point_ind = 0;
            for (ind, p) in points.iter().enumerate() {
                let dist = ((p.fields["x"] - x).powi(2) + (p.fields["y"] - y).powi(2)).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    min_point_ind = ind;
                }
            }
            class[min_point_ind] = 1;
        }
    }
    class
}

// fn add_grid_as_points(grid: &Arc<Mutex<Option<Grid>>>, points: &mut
// Vec<ParsedPoint>, args: &Args) {     let guard = block_on(grid.lock());
//     if guard.is_none() {
//         return;
//     }
//     let (g, _) = guard.as_ref().unwrap();
//     for i in 0..g.len() {
//         for j in 0..g[i].len() {
//             if !g[i][j] {
//                 continue;
//             }
//             // center of grid
//             let (x, y) = if args.model_polar {
//                 let angle = args.angle_bin_limit[0] + args.angle_bin_width *
// (j as f64 + 0.5);                 let range = args.range_bin_limit[0] +
// args.range_bin_width * (i as f64 + 0.5);                 let x =
// (-angle).to_radians().cos() * range;                 let y =
// (-angle).to_radians().sin() * range;                 (x, y)
//             } else {
//                 let x = args.range_bin_limit[0] + args.range_bin_width * (i
// as f64 + 0.5);                 let y = -(args.range_bin_limit[0] -
// args.range_bin_limit[1] / 2.0
//                     + args.range_bin_width * (j as f64 + 0.5));
//                 (x, y)
//             };
//             let mut p = ParsedPoint {
//                 fields: HashMap::new(),
//                 angle: 0.0,
//                 range: 0.0,
//             };
//             p.fields.insert("x".to_string(), x);
//             p.fields.insert("y".to_string(), y);
//             p.fields.insert("z".to_string(), 0.0);
//             p.fields.insert("speed".to_string(), 0.0);
//             p.fields.insert("class".to_string(), 10.0);
//             p.fields.insert("count".to_string(), 1.0);
//             p.fields.insert("cluster_id".to_string(), 0.0);
//             p.fields.insert("radar_class".to_string(), 10.0);
//             points.push(p);
//         }
//     }
// }

// Return the centroid of clusters that have class_id. All points in a class
// should have the same class_id
fn get_occupied_cluster(
    header: Header,
    points: &[ParsedPoint],
    cluster_ids: &HashMap<i32, Vec<usize>>,
) -> Value {
    let mut centroid_points = Vec::new();
    for id in cluster_ids {
        // sanity check, should not have cluster_ids with no points
        if id.1.is_empty() {
            continue;
        }
        let class = points[id.1[0]].fields["class"];
        let mut xyzv = id.1.iter().fold([0.0, 0.0, 0.0, 0.0], |mut xyzv, ind| {
            xyzv[0] += points[*ind].fields["x"];
            xyzv[1] += points[*ind].fields["y"];
            xyzv[2] += points[*ind].fields["z"];
            if let Some(v) = points[*ind].fields.get("speed") {
                xyzv[3] += v;
            }
            xyzv
        });
        for v in xyzv.iter_mut() {
            *v /= id.1.len() as f64
        }
        let mut p = ParsedPoint {
            fields: HashMap::new(),
            angle: 0.0,
            range: DEFAULT_PCD_RANGE,
        };

        p.fields.insert("x".to_string(), xyzv[0]);
        p.fields.insert("y".to_string(), xyzv[1]);
        p.fields.insert("z".to_string(), xyzv[2]);
        p.fields.insert("speed".to_string(), xyzv[3]);
        p.fields.insert("class".to_string(), class);
        p.fields.insert("count".to_string(), id.1.len() as f64);
        centroid_points.push(p);
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
    for char in ["x", "y", "z", "class", "speed", "count"] {
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

    let data = serialize_pcd(&centroid_points, &centroid_pcd.fields);
    centroid_pcd.row_step = data.len() as u32;
    centroid_pcd.data = data;

    Value::from(cdr::serialize::<_, _, CdrLe>(&centroid_pcd, Infinite).unwrap()).encoding(
        Encoding::WithSuffix(
            KnownEncoding::AppOctetStream,
            "sensor_msgs/msg/PointCloud2".into(),
        ),
    )
}

// Do a grid and highlight the grid based on point classes
fn get_occupied_no_cluster(
    header: Header,
    points: &Vec<ParsedPoint>,
    bins: &mut [Vec<Bin>],
    frame_index: u128,
    args: &Args,
) -> Value {
    for p in points {
        if p.fields["class"] <= 0.0 {
            continue;
        }
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
        let class = p.fields["class"] as usize;
        bins[i][j].classes.push(class);

        if let Some(speed) = p.fields.get("speed") {
            bins[i][j].speeds.push(*speed);
        }
    }
    let mut grid_points = Vec::new();

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
                    grid_points.push(draw_point(bins, i, j, args));
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
    for char in ["x", "y", "z", "class", "speed", "count"] {
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

    let data = serialize_pcd(&grid_points, &grid_pcd.fields);
    grid_pcd.row_step = data.len() as u32;
    grid_pcd.data = data;

    Value::from(cdr::serialize::<_, _, CdrLe>(&grid_pcd, Infinite).unwrap()).encoding(
        Encoding::WithSuffix(
            KnownEncoding::AppOctetStream,
            "sensor_msgs/msg/PointCloud2".into(),
        ),
    )
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

fn max_slice<T: Ord>(numbers: &[T]) -> Option<&T> {
    match argmax_slice(numbers) {
        Some(v) => Some(v.1),
        None => None,
    }
}
fn argmax_slice<T: Ord>(numbers: &[T]) -> Option<(usize, &T)> {
    numbers.iter().enumerate().max_by(|(_, a), (_, b)| a.cmp(b))
}
