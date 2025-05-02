use edgefirst_schemas::sensor_msgs::{point_field, PointCloud2, PointField};
use log::error;
use std::collections::HashMap;

use crate::{FUSION_CLASS, VISION_CLASS};

pub struct ParsedPoint {
    pub fields: HashMap<String, f32>,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub id: Option<usize>,
    pub angle: f32,
    pub range: f32,
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
const DEFAULT_PCD_RANGE: f32 = 100000.0;

fn parse_point_be(fields: &Vec<PointField>, data: &[u8]) -> ParsedPoint {
    let mut p = ParsedPoint {
        fields: HashMap::new(),
        x: 0.0,
        y: 0.0,
        z: 0.0,
        id: None,
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
                    .unwrap_or_else(|e| panic!("Expected slice with 2 element: {:?}", e));
                i16::from_be_bytes(bytes) as f32
            }
            point_field::UINT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 2 element: {:?}", e));
                u16::from_be_bytes(bytes) as f32
            }
            point_field::INT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {:?}", e));
                i32::from_be_bytes(bytes) as f32
            }
            point_field::UINT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {:?}", e));
                u32::from_be_bytes(bytes) as f32
            }
            point_field::FLOAT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {:?}", e));
                f32::from_be_bytes(bytes)
            }
            point_field::FLOAT64 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT64 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 8 element: {:?}", e));
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
            "cluster_id" => {
                let _ = p.id.insert(val as usize);
            }
            _ => {
                let _ = p.fields.insert(f.name.clone(), val);
            }
        }
    }

    p.range = (p.x.powi(2) + p.y.powi(2) + p.z.powi(2)).sqrt();
    p.angle = p.y.atan2(p.x).to_degrees();

    p
}

fn parse_point_le(fields: &Vec<PointField>, data: &[u8]) -> ParsedPoint {
    let mut p = ParsedPoint {
        fields: HashMap::new(),
        x: 0.0,
        y: 0.0,
        z: 0.0,
        id: None,
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
                    .unwrap_or_else(|e| panic!("Expected slice with 2 element: {:?}", e));
                i16::from_le_bytes(bytes) as f32
            }
            point_field::UINT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 2 element: {:?}", e));
                u16::from_le_bytes(bytes) as f32
            }
            point_field::INT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {:?}", e));
                i32::from_le_bytes(bytes) as f32
            }
            point_field::UINT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {:?}", e));
                u32::from_le_bytes(bytes) as f32
            }
            point_field::FLOAT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {:?}", e));
                f32::from_le_bytes(bytes)
            }
            point_field::FLOAT64 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT64 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 8 element: {:?}", e));
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
            "cluster_id" => {
                let _ = p.id.insert(val as usize);
            }
            _ => {
                let _ = p.fields.insert(f.name.clone(), val);
            }
        }
    }

    p.range = (p.x.powi(2) + p.y.powi(2) + p.z.powi(2)).sqrt();
    p.angle = p.y.atan2(p.x).to_degrees();
    p
}

pub fn parse_pcd(pcd: &PointCloud2) -> Vec<ParsedPoint> {
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

pub fn serialize_pcd(
    points: &[ParsedPoint],
    fields: &[PointField],
    vision_class: &[u8],
    fusion_class: &[u8],
) -> Vec<u8> {
    let mut name_to_field = HashMap::new();
    let mut point_step = 0;
    for f in fields {
        point_step = point_step.max(f.offset as usize + SIZE_OF_DATATYPE[f.datatype as usize]);
        name_to_field.insert(f.name.clone(), f);
    }
    let row_step = point_step * points.len();
    let mut buf = vec![0u8; row_step];

    let mut point_offset = 0usize;
    for (i, p) in points.iter().enumerate() {
        for f in fields {
            match f.name.as_str() {
                "x" => serialize_field_f32(f, &p.x, point_offset, &mut buf),
                "y" => serialize_field_f32(f, &p.y, point_offset, &mut buf),
                "z" => serialize_field_f32(f, &p.z, point_offset, &mut buf),
                "cluster_id" => {
                    serialize_field_usize(f, &p.id.unwrap_or_default(), point_offset, &mut buf)
                }
                VISION_CLASS => serialize_field_u8(f, &vision_class[i], point_offset, &mut buf),
                FUSION_CLASS => serialize_field_u8(f, &fusion_class[i], point_offset, &mut buf),
                s => {
                    if let Some(v) = p.fields.get(s) {
                        serialize_field_f32(f, v, point_offset, &mut buf);
                    }
                }
            }
        }
        point_offset += point_step;
    }
    buf
}

// TODO: see if there is some trait which can make this better than three copy
// and pasted functions
fn serialize_field_f32(field: &PointField, val: &f32, point_offset: usize, buf: &mut [u8]) {
    let start = point_offset + field.offset as usize;
    let end = start + SIZE_OF_DATATYPE[field.datatype as usize];
    match field.datatype {
        point_field::INT8 => {
            let d = (*val as i8).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::UINT8 => {
            let d = (*val as u8).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::INT16 => {
            let d = (*val as i16).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::UINT16 => {
            let d = (*val as u16).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::INT32 => {
            let d = (*val as i32).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::UINT32 => {
            let d = (*val as u32).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::FLOAT32 => {
            let d = (*val as f32).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::FLOAT64 => {
            let d = (*val as f64).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        d => {
            error!("Unknown datatype in PointField: {}", d);
        }
    }
}

fn serialize_field_u8(field: &PointField, val: &u8, point_offset: usize, buf: &mut [u8]) {
    let start = point_offset + field.offset as usize;
    let end = start + SIZE_OF_DATATYPE[field.datatype as usize];
    match field.datatype {
        point_field::INT8 => {
            let d = (*val as i8).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::UINT8 => {
            let d = (*val as u8).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::INT16 => {
            let d = (*val as i16).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::UINT16 => {
            let d = (*val as u16).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::INT32 => {
            let d = (*val as i32).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::UINT32 => {
            let d = (*val as u32).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::FLOAT32 => {
            let d = (*val as f32).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::FLOAT64 => {
            let d = (*val as f64).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        d => {
            error!("Unknown datatype in PointField: {}", d);
        }
    }
}

fn serialize_field_usize(field: &PointField, val: &usize, point_offset: usize, buf: &mut [u8]) {
    let start = point_offset + field.offset as usize;
    let end = start + SIZE_OF_DATATYPE[field.datatype as usize];
    match field.datatype {
        point_field::INT8 => {
            let d = (*val as i8).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::UINT8 => {
            let d = (*val as u8).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::INT16 => {
            let d = (*val as i16).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::UINT16 => {
            let d = (*val as u16).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::INT32 => {
            let d = (*val as i32).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::UINT32 => {
            let d = (*val as u32).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::FLOAT32 => {
            let d = (*val as f32).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::FLOAT64 => {
            let d = (*val as f64).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        d => {
            error!("Unknown datatype in PointField: {}", d);
        }
    }
}

pub fn insert_field(pcd: &mut PointCloud2, new_field: PointField) {
    pcd.fields.push(PointField {
        name: new_field.name,
        offset: pcd.point_step,
        datatype: new_field.datatype,
        count: new_field.count,
    });
    pcd.point_step += SIZE_OF_DATATYPE[new_field.datatype as usize] as u32 * new_field.count;
}
