// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use edgefirst_schemas::sensor_msgs::{point_field, PointCloud2, PointField};
use log::error;
use std::collections::HashMap;
use tracing::instrument;

use crate::{FUSION_CLASS, INSTANCE_ID, VISION_CLASS};

pub struct ParsedPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub id: Option<u32>,
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
        id: None,
    };
    for f in fields {
        let start = f.offset as usize;

        let val = match f.datatype {
            point_field::INT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {e:?}"));
                i8::from_be_bytes(bytes) as f32
            }
            point_field::UINT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {e:?}"));
                u8::from_be_bytes(bytes) as f32
            }
            point_field::INT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 2 element: {e:?}"));
                i16::from_be_bytes(bytes) as f32
            }
            point_field::UINT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 2 element: {e:?}"));
                u16::from_be_bytes(bytes) as f32
            }
            point_field::INT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {e:?}"));
                i32::from_be_bytes(bytes) as f32
            }
            point_field::UINT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {e:?}"));
                u32::from_be_bytes(bytes) as f32
            }
            point_field::FLOAT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {e:?}"));
                f32::from_be_bytes(bytes)
            }
            point_field::FLOAT64 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT64 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 8 element: {e:?}"));
                f64::from_be_bytes(bytes) as f32
            }
            d => {
                error!("Unknown datatype in PointField: {d}");
                continue;
            }
        };
        match f.name.as_str() {
            "x" => p.x = val,
            "y" => p.y = val,
            "z" => p.z = val,
            "cluster_id" => {
                let _ = p.id.insert(val as u32);
            }
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
        id: None,
    };
    for f in fields {
        let start = f.offset as usize;
        let val = match f.datatype {
            point_field::INT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {e:?}"));
                i8::from_le_bytes(bytes) as f32
            }
            point_field::UINT8 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT8 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 1 element: {e:?}"));
                u8::from_le_bytes(bytes) as f32
            }
            point_field::INT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 2 element: {e:?}"));
                i16::from_le_bytes(bytes) as f32
            }
            point_field::UINT16 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT16 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 2 element: {e:?}"));
                u16::from_le_bytes(bytes) as f32
            }
            point_field::INT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::INT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {e:?}"));
                i32::from_le_bytes(bytes) as f32
            }
            point_field::UINT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::UINT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {e:?}"));
                u32::from_le_bytes(bytes) as f32
            }
            point_field::FLOAT32 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT32 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 4 element: {e:?}"));
                f32::from_le_bytes(bytes)
            }
            point_field::FLOAT64 => {
                let bytes = data[start..start + SIZE_OF_DATATYPE[point_field::FLOAT64 as usize]]
                    .try_into()
                    .unwrap_or_else(|e| panic!("Expected slice with 8 element: {e:?}"));
                f64::from_le_bytes(bytes) as f32
            }
            d => {
                error!("Unknown datatype in PointField: {d}");
                continue;
            }
        };
        match f.name.as_str() {
            "x" => p.x = val,
            "y" => p.y = val,
            "z" => p.z = val,
            "cluster_id" => {
                let _ = p.id.insert(val as u32);
            }
            _ => {}
        }
    }

    p
}

#[instrument(skip_all)]
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

#[instrument(skip_all)]
pub fn serialize_pcd(
    points: &[ParsedPoint],
    fields: &[PointField],
    vision_class: &[u8],
    fusion_class: &[u8],
    instance_id: &[u32],
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
                    serialize_field_u32(f, &p.id.unwrap_or_default(), point_offset, &mut buf)
                }
                VISION_CLASS => serialize_field_u8(f, &vision_class[i], point_offset, &mut buf),
                FUSION_CLASS => serialize_field_u8(f, &fusion_class[i], point_offset, &mut buf),
                INSTANCE_ID => {
                    serialize_field_u32(f, &instance_id[i], point_offset, &mut buf)
                }
                _ => {}
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
            let d = { *val }.to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        point_field::FLOAT64 => {
            let d = (*val as f64).to_ne_bytes();
            buf[start..end].copy_from_slice(&d);
        }
        d => {
            error!("Unknown datatype in PointField: {d}");
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
            let d = { *val }.to_ne_bytes();
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
            error!("Unknown datatype in PointField: {d}");
        }
    }
}

fn serialize_field_u32(field: &PointField, val: &u32, point_offset: usize, buf: &mut [u8]) {
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
            let d = (*val).to_ne_bytes();
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
            error!("Unknown datatype in PointField: {d}");
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

/// Insert the standard fusion output fields into a PointCloud2 message.
/// Resets `fields` and `point_step` before inserting.
pub fn insert_standard_fields(pcd: &mut PointCloud2) {
    pcd.fields = Vec::new();
    pcd.point_step = 0;
    for (field_name, datatype) in [
        ("x", point_field::FLOAT32),
        ("y", point_field::FLOAT32),
        ("z", point_field::FLOAT32),
        ("cluster_id", point_field::UINT32),
        (FUSION_CLASS, point_field::UINT8),
        (VISION_CLASS, point_field::UINT8),
        (INSTANCE_ID, point_field::UINT32),
    ] {
        insert_field(
            pcd,
            PointField {
                name: field_name.to_string(),
                offset: 0, // offset is calculated by insert_field
                datatype,
                count: 1,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the standard output field list including instance_id
    fn make_fields() -> (Vec<PointField>, u32) {
        let mut offset = 0u32;
        let mut fields = Vec::new();
        for (name, datatype) in [
            ("x", point_field::FLOAT32),
            ("y", point_field::FLOAT32),
            ("z", point_field::FLOAT32),
            ("cluster_id", point_field::UINT32),
            (FUSION_CLASS, point_field::UINT8),
            (VISION_CLASS, point_field::UINT8),
            (INSTANCE_ID, point_field::UINT32),
        ] {
            fields.push(PointField {
                name: name.to_string(),
                offset,
                datatype,
                count: 1,
            });
            offset += SIZE_OF_DATATYPE[datatype as usize] as u32;
        }
        (fields, offset)
    }

    #[test]
    fn serialize_pcd_writes_instance_id_at_expected_offset() {
        let points = vec![
            ParsedPoint {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                id: Some(10),
            },
            ParsedPoint {
                x: 4.0,
                y: 5.0,
                z: 6.0,
                id: Some(20),
            },
        ];
        let vision_class = vec![1u8, 2u8];
        let fusion_class = vec![3u8, 4u8];
        let instance_id = vec![42u32, 99u32];

        let (fields, point_step) = make_fields();
        let buf = serialize_pcd(&points, &fields, &vision_class, &fusion_class, &instance_id);

        // Total buffer size should be point_step * num_points
        assert_eq!(buf.len(), point_step as usize * points.len());

        // instance_id field is the last field: offset = 3*4 (xyz) + 4 (cluster_id) + 1 (fusion) + 1 (vision) = 18
        let instance_id_offset = 18usize;

        // Point 0: instance_id = 42
        let start = 0 * point_step as usize + instance_id_offset;
        let val = u32::from_ne_bytes(buf[start..start + 4].try_into().unwrap());
        assert_eq!(val, 42);

        // Point 1: instance_id = 99
        let start = 1 * point_step as usize + instance_id_offset;
        let val = u32::from_ne_bytes(buf[start..start + 4].try_into().unwrap());
        assert_eq!(val, 99);
    }

    #[test]
    fn serialize_pcd_zero_instance_ids() {
        let points = vec![
            ParsedPoint {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                id: None,
            },
            ParsedPoint {
                x: 1.0,
                y: 1.0,
                z: 1.0,
                id: None,
            },
            ParsedPoint {
                x: 2.0,
                y: 2.0,
                z: 2.0,
                id: None,
            },
        ];
        let vision_class = vec![0u8; 3];
        let fusion_class = vec![0u8; 3];
        let instance_id = vec![0u32; 3];

        let (fields, point_step) = make_fields();
        let buf = serialize_pcd(&points, &fields, &vision_class, &fusion_class, &instance_id);

        let instance_id_offset = 18usize;

        for i in 0..3 {
            let start = i * point_step as usize + instance_id_offset;
            let val = u32::from_ne_bytes(buf[start..start + 4].try_into().unwrap());
            assert_eq!(val, 0, "instance_id for point {i} should be 0");
        }
    }

    #[test]
    fn serialize_pcd_preserves_other_fields_with_instance_id() {
        let points = vec![ParsedPoint {
            x: 1.5,
            y: 2.5,
            z: 3.5,
            id: Some(7),
        }];
        let vision_class = vec![5u8];
        let fusion_class = vec![9u8];
        let instance_id = vec![100u32];

        let (fields, point_step) = make_fields();
        let buf = serialize_pcd(&points, &fields, &vision_class, &fusion_class, &instance_id);

        assert_eq!(buf.len(), point_step as usize);

        // x at offset 0
        let val = f32::from_ne_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(val, 1.5);

        // y at offset 4
        let val = f32::from_ne_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(val, 2.5);

        // z at offset 8
        let val = f32::from_ne_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(val, 3.5);

        // cluster_id at offset 12
        let val = u32::from_ne_bytes(buf[12..16].try_into().unwrap());
        assert_eq!(val, 7);

        // fusion_class at offset 16
        assert_eq!(buf[16], 9);

        // vision_class at offset 17
        assert_eq!(buf[17], 5);

        // instance_id at offset 18
        let val = u32::from_ne_bytes(buf[18..22].try_into().unwrap());
        assert_eq!(val, 100);
    }
}
