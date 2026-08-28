// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use edgefirst_schemas::builtin_interfaces::Time;
use edgefirst_schemas::sensor_msgs::{point_field, PointCloud2, PointFieldView};
use log::warn;
use tracing::instrument;

use crate::mask::UNCLASSIFIED;
use crate::{FUSION_CLASS, INSTANCE_ID, TRACK_ID, VISION_CLASS};

/// Owned timestamp and frame_id extracted from a buffer-backed PCD header.
#[derive(Debug, Clone)]
pub struct CloudHeader {
    pub stamp: Time,
    pub frame_id: String,
}

/// Structure-of-Arrays representation of a point cloud frame with fusion
/// annotations. All vectors are parallel and indexed by point index.
pub struct FusionFrame {
    pub len: usize,
    // Core geometry (from input PCD)
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    // Cluster IDs from input (empty if no clustering)
    pub cluster_id: Vec<u32>,
    // Projection results (filled by transform_and_project_points)
    pub proj_u: Vec<f32>,
    pub proj_v: Vec<f32>,
    // Fusion results (filled by fusion algorithms)
    pub vision_class: Vec<u8>,
    pub fusion_class: Vec<u8>,
    pub instance_id: Vec<u16>,
    pub track_id: Vec<u32>,
}

impl FusionFrame {
    /// Create a new frame with pre-allocated capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            len: 0,
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
            cluster_id: Vec::new(),
            proj_u: Vec::new(),
            proj_v: Vec::new(),
            vision_class: vec![UNCLASSIFIED; capacity],
            fusion_class: vec![UNCLASSIFIED; capacity],
            instance_id: vec![0; capacity],
            track_id: vec![0; capacity],
        }
    }

    /// Returns true if the input PCD had cluster_id fields.
    pub fn has_clusters(&self) -> bool {
        !self.cluster_id.is_empty()
    }
}

/// Resolved byte offsets for fields within a single point record.
struct FieldOffsets {
    x: usize,
    y: usize,
    z: usize,
    cluster_id: Option<usize>,
}

/// Scan `PointCloud2.fields` once to find byte offsets for known fields.
fn resolve_offsets(fields: &[PointFieldView<'_>]) -> FieldOffsets {
    let mut offsets = FieldOffsets {
        x: 0,
        y: 4,
        z: 8,
        cluster_id: None,
    };
    for f in fields {
        match f.name {
            "x" => offsets.x = f.offset as usize,
            "y" => offsets.y = f.offset as usize,
            "z" => offsets.z = f.offset as usize,
            "cluster_id" => offsets.cluster_id = Some(f.offset as usize),
            _ => {}
        }
    }
    offsets
}

/// Read an f32 from a byte slice at the given offset using the appropriate
/// endianness.
#[inline(always)]
fn read_f32(data: &[u8], offset: usize, big_endian: bool) -> f32 {
    let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
    if big_endian {
        f32::from_be_bytes(bytes)
    } else {
        f32::from_le_bytes(bytes)
    }
}

/// Read a u32 from a byte slice at the given offset using the appropriate
/// endianness.
#[inline(always)]
fn read_u32(data: &[u8], offset: usize, big_endian: bool) -> u32 {
    let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
    if big_endian {
        u32::from_be_bytes(bytes)
    } else {
        u32::from_le_bytes(bytes)
    }
}

/// Parse a PointCloud2 message into a FusionFrame using branchless
/// offset-based gather. Field offsets are resolved once, then each point
/// is read with a stride loop — no per-field string matching.
#[instrument(skip_all)]
pub fn parse_pcd<B: AsRef<[u8]>>(pcd: &PointCloud2<B>) -> FusionFrame {
    let fields = pcd.fields();
    let offsets = resolve_offsets(&fields);
    let stride = pcd.point_step() as usize;
    let n = (pcd.height() * pcd.width()) as usize;
    let data = pcd.data();
    let be = pcd.is_bigendian();
    let has_cid = offsets.cluster_id.is_some();

    let mut frame = FusionFrame::new(n);
    if has_cid {
        frame.cluster_id.reserve(n);
    }

    for i in 0..n {
        let base = i * stride;
        frame.x.push(read_f32(data, base + offsets.x, be));
        frame.y.push(read_f32(data, base + offsets.y, be));
        frame.z.push(read_f32(data, base + offsets.z, be));
        if let Some(cid_off) = offsets.cluster_id {
            frame.cluster_id.push(read_u32(data, base + cid_off, be));
        }
    }
    frame.len = n;
    frame
}

fn pf(name: &'static str, offset: u32, datatype: u8) -> PointFieldView<'static> {
    PointFieldView {
        name,
        offset,
        datatype,
        count: 1,
    }
}

fn build_pcd(
    header: &CloudHeader,
    width: u32,
    fields: &[PointFieldView<'_>],
    point_step: u32,
    data: &[u8],
) -> PointCloud2<Vec<u8>> {
    PointCloud2::builder()
        .stamp(header.stamp)
        .frame_id(header.frame_id.as_str())
        .height(1)
        .width(width)
        .fields(fields)
        .is_bigendian(false)
        .point_step(point_step)
        .row_step(data.len() as u32)
        .data(data)
        .is_dense(true)
        .build()
        .expect("valid PointCloud2")
}

/// Classes topic point layout: x(f32) y(f32) z(f32) fusion_class(u8)
/// vision_class(u8) instance_id(u16) = 16 bytes/point
const CLASSES_POINT_STEP: u32 = 16;

/// Build the PointField descriptors for the classes topic.
fn classes_fields() -> [PointFieldView<'static>; 6] {
    [
        pf("x", 0, point_field::FLOAT32),
        pf("y", 4, point_field::FLOAT32),
        pf("z", 8, point_field::FLOAT32),
        pf(FUSION_CLASS, 12, point_field::UINT8),
        pf(VISION_CLASS, 13, point_field::UINT8),
        pf(INSTANCE_ID, 14, point_field::UINT16),
    ]
}

/// Late-fusion point layout: x(f32) y(f32) z(f32) vision_class(u16)
/// instance_id(u16) = 16 bytes/point
const LATE_FUSION_POINT_STEP: u32 = 16;

/// Late-fusion point layout with tracking: x(f32) y(f32) z(f32)
/// vision_class(u16) instance_id(u16) track_id(u32) = 20 bytes/point
const LATE_FUSION_TRACKED_POINT_STEP: u32 = 20;

fn late_fusion_fields() -> [PointFieldView<'static>; 5] {
    [
        pf("x", 0, point_field::FLOAT32),
        pf("y", 4, point_field::FLOAT32),
        pf("z", 8, point_field::FLOAT32),
        pf(VISION_CLASS, 12, point_field::UINT16),
        pf(INSTANCE_ID, 14, point_field::UINT16),
    ]
}

fn late_fusion_tracked_fields() -> [PointFieldView<'static>; 6] {
    [
        pf("x", 0, point_field::FLOAT32),
        pf("y", 4, point_field::FLOAT32),
        pf("z", 8, point_field::FLOAT32),
        pf(VISION_CLASS, 12, point_field::UINT16),
        pf(INSTANCE_ID, 14, point_field::UINT16),
        pf(TRACK_ID, 16, point_field::UINT32),
    ]
}

/// Serialize late-fusion output: x y z vision_class(u16) instance_id(u16).
/// When `tracking` is true, appends track_id(u32) for 20 bytes/point.
/// Used when no fusion model is active (no fusion_class field).
#[instrument(skip_all)]
pub fn serialize_late_fusion(
    frame: &FusionFrame,
    header: &CloudHeader,
    tracking: bool,
) -> PointCloud2<Vec<u8>> {
    let n = frame.len;
    let tracked_fields = late_fusion_tracked_fields();
    let untracked_fields = late_fusion_fields();
    let (point_step, fields): (u32, &[PointFieldView]) = if tracking {
        (LATE_FUSION_TRACKED_POINT_STEP, &tracked_fields)
    } else {
        (LATE_FUSION_POINT_STEP, &untracked_fields)
    };
    let step = point_step as usize;
    let mut data = vec![0u8; n * step];

    for i in 0..n {
        let base = i * step;
        data[base..base + 4].copy_from_slice(&frame.x[i].to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&frame.y[i].to_le_bytes());
        data[base + 8..base + 12].copy_from_slice(&frame.z[i].to_le_bytes());
        data[base + 12..base + 14].copy_from_slice(&(frame.vision_class[i] as u16).to_le_bytes());
        data[base + 14..base + 16].copy_from_slice(&frame.instance_id[i].to_le_bytes());
        if tracking {
            data[base + 16..base + 20].copy_from_slice(&frame.track_id[i].to_le_bytes());
        }
    }

    build_pcd(header, n as u32, fields, point_step, &data)
}

/// Serialize xyzc (16 bytes/point) for rt/fusion/classes.
///
/// Layout per point: x(f32) y(f32) z(f32) fusion_class(u8) vision_class(u8)
/// instance_id(u16)
#[instrument(skip_all)]
pub fn serialize_classes(frame: &FusionFrame, header: &CloudHeader) -> PointCloud2<Vec<u8>> {
    let n = frame.len;
    let step = CLASSES_POINT_STEP as usize;
    let mut data = vec![0u8; n * step];

    for i in 0..n {
        let base = i * step;
        data[base..base + 4].copy_from_slice(&frame.x[i].to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&frame.y[i].to_le_bytes());
        data[base + 8..base + 12].copy_from_slice(&frame.z[i].to_le_bytes());
        data[base + 12] = frame.fusion_class[i];
        data[base + 13] = frame.vision_class[i];
        data[base + 14..base + 16].copy_from_slice(&frame.instance_id[i].to_le_bytes());
    }

    build_pcd(
        header,
        n as u32,
        &classes_fields(),
        CLASSES_POINT_STEP,
        &data,
    )
}

/// Serialize a FusionFrame for the occupancy grid topic. Uses the same
/// field layout as the legacy format for backward compatibility with grid
/// consumers: x y z cluster_id fusion_class vision_class instance_id track_id
/// (24 bytes/point).
#[instrument(skip_all)]
pub fn serialize_grid(frame: &FusionFrame, header: &CloudHeader) -> PointCloud2<Vec<u8>> {
    let n = frame.len;
    let step = 24usize;
    let mut data = vec![0u8; n * step];

    for i in 0..n {
        let base = i * step;
        data[base..base + 4].copy_from_slice(&frame.x[i].to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&frame.y[i].to_le_bytes());
        data[base + 8..base + 12].copy_from_slice(&frame.z[i].to_le_bytes());
        let cid = if i < frame.cluster_id.len() {
            frame.cluster_id[i]
        } else {
            0
        };
        data[base + 12..base + 16].copy_from_slice(&cid.to_le_bytes());
        data[base + 16] = frame.fusion_class[i];
        data[base + 17] = frame.vision_class[i];
        data[base + 18..base + 20].copy_from_slice(&frame.instance_id[i].to_le_bytes());
        data[base + 20..base + 24].copy_from_slice(&frame.track_id[i].to_le_bytes());
    }

    build_pcd(header, n as u32, &grid_fields(), 24, &data)
}

/// Build the PointField descriptors for the grid/legacy format.
fn grid_fields() -> [PointFieldView<'static>; 8] {
    [
        pf("x", 0, point_field::FLOAT32),
        pf("y", 4, point_field::FLOAT32),
        pf("z", 8, point_field::FLOAT32),
        pf("cluster_id", 12, point_field::UINT32),
        pf(FUSION_CLASS, 16, point_field::UINT8),
        pf(VISION_CLASS, 17, point_field::UINT8),
        pf(INSTANCE_ID, 18, point_field::UINT16),
        pf(TRACK_ID, 20, point_field::UINT32),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_header() -> CloudHeader {
        CloudHeader {
            stamp: Time { sec: 0, nanosec: 0 },
            frame_id: "test".to_string(),
        }
    }

    /// Build a PointCloud2 with xyz + optional cluster_id fields, little-endian.
    fn make_input_pcd(
        points: &[(f32, f32, f32)],
        cluster_ids: Option<&[u32]>,
    ) -> PointCloud2<Vec<u8>> {
        let has_cid = cluster_ids.is_some();
        let point_step: u32 = if has_cid { 16 } else { 12 };
        let n = points.len();
        let mut data = vec![0u8; n * point_step as usize];

        for (i, (x, y, z)) in points.iter().enumerate() {
            let base = i * point_step as usize;
            data[base..base + 4].copy_from_slice(&x.to_le_bytes());
            data[base + 4..base + 8].copy_from_slice(&y.to_le_bytes());
            data[base + 8..base + 12].copy_from_slice(&z.to_le_bytes());
            if let Some(cids) = cluster_ids {
                data[base + 12..base + 16].copy_from_slice(&cids[i].to_le_bytes());
            }
        }

        let xyz = [
            pf("x", 0, point_field::FLOAT32),
            pf("y", 4, point_field::FLOAT32),
            pf("z", 8, point_field::FLOAT32),
        ];
        let with_cid = [
            pf("x", 0, point_field::FLOAT32),
            pf("y", 4, point_field::FLOAT32),
            pf("z", 8, point_field::FLOAT32),
            pf("cluster_id", 12, point_field::UINT32),
        ];
        let fields: &[PointFieldView] = if has_cid { &with_cid } else { &xyz };
        let header = test_header();
        let built = build_pcd(&header, n as u32, fields, point_step, &data);
        PointCloud2::from_cdr(built.into_cdr()).expect("PointCloud2 roundtrip")
    }

    #[test]
    fn parse_pcd_xyz_only() {
        let pcd = make_input_pcd(&[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], None);
        let frame = parse_pcd(&pcd);
        assert_eq!(frame.len, 2);
        assert_eq!(frame.x, vec![1.0, 4.0]);
        assert_eq!(frame.y, vec![2.0, 5.0]);
        assert_eq!(frame.z, vec![3.0, 6.0]);
        assert!(!frame.has_clusters());
    }

    #[test]
    fn parse_pcd_with_cluster_ids() {
        let pcd = make_input_pcd(&[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], Some(&[10, 20]));
        let frame = parse_pcd(&pcd);
        assert_eq!(frame.len, 2);
        assert!(frame.has_clusters());
        assert_eq!(frame.cluster_id, vec![10, 20]);
    }

    #[test]
    fn serialize_classes_roundtrip() {
        let mut frame = FusionFrame::new(2);
        frame.x = vec![1.5, 4.0];
        frame.y = vec![2.5, 5.0];
        frame.z = vec![3.5, 6.0];
        frame.fusion_class = vec![9, 4];
        frame.vision_class = vec![5, 2];
        frame.instance_id = vec![100, 99];
        frame.track_id = vec![5555, 2000];
        frame.len = 2;

        let header = test_header();
        let pcd = serialize_classes(&frame, &header);
        let data = pcd.data();

        assert_eq!(pcd.point_step(), 16);
        assert_eq!(pcd.width(), 2);
        assert_eq!(data.len(), 32);

        // Point 0
        let val = f32::from_le_bytes(data[0..4].try_into().unwrap());
        assert_eq!(val, 1.5);
        let val = f32::from_le_bytes(data[4..8].try_into().unwrap());
        assert_eq!(val, 2.5);
        let val = f32::from_le_bytes(data[8..12].try_into().unwrap());
        assert_eq!(val, 3.5);
        assert_eq!(data[12], 9); // fusion_class
        assert_eq!(data[13], 5); // vision_class
        let val = u16::from_le_bytes(data[14..16].try_into().unwrap());
        assert_eq!(val, 100); // instance_id

        // Point 1
        let val = f32::from_le_bytes(data[16..20].try_into().unwrap());
        assert_eq!(val, 4.0);
        assert_eq!(data[28], 4); // fusion_class
        assert_eq!(data[29], 2); // vision_class
    }

    #[test]
    fn serialize_late_fusion_roundtrip() {
        let mut frame = FusionFrame::new(2);
        frame.x = vec![1.5, 4.0];
        frame.y = vec![2.5, 5.0];
        frame.z = vec![3.5, 6.0];
        frame.vision_class = vec![5, 200];
        frame.instance_id = vec![100, 99];
        frame.len = 2;

        let header = test_header();
        let pcd = serialize_late_fusion(&frame, &header, false);
        let data = pcd.data();
        let fields = pcd.fields();

        assert_eq!(pcd.point_step(), 16);
        assert_eq!(pcd.width(), 2);
        assert_eq!(data.len(), 32);
        assert_eq!(fields.len(), 5);
        assert_eq!(fields[3].name, "vision_class");
        assert_eq!(fields[3].datatype, point_field::UINT16);

        // Point 0
        assert_eq!(f32::from_le_bytes(data[0..4].try_into().unwrap()), 1.5);
        assert_eq!(u16::from_le_bytes(data[12..14].try_into().unwrap()), 5);
        assert_eq!(u16::from_le_bytes(data[14..16].try_into().unwrap()), 100);

        // Point 1
        assert_eq!(f32::from_le_bytes(data[16..20].try_into().unwrap()), 4.0);
        assert_eq!(u16::from_le_bytes(data[28..30].try_into().unwrap()), 200);
        assert_eq!(u16::from_le_bytes(data[30..32].try_into().unwrap()), 99);
    }

    #[test]
    fn serialize_late_fusion_tracked_roundtrip() {
        let mut frame = FusionFrame::new(2);
        frame.x = vec![1.5, 4.0];
        frame.y = vec![2.5, 5.0];
        frame.z = vec![3.5, 6.0];
        frame.vision_class = vec![5, 200];
        frame.instance_id = vec![100, 99];
        frame.track_id = vec![0xDEAD_BEEF, 42];
        frame.len = 2;

        let header = test_header();
        let pcd = serialize_late_fusion(&frame, &header, true);
        let data = pcd.data();
        let fields = pcd.fields();

        assert_eq!(pcd.point_step(), 20);
        assert_eq!(pcd.width(), 2);
        assert_eq!(data.len(), 40);
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[5].name, "track_id");
        assert_eq!(fields[5].datatype, point_field::UINT32);
        assert_eq!(fields[5].offset, 16);

        // Point 0
        assert_eq!(f32::from_le_bytes(data[0..4].try_into().unwrap()), 1.5);
        assert_eq!(u16::from_le_bytes(data[12..14].try_into().unwrap()), 5);
        assert_eq!(u16::from_le_bytes(data[14..16].try_into().unwrap()), 100);
        assert_eq!(
            u32::from_le_bytes(data[16..20].try_into().unwrap()),
            0xDEAD_BEEF
        );

        // Point 1
        assert_eq!(f32::from_le_bytes(data[20..24].try_into().unwrap()), 4.0);
        assert_eq!(u16::from_le_bytes(data[32..34].try_into().unwrap()), 200);
        assert_eq!(u16::from_le_bytes(data[34..36].try_into().unwrap()), 99);
        assert_eq!(u32::from_le_bytes(data[36..40].try_into().unwrap()), 42);
    }

    #[test]
    fn serialize_grid_preserves_all_fields() {
        let mut frame = FusionFrame::new(1);
        frame.x = vec![1.5];
        frame.y = vec![2.5];
        frame.z = vec![3.5];
        frame.cluster_id = vec![7];
        frame.fusion_class = vec![9];
        frame.vision_class = vec![5];
        frame.instance_id = vec![100];
        frame.track_id = vec![5555];
        frame.len = 1;

        let header = test_header();
        let pcd = serialize_grid(&frame, &header);
        let buf = pcd.data();

        assert_eq!(pcd.point_step(), 24);
        assert_eq!(buf.len(), 24);

        assert_eq!(f32::from_le_bytes(buf[0..4].try_into().unwrap()), 1.5);
        assert_eq!(f32::from_le_bytes(buf[4..8].try_into().unwrap()), 2.5);
        assert_eq!(f32::from_le_bytes(buf[8..12].try_into().unwrap()), 3.5);
        assert_eq!(u32::from_le_bytes(buf[12..16].try_into().unwrap()), 7);
        assert_eq!(buf[16], 9); // fusion_class
        assert_eq!(buf[17], 5); // vision_class
        assert_eq!(u16::from_le_bytes(buf[18..20].try_into().unwrap()), 100);
        assert_eq!(u32::from_le_bytes(buf[20..24].try_into().unwrap()), 5555);
    }
}
