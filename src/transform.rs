// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

// From https://github.com/smilerobotics/tf_rosrust/blob/main/tf_rosrust/src/transforms.rs
// licensed under MIT
// Modified by Au-Zone Technologies 2025

use edgefirst_schemas::geometry_msgs::Transform;
use nalgebra::geometry::{Isometry3, Translation3, UnitQuaternion};
use tracing::instrument;

use crate::pcd::FusionFrame;

pub fn isometry_from_transform(tf: &Transform) -> Isometry3<f32> {
    let trans = Translation3::new(
        tf.translation.x as f32,
        tf.translation.y as f32,
        tf.translation.z as f32,
    );
    let rot = UnitQuaternion::new_normalize(nalgebra::geometry::Quaternion::new(
        tf.rotation.w as f32,
        tf.rotation.x as f32,
        tf.rotation.y as f32,
        tf.rotation.z as f32,
    ));

    Isometry3::from_parts(trans, rot)
}

/// Flat row-major 4x4 matrix for direct SIMD-friendly access.
/// Element at (row, col) is stored at index `row * 4 + col`.
struct Mat4([f32; 16]);

impl Mat4 {
    /// Convert from a nalgebra 4x4 column-major matrix.
    fn from_nalgebra(m: &nalgebra::Matrix4<f32>) -> Self {
        let mut out = [0.0f32; 16];
        for r in 0..4 {
            for c in 0..4 {
                out[r * 4 + c] = m[(r, c)];
            }
        }
        Self(out)
    }

    /// Transform a point (x, y, z, 1) returning (rx, ry, rz).
    #[inline(always)]
    fn transform_point(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let m = &self.0;
        let rx = m[0] * x + m[1] * y + m[2] * z + m[3];
        let ry = m[4] * x + m[5] * y + m[6] * z + m[7];
        let rz = m[8] * x + m[9] * y + m[10] * z + m[11];
        (rx, ry, rz)
    }
}

/// Transform points from their source frame to base_link (in-place), then
/// project them into normalized [0,1] image coordinates using the camera
/// optical frame transform and intrinsics.
///
/// `lidar_transform`: tf_static base_link -> lidar (transforms lidar points to base_link)
/// `cam_transform`:   tf_static base_link -> camera_optical
/// `cam_mtx`:         3x3 intrinsic matrix K (row-major: fx, 0, cx, 0, fy, cy, 0, 0, 1)
/// `image_dims`:      (width, height) in pixels
///
/// After this call, `frame.x/y/z` are in the base_link frame.
/// Projection results are stored in `frame.proj_u` and `frame.proj_v`.
/// Points behind the camera get coordinates outside [0,1].
#[instrument(skip_all)]
pub(crate) fn transform_and_project_points(
    frame: &mut FusionFrame,
    lidar_transform: &Transform,
    cam_transform: &Transform,
    cam_mtx: &[f32; 9],
    image_dims: (f32, f32),
) {
    let n = frame.len;

    // Pre-compose matrices once
    let t_base_lidar = isometry_from_transform(lidar_transform);
    let t_base_cam = isometry_from_transform(cam_transform);
    let t_cam_lidar = t_base_cam.inverse() * t_base_lidar;

    let base_m = Mat4::from_nalgebra(&t_base_lidar.to_matrix());
    let cam_m = Mat4::from_nalgebra(&t_cam_lidar.to_matrix());

    let fx = cam_mtx[0];
    let cx = cam_mtx[2];
    let fy = cam_mtx[4];
    let cy = cam_mtx[5];
    let (width, height) = image_dims;

    frame.proj_u.resize(n, 0.0);
    frame.proj_v.resize(n, 0.0);

    #[cfg(target_arch = "aarch64")]
    {
        let chunks = n / 4;
        for c in 0..chunks {
            let i = c * 4;
            unsafe {
                neon_transform_project_4(frame, i, &base_m, &cam_m, fx, fy, cx, cy, width, height);
            }
        }
        // Scalar remainder
        for i in (chunks * 4)..n {
            scalar_transform_project(frame, i, &base_m, &cam_m, fx, fy, cx, cy, width, height);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            scalar_transform_project(frame, i, &base_m, &cam_m, fx, fy, cx, cy, width, height);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn scalar_transform_project(
    frame: &mut FusionFrame,
    i: usize,
    base_m: &Mat4,
    cam_m: &Mat4,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
) {
    let ox = frame.x[i];
    let oy = frame.y[i];
    let oz = frame.z[i];

    // Transform to base_link
    let (bx, by, bz) = base_m.transform_point(ox, oy, oz);
    frame.x[i] = bx;
    frame.y[i] = by;
    frame.z[i] = bz;

    // Transform to camera frame for projection
    let (camx, camy, camz) = cam_m.transform_point(ox, oy, oz);

    if camz <= 0.0 {
        frame.proj_u[i] = 2.0;
        frame.proj_v[i] = 2.0;
    } else {
        let inv_z = 1.0 / camz;
        frame.proj_u[i] = (fx * camx * inv_z + cx) / width;
        frame.proj_v[i] = (fy * camy * inv_z + cy) / height;
    }
}

/// NEON: transform and project 4 points at once.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_transform_project_4(
    frame: &mut FusionFrame,
    i: usize,
    base_m: &Mat4,
    cam_m: &Mat4,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
) {
    use std::arch::aarch64::*;

    let bm = &base_m.0;
    let cm = &cam_m.0;

    // Load 4 points from SoA columns
    let x4 = vld1q_f32(frame.x.as_ptr().add(i));
    let y4 = vld1q_f32(frame.y.as_ptr().add(i));
    let z4 = vld1q_f32(frame.z.as_ptr().add(i));

    // Base-link transform: result = M * [x, y, z, 1]^T (row-major)
    // row 0: bm[0]*x + bm[1]*y + bm[2]*z + bm[3]
    let bx = vfmaq_n_f32(
        vfmaq_n_f32(vfmaq_n_f32(vdupq_n_f32(bm[3]), x4, bm[0]), y4, bm[1]),
        z4,
        bm[2],
    );
    let by = vfmaq_n_f32(
        vfmaq_n_f32(vfmaq_n_f32(vdupq_n_f32(bm[7]), x4, bm[4]), y4, bm[5]),
        z4,
        bm[6],
    );
    let bz = vfmaq_n_f32(
        vfmaq_n_f32(vfmaq_n_f32(vdupq_n_f32(bm[11]), x4, bm[8]), y4, bm[9]),
        z4,
        bm[10],
    );

    // Store base_link coords
    vst1q_f32(frame.x.as_mut_ptr().add(i), bx);
    vst1q_f32(frame.y.as_mut_ptr().add(i), by);
    vst1q_f32(frame.z.as_mut_ptr().add(i), bz);

    // Camera-frame transform (from original lidar coords)
    let cx4 = vfmaq_n_f32(
        vfmaq_n_f32(vfmaq_n_f32(vdupq_n_f32(cm[3]), x4, cm[0]), y4, cm[1]),
        z4,
        cm[2],
    );
    let cy4 = vfmaq_n_f32(
        vfmaq_n_f32(vfmaq_n_f32(vdupq_n_f32(cm[7]), x4, cm[4]), y4, cm[5]),
        z4,
        cm[6],
    );
    let cz4 = vfmaq_n_f32(
        vfmaq_n_f32(vfmaq_n_f32(vdupq_n_f32(cm[11]), x4, cm[8]), y4, cm[9]),
        z4,
        cm[10],
    );

    // Pinhole projection with behind-camera handling
    let zero = vdupq_n_f32(0.0);
    let behind = vcleq_f32(cz4, zero);

    // Reciprocal of z (Newton-Raphson refined)
    let inv_z_est = vrecpeq_f32(cz4);
    let inv_z = vmulq_f32(vrecpsq_f32(cz4, inv_z_est), inv_z_est);

    // u = (fx * cx/cz + cx_intr) / width, v = (fy * cy/cz + cy_intr) / height
    let u4 = vmulq_n_f32(
        vfmaq_n_f32(vdupq_n_f32(cx), vmulq_f32(cx4, inv_z), fx),
        1.0 / width,
    );
    let v4 = vmulq_n_f32(
        vfmaq_n_f32(vdupq_n_f32(cy), vmulq_f32(cy4, inv_z), fy),
        1.0 / height,
    );

    // Points behind camera get (2.0, 2.0)
    let oob = vdupq_n_f32(2.0);
    let u_out = vbslq_f32(behind, oob, u4);
    let v_out = vbslq_f32(behind, oob, v4);

    vst1q_f32(frame.proj_u.as_mut_ptr().add(i), u_out);
    vst1q_f32(frame.proj_v.as_mut_ptr().add(i), v_out);
}

#[cfg(test)]
mod projection_test {

    use edgefirst_schemas::geometry_msgs::{Quaternion, Vector3};

    use super::*;

    fn identity_transform() -> Transform {
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
    }

    #[test]
    fn test_basic() {
        // Point at (0, 0, 10) in camera optical frame (10m forward along Z)
        // With fx=1260, fy=1260, cx=960, cy=540 on 1920x1080 image
        // Expected: u = (1260*0/10 + 960) / 1920 = 0.5
        //           v = (1260*0/10 + 540) / 1080 = 0.5
        let mut frame = FusionFrame::new(1);
        frame.x = vec![0.0];
        frame.y = vec![0.0];
        frame.z = vec![10.0];
        frame.len = 1;

        let cam_mtx = [1260.0, 0.0, 960.0, 0.0, 1260.0, 540.0, 0.0, 0.0, 1.0];
        let identity = identity_transform();
        transform_and_project_points(&mut frame, &identity, &identity, &cam_mtx, (1920.0, 1080.0));
        assert!(
            (frame.proj_u[0] - 0.5).abs() < 0.001,
            "center u={}",
            frame.proj_u[0]
        );
        assert!(
            (frame.proj_v[0] - 0.5).abs() < 0.001,
            "center v={}",
            frame.proj_v[0]
        );
    }

    #[test]
    fn test_behind_camera_points_out_of_bounds() {
        let mut frame = FusionFrame::new(2);
        frame.x = vec![0.0, 0.0];
        frame.y = vec![0.0, 0.0];
        frame.z = vec![10.0, -5.0];
        frame.len = 2;

        let cam_mtx = [1260.0, 0.0, 960.0, 0.0, 1260.0, 540.0, 0.0, 0.0, 1.0];
        let identity = identity_transform();
        transform_and_project_points(&mut frame, &identity, &identity, &cam_mtx, (1920.0, 1080.0));
        // In-front point should have valid [0,1] projection
        assert!(
            frame.proj_u[0] >= 0.0 && frame.proj_u[0] <= 1.0,
            "front point u={}",
            frame.proj_u[0]
        );
        assert!(
            frame.proj_v[0] >= 0.0 && frame.proj_v[0] <= 1.0,
            "front point v={}",
            frame.proj_v[0]
        );
        // Behind-camera point should NOT be in [0,1] bounds
        assert!(
            frame.proj_u[1] < 0.0
                || frame.proj_u[1] > 1.0
                || frame.proj_v[1] < 0.0
                || frame.proj_v[1] > 1.0,
            "behind-camera point should be out of bounds but got ({}, {})",
            frame.proj_u[1],
            frame.proj_v[1]
        );
    }

    #[test]
    fn test_with_lidar_and_camera_transforms() {
        let lidar_tf = identity_transform();
        let cam_tf = Transform {
            translation: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: Quaternion {
                x: -0.5,
                y: 0.5,
                z: -0.5,
                w: 0.5,
            },
        };

        // Point at (10, 0, 0) in base_link = (0, 0, 10) in camera optical
        let mut frame = FusionFrame::new(1);
        frame.x = vec![10.0];
        frame.y = vec![0.0];
        frame.z = vec![0.0];
        frame.len = 1;

        let cam_mtx = [1260.0, 0.0, 960.0, 0.0, 1260.0, 540.0, 0.0, 0.0, 1.0];
        transform_and_project_points(&mut frame, &lidar_tf, &cam_tf, &cam_mtx, (1920.0, 1080.0));

        // Should project to image center (0.5, 0.5)
        assert!(
            (frame.proj_u[0] - 0.5).abs() < 0.01,
            "u={} expected ~0.5",
            frame.proj_u[0]
        );
        assert!(
            (frame.proj_v[0] - 0.5).abs() < 0.01,
            "v={} expected ~0.5",
            frame.proj_v[0]
        );

        // frame.x/y/z should still be in base_link frame (unchanged since lidar_tf is identity)
        assert!((frame.x[0] - 10.0).abs() < 0.001);
        assert!((frame.y[0] - 0.0).abs() < 0.001);
        assert!((frame.z[0] - 0.0).abs() < 0.001);
    }
}
