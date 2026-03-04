// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

// From https://github.com/smilerobotics/tf_rosrust/blob/main/tf_rosrust/src/transforms.rs
// licensed under MIT
// Modified by Au-Zone Technologies 2025

use edgefirst_schemas::geometry_msgs::Transform;
use nalgebra::{
    self,
    geometry::{Isometry3, Translation3, UnitQuaternion},
    Matrix4xX,
};
use tracing::instrument;

use crate::ParsedPoint;

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

/// Transform points from their source frame to base_link (in-place), then
/// project them into normalized [0,1] image coordinates using the camera
/// optical frame transform and intrinsics.
///
/// `lidar_transform`: tf_static base_link → lidar (transforms lidar points to base_link)
/// `cam_transform`:   tf_static base_link → camera_optical
/// `cam_mtx`:         3×3 intrinsic matrix K (row-major: fx, 0, cx, 0, fy, cy, 0, 0, 1)
/// `image_dims`:      (width, height) in pixels
///
/// After this call, `points[i].x/y/z` are in the base_link frame.
/// Returns per-point normalized image coordinates [u, v] in [0,1].
/// Points behind the camera get coordinates outside [0,1].
#[instrument(skip_all)]
pub(crate) fn transform_and_project_points(
    points: &mut [ParsedPoint],
    lidar_transform: &Transform,
    cam_transform: &Transform,
    cam_mtx: &[f32; 9],
    image_dims: (f32, f32),
) -> Vec<[f32; 2]> {
    // Step 1: Transform points from lidar frame to base_link frame (in-place)
    let t_base_lidar = isometry_from_transform(lidar_transform);
    let base_mtx = t_base_lidar.to_matrix();

    let mut xyz1 = Vec::with_capacity(points.len() * 4);
    for p in points.iter() {
        xyz1.push(p.x);
        xyz1.push(p.y);
        xyz1.push(p.z);
        xyz1.push(1.0);
    }

    let xyz1 = Matrix4xX::from_vec(xyz1);
    let base_pts = base_mtx * &xyz1;
    for (col, p) in base_pts.column_iter().zip(points.iter_mut()) {
        p.x = col[0];
        p.y = col[1];
        p.z = col[2];
    }

    // Step 2: Compose lidar→camera: inv(T_base_cam) * T_base_lidar
    let t_base_cam = isometry_from_transform(cam_transform);
    let t_cam_lidar = t_base_cam.inverse() * t_base_lidar;
    let cam_frame_mtx = t_cam_lidar.to_matrix();

    // Transform original lidar points to camera optical frame (for projection only)
    let cam_pts = cam_frame_mtx * xyz1;

    // Step 3: Standard pinhole projection → normalized [0,1] coordinates
    // K = [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    let fx = cam_mtx[0];
    let cx = cam_mtx[2];
    let fy = cam_mtx[4];
    let cy = cam_mtx[5];
    let (width, height) = image_dims;

    cam_pts
        .column_iter()
        .map(|col| {
            let (x, y, z) = (col[0], col[1], col[2]);
            if z <= 0.0 {
                // Behind camera: return out-of-bounds coordinates
                [2.0, 2.0]
            } else {
                let u_px = fx * (x / z) + cx;
                let v_px = fy * (y / z) + cy;
                [u_px / width, v_px / height]
            }
        })
        .collect()
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
        let mut points = vec![ParsedPoint {
            x: 0.0,
            y: 0.0,
            z: 10.0,
            id: None,
        }];
        let cam_mtx = [1260.0, 0.0, 960.0, 0.0, 1260.0, 540.0, 0.0, 0.0, 1.0];
        let identity = identity_transform();
        let proj = transform_and_project_points(
            &mut points,
            &identity,
            &identity,
            &cam_mtx,
            (1920.0, 1080.0),
        );
        assert!((proj[0][0] - 0.5).abs() < 0.001, "center u={}", proj[0][0]);
        assert!((proj[0][1] - 0.5).abs() < 0.001, "center v={}", proj[0][1]);
    }

    #[test]
    fn test_behind_camera_points_out_of_bounds() {
        let mut points = vec![
            ParsedPoint {
                x: 0.0,
                y: 0.0,
                z: 10.0,
                id: None,
            }, // in front (Z > 0)
            ParsedPoint {
                x: 0.0,
                y: 0.0,
                z: -5.0,
                id: None,
            }, // behind (Z < 0)
        ];
        let cam_mtx = [1260.0, 0.0, 960.0, 0.0, 1260.0, 540.0, 0.0, 0.0, 1.0];
        let identity = identity_transform();
        let proj = transform_and_project_points(
            &mut points,
            &identity,
            &identity,
            &cam_mtx,
            (1920.0, 1080.0),
        );
        // In-front point should have valid [0,1] projection
        assert!(
            proj[0][0] >= 0.0 && proj[0][0] <= 1.0,
            "front point x={}",
            proj[0][0]
        );
        assert!(
            proj[0][1] >= 0.0 && proj[0][1] <= 1.0,
            "front point y={}",
            proj[0][1]
        );
        // Behind-camera point should NOT be in [0,1] bounds
        let behind = &proj[1];
        assert!(
            behind[0] < 0.0 || behind[0] > 1.0 || behind[1] < 0.0 || behind[1] > 1.0,
            "behind-camera point should be out of bounds but got ({}, {})",
            behind[0],
            behind[1]
        );
    }

    #[test]
    fn test_with_lidar_and_camera_transforms() {
        // Lidar at base_link with identity transform, camera optical frame
        // rotated so that base_link X (forward) maps to camera Z (forward)
        // This is the standard base_link→optical rotation:
        // optical_x = -base_y, optical_y = -base_z, optical_z = base_x
        // quaternion for this rotation: (x=-0.5, y=0.5, z=-0.5, w=0.5)
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
        let mut points = vec![ParsedPoint {
            x: 10.0,
            y: 0.0,
            z: 0.0,
            id: None,
        }];
        let cam_mtx = [1260.0, 0.0, 960.0, 0.0, 1260.0, 540.0, 0.0, 0.0, 1.0];
        let proj = transform_and_project_points(
            &mut points,
            &lidar_tf,
            &cam_tf,
            &cam_mtx,
            (1920.0, 1080.0),
        );

        // Should project to image center (0.5, 0.5)
        assert!(
            (proj[0][0] - 0.5).abs() < 0.01,
            "u={} expected ~0.5",
            proj[0][0]
        );
        assert!(
            (proj[0][1] - 0.5).abs() < 0.01,
            "v={} expected ~0.5",
            proj[0][1]
        );

        // points[0].x/y/z should still be in base_link frame (unchanged since lidar_tf is identity)
        assert!((points[0].x - 10.0).abs() < 0.001);
        assert!((points[0].y - 0.0).abs() < 0.001);
        assert!((points[0].z - 0.0).abs() < 0.001);
    }
}
