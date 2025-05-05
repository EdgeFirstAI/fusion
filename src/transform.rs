// From https://github.com/smilerobotics/tf_rosrust/blob/main/tf_rosrust/src/transforms.rs
// licensed under MIT
// Modified by Au-Zone Technologies 2025

use edgefirst_schemas::geometry_msgs::Transform;
use nalgebra::{
    self,
    geometry::{Isometry3, Translation3, UnitQuaternion},
    ArrayStorage, Matrix2x3, Matrix3, Matrix3x4, Matrix4xX,
};

#[cfg(feature = "profiling")]
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

const REORDER_ROWS: Matrix3x4<f32> = Matrix3x4::from_array_storage(ArrayStorage([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
]));

// Converts (x, y, 1) column vector from camera coordinate system to the image
// coordinate system.
const CONVERT_COORD: Matrix2x3<f32> =
    Matrix2x3::from_array_storage(ArrayStorage([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]));

// Applies the given transforms onto points. Then projects the points on the
// given camera matrix and returns the projected coordinates
pub(crate) fn transform_and_project_points(
    points: &mut [ParsedPoint],
    transforms: &[Transform],
    cam_mtx: &[f32; 9],
    image_dims: (f32, f32),
) -> Vec<[f32; 2]> {
    let mut final_transform = Isometry3::<f32>::identity();
    for t in transforms {
        let tf = isometry_from_transform(t);
        final_transform *= tf;
    }
    let transform = final_transform.to_matrix();

    let mut xyz1 = Vec::with_capacity(points.len() * 4);
    for p in points.iter() {
        xyz1.push(p.x);
        xyz1.push(p.y);
        xyz1.push(p.z);
        xyz1.push(1.0);
    }

    let xyz1 = Matrix4xX::from_vec(xyz1);
    let xyz1 = transform * xyz1;
    for (col, p) in xyz1.column_iter().zip(points.iter_mut()) {
        p.x = col[0];
        p.y = col[1];
        p.z = col[2];
    }
    // get the normalized camera matrix
    let mut cam_mtx = Matrix3::from_row_slice(cam_mtx);
    cam_mtx[(0, 0)] /= image_dims.0;
    cam_mtx[(0, 2)] /= image_dims.0;
    cam_mtx[(1, 1)] /= image_dims.1;
    cam_mtx[(1, 2)] /= image_dims.1;

    let mut xyz = cam_mtx * REORDER_ROWS * xyz1;

    for mut c in xyz.column_iter_mut() {
        let z = c.z;
        c /= z;
    }

    let xy = CONVERT_COORD * xyz;
    xy.as_slice()
        .chunks_exact(2)
        .map(|x| [x[0], x[1]])
        .collect()
}

#[cfg(test)]
mod projection_test {
    use std::collections::HashMap;

    use edgefirst_schemas::geometry_msgs::{Quaternion, Vector3};

    use super::*;

    #[test]
    fn test_basic() {
        let mut points = Vec::new();
        points.push(ParsedPoint {
            fields: HashMap::new(),
            x: 10.0,
            y: 20.0,
            z: 30.0,
            id: None,
            angle: 0.0,
            range: 1000.0,
        });

        points.push(ParsedPoint {
            fields: HashMap::new(),
            x: 1.0,
            y: 2.0,
            z: 3.0,
            id: None,
            angle: 0.0,
            range: 1000.0,
        });
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
        let transform = Transform {
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
        };
        let proj = transform_and_project_points(&mut points, &[transform], &cam_mtx, (1.0, 1.0));
        println!("Projected values: {:?}", proj);
    }
}
