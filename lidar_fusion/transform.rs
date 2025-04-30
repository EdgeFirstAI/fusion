// From https://github.com/smilerobotics/tf_rosrust/blob/main/tf_rosrust/src/transforms.rs
// licensed under MIT
// Modified by Au-Zone Technologies 2025

use edgefirst_schemas::geometry_msgs::{Quaternion, Transform, Vector3};
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

pub fn interpolate(t1: Transform, t2: Transform, weight: f64) -> Transform {
    let r1 = nalgebra::geometry::Quaternion::new(
        t1.rotation.w,
        t1.rotation.x,
        t1.rotation.y,
        t1.rotation.z,
    );
    let r2 = nalgebra::geometry::Quaternion::new(
        t2.rotation.w,
        t2.rotation.x,
        t2.rotation.y,
        t2.rotation.z,
    );
    let r1 = UnitQuaternion::from_quaternion(r1);
    let r2 = UnitQuaternion::from_quaternion(r2);
    let res = r1.try_slerp(&r2, weight, 1e-9);
    match res {
        Some(qt) => Transform {
            translation: Vector3 {
                x: t1.translation.x * weight + t2.translation.x * (1.0 - weight),
                y: t1.translation.y * weight + t2.translation.y * (1.0 - weight),
                z: t1.translation.z * weight + t2.translation.z * (1.0 - weight),
            },
            rotation: Quaternion {
                x: qt.coords[0],
                y: qt.coords[1],
                z: qt.coords[2],
                w: qt.coords[3],
            },
        },
        None => {
            if weight > 0.5 {
                Transform {
                    translation: Vector3 {
                        x: t1.translation.x * weight + t2.translation.x * (1.0 - weight),
                        y: t1.translation.y * weight + t2.translation.y * (1.0 - weight),
                        z: t1.translation.z * weight + t2.translation.z * (1.0 - weight),
                    },
                    rotation: t1.rotation,
                }
            } else {
                Transform {
                    translation: Vector3 {
                        x: t1.translation.x * weight + t2.translation.x * (1.0 - weight),
                        y: t1.translation.y * weight + t2.translation.y * (1.0 - weight),
                        z: t1.translation.z * weight + t2.translation.z * (1.0 - weight),
                    },
                    rotation: t2.rotation,
                }
            }
        }
    }
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
    // reorders the rows and removes the last one
    let xyz = REORDER_ROWS * xyz1;

    // get the normalized camera matrix
    let mut cam_mtx = Matrix3::from_row_slice(cam_mtx);
    cam_mtx[(0, 0)] /= image_dims.0;
    cam_mtx[(0, 2)] /= image_dims.0;
    cam_mtx[(1, 1)] /= image_dims.1;
    cam_mtx[(1, 2)] /= image_dims.1;

    let mut xyz = cam_mtx * xyz;

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
mod test {
    use super::*;

    #[test]
    fn test_basic_interpolation() {
        let tf1 = Transform {
            translation: Vector3 {
                x: 1f64,
                y: 1f64,
                z: 0f64,
            },
            rotation: Quaternion {
                x: 0f64,
                y: 0f64,
                z: 0f64,
                w: 1f64,
            },
        };
        let tf2 = Transform {
            translation: Vector3 {
                x: 2f64,
                y: 2f64,
                z: 0f64,
            },
            rotation: Quaternion {
                x: 0f64,
                y: 0f64,
                z: 0f64,
                w: 1f64,
            },
        };
        let expected = Transform {
            translation: Vector3 {
                x: 1.5f64,
                y: 1.5f64,
                z: 0f64,
            },
            rotation: Quaternion {
                x: 0f64,
                y: 0f64,
                z: 0f64,
                w: 1f64,
            },
        };
        assert_eq!(interpolate(tf1, tf2, 0.5), expected);
    }
}
