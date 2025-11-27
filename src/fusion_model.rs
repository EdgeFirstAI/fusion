// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use log::{error, info};
use ndarray::{
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
    s, Array,
};
use std::{
    f32::consts::E,
    io,
    sync::Arc,
    thread::{self, JoinHandle},
};
use tflitec_sys::{LibloadingError, TfLiteError};
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{info_span, instrument};
use zenoh::Session;

use crate::{
    args::Args, rtm_model::run_rtm_fusion_model, tflite_model::run_tflite_fusion_model, Grid,
};

pub fn spawn_fusion_model_thread(
    session: Session,
    args: Args,
    grid: Arc<Mutex<Option<Grid>>>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("model".to_string())
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(run_fusion_model(session, args, grid));
        })
        .unwrap()
}

pub async fn run_fusion_model(session: Session, args: Args, grid: Arc<Mutex<Option<Grid>>>) {
    if args.model.is_none() {
        return;
    }

    let model_name = args.model.as_ref().unwrap().clone();
    match model_name.extension() {
        Some(v) if v.eq_ignore_ascii_case("tflite") => {
            info!("Using TFLite model type for {model_name:?}");
            let _ = run_tflite_fusion_model(session, args, grid).await;
        }
        Some(v) if v.eq_ignore_ascii_case("rtm") => {
            info!("Using RTM model type for {model_name:?}");
            let _ = run_rtm_fusion_model(session, args, grid).await;
        }
        Some(_) => {
            error!("Unknown model type extension for {model_name:?}");
        }
        None => {
            error!("No extension for {model_name:?}");
        }
    }
}

#[instrument(skip_all)]
pub fn preprocess_cube(cube: &[i16], shape: &[usize], input_size: &[usize]) -> Vec<f32> {
    let cube = normalize_cube(cube);
    let mut cube = Array::from_shape_vec(
        [shape[0], shape[1], shape[2], shape[3] / 2, 2],
        cube.to_owned(),
    )
    .unwrap();

    let mut cube = info_span!("swap_axes").in_scope(|| {
        // need to convert axis (0, 1, 2, 3, 4) into (1, 3, 0, 2, 4)
        // (0, 1, 2, 3, 4)
        cube.swap_axes(0, 1);
        // (1, 0, 2, 3, 4)
        cube.swap_axes(1, 3);
        // (1, 3, 2, 0, 4)
        cube.swap_axes(2, 3);
        // (1, 3, 0, 2, 4)

        cube.as_standard_layout()
    });

    assert_eq!(
        input_size.len(),
        4,
        "Model input size was: {:?}. Expected 4 dims",
        input_size
    );

    assert!(
        input_size[1] <= cube.dim().0,
        "Model second dim was: {:?}. Expected to be <= {}",
        input_size[1],
        cube.dim().0
    );

    if input_size[1] < cube.dim().0 {
        cube = cube.slice_move(s![..input_size[1], .., .., .., ..]);
    }

    info_span!("cube_flatten").in_scope(|| cube.flatten().to_vec())
}

#[instrument(skip_all)]
fn normalize_cube(cube: &[i16]) -> Vec<f32> {
    cube.par_iter()
        .map(|v| {
            let v = *v as f32;
            (v.abs() + 1.0).log(E) * v.signum()
        })
        .collect::<Vec<_>>()
}

pub(crate) fn apply_sigmoid(mask: &mut [f32]) {
    for v in mask.iter_mut() {
        *v = v.exp() / (1.0 + v.exp())
    }
}

#[derive(Error, Debug)]
pub enum FusionError {
    #[error("{0}")]
    String(String),
    #[error("TfLite Error: {0:?}")]
    TfLite(#[from] TfLiteError),
    #[error("Rtm Error: {0:?}")]
    Rtm(#[from] deepviewrt::error::Error),
    #[error("LibLoading Error: {0:?}")]
    LibLoading(#[from] LibloadingError),
    #[error("IO Error: {0:?}")]
    Io(#[from] io::Error),
}

impl From<String> for FusionError {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for FusionError {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

#[cfg(test)]
mod swap_axes_test {

    use super::*;

    #[test]
    fn test_log1p() {
        let mut cube = Array::from_shape_vec(
            [2, 200, 4, 256 / 2, 2],
            (0..(2 * 200 * 4 * 256)).map(|v| v as f32 / 256.0).collect(),
        )
        .unwrap();
        let cube1 = (cube.abs() + 1.0).log(E) * cube.signum();
        cube.par_mapv_inplace(|v| (v.abs() + 1.0).log(E) * v.signum());
        println!("{}", cube1 == cube);
    }
    // #[test]
    // fn test_range_crop() {
    //     let mut cube = Array::from_shape_vec(
    //         [2, 200, 4, 256 / 2, 2],
    //         (0..(2 * 200 * 4 * 256)).map(|v| v as f32).collect(),
    //     )
    //     .unwrap();
    //     let input_size = [1, 128, 128, 16];
    //     let cube = preprocess_cube(&mut cube, &input_size);
    //     assert_eq!(
    //         cube.dim(),
    //         [128, 128, 16].into(),
    //         "Dims was not (1, 128, 128, 16)"
    //     );
    //     println!("{}", cube.flatten());
    //     // assert_eq!(
    //     //     cube.flatten()[1],
    //     //     0.7615942,
    //     //     "Second value was not 0.7615942"
    //     // );
    //     println!("len={}", cube.flatten().as_slice().unwrap().len())
    // }
    // #[test]
    // fn test_basic() {
    //     let data = include_str!("testdata/before_radar.txt");

    //     let flat_cube: Vec<f32> = data.lines().map(|s|
    // s.parse().unwrap()).collect();     let mut cube =
    // Array::from_shape_vec(         [2, 200, 4, 256 / 2, 2],
    //         (0..(2 * 200 * 4 * 256)).map(|v| flat_cube[v]).collect(),
    //     )
    //     .unwrap();
    //     let input_size = [1, 200, 128, 16];
    //     let start = Instant::now();
    //     let cube = preprocess_cube(&mut cube, &input_size);
    //     println!("Preprocessing Cube takes {:?}", start.elapsed());
    //     assert_eq!(
    //         cube.dim(),
    //         [200, 128, 16].into(),
    //         "Dims was not (200, 128, 16)"
    //     );
    //     println!("{}", cube.flatten());
    //     // assert_eq!(
    //     //     cube.flatten()[1],
    //     //     0.7615942,
    //     //     "Second value was not 0.7615942"
    //     // );
    //     println!("len={}", cube.flatten().as_slice().unwrap().len())
    // }

    // #[test]
    // fn test_data() {
    //     println!(
    //         "{}",
    //         env!("CARGO_MANIFEST_DIR").to_owned() +
    // "/src/testdata/before_radar.txt"     );
    //     let file =
    //         File::open(env!("CARGO_MANIFEST_DIR").to_owned() +
    // "/src/testdata/before_radar.txt")             .unwrap();
    //     let reader = BufReader::new(file);
    //     let flat_cube: Vec<f32> = reader
    //         .lines()
    //         .map(|s| s.unwrap().parse().unwrap())
    //         .collect();
    //     let mut cube = Array::from_shape_vec([2, 200, 4, 256 / 2, 2],
    // flat_cube).unwrap();     let input_size = [1, 200, 128, 16];
    //     let cube = preprocess_cube(&mut cube, &input_size);
    //     assert_eq!(
    //         cube.dim(),
    //         [200, 128, 16].into(),
    //         "Dims was not (200, 128, 16)"
    //     );

    //     let file =
    //         File::open(env!("CARGO_MANIFEST_DIR").to_owned() +
    // "/src/testdata/after_radar.txt")             .unwrap();
    //     let reader = BufReader::new(file);
    //     let flat_output: Vec<f32> = reader
    //         .lines()
    //         .map(|s| s.unwrap().parse().unwrap())
    //         .collect();
    //     let flat_out: Vec<f32> = flat_output.to_vec();

    //     let flat = cube.flatten();
    //     let flat_cube: Vec<f32> = flat.as_slice().unwrap().to_vec();
    //     assert!(
    //         vec_compare(&flat_cube, &flat_out),
    //         "Output not equal:\nRust: {:?}\nNump: {:?}",
    //         flat_cube,
    //         flat_out
    //     );
    // }

    // fn vec_compare(va: &[f32], vb: &[f32]) -> bool {
    //     (va.len() == vb.len()) &&  // zip stops at the shortest
    //  va.iter()
    //    .zip(vb)
    //    .all(|(a,b)| *a-*b < 0.0001)
    // }
}
