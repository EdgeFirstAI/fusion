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

#[cfg(feature = "deepviewrt")]
use crate::rtm_model::run_rtm_fusion_model;
use crate::{args::Args, tflite_model::run_tflite_fusion_model, Grid};

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
        #[cfg(feature = "deepviewrt")]
        Some(v) if v.eq_ignore_ascii_case("rtm") => {
            info!("Using RTM model type for {model_name:?}");
            let _ = run_rtm_fusion_model(session, args, grid).await;
        }
        #[cfg(not(feature = "deepviewrt"))]
        Some(v) if v.eq_ignore_ascii_case("rtm") => {
            error!("Model {model_name:?} requires the `deepviewrt` feature. Rebuild with `--features deepviewrt`.");
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
        *v = if *v >= 0.0 {
            1.0 / (1.0 + (-*v).exp())
        } else {
            let ev = v.exp();
            ev / (1.0 + ev)
        };
    }
}

#[derive(Error, Debug)]
pub enum FusionError {
    #[error("{0}")]
    String(String),
    #[error("TfLite Error: {0:?}")]
    TfLite(#[from] TfLiteError),
    #[cfg(feature = "deepviewrt")]
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
}
