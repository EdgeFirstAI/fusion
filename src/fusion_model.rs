use async_std::sync::Mutex;
use log::{error, info, trace};
use ndarray::{s, Array};
use std::{f32::consts::E, sync::Arc, thread::spawn, time::Instant};
use zenoh::Session;

use crate::{
    rtm_model::run_rtm_fusion_model, setup::Args, tflite_model::run_tflite_fusion_model, Grid,
};

pub fn spawn_fusion_model_thread(
    session: Arc<Session>,
    args: Args,
    grid: Arc<Mutex<Option<Grid>>>,
) {
    spawn(move || run_fusion_model(session, args, grid));
}
pub fn run_fusion_model(session: Arc<Session>, args: Args, grid: Arc<Mutex<Option<Grid>>>) {
    let model_name = args.model.as_ref().unwrap().clone();
    match model_name.extension() {
        Some(v) if v.eq_ignore_ascii_case("tflite") => {
            info!("Using TFLite model type for {:?}", model_name);
            run_tflite_fusion_model(session, args, grid)
        }
        Some(v) if v.eq_ignore_ascii_case("rtm") => {
            info!("Using RTM model type for {:?}", model_name);
            run_rtm_fusion_model(session, args, grid);
        }
        Some(_) => {
            error!("Unknown model type extension for {:?}", model_name);
        }
        None => {
            error!("No extension for {:?}", model_name);
        }
    }
}

pub fn preprocess_cube(
    cube: &mut Array<f32, ndarray::Dim<[usize; 5]>>,
    input_size: &[usize],
) -> Array<f32, ndarray::Dim<[usize; 3]>> {
    // need to convert axis (0, 1, 2, 3, 4) into (1, 3, 0, 2, 4)
    let start = Instant::now();
    // (0, 1, 2, 3, 4)
    cube.swap_axes(0, 1);
    // (1, 0, 2, 3, 4)
    cube.swap_axes(1, 3);
    // (1, 3, 2, 0, 4)
    cube.swap_axes(2, 3);
    // (1, 3, 0, 2, 4)

    let mut cube = cube.to_shape([200, 128, 2 * 4 * 2]).unwrap().to_owned();
    trace!("transpose and reshape takes {:?}", start.elapsed());
    let start = Instant::now();
    // this takes about 24ms
    cube.par_mapv_inplace(|v| (v.abs() + 1.0).log(E) * v.signum());

    // this takes about 40 ms
    // let mut cube = (cube.abs() + 1.0).log(E) * cube.signum();
    if input_size.len() != 4 {
        error!("Model input size was: {:?}. Expected 4 dims", input_size);
        return cube;
    }
    if input_size[1] > cube.dim().0 {
        error!(
            "Model second dim was: {:?}. Expected to be <= {}",
            input_size[1],
            cube.dim().0
        );
        return cube;
    }
    if input_size[1] < cube.dim().0 {
        cube = cube.slice_move(s![..input_size[1], .., ..]);
    }
    trace!("the rest takes {:?}", start.elapsed());
    cube
}

#[cfg(test)]
mod swap_axes_test {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

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
    #[test]
    fn test_range_crop() {
        let mut cube = Array::from_shape_vec(
            [2, 200, 4, 256 / 2, 2],
            (0..(2 * 200 * 4 * 256)).map(|v| v as f32).collect(),
        )
        .unwrap();
        let input_size = [1, 128, 128, 16];
        let cube = preprocess_cube(&mut cube, &input_size);
        assert_eq!(
            cube.dim(),
            [128, 128, 16].into(),
            "Dims was not (1, 128, 128, 16)"
        );
        println!("{}", cube.flatten());
        // assert_eq!(
        //     cube.flatten()[1],
        //     0.7615942,
        //     "Second value was not 0.7615942"
        // );
        println!("len={}", cube.flatten().as_slice().unwrap().len())
    }
    #[test]
    fn test_basic() {
        let data = include_str!("testdata/before_radar.txt");

        let flat_cube: Vec<f32> = data.lines().map(|s| s.parse().unwrap()).collect();
        let mut cube = Array::from_shape_vec(
            [2, 200, 4, 256 / 2, 2],
            (0..(2 * 200 * 4 * 256)).map(|v| flat_cube[v]).collect(),
        )
        .unwrap();
        let input_size = [1, 200, 128, 16];
        let start = Instant::now();
        let cube = preprocess_cube(&mut cube, &input_size);
        println!("Preprocessing Cube takes {:?}", start.elapsed());
        assert_eq!(
            cube.dim(),
            [200, 128, 16].into(),
            "Dims was not (200, 128, 16)"
        );
        println!("{}", cube.flatten());
        // assert_eq!(
        //     cube.flatten()[1],
        //     0.7615942,
        //     "Second value was not 0.7615942"
        // );
        println!("len={}", cube.flatten().as_slice().unwrap().len())
    }

    #[test]
    fn test_data() {
        println!(
            "{}",
            env!("CARGO_MANIFEST_DIR").to_owned() + "/src/testdata/before_radar.txt"
        );
        let file =
            File::open(env!("CARGO_MANIFEST_DIR").to_owned() + "/src/testdata/before_radar.txt")
                .unwrap();
        let reader = BufReader::new(file);
        let flat_cube: Vec<f32> = reader
            .lines()
            .map(|s| s.unwrap().parse().unwrap())
            .collect();
        let mut cube = Array::from_shape_vec([2, 200, 4, 256 / 2, 2], flat_cube).unwrap();
        let input_size = [1, 200, 128, 16];
        let cube = preprocess_cube(&mut cube, &input_size);
        assert_eq!(
            cube.dim(),
            [200, 128, 16].into(),
            "Dims was not (200, 128, 16)"
        );

        let file =
            File::open(env!("CARGO_MANIFEST_DIR").to_owned() + "/src/testdata/after_radar.txt")
                .unwrap();
        let reader = BufReader::new(file);
        let flat_output: Vec<f32> = reader
            .lines()
            .map(|s| s.unwrap().parse().unwrap())
            .collect();
        let flat_out: Vec<f32> = flat_output.iter().map(|v| *v).collect();

        let flat = cube.flatten();
        let flat_cube: Vec<f32> = flat.as_slice().unwrap().iter().map(|v| *v).collect();
        assert!(
            vec_compare(&flat_cube, &flat_out),
            "Output not equal:\nRust: {:?}\nNump: {:?}",
            flat_cube,
            flat_out
        );
    }

    fn vec_compare(va: &[f32], vb: &[f32]) -> bool {
        (va.len() == vb.len()) &&  // zip stops at the shortest
     va.iter()
       .zip(vb)
       .all(|(a,b)| *a-*b < 0.0001)
    }
}
