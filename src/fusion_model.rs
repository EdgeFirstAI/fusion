use async_std::{sync::Mutex, task::block_on};
use deepviewrt::model;
use edgefirst_schemas::edgefirst_msgs::RadarCube;
use log::{debug, error, info};
use ndarray::{s, Array, ArrayBase};
use std::{sync::Arc, thread::spawn, time::Duration};
use vaal::Context;
use zenoh::{
    prelude::{r#async::*, sync::*},
    Session,
};

#[cfg(feature = "model_output")]
use cdr::{CdrLe, Infinite};
#[cfg(feature = "model_output")]
use edgefirst_schemas::edgefirst_msgs::Mask;

use crate::{setup::Args, Grid};

pub fn spawn_fusion_model_thread(
    session: Arc<Session>,
    args: Args,
    grid: Arc<Mutex<Option<Grid>>>,
) {
    spawn(move || run_fusion_model(session, args, grid));
}
pub fn run_fusion_model(session: Arc<Session>, args: Args, grid: Arc<Mutex<Option<Grid>>>) {
    let mut backbone = match Context::new(&args.engine) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open VAAL Context on {}, {:?}", args.engine, e);
            return;
        }
    };
    info!("Opened DeepViewRT Context on {}", args.engine);

    if args.model.is_none() {
        info!("No radar model was given");
        return;
    }
    let filepath = args.model.unwrap().clone();
    let filename = match filepath.to_str() {
        Some(v) => v,
        None => {
            error!(
                "Cannot use file {:?}, please use only utf8 characters in file path",
                filepath
            );
            return;
        }
    };

    match backbone.load_model_file(filename) {
        Ok(_) => info!("Loaded backbone model {:?}", filename),
        Err(e) => {
            error!("Could not load model file {}: {:?}", filename, e);
            return;
        }
    }

    let sub_radarcube = session
        .declare_subscriber(&args.radarcube_topic)
        .res_sync()
        .unwrap();
    info!("Declared subscriber on {:?}", &args.radarcube_topic);

    #[cfg(feature = "model_output")]
    let publ_mask = match session.declare_publisher("rt/fusion/mask_test").res_sync() {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error while declaring detection publisher rt/fusion/mask_test: {:?}",
                e
            );
            return;
        }
    };

    let input_tensor_index = model::inputs(backbone.model().unwrap()).unwrap();
    let input_shape: Vec<_> = match backbone
        .dvrt_context()
        .unwrap()
        .tensor_index_mut(input_tensor_index[0] as usize)
    {
        Ok(v) => v.shape().iter().map(|v| *v as usize).collect(),
        Err(e) => {
            error!("Could not get input 0 from model: {:?}", e);
            return;
        }
    };
    debug!("got input tensor shape");

    let timeout = Duration::from_millis(2000);
    loop {
        let sample = if let Some(v) = sub_radarcube.drain().last() {
            v
        } else {
            match sub_radarcube.recv_timeout(timeout) {
                Ok(v) => v,
                Err(e) => {
                    error!(
                        "error receiving radar cube on {}: {:?}",
                        sub_radarcube.key_expr(),
                        e
                    );
                    continue;
                }
            }
        };

        let radarcube: RadarCube = match cdr::deserialize(&sample.payload.contiguous()) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize message: {:?}", e);
                continue;
            }
        };

        debug!("deserialized radarcube");
        let mut cube = Array::from_shape_vec(
            [
                radarcube.shape[0] as usize,
                radarcube.shape[1] as usize,
                radarcube.shape[2] as usize,
                radarcube.shape[3] as usize / 2,
                2,
            ],
            radarcube.cube.iter().map(|v| *v as f32).collect(),
        )
        .unwrap();

        let cube = preprocess_cube(&mut cube, &input_shape);
        let cube = cube.to_owned().into_flat();
        let cube = cube.as_slice().unwrap();
        debug!("preprocessed radarcube: len={:?}", cube.len());

        let input_tensor = match backbone
            .dvrt_context()
            .unwrap()
            .tensor_index_mut(input_tensor_index[0] as usize)
        {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get input 0 from model: {:?}", e);
                return;
            }
        };
        let mut input_tensor_map = input_tensor.maprw_f32().unwrap();
        debug!("mapped input tensor: len={:?}", input_tensor_map.len());
        input_tensor_map.copy_from_slice(cube);
        drop(input_tensor_map);

        if let Err(e) = backbone.run_model() {
            error!("Failed to run model: {}", e);
            return;
        }

        let mut output_shape: Vec<u32> = vec![0, 0, 0, 0];
        let mask = if let Some(tensor) = backbone.output_tensor(0) {
            output_shape = tensor.shape().iter().map(|x| *x as u32).collect();
            let data = tensor.mapro_f32().unwrap();
            let len = data.len();
            let mut buffer = vec![0.0; len];
            buffer.copy_from_slice(&data);
            buffer
        } else {
            error!("Did not find model output");
            Vec::new()
        };

        #[cfg(feature = "model_output")]
        {
            let mask = mask
                .iter()
                .flat_map(|v| [128, (255.0 * v).min(255.0) as u8])
                .collect();
            let msg = Mask {
                height: output_shape[1],
                width: output_shape[2],
                length: 1,
                encoding: "".to_string(),
                mask,
            };
            let val = Value::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap()).encoding(
                Encoding::WithSuffix(
                    KnownEncoding::AppOctetStream,
                    "edgefirst_msgs/msg/Mask".into(),
                ),
            );
            let _ = publ_mask.put(val).res_sync();
            debug!("sent model output on {}", publ_mask.key_expr());
        }

        let mut occupied_ = mask.into_iter().map(|v| v > 0.5);
        let mut occupied = Vec::new();
        for i in 0..output_shape[2] as usize {
            occupied.push(Vec::new());
            for _ in 0..output_shape[1] {
                let item = occupied_.next().unwrap();
                occupied[i].push(item)
            }
        }
        let timestamp = radarcube.header.stamp.nanosec as u64
            + radarcube.header.stamp.sec as u64 * 1_000_000_000;
        let mut guard = block_on(grid.lock());
        *guard = Some((occupied, timestamp));
    }
}

fn preprocess_cube<'a>(
    cube: &'a mut ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 5]>>,
    input_size: &[usize],
) -> ArrayBase<ndarray::CowRepr<'a, f32>, ndarray::Dim<[usize; 4]>> {
    // need to convert axis (0, 1, 2, 3, 4) into (0, 2, 4, 1, 3)

    // (0, 1, 2, 3, 4)
    cube.swap_axes(2, 1);
    // (0, 2, 1, 3, 4)
    cube.swap_axes(4, 2);
    // (0, 2, 4, 3, 1)
    cube.swap_axes(4, 3);
    // (0, 2, 4, 1, 3)

    let mut cube = cube.to_shape([1, 2 * 4 * 2, 200, 128]).unwrap();
    cube.par_mapv_inplace(|v| v.tanh());

    // need to convert axis (0, 1, 2, 3) into (0, 2, 3, 1)

    // (0, 1, 2, 3)
    cube.swap_axes(2, 1);
    // (0, 2, 1, 3)
    cube.swap_axes(3, 2);
    // (0, 2, 3, 1)

    if input_size.len() != 4 {
        error!("Model input size was: {:?}. Expected 4 dims", input_size);
        return cube;
    }
    if input_size[1] > cube.dim().1 {
        error!(
            "Model second dim was: {:?}. Expected to be <= {}",
            input_size[1],
            cube.dim().1
        );
        return cube;
    }
    cube = cube.slice_move(s![.., ..input_size[1], .., ..]);
    cube
}

#[cfg(test)]
mod swap_axes_test {
    use super::*;

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
            [1, 128, 128, 16].into(),
            "Dims was not (1, 128, 128, 16)"
        );
        println!("{}", cube.flatten());
        assert_eq!(
            cube.flatten()[1],
            0.7615942,
            "Second value was not 0.7615942"
        );
        println!("len={}", cube.flatten().as_slice().unwrap().len())
    }
    #[test]
    fn test_basic() {
        let mut cube = Array::from_shape_vec(
            [2, 200, 4, 256 / 2, 2],
            (0..(2 * 200 * 4 * 256)).map(|v| v as f32).collect(),
        )
        .unwrap();
        let input_size = [1, 200, 128, 16];
        let cube = preprocess_cube(&mut cube, &input_size);
        assert_eq!(
            cube.dim(),
            [1, 200, 128, 16].into(),
            "Dims was not (1, 200, 128, 16)"
        );
        println!("{}", cube.flatten());
        assert_eq!(
            cube.flatten()[1],
            0.7615942,
            "Second value was not 0.7615942"
        );
        println!("len={}", cube.flatten().as_slice().unwrap().len())
    }
}
