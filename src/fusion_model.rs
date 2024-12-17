use async_pidfd::PidFd;
use async_std::{sync::Mutex, task::block_on};
use deepviewrt::{model, tensor::Tensor};
use edgefirst_schemas::edgefirst_msgs::{DmaBuf, RadarCube};
use log::{debug, error, info, trace};
use ndarray::{s, Array};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    f32::consts::E,
    os::fd::AsRawFd,
    sync::Arc,
    thread::spawn,
    time::{Duration, Instant},
};
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

const VAAL_IMAGE_PROC_UNSIGNED_NORM: u32 = 0x0001;

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

    let input_names: Vec<_> = input_tensor_index
        .iter()
        .map(|v| model::layer_name(backbone.model().unwrap(), *v as usize).unwrap_or("NO_NAME"))
        .collect();

    let mut radar_input_index = 0;
    let mut camera_input_index = None;

    for (i, name) in input_names.iter().enumerate() {
        debug!("Input #{} has name: {}", i, name);
        if name.contains("radar") {
            radar_input_index = i;
            debug!("setting radar input index to {}", i);
        } else if name.contains("camera") {
            let _ = camera_input_index.insert(i);
            debug!("setting camera input index to {}", i);
        }
    }

    let radar_input_shape: Vec<_> = match backbone
        .dvrt_context()
        .unwrap()
        .tensor_index(input_tensor_index[radar_input_index] as usize)
    {
        Ok(v) => v.shape().iter().map(|v| *v as usize).collect(),
        Err(e) => {
            error!("Could not get input 0 from model: {:?}", e);
            return;
        }
    };
    debug!("got input tensor shape");

    let sub_camera = if camera_input_index.is_some() {
        let s = session
            .declare_subscriber(&args.camera_topic)
            .res_sync()
            .unwrap();
        info!("Declared subscriber on {:?}", &args.camera_topic);
        Some(s)
    } else {
        None
    };

    let camera_input_tensor = if let Some(camera_input_index) = camera_input_index {
        match backbone
            .dvrt_context()
            .unwrap()
            .tensor_index(input_tensor_index[camera_input_index] as usize)
        {
            Ok(v) => {
                // needed because the dvrt borrow is still mutable even though the Tensor
                // pointer itself isn't mutable
                let tensor = unsafe { Tensor::from_ptr(v.to_mut_ptr(), false).unwrap() };
                Some(tensor)
            }
            Err(e) => {
                error!(
                    "Could not get input {} from model: {:?}",
                    camera_input_index, e
                );
                return;
            }
        }
    } else {
        None
    };
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
        let start = Instant::now();
        let radarcube: RadarCube = match cdr::deserialize(&sample.payload.contiguous()) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize message: {:?}", e);
                continue;
            }
        };

        debug!("deserialized radarcube, took {:?}", start.elapsed()); // takes about 4-5ms
        let start = Instant::now();
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

        let cube = preprocess_cube(&mut cube, &radar_input_shape);
        let cube = cube.into_flat();
        let cube = cube.as_slice().unwrap();
        debug!("preprocessed radarcube: took {:?}", start.elapsed()); // takes about 28-30ms

        let radar_input_tensor = match backbone
            .dvrt_context()
            .unwrap()
            .tensor_index_mut(input_tensor_index[radar_input_index] as usize)
        {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get input 0 from model: {:?}", e);
                return;
            }
        };
        let mut input_tensor_map = radar_input_tensor.maprw_f32().unwrap();
        debug!("mapped input tensor: len={:?}", input_tensor_map.len());
        input_tensor_map.copy_from_slice(cube);
        drop(input_tensor_map);

        if camera_input_index.is_some() {
            let camera_input_tensor = camera_input_tensor.as_ref().unwrap();
            let sub_camera = sub_camera.as_ref().unwrap();
            let sample = if let Some(v) = sub_camera.drain().last() {
                v
            } else {
                match sub_camera.recv_timeout(timeout) {
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
            let mut cam_buffer: DmaBuf = match cdr::deserialize(&sample.payload.contiguous()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    continue;
                }
            };
            debug!("Got DMA Buffer: {:?}", cam_buffer);

            let pidfd: PidFd = match PidFd::from_pid(cam_buffer.pid as i32) {
                Ok(v) => v,
                Err(e) => {
                    error!(
                    "Error getting PID {:?}, please check if the camera process is running: {:?}",
                    cam_buffer.pid, e
                );
                    continue;
                }
            };
            let fd = match get_file_from_pidfd(
                pidfd.as_raw_fd(),
                cam_buffer.fd,
                GetFdFlags::empty(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    error!(
                    "Error getting Camera DMA file descriptor, please check if current process is running with same permissions as camera: {:?}",
                    e
                );
                    continue;
                }
            };

            cam_buffer.fd = fd.as_raw_fd();
            debug!("Updated dma fd to {}", cam_buffer.fd);

            info!("Start load_frame_dmabuf");
            match backbone.load_frame_dmabuf(
                Some(camera_input_tensor),
                cam_buffer.fd,
                cam_buffer.fourcc,
                cam_buffer.width as i32,
                cam_buffer.height as i32,
                None,
                VAAL_IMAGE_PROC_UNSIGNED_NORM,
            ) {
                Ok(_) => {}
                Err(e) => error!("Could not load camera frame: {:?}", e),
            }
        }
        info!("finished load_frame_dmabuf");
        if let Err(e) = backbone.run_model() {
            error!("Failed to run model: {}", e);
            return;
        }
        info!("finished run model");
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
                .flat_map(|v| {
                    [
                        (255.0 * args.model_threshold) as u8,
                        (255.0 * v).min(255.0) as u8,
                    ]
                })
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

        let mut occupied_ = mask.into_iter();
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

fn preprocess_cube(
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
