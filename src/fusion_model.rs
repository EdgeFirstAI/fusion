use async_pidfd::PidFd;
use async_std::{sync::Mutex, task::block_on};
use deepviewrt::{
    context::Context,
    engine::Engine,
    model,
    tensor::{Tensor, TensorType},
};
use edgefirst_schemas::edgefirst_msgs::{DmaBuf, RadarCube};
use log::{debug, error, info, trace};
use ndarray::{s, Array};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    f32::consts::E,
    fs::read,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::io::OwnedFd,
    },
    sync::Arc,
    thread::spawn,
    time::{Duration, Instant},
};
use zenoh::{
    prelude::{r#async::*, sync::*},
    Session,
};

#[cfg(feature = "model_output")]
use cdr::{CdrLe, Infinite};
#[cfg(feature = "model_output")]
use edgefirst_schemas::edgefirst_msgs::Mask;

use crate::{
    image::{Image, ImageManager, Rotation, RGB3},
    setup::Args,
    Grid,
};

static RGB_MEANS_IMAGENET: [f32; 3] = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0];

static RGB_STDS_IMAGENET: [f32; 3] = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0];

pub fn spawn_fusion_model_thread(
    session: Arc<Session>,
    args: Args,
    grid: Arc<Mutex<Option<Grid>>>,
) {
    spawn(move || run_fusion_model(session, args, grid));
}
pub fn run_fusion_model(session: Arc<Session>, args: Args, grid: Arc<Mutex<Option<Grid>>>) {
    if args.model.is_none() {
        info!("No radar model was given");
        return;
    }
    let model_name = args.model.as_ref().unwrap().clone();
    let model_data = match read(&model_name) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open `{:?}` file: {:?}", model_name, e);
            return;
        }
    };

    info!("Model read from file");

    let engine = if args.engine.to_lowercase() == "npu" {
        Engine::new("deepview-rt-openvx.so")
            .expect("Initializing deepview-rt-openvx.so engine failed")
    } else {
        Engine::new("").expect("Initializing cpu engine failed")
    };
    let mut nn_context = Context::new(Some(engine), model::memory_size(&model_data), 4096 * 1024)
        .expect("NNContext init failed");
    info!("NNContext initialized");

    match nn_context.load_model(model_data) {
        Ok(_) => info!("Loaded backbone model {:?}", model_name),
        Err(e) => {
            error!("Could not load model file {:?}: {:?}", model_name, e);
            return;
        }
    }

    let sub_radarcube = session
        .declare_subscriber(&args.radarcube_topic)
        .res_sync()
        .unwrap();
    info!("Declared subscriber on {:?}", &args.radarcube_topic);

    #[cfg(feature = "model_output")]
    let publ_mask = match session
        .declare_publisher(args.model_output_topic.clone())
        .res_sync()
    {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error while declaring detection publisher {}: {:?}",
                &args.model_output_topic, e
            );
            return;
        }
    };

    let input_tensor_index = model::inputs(nn_context.model()).unwrap();

    let input_names: Vec<_> = input_tensor_index
        .iter()
        .map(|v| model::layer_name(nn_context.model(), *v as usize).unwrap_or("NO_NAME"))
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

    let radar_input_shape: Vec<_> =
        match nn_context.tensor_index(input_tensor_index[radar_input_index] as usize) {
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

    let mut camera_input_tensor = if let Some(camera_input_index) = camera_input_index {
        match nn_context.tensor_index(input_tensor_index[camera_input_index] as usize) {
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
    let camera_input_shape = if let Some(ref camera_input_tensor) = camera_input_tensor {
        camera_input_tensor
            .shape()
            .iter()
            .map(|v| *v as usize)
            .collect()
    } else {
        vec![1, 1, 1, 1]
    };

    let img_mgr = match ImageManager::new() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open G2D: {:?}", e);
            return;
        }
    };

    let g2dbuf = match img_mgr.alloc(camera_input_shape[2], camera_input_shape[1], 4) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not alloc G2D buf: {:?}", e);
            return;
        }
    };
    let dest = Image::new_preallocated(
        img_mgr.g2d_buf_fd(&g2dbuf),
        camera_input_shape[2] as u32,
        camera_input_shape[1] as u32,
        RGB3,
    );

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

        let radar_input_tensor =
            match nn_context.tensor_index_mut(input_tensor_index[radar_input_index] as usize) {
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
            let camera_input_tensor = camera_input_tensor.as_mut().unwrap();
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

            debug!("Start load_frame_dmabuf");

            match load_frame_dmabuf(
                camera_input_tensor,
                &img_mgr,
                &dest,
                &cam_buffer,
                Preprocessing::UnsignedNorm,
            ) {
                Ok(_) => {}
                Err(e) => {
                    error!("Error loading camera frame into input: {:?}", e);
                    continue;
                }
            }
            debug!("finished load_frame_dmabuf");
        }

        if let Err(e) = nn_context.run() {
            error!("Failed to run model: {}", e);
            return;
        }
        debug!("finished run model");
        let mut output_shape: Vec<u32> = vec![0, 0, 0, 0];
        let mask = if let Ok(tensor) = nn_context.output(0) {
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

#[allow(dead_code)]
pub enum Preprocessing {
    Raw = 0x0,
    UnsignedNorm = 0x1,
    SignedNorm = 0x2,
    ImageNet = 0x8,
}

fn load_frame_dmabuf(
    tensor: &mut Tensor,
    img_mgr: &ImageManager,
    dest: &Image,
    dma_buf: &DmaBuf,
    preprocess: Preprocessing,
) -> Result<(), String> {
    if dest.height() as i32 != tensor.shape()[1] {
        return Err(
            "The height of the destination buffer is not equal to the height of the tensor"
                .to_owned(),
        );
    }
    if dest.width() as i32 != tensor.shape()[2] {
        return Err(
            "The width of the destination buffer is not equal to the width of the tensor"
                .to_owned(),
        );
    }
    if dest.format() != RGB3 {
        return Err("The format of destination buffer is not RGB3".to_owned());
    }
    let input = Image::new_preallocated(
        unsafe { OwnedFd::from_raw_fd(dma_buf.fd) },
        dma_buf.width,
        dma_buf.height,
        dma_buf.fourcc.into(),
    );
    match img_mgr.convert(&input, dest, None, Rotation::Rotation0) {
        Ok(_) => {}
        Err(e) => {
            error!(
                "Could not g2d convert from {:?} to {:?}: {:?}",
                input, dest, e
            )
        }
    }
    debug!("Dest size: {}", dest.size());
    debug!("Tensor size: {}", tensor.volume());
    match tensor.tensor_type() {
        TensorType::U8 => {
            let mut tensor_mapped = match tensor.maprw() {
                Ok(v) => v,
                Err(e) => return Err(e.to_string()),
            };
            let dest_mapped = dest.dmabuf().memory_map().unwrap();

            match dest_mapped.read(
                |data, tensor_mapped| {
                    let tensor_mapped = tensor_mapped.unwrap();
                    tensor_mapped.copy_from_slice(data);
                    Ok(())
                },
                Some(&mut tensor_mapped),
            ) {
                Ok(_) => {}
                Err(e) => {
                    error!("Could not copy from g2d dest buffer to tensor: {:?}", e)
                }
            }
        }
        TensorType::I16 => todo!(),
        TensorType::U16 => todo!(),
        TensorType::I32 => todo!(),
        TensorType::RAW => todo!(),
        TensorType::STR => todo!(),
        TensorType::I8 => {
            let mut tensor_mapped = match tensor.maprw() {
                Ok(v) => v,
                Err(e) => return Err(e.to_string()),
            };
            let dest_mapped = dest.dmabuf().memory_map().unwrap();

            match dest_mapped.read(
                |data, tensor_mapped| {
                    let tensor_mapped = tensor_mapped.unwrap();
                    for i in 0..tensor_mapped.len() {
                        tensor_mapped[i] = (data[i] as i16 - 128) as i8;
                    }

                    Ok(())
                },
                Some(&mut tensor_mapped),
            ) {
                Ok(_) => {}
                Err(e) => {
                    error!("Could not copy from g2d dest buffer to tensor: {:?}", e)
                }
            }
        }
        TensorType::U32 => todo!(),
        TensorType::I64 => todo!(),
        TensorType::U64 => todo!(),
        TensorType::F16 => todo!(),
        TensorType::F32 => {
            let mut tensor_mapped = match tensor.maprw() {
                Ok(v) => v,
                Err(e) => return Err(e.to_string()),
            };
            let dest_mapped = dest.dmabuf().memory_map().unwrap();
            match dest_mapped.read(
                |data, tensor_mapped| {
                    let tensor_mapped = tensor_mapped.unwrap();
                    match preprocess {
                        Preprocessing::Raw => {
                            for i in 0..tensor_mapped.len() {
                                tensor_mapped[i] = data[i] as f32;
                            }
                        }
                        Preprocessing::UnsignedNorm => {
                            for i in 0..tensor_mapped.len() {
                                tensor_mapped[i] = data[i] as f32 / 255.0;
                            }
                        }
                        Preprocessing::SignedNorm => {
                            for i in 0..tensor_mapped.len() {
                                tensor_mapped[i] = data[i] as f32 / 127.5 - 1.0;
                            }
                        }
                        Preprocessing::ImageNet => {
                            for i in (0..tensor_mapped.len()).step_by(3) {
                                tensor_mapped[i] =
                                    (data[i] as f32 - RGB_MEANS_IMAGENET[0]) / RGB_STDS_IMAGENET[0];
                                tensor_mapped[i + 1] =
                                    (data[i] as f32 - RGB_MEANS_IMAGENET[1]) / RGB_STDS_IMAGENET[1];
                                tensor_mapped[i + 2] =
                                    (data[i] as f32 - RGB_MEANS_IMAGENET[2]) / RGB_STDS_IMAGENET[2];
                            }
                        }
                    }
                    Ok(())
                },
                Some(&mut tensor_mapped),
            ) {
                Ok(_) => {}
                Err(e) => {
                    error!("Could not copy from g2d dest buffer to tensor: {:?}", e)
                }
            }
        }
        TensorType::F64 => todo!(),
    };

    Ok(())
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
