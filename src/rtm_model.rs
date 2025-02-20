use async_pidfd::PidFd;
use deepviewrt::{
    context::Context,
    engine::Engine,
    model,
    tensor::{Tensor, TensorType},
};
use edgefirst_schemas::edgefirst_msgs::{DmaBuf, RadarCube};
use libc::memcpy;
use log::{debug, error, info, trace, warn};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    ffi::c_void,
    fs::read,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::io::OwnedFd,
    },
    sync::Arc,
    time::Duration,
};
use tokio::sync::Mutex;
use tracing::{info_span, instrument};
use zenoh::{
    bytes::{Encoding, ZBytes},
    Session,
};

use cdr::{CdrLe, Infinite};

use edgefirst_schemas::edgefirst_msgs::Mask;

use crate::{
    args::Args,
    fusion_model::{apply_sigmoid, preprocess_cube},
    image::{Image, ImageManager, Rotation, RGBA},
    Grid,
};

pub async fn run_rtm_fusion_model(session: Session, args: Args, grid: Arc<Mutex<Option<Grid>>>) {
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

    let sub_radarcube = session
        .declare_subscriber(&args.radarcube_topic)
        .await
        .unwrap();
    info!("Declared subscriber on {:?}", &args.radarcube_topic);

    let publ_mask = match session
        .declare_publisher(args.model_output_topic.clone())
        .await
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

    let engine = if args.engine.to_lowercase() == "npu" {
        Some(
            Engine::new("deepview-rt-openvx.so")
                .expect("Initializing deepview-rt-openvx.so engine failed"),
        )
    } else {
        None
    };
    let mut backbone = Context::new(engine, model::memory_size(&model_data), 4096 * 1024)
        .expect("NNContext init failed");
    info!("NNContext for backbone initialized");

    match backbone.load_model(model_data) {
        Ok(_) => info!("Loaded backbone model {:?}", model_name),
        Err(e) => {
            error!("Could not load model file {:?}: {:?}", model_name, e);
            return;
        }
    }

    let mut decoder = None;
    if let Some(ref path) = args.model_decoder {
        let model_name = path.clone();
        let model_data = match read(&model_name) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not open `{:?}` file: {:?}", model_name, e);
                return;
            }
        };
        let mut decoder_ctx = Context::new(None, model::memory_size(&model_data), 4096 * 1024)
            .expect("NNContext init failed");
        info!("NNContext for decoder initialized");
        match decoder_ctx.load_model(model_data) {
            Ok(_) => info!("Loaded decoder model {:?}", model_name),
            Err(e) => {
                error!(
                    "Could not load decoder model file {:?}: {:?}",
                    model_name, e
                );
                return;
            }
        }
        let _ = decoder.insert(decoder_ctx);
    }

    let input_match = match get_input_match(&backbone, &decoder) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not match backbone outputs to decoder inputs: {}", e);
            return;
        }
    };

    let input_tensor_index = model::inputs(backbone.model()).unwrap();

    let input_names: Vec<_> = input_tensor_index
        .iter()
        .map(|v| model::layer_name(backbone.model(), *v as usize).unwrap_or("NO_NAME"))
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
        match backbone.tensor_index(input_tensor_index[radar_input_index] as usize) {
            Ok(v) => v.shape().iter().map(|v| *v as usize).collect(),
            Err(e) => {
                error!("Could not get input 0 from model: {:?}", e);
                return;
            }
        };
    debug!("got input tensor shape: {:?}", radar_input_shape);
    let sub_camera = if camera_input_index.is_some() {
        let s = session
            .declare_subscriber(&args.camera_topic)
            .await
            .unwrap();
        info!("Declared subscriber on {:?}", &args.camera_topic);
        Some(s)
    } else {
        None
    };
    let mut camera_input_tensor = if let Some(camera_input_index) = camera_input_index {
        match backbone.tensor_index(input_tensor_index[camera_input_index] as usize) {
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

    let mut dest = match Image::new(
        camera_input_shape[2] as u32,
        camera_input_shape[1] as u32,
        RGBA,
    ) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not alloc CMA heap: {:?}", e);
            return;
        }
    };

    let timeout = Duration::from_millis(2000);
    loop {
        let sample = if let Some(v) = sub_radarcube.drain().last() {
            v
        } else {
            match sub_radarcube.recv_timeout(timeout) {
                Ok(v) => match v {
                    Some(v) => v,
                    None => {
                        warn!("Timeout on radar cube");
                        continue;
                    }
                },
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

        let radarcube = info_span!("cube_deserialize")
            .in_scope(|| cdr::deserialize::<RadarCube>(&sample.payload().to_bytes()).unwrap());
        let cube_shape = radarcube
            .shape
            .iter()
            .map(|v| *v as usize)
            .collect::<Vec<_>>();
        let cube = preprocess_cube(&radarcube.cube, &cube_shape, &radar_input_shape);

        info_span!("cube_load").in_scope(|| {
            let radar_input_tensor =
                match backbone.tensor_index_mut(input_tensor_index[radar_input_index] as usize) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("Could not get input 0 from model: {:?}", e);
                        return;
                    }
                };
            let mut input_tensor_map = radar_input_tensor.maprw_f32().unwrap();
            trace!("mapped input tensor: len={:?}", input_tensor_map.len());
            input_tensor_map.copy_from_slice(&cube);
            drop(input_tensor_map);
        });

        if camera_input_index.is_some() {
            let camera_input_tensor = camera_input_tensor.as_mut().unwrap();
            let sub_camera = sub_camera.as_ref().unwrap();
            let sample = if let Some(v) = sub_camera.drain().last() {
                v
            } else {
                match sub_camera.recv_timeout(timeout) {
                    Ok(v) => match v {
                        Some(v) => v,
                        None => {
                            warn!("Timeout on camera frame");
                            continue;
                        }
                    },
                    Err(e) => {
                        error!(
                            "error receiving camera frame on {}: {:?}",
                            sub_radarcube.key_expr(),
                            e
                        );
                        continue;
                    }
                }
            };

            let mut cam_buffer = info_span!("camera_deserialize")
                .in_scope(|| cdr::deserialize::<DmaBuf>(&sample.payload().to_bytes()).unwrap());

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
            trace!("Updated dma fd to {}", cam_buffer.fd);

            match info_span!("camera_load").in_scope(|| {
                load_frame_dmabuf(
                    camera_input_tensor,
                    &img_mgr,
                    &mut dest,
                    &cam_buffer,
                    Preprocessing::UnsignedNorm,
                )
            }) {
                Ok(_) => {}
                Err(e) => {
                    error!("Error loading camera frame into input: {:?}", e);
                    continue;
                }
            }
        }

        if let Err(e) = run_model(&backbone, &mut decoder, &input_match) {
            error!("Failed to run model: {}", e);
            return;
        }

        let output_ctx = match decoder {
            Some(ref v) => v,
            None => &backbone,
        };

        let mut output_shape: Vec<u32> = vec![0, 0, 0, 0];

        let mask = info_span!("model_output").in_scope(|| {
            let mut mask = if let Ok(tensor) = output_ctx.output(0) {
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

            if args.logits {
                apply_sigmoid(&mut mask);
            }

            mask
        });

        let (buf, enc) = info_span!("model_publish").in_scope(|| {
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
            let buf = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap());
            let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

            (buf, enc)
        });

        publ_mask.put(buf).encoding(enc).await.unwrap();

        let mut occupied_ = mask.into_iter();
        let mut occupied = Vec::new();
        for i in 0..output_shape[1] as usize {
            occupied.push(Vec::new());
            for _ in 0..output_shape[2] {
                let item = occupied_.next().unwrap();
                occupied[i].push(item)
            }
        }
        let timestamp = radarcube.header.stamp.nanosec as u64
            + radarcube.header.stamp.sec as u64 * 1_000_000_000;
        let mut guard = grid.lock().await;
        *guard = Some((occupied, timestamp));
    }
}

fn get_input_match(
    backbone: &Context,
    decoder: &Option<Context>,
) -> Result<Vec<(usize, usize)>, String> {
    if decoder.is_none() {
        return Ok(Vec::new());
    }
    let decoder = decoder.as_ref().unwrap();
    let backbone_outputs = model::outputs(backbone.model()).unwrap_or_default();
    let decoder_inputs = model::inputs(decoder.model()).unwrap_or_default();
    if backbone_outputs.len() != decoder_inputs.len() {
        return Err("backbone output count and decoder input count are not equal".to_string());
    }
    let mut matching = Vec::new();
    for bb_out in backbone_outputs {
        let bb_out_shape = match backbone.tensor_index(bb_out as usize) {
            Ok(v) => v.shape(),
            Err(_) => continue,
        };
        let mut found = false;
        for dc_in in decoder_inputs.iter() {
            let dc_in_shape = match decoder.tensor_index(*dc_in as usize) {
                Ok(v) => v.shape(),
                Err(_) => continue,
            };
            if bb_out_shape == dc_in_shape {
                matching.push((bb_out as usize, *dc_in as usize));
                found = true;
                break;
            }
        }
        if !found {
            return Err(format!(
                "could not find matching decoder input for backbone output with shape {}",
                bb_out
            ));
        }
    }

    Ok(matching)
}

#[instrument(skip_all)]
fn run_model(
    backbone: &Context,
    decoder: &mut Option<Context>,
    input_match: &[(usize, usize)],
) -> Result<(), deepviewrt::error::Error> {
    backbone.run()?;
    if decoder.is_none() {
        return Ok(());
    }
    let decoder = decoder.as_mut().unwrap();
    for (bb_out, dc_in) in input_match {
        let output = match backbone.tensor_index(*bb_out) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get output tensor from backbone");
                return Err(e);
            }
        };
        let input = match decoder.tensor_index_mut(*dc_in) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get input tensor from decoder");
                return Err(e);
            }
        };
        let tensor_size = output.size() as usize;
        let output_map = match output.mapro::<u8>() {
            Ok(v) => v,
            Err(e) => {
                error!("Could not map output tensor from backbone");
                return Err(e);
            }
        };
        let mut input_map = match input.maprw::<u8>() {
            Ok(v) => v,
            Err(e) => {
                error!("Could not map input tensor from decoder");
                return Err(e);
            }
        };

        unsafe {
            memcpy(
                input_map.as_mut_ptr() as *mut c_void,
                output_map.as_ptr() as *const c_void,
                tensor_size,
            );
        }
    }
    decoder.run()
}

#[allow(dead_code)]
pub enum Preprocessing {
    Raw = 0x0,
    UnsignedNorm = 0x1,
    SignedNorm = 0x2,
    ImageNet = 0x8,
}

static RGB_MEANS_IMAGENET: [f32; 4] = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0, 128.0]; // last value is for Alpha channel when needed
static RGB_STDS_IMAGENET: [f32; 4] = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0, 64.0]; // last value is for Alpha channel when needed

fn load_frame_dmabuf(
    tensor: &mut Tensor,
    img_mgr: &ImageManager,
    dest: &mut Image,
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
    if dest.format() != RGBA {
        return Err("The format of destination buffer is not RGBA".to_owned());
    }
    const DATA_CHANNELS: usize = 4; // RGBA is 4 channels

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
    trace!("Dest size: {}", dest.size());
    let tensor_vol = tensor.volume() as usize;
    trace!("Tensor volume: {}", tensor_vol);
    let tensor_channels = *tensor.shape().last().unwrap_or(&3) as usize;
    match tensor_channels {
        3 | 4 => {}
        _ => {
            return Err(format!(
                "Input tensor has an invalid number of channels for images: {tensor_channels}"
            ))
        }
    }
    match tensor.tensor_type() {
        TensorType::U8 => {
            let mut tensor_mapped = match tensor.maprw() {
                Ok(v) => v,
                Err(e) => return Err(e.to_string()),
            };
            let mut dest_mapped = dest.mmap();
            let data = dest_mapped.as_slice_mut();

            for i in 0..tensor_vol / tensor_channels {
                for j in 0..tensor_channels {
                    tensor_mapped[i * tensor_channels + j] = data[i * DATA_CHANNELS + j];
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

            let mut dest_mapped = dest.mmap();
            let data = dest_mapped.as_slice_mut();
            for i in 0..tensor_vol / tensor_channels {
                for j in 0..tensor_channels {
                    tensor_mapped[i * tensor_channels + j] =
                        (data[i * DATA_CHANNELS + j] as i16 - 128) as i8;
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
            let mut dest_mapped = dest.mmap();
            let data = dest_mapped.as_slice_mut();
            match preprocess {
                Preprocessing::Raw => {
                    for i in 0..tensor_vol / tensor_channels {
                        for j in 0..tensor_channels {
                            tensor_mapped[i * tensor_channels + j] =
                                data[i * DATA_CHANNELS + j] as f32;
                        }
                    }
                }
                Preprocessing::UnsignedNorm => {
                    for i in 0..tensor_vol / tensor_channels {
                        for j in 0..tensor_channels {
                            tensor_mapped[i * tensor_channels + j] =
                                data[i * DATA_CHANNELS + j] as f32 / 255.0;
                        }
                    }
                }
                Preprocessing::SignedNorm => {
                    for i in 0..tensor_vol / tensor_channels {
                        for j in 0..tensor_channels {
                            tensor_mapped[i * tensor_channels + j] =
                                data[i * DATA_CHANNELS + j] as f32 / 127.5 - 1.0;
                        }
                    }
                }
                Preprocessing::ImageNet => {
                    for i in 0..tensor_vol / tensor_channels {
                        for j in 0..tensor_channels {
                            tensor_mapped[i * tensor_channels + j] =
                                (data[i * DATA_CHANNELS + j] as f32 - RGB_MEANS_IMAGENET[j])
                                    / RGB_STDS_IMAGENET[j];
                        }
                    }
                }
            }
        }
        TensorType::F64 => todo!(),
    };

    Ok(())
}
