use async_pidfd::PidFd;
use cdr::{CdrLe, Infinite};
use edgefirst_schemas::edgefirst_msgs::{DmaBuf, Mask, RadarCube};
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
use tflitec_sys::{
    delegate::Delegate,
    tensor::{TensorMut, TensorType},
    Interpreter, TFLiteLib,
};
use tokio::sync::Mutex;
use tracing::{info_span, instrument};
use tracy_client::secondary_frame_mark;
use zenoh::{
    bytes::{Encoding, ZBytes},
    Session,
};

use crate::{
    args::Args,
    fusion_model::{apply_sigmoid, preprocess_cube},
    image::{Image, ImageManager, Rotation, RGBA},
    Grid,
};

static NPU_PATH: &str = "libvx_delegate.so";
static TFLITEC_PATH: &str = "libtensorflowlite_c.so";

pub async fn run_tflite_fusion_model(session: Session, args: Args, grid: Arc<Mutex<Option<Grid>>>) {
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

    let tflite_lib = match TFLiteLib::new(TFLITEC_PATH) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open TFLite library: {e}");
            return;
        }
    };

    let model = match tflite_lib.new_model_from_mem(model_data) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not create TFLite model from {:?}: {e}", model_name);
            return;
        }
    };

    let sub_radarcube = session
        .declare_subscriber(&args.radarcube_topic)
        .await
        .unwrap();
    info!("Declared subscriber on {:?}", &args.radarcube_topic);

    let publ_mask = session
        .declare_publisher(args.model_output_topic.clone())
        .await
        .unwrap();

    let mut builder = match tflite_lib.new_interpreter_builder() {
        Ok(v) => v,
        Err(e) => {
            error!("Error while building backbone: {}", e);
            return;
        }
    };

    if args.engine.to_lowercase() == "npu" {
        info!("Using delegate {:?}", NPU_PATH);
        let delegate = Delegate::load_external(NPU_PATH).unwrap();
        builder.add_owned_delegate(delegate);
    }
    let mut backbone = match builder.build(&model) {
        Ok(v) => v,
        Err(e) => {
            error!("Error while building backbone: {}", e);
            return;
        }
    };

    info!("TFLite context for backbone initialized");

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
        let model = match tflite_lib.new_model_from_mem(model_data) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not create TFLite model from {:?}: {e}", model_name);
                return;
            }
        };
        let builder = match tflite_lib.new_interpreter_builder() {
            Ok(v) => v,
            Err(e) => {
                error!("Error while building decoder: {}", e);
                return;
            }
        };
        let decoder_ctx = match builder.build(&model) {
            Ok(v) => v,
            Err(e) => {
                error!("Error while building decoder: {}", e);
                return;
            }
        };
        info!("TFLite context for decoder initialized");
        let _ = decoder.insert(decoder_ctx);
    }

    let input_match = match get_input_match(&backbone, &decoder) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not match backbone outputs to decoder inputs: {}", e);
            return;
        }
    };

    let mut radar_input_index = 0;
    let mut camera_input_index = None;

    let inputs = match backbone.inputs_mut() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not get backbone inputs: {}", e);
            return;
        }
    };
    for (i, inp) in inputs.iter().enumerate() {
        debug!("found input: {:?}", inp);
        if inp.name().contains("radar") {
            radar_input_index = i;
            debug!("setting radar input index to {}", i);
        }
        if inp.name().contains("camera") {
            let _ = camera_input_index.insert(i);
            debug!("setting camera input index to {}", i);
        }
    }
    let radar_input_shape: Vec<_> = match inputs[radar_input_index].shape() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not get input shape: {}", e);
            return;
        }
    };
    debug!("got input tensor shape: {:?}", radar_input_shape);

    let camera_input_shape = if let Some(ref camera_input_index) = camera_input_index {
        match inputs[*camera_input_index].shape() {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get input shape: {}", e);
                return;
            }
        }
    } else {
        vec![1, 1, 1, 1]
    };
    drop(inputs);
    // warmup the model. Tflite models load on first run, instead of on load.
    if let Err(e) = run_model(&mut backbone, &mut decoder, &input_match) {
        error!("Failed to run model: {}", e);
        return;
    }

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
                        warn!("timeout on radar cube");
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

        let mut backbone_inputs = match backbone.inputs_mut() {
            Ok(v) => v,
            Err(e) => {
                error!("Error while getting inputs of backbone: {:?}", e);
                return;
            }
        };

        info_span!("cube_load").in_scope(|| {
            let radar_input_tensor = &mut backbone_inputs[radar_input_index];
            let input_tensor_map = match radar_input_tensor.maprw() {
                Ok(v) => v,
                Err(e) => {
                    error!("Could not map radar input: {:?}", e);
                    return;
                }
            };
            input_tensor_map.copy_from_slice(&cube);
            let _ = input_tensor_map;
        });

        if camera_input_index.is_some() {
            let camera_input_tensor = &mut backbone_inputs[camera_input_index.unwrap()];

            let sub_camera = sub_camera.as_ref().unwrap();
            let sample = if let Some(v) = sub_camera.drain().last() {
                v
            } else {
                match sub_camera.recv_timeout(timeout) {
                    Ok(v) => match v {
                        Some(v) => v,
                        None => {
                            warn!("timeout on camera frame");
                            continue;
                        }
                    },
                    Err(e) => {
                        error!(
                            "error receiving camera frame on {}: {:?}",
                            sub_camera.key_expr(),
                            e
                        );
                        continue;
                    }
                }
            };

            let mut cam_buffer: DmaBuf = match cdr::deserialize(&sample.payload().to_bytes()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    continue;
                }
            };

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

            match load_frame_dmabuf(
                camera_input_tensor,
                &img_mgr,
                &mut dest,
                &cam_buffer,
                Preprocessing::UnsignedNorm,
            ) {
                Ok(_) => {}
                Err(e) => {
                    error!("Error loading camera frame into input: {:?}", e);
                    continue;
                }
            }
            trace!("finished load_frame_dmabuf");
        }

        drop(backbone_inputs);

        if let Err(e) = run_model(&mut backbone, &mut decoder, &input_match) {
            error!("Failed to run model: {}", e);
            return;
        }

        let output_ctx = match decoder {
            Some(ref v) => v,
            None => &backbone,
        };

        let mut output_shape: Vec<usize> = vec![0, 0, 0, 0];
        let (mask, buf, enc) = info_span!("publish_output").in_scope(|| {
            let outputs = output_ctx.outputs().unwrap();

            let mut mask = if !outputs.is_empty() {
                let tensor = &outputs[0];
                output_shape = tensor.shape().unwrap();
                let data = tensor.mapro().unwrap();
                let len = data.len();
                let mut buffer = vec![0.0f32; len];
                buffer.copy_from_slice(data);
                buffer
            } else {
                error!("Did not find model output");
                Vec::new()
            };

            if args.logits {
                apply_sigmoid(&mut mask);
            }

            let msg = Mask {
                height: output_shape[1] as u32,
                width: output_shape[2] as u32,
                length: 1,
                encoding: "".to_string(),
                mask: mask
                    .iter()
                    .flat_map(|v| {
                        [
                            (255.0 * args.model_threshold) as u8,
                            (255.0 * v).min(255.0) as u8,
                        ]
                    })
                    .collect(),
                boxed: false,
            };

            let buf = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap());
            let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

            (mask, buf, enc)
        });

        publ_mask.put(buf).encoding(enc).await.unwrap();

        let mut occupied_ = mask.into_iter();
        let mut occupied = Vec::new();
        for i in 0..output_shape[1] {
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

        args.tracy.then(|| secondary_frame_mark!("model"));
    }
}

fn get_input_match(
    backbone: &Interpreter,
    decoder: &Option<Interpreter>,
) -> Result<Vec<(usize, usize)>, String> {
    if decoder.is_none() {
        return Ok(Vec::new());
    }
    let decoder = decoder.as_ref().unwrap();
    let backbone_outputs = backbone.outputs()?;
    let decoder_inputs = decoder.inputs_mut()?;
    if backbone_outputs.len() != decoder_inputs.len() {
        return Err("backbone output count and decoder input count are not equal".to_string());
    }
    let mut matching = Vec::new();
    for (bb_out, outp) in backbone_outputs.iter().enumerate() {
        let bb_out_shape = outp.shape()?;
        let mut found = false;
        for (dc_in, inp) in decoder_inputs.iter().enumerate() {
            let dc_in_shape = inp.shape()?;
            if bb_out_shape == dc_in_shape {
                matching.push((bb_out, dc_in));
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
    backbone: &mut Interpreter,
    decoder: &mut Option<Interpreter>,
    input_match: &[(usize, usize)],
) -> Result<(), String> {
    backbone.invoke()?;
    if decoder.is_none() {
        return Ok(());
    }
    let decoder = decoder.as_mut().unwrap();
    for (bb_out, dc_in) in input_match {
        let output = &backbone.outputs()?[*bb_out];
        let input = &mut decoder.inputs_mut()?[*dc_in];
        let tensor_size = output.byte_size();
        let output_map = match output.mapro::<u8>() {
            Ok(v) => v,
            Err(e) => {
                error!("Could not map output tensor from backbone");
                return Err(e);
            }
        };
        let input_map = match input.maprw::<u8>() {
            Ok(v) => v,
            Err(e) => {
                error!("Could not map input tensor from decoder");
                return Err(e);
            }
        };
        // TODO: what happens if the types are different?
        unsafe {
            memcpy(
                input_map.as_mut_ptr() as *mut c_void,
                output_map.as_ptr() as *const c_void,
                tensor_size,
            );
        }
    }
    decoder.invoke()
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

#[instrument(skip_all)]
fn load_frame_dmabuf(
    tensor: &mut TensorMut,
    img_mgr: &ImageManager,
    dest: &mut Image,
    dma_buf: &DmaBuf,
    preprocess: Preprocessing,
) -> Result<(), String> {
    if dest.height() as usize != tensor.shape()?[1] {
        return Err(
            "The height of the destination buffer is not equal to the height of the tensor"
                .to_owned(),
        );
    }
    if dest.width() as usize != tensor.shape()?[2] {
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
    let tensor_vol = tensor.volume()?;
    trace!("Tensor volume: {}", tensor_vol);

    let tensor_channels = { *tensor.shape()?.last().unwrap_or(&3) };
    match tensor_channels {
        3 | 4 => {}
        _ => {
            return Err(format!(
                "Input tensor has an invalid number of channels for images: {tensor_channels}"
            ))
        }
    };

    match tensor.tensor_type() {
        TensorType::Int8 => {
            let tensor_mapped = tensor.maprw().unwrap();

            let mut dest_mapped = dest.mmap();
            let data = dest_mapped.as_slice_mut();
            for i in 0..tensor_vol / tensor_channels {
                for j in 0..tensor_channels {
                    tensor_mapped[i * tensor_channels + j] =
                        (data[i * DATA_CHANNELS + j] as i16 - 128) as i8;
                }
            }
        }
        TensorType::Float32 => {
            let tensor_mapped = tensor.maprw().unwrap();
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
        t => {
            return Err(format!("Input tensor unsupported type: {:?}", t));
        }
    };

    Ok(())
}
