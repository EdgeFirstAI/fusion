use async_pidfd::PidFd;
use cdr::{CdrLe, Infinite};
use edgefirst_schemas::edgefirst_msgs::{DmaBuf, Mask, RadarCube};
use libc::memcpy;
use log::{debug, error, info, trace, warn};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    ffi::c_void,
    fs::{read, File},
    io,
    os::fd::AsRawFd,
    path::PathBuf,
    sync::Arc,
};
use tflitec_sys::{
    delegate::Delegate,
    tensor::{Tensor, TensorMut, TensorType},
    Interpreter, TFLiteLib,
};
use tokio::sync::Mutex;
use tracing::{info_span, instrument};
use tracy_client::secondary_frame_mark;
use zenoh::{
    bytes::{Encoding, ZBytes},
    handlers::FifoChannelHandler,
    pubsub::Subscriber,
    sample::Sample,
    Session,
};

use crate::{
    args::Args,
    drain_recv,
    fusion_model::{apply_sigmoid, preprocess_cube, FusionError},
    image::{Image, ImageManager, Rotation, RGBA},
    DrainRecvTimeoutSettings, Grid,
};

static NPU_PATH: &str = "libvx_delegate.so";

#[instrument(skip_all)]
fn load_model(
    model_name: Option<PathBuf>,
    engine: String,
    tflite_lib: &TFLiteLib,
) -> Option<Interpreter<'_>> {
    // let model_name = args.model.as_ref().unwrap().clone();
    if model_name.is_none() {
        info!("No radar model was given");
        return None;
    }
    let model_name = model_name.unwrap();
    let model_data = match read(&model_name) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open `{model_name:?}` file: {e:?}");
            return None;
        }
    };

    info!("Model read from file");

    let model = match tflite_lib.new_model_from_mem(model_data) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not create TFLite model from {model_name:?}: {e}");
            return None;
        }
    };

    let mut builder = match tflite_lib.new_interpreter_builder() {
        Ok(v) => v,
        Err(e) => {
            error!("Error while building backbone: {e}");
            return None;
        }
    };

    if engine.to_lowercase() == "npu" {
        info!("Using delegate {NPU_PATH:?}");
        let delegate = Delegate::load_external(NPU_PATH).unwrap();
        builder.add_owned_delegate(delegate);
    }

    let backbone = match builder.build(model) {
        Ok(v) => v,
        Err(e) => {
            error!("Error while building backbone: {e}");
            return None;
        }
    };
    Some(backbone)
}

#[instrument(skip_all)]
fn identify_inputs(inputs: &[TensorMut]) -> (usize, Option<usize>) {
    let mut radar_input_index = 0;
    let mut camera_input_index = None;
    for (i, inp) in inputs.iter().enumerate() {
        debug!("found input: {inp:?}");
        if inp.name().contains("radar") {
            radar_input_index = i;
            debug!("setting radar input index to {i}");
        }
        if inp.name().contains("camera") {
            let _ = camera_input_index.insert(i);
            debug!("setting camera input index to {i}");
        }
    }
    (radar_input_index, camera_input_index)
}

#[instrument(skip_all)]
fn get_input_shape(
    inputs: &[TensorMut],
    input_index: Option<usize>,
) -> Result<Vec<usize>, FusionError> {
    if let Some(ref index) = input_index {
        match inputs[*index].shape() {
            Ok(v) => {
                debug!("got input tensor shape: {v:?}");
                Ok(v)
            }
            Err(e) => {
                error!("Could not get input shape: {e}");
                Err(e.into())
            }
        }
    } else {
        Ok(vec![1, 1, 1, 1])
    }
}

#[instrument(skip_all)]
fn initialize_g2d(camera_input_shape: &[usize]) -> Result<(ImageManager, Image), FusionError> {
    let img_mgr = match ImageManager::new() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open G2D: {e:?}");
            return Err(e.to_string().into());
        }
    };
    info!("Opened G2D with version {}", img_mgr.version());

    let dest = match Image::new(
        camera_input_shape[2] as u32,
        camera_input_shape[1] as u32,
        RGBA,
    ) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not alloc CMA heap: {e:?}");
            return Err(e.to_string().into());
        }
    };
    Ok((img_mgr, dest))
}

#[instrument(skip_all)]
pub async fn run_tflite_fusion_model(
    session: Session,
    args: Args,
    grid: Arc<Mutex<Option<Grid>>>,
) -> Result<(), FusionError> {
    if args.model.is_none() {
        info!("No radar model was given");
        return Err("No radar model was given".into());
    }

    let tflite_lib = match TFLiteLib::new() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open TFLite library: {e}");
            return Err(e.into());
        }
    };

    let mut backbone = load_model(args.model.clone(), args.engine.clone(), &tflite_lib).unwrap();
    info!("TFLite context for backbone initialized");
    let mut decoder = None;
    if args.model_decoder.is_some() {
        decoder = load_model(args.model_decoder.clone(), "cpu".to_string(), &tflite_lib);
        info!("TFLite context for decoder initialized");
    }
    let input_match = get_input_match(&backbone, &decoder)?;
    let inputs = match backbone.inputs_mut() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not get backbone inputs: {e}");
            return Err(e.into());
        }
    };

    let (radar_input_index, camera_input_index) = identify_inputs(&inputs);

    let radar_input_shape: Vec<_> = get_input_shape(&inputs, Some(radar_input_index))?;

    let camera_input_shape = get_input_shape(&inputs, camera_input_index)?;
    drop(inputs);

    // warmup the model. Tflite models load on first run, instead of on load.
    if let Err(e) = run_model(&mut backbone, &mut decoder, &input_match) {
        error!("Failed to run model: {e}");
        return Err(e);
    }

    let sub_radarcube = session
        .declare_subscriber(&args.radarcube_topic)
        .await
        .unwrap();
    info!("Declared subscriber on {:?}", &args.radarcube_topic);

    let publ_mask = session
        .declare_publisher(args.model_output_topic.clone())
        .await
        .unwrap();

    let mut sub_camera = None;
    if camera_input_index.is_some() {
        let s = session
            .declare_subscriber(&args.camera_topic)
            .await
            .unwrap();
        info!("Declared subscriber on {:?}", &args.camera_topic);
        let _ = sub_camera.insert(s);
    }

    let (img_mgr, mut dest) = initialize_g2d(&camera_input_shape)?;

    let mut timeout_radarcube = DrainRecvTimeoutSettings::default();
    let mut timeout_camera = DrainRecvTimeoutSettings::default();
    loop {
        let sample = match drain_recv(&sub_radarcube, &mut timeout_radarcube).await {
            Some(v) => v,
            None => continue,
        };

        let radarcube = info_span!("cube_deserialize")
            .in_scope(|| cdr::deserialize::<RadarCube>(&sample.payload().to_bytes()).unwrap());
        let cube_shape = radarcube
            .shape
            .iter()
            .map(|v| *v as usize)
            .collect::<Vec<_>>();
        let cube = preprocess_cube(&radarcube.cube, &cube_shape, &radar_input_shape);

        let mut backbone_inputs = backbone.inputs_mut()?;

        load_cube(&mut backbone_inputs, radar_input_index, &cube);

        if camera_input_index.is_some() {
            let camera_input_tensor = &mut backbone_inputs[camera_input_index.unwrap()];
            let sub_camera = sub_camera.as_ref().unwrap();
            load_camera_frame(
                camera_input_tensor,
                sub_camera,
                &mut timeout_camera,
                &img_mgr,
                &mut dest,
            )
            .await;
        }

        drop(backbone_inputs);

        if let Err(e) = run_model(&mut backbone, &mut decoder, &input_match) {
            error!("Failed to run model: {e}");
            return Err(e);
        }

        let output_ctx = match decoder {
            Some(ref v) => v,
            None => &backbone,
        };

        let outputs = output_ctx.outputs()?;

        let (mask, output_shape) = get_model_output(&outputs, args.logits);

        let (mask, buf, enc) = info_span!("publish_output").in_scope(|| {
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

        let occupied = build_occupancy_grid(&mask, &output_shape);

        let timestamp = radarcube.header.stamp.nanosec as u64
            + radarcube.header.stamp.sec as u64 * 1_000_000_000;
        let mut guard = grid.lock().await;
        *guard = Some((occupied, timestamp));

        args.tracy.then(|| secondary_frame_mark!("model"));
    }
}

#[instrument(skip_all)]
fn load_cube(backbone_inputs: &mut [TensorMut], radar_input_index: usize, cube: &[f32]) {
    let radar_input_tensor = &mut backbone_inputs[radar_input_index];
    let input_tensor_map = match radar_input_tensor.maprw() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not map radar input: {e:?}");
            return;
        }
    };
    input_tensor_map.copy_from_slice(cube);
}

#[instrument(skip_all)]
fn build_occupancy_grid(mask: &[f32], output_shape: &[usize]) -> Vec<Vec<f32>> {
    let mut occupied_ = mask.iter();
    let mut occupied = Vec::new();
    for i in 0..output_shape[1] {
        occupied.push(Vec::new());
        for _ in 0..output_shape[2] {
            let item = occupied_.next().unwrap();
            occupied[i].push(*item)
        }
    }
    occupied
}

#[instrument(skip_all)]
fn get_model_output(outputs: &[Tensor], logits: bool) -> (Vec<f32>, Vec<usize>) {
    let mut output_shape: Vec<usize> = vec![0, 0, 0, 0];
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

    if logits {
        apply_sigmoid(&mut mask);
    }

    (mask, output_shape)
}

#[instrument(skip_all)]
fn process_dmabuffer(cam_buffer: &mut DmaBuf) -> Result<File, io::Error> {
    let pidfd: PidFd = match PidFd::from_pid(cam_buffer.pid as i32) {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error getting PID {:?}, please check if the camera process is running: {:?}",
                cam_buffer.pid, e
            );
            return Err(e);
        }
    };

    let fd = match get_file_from_pidfd(pidfd.as_raw_fd(), cam_buffer.fd, GetFdFlags::empty()) {
        Ok(v) => v,
        Err(e) => {
            error!(
            "Error getting Camera DMA file descriptor, please check if current process is running with same permissions as camera: {e:?}"
            );
            return Err(e);
        }
    };

    cam_buffer.fd = fd.as_raw_fd();
    debug!("Updated dma fd to {}", cam_buffer.fd);
    Ok(fd)
}

#[instrument(skip_all)]
fn get_input_match(
    backbone: &Interpreter,
    decoder: &Option<Interpreter>,
) -> Result<Vec<(usize, usize)>, FusionError> {
    if decoder.is_none() {
        return Ok(Vec::new());
    }
    let decoder = decoder.as_ref().unwrap();
    let backbone_outputs = backbone.outputs()?;
    let decoder_inputs = decoder.inputs_mut()?;
    if backbone_outputs.len() != decoder_inputs.len() {
        error!("backbone output count and decoder input count are not equal");
        return Err("backbone output count and decoder input count are not equal".into());
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
            error!("could not find matching decoder input for backbone output with shape {bb_out}");
            return Err(format!(
                "could not find matching decoder input for backbone output with shape {bb_out}"
            )
            .into());
        }
    }

    Ok(matching)
}

#[instrument(skip_all)]
fn run_model(
    backbone: &mut Interpreter,
    decoder: &mut Option<Interpreter>,
    input_match: &[(usize, usize)],
) -> Result<(), FusionError> {
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
                return Err(e.into());
            }
        };
        let input_map = match input.maprw::<u8>() {
            Ok(v) => v,
            Err(e) => {
                error!("Could not map input tensor from decoder");
                return Err(e.into());
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
    Ok(decoder.invoke()?)
}

#[instrument(skip_all)]
async fn load_camera_frame(
    camera_input_tensor: &mut TensorMut<'_>,
    sub_camera: &Subscriber<FifoChannelHandler<Sample>>,
    timeout_camera: &mut DrainRecvTimeoutSettings,
    img_mgr: &ImageManager,
    dest: &mut Image,
) {
    let sample = match drain_recv(sub_camera, timeout_camera).await {
        Some(v) => v,
        None => return,
    };

    let cam_buffer = info_span!("camera_deserialize")
        .in_scope(|| cdr::deserialize::<DmaBuf>(&sample.payload().to_bytes()).unwrap());

    match load_frame_dmabuf(
        camera_input_tensor,
        img_mgr,
        dest,
        &cam_buffer,
        Preprocessing::UnsignedNorm,
    ) {
        Ok(_) => {}
        Err(e) => {
            error!("Error loading camera frame into input: {e:?}");
        }
    }
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
) -> Result<(), FusionError> {
    if dest.height() as usize != tensor.shape()?[1] {
        return Err(
            "The height of the destination buffer is not equal to the height of the tensor".into(),
        );
    }
    if dest.width() as usize != tensor.shape()?[2] {
        return Err(
            "The width of the destination buffer is not equal to the width of the tensor".into(),
        );
    }
    if dest.format() != RGBA {
        return Err("The format of destination buffer is not RGBA".into());
    }
    const DATA_CHANNELS: usize = 4; // RGBA is 4 channels

    let input = dma_buf.try_into()?;
    match img_mgr.convert(&input, dest, None, Rotation::Rotation0) {
        Ok(_) => {}
        Err(e) => {
            error!("Could not g2d convert from {input:?} to {dest:?}: {e:?}")
        }
    }
    trace!("Dest size: {}", dest.size());
    let tensor_vol = tensor.volume()?;
    trace!("Tensor volume: {}", tensor_vol);
    let tensor_channels = *tensor.shape()?.last().unwrap_or(&3);
    match tensor_channels {
        3 | 4 => {}
        _ => {
            return Err(format!(
                "Input tensor has an invalid number of channels for images: {tensor_channels}"
            )
            .into())
        }
    }
    load_input(
        dest,
        DATA_CHANNELS,
        tensor,
        tensor_vol,
        tensor_channels,
        preprocess,
    )?;
    Ok(())
}

#[instrument(skip_all)]
fn load_input(
    dest: &mut Image,
    data_channels: usize,
    tensor: &mut TensorMut,
    tensor_vol: usize,
    tensor_channels: usize,
    preprocess: Preprocessing,
) -> Result<(), FusionError> {
    match tensor.tensor_type() {
        TensorType::UInt8 => {
            load_input_u8(dest, data_channels, tensor, tensor_vol, tensor_channels)?
        }
        TensorType::Int8 => {
            load_input_i8(dest, data_channels, tensor, tensor_vol, tensor_channels)?
        }
        TensorType::Float32 => load_input_f32(
            dest,
            data_channels,
            tensor,
            tensor_vol,
            tensor_channels,
            preprocess,
        )?,
        TensorType::UnknownType => todo!(),
        TensorType::NoType => todo!(),
        TensorType::Int32 => todo!(),
        TensorType::Int64 => todo!(),
        TensorType::String => todo!(),
        TensorType::Bool => todo!(),
        TensorType::Int16 => todo!(),
        TensorType::Complex64 => todo!(),
        TensorType::Float16 => todo!(),
        TensorType::Float64 => todo!(),
        TensorType::Complex128 => todo!(),
        TensorType::UInt64 => todo!(),
        TensorType::Resource => todo!(),
        TensorType::Variant => todo!(),
        TensorType::UInt32 => todo!(),
        TensorType::UInt16 => todo!(),
        TensorType::Int4 => todo!(),
        TensorType::BFloat16 => todo!(),
    };
    Ok(())
}

#[instrument(skip_all)]
fn load_input_u8(
    dest: &mut Image,
    data_channels: usize,
    tensor: &mut TensorMut,
    tensor_vol: usize,
    tensor_channels: usize,
) -> Result<(), FusionError> {
    let tensor_mapped = tensor.maprw()?;
    let mut dest_mapped = dest.mmap();
    let data = dest_mapped.as_slice_mut();
    if tensor_channels == data_channels {
        tensor_mapped.copy_from_slice(&data[0..tensor_vol]);
        return Ok(());
    }
    for i in 0..tensor_vol / tensor_channels {
        for j in 0..tensor_channels {
            tensor_mapped[i * tensor_channels + j] = data[i * data_channels + j];
        }
    }
    Ok(())
}

#[instrument(skip_all)]
fn load_input_i8(
    dest: &mut Image,
    data_channels: usize,
    tensor: &mut TensorMut,
    tensor_vol: usize,
    tensor_channels: usize,
) -> Result<(), FusionError> {
    let tensor_mapped = tensor.maprw()?;

    let mut dest_mapped = dest.mmap();
    let data = dest_mapped.as_slice_mut();
    for i in 0..tensor_vol / tensor_channels {
        for j in 0..tensor_channels {
            tensor_mapped[i * tensor_channels + j] =
                (data[i * data_channels + j] as i16 - 128) as i8;
        }
    }
    Ok(())
}

#[instrument(skip_all)]
fn load_input_f32(
    dest: &mut Image,
    data_channels: usize,
    tensor: &mut TensorMut,
    tensor_vol: usize,
    tensor_channels: usize,
    preprocess: Preprocessing,
) -> Result<(), FusionError> {
    let tensor_mapped = tensor.maprw()?;
    let mut dest_mapped = dest.mmap();
    let data = dest_mapped.as_slice_mut();
    for i in 0..tensor_vol / tensor_channels {
        for j in 0..tensor_channels {
            match preprocess {
                Preprocessing::Raw => {
                    tensor_mapped[i * tensor_channels + j] = data[i * data_channels + j] as f32;
                }
                Preprocessing::UnsignedNorm => {
                    tensor_mapped[i * tensor_channels + j] =
                        data[i * data_channels + j] as f32 / 255.0;
                }
                Preprocessing::SignedNorm => {
                    tensor_mapped[i * tensor_channels + j] =
                        data[i * data_channels + j] as f32 / 127.5 - 1.0;
                }
                Preprocessing::ImageNet => {
                    tensor_mapped[i * tensor_channels + j] = (data[i * data_channels + j] as f32
                        - RGB_MEANS_IMAGENET[j])
                        / RGB_STDS_IMAGENET[j];
                }
            }
        }
    }
    Ok(())
}
