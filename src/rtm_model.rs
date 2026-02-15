// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use deepviewrt::{
    context::Context,
    engine::Engine,
    model,
    tensor::{Tensor, TensorType},
};
use edgefirst_schemas::{
    edgefirst_msgs::{DmaBuffer, Mask, RadarCube},
    serde_cdr,
};
use log::{debug, error, info, trace};
use std::{fs::read, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use tracing::{info_span, instrument};
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

#[instrument(skip_all)]
fn load_model(model_name: Option<PathBuf>, engine: String) -> Option<Context> {
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

    let engine = if engine.to_lowercase() == "npu" {
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
        Ok(_) => info!("Loaded backbone model {model_name:?}"),
        Err(e) => {
            error!("Could not load model file {model_name:?}: {e:?}");
            return None;
        }
    }
    Some(backbone)
}

#[instrument(skip_all)]
fn identify_inputs(input_names: &[&str]) -> (usize, Option<usize>) {
    let mut radar_input_index = 0;
    let mut camera_input_index = None;
    for (i, name) in input_names.iter().enumerate() {
        debug!("Input #{i} has name: {name}");
        if name.contains("radar") {
            radar_input_index = i;
            debug!("setting radar input index to {i}");
        } else if name.contains("camera") {
            let _ = camera_input_index.insert(i);
            debug!("setting camera input index to {i}");
        }
    }
    (radar_input_index, camera_input_index)
}

#[instrument(skip_all)]
fn get_camera_input(
    backbone: &Context,
    input_tensor_index: &[u32],
    camera_input_index: Option<usize>,
) -> Result<(Option<Tensor>, Vec<usize>), FusionError> {
    let mut camera_input_tensor = None;
    if let Some(camera_input_index) = camera_input_index {
        match backbone.tensor_index(input_tensor_index[camera_input_index] as usize) {
            Ok(v) => {
                // needed because the dvrt borrow is still mutable even though the Tensor
                // pointer itself isn't mutable
                let tensor = unsafe { Tensor::from_ptr(v.to_mut_ptr(), false).unwrap() };
                camera_input_tensor = Some(tensor)
            }
            Err(e) => {
                error!("Could not get input {camera_input_index} from model: {e:?}");
                return Err(
                    format!("Could not get input {camera_input_index} from model: {e:?}").into(),
                );
            }
        }
    }
    let mut camera_input_shape = vec![1, 1, 1, 1];
    if let Some(ref camera_input_tensor) = camera_input_tensor {
        camera_input_shape = camera_input_tensor
            .shape()
            .iter()
            .map(|v| *v as usize)
            .collect()
    }
    Ok((camera_input_tensor, camera_input_shape))
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
pub async fn run_rtm_fusion_model(
    session: Session,
    args: Args,
    grid: Arc<Mutex<Option<Grid>>>,
) -> Result<(), FusionError> {
    let mut backbone = load_model(args.model.clone(), args.engine.clone()).unwrap();

    let mut decoder = None;
    if args.model_decoder.is_some() {
        decoder = load_model(args.model_decoder.clone(), "cpu".to_string());
    }

    let input_match = get_input_match(&backbone, &decoder)?;

    let input_tensor_index = model::inputs(backbone.model())?;

    let input_names: Vec<_> = input_tensor_index
        .iter()
        .map(|v| model::layer_name(backbone.model(), *v as usize).unwrap_or("NO_NAME"))
        .collect();

    let (radar_input_index, camera_input_index) = identify_inputs(&input_names);

    let radar_input_shape: Vec<_> =
        match backbone.tensor_index(input_tensor_index[radar_input_index] as usize) {
            Ok(v) => v.shape().iter().map(|v| *v as usize).collect(),
            Err(e) => {
                error!("Could not get input 0 from model: {e:?}");
                return Err(e.into());
            }
        };
    debug!("got input tensor shape: {:?}", radar_input_shape);

    let sub_radarcube = session
        .declare_subscriber(&args.radarcube_topic)
        .await
        .unwrap();
    info!("Declared subscriber on {:?}", &args.radarcube_topic);

    let mut sub_camera = None;
    if camera_input_index.is_some() {
        let s = session
            .declare_subscriber(&args.camera_topic)
            .await
            .unwrap();
        info!("Declared subscriber on {:?}", &args.camera_topic);
        let _ = sub_camera.insert(s);
    }

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
            return Err(e.to_string().into());
        }
    };

    let (mut camera_input_tensor, camera_input_shape) =
        get_camera_input(&backbone, &input_tensor_index, camera_input_index)?;

    let (img_mgr, mut dest) = initialize_g2d(&camera_input_shape)?;

    let mut timeout_radarcube = DrainRecvTimeoutSettings::default();
    let mut timeout_camera = DrainRecvTimeoutSettings::default();
    loop {
        let sample = match drain_recv(&sub_radarcube, &mut timeout_radarcube).await {
            Some(v) => v,
            None => continue,
        };

        let radarcube = info_span!("cube_deserialize").in_scope(|| {
            serde_cdr::deserialize::<RadarCube>(&sample.payload().to_bytes()).unwrap()
        });
        let cube_shape = radarcube
            .shape
            .iter()
            .map(|v| *v as usize)
            .collect::<Vec<_>>();
        let cube = preprocess_cube(&radarcube.cube, &cube_shape, &radar_input_shape);

        load_cube(&mut backbone, &input_tensor_index, radar_input_index, &cube);

        if camera_input_index.is_some() {
            let camera_input_tensor = camera_input_tensor.as_mut().unwrap();
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

        if let Err(e) = run_model(&backbone, &mut decoder, &input_match) {
            error!("Failed to run model: {e}");
            return Err(e.into());
        }

        let output_ctx = match decoder {
            Some(ref v) => v,
            None => &backbone,
        };

        let (mask, output_shape) = get_model_output(output_ctx, args.logits);

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
                boxed: false,
            };
            let buf = ZBytes::from(serde_cdr::serialize(&msg).unwrap());
            let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

            (buf, enc)
        });

        publ_mask.put(buf).encoding(enc).await.unwrap();

        let occupied = build_occupancy_grid(&mask, &output_shape);
        let timestamp = radarcube.header.stamp.nanosec as u64
            + radarcube.header.stamp.sec as u64 * 1_000_000_000;
        let mut guard = grid.lock().await;
        *guard = Some((occupied, timestamp));
    }
}

#[instrument(skip_all)]
fn load_cube(
    backbone: &mut Context,
    input_tensor_index: &[u32],
    radar_input_index: usize,
    cube: &[f32],
) {
    let radar_input_tensor =
        match backbone.tensor_index_mut(input_tensor_index[radar_input_index] as usize) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get input 0 from model: {e:?}");
                return;
            }
        };
    let mut input_tensor_map = radar_input_tensor.maprw_f32().unwrap();
    trace!("mapped input tensor: len={:?}", input_tensor_map.len());
    input_tensor_map.copy_from_slice(cube);
}

#[instrument(skip_all)]
fn build_occupancy_grid(mask: &[f32], output_shape: &[u32]) -> Vec<Vec<f32>> {
    let mut occupied_ = mask.iter();
    let mut occupied = Vec::new();
    for i in 0..output_shape[1] as usize {
        occupied.push(Vec::new());
        for _ in 0..output_shape[2] {
            let item = occupied_.next().unwrap();
            occupied[i].push(*item)
        }
    }
    occupied
}

#[instrument(skip_all)]
fn get_model_output(output_ctx: &Context, logits: bool) -> (Vec<f32>, Vec<u32>) {
    let mut output_shape: Vec<u32> = vec![0, 0, 0, 0];
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

    if logits {
        apply_sigmoid(&mut mask);
    }

    (mask, output_shape)
}

#[instrument(skip_all)]
async fn load_camera_frame(
    camera_input_tensor: &mut Tensor,
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
        .in_scope(|| serde_cdr::deserialize::<DmaBuffer>(&sample.payload().to_bytes()).unwrap());

    match info_span!("camera_load").in_scope(|| {
        load_frame_dmabuf(
            camera_input_tensor,
            img_mgr,
            dest,
            &cam_buffer,
            Preprocessing::UnsignedNorm,
        )
    }) {
        Ok(_) => {}
        Err(e) => {
            error!("Error loading camera frame into input: {e:?}");
        }
    }
}

#[instrument(skip_all)]
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
        error!("backbone output count and decoder input count are not equal");
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
            error!("could not find matching decoder input for backbone output with shape {bb_out}");
            return Err(format!(
                "could not find matching decoder input for backbone output with shape {bb_out}"
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

        assert!(
            output_map.len() >= tensor_size && input_map.len() >= tensor_size,
            "Tensor buffer size mismatch: output={}, input={}, needed={}",
            output_map.len(),
            input_map.len(),
            tensor_size
        );
        input_map[..tensor_size].copy_from_slice(&output_map[..tensor_size]);
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

#[instrument(skip_all)]
fn load_frame_dmabuf(
    tensor: &mut Tensor,
    img_mgr: &ImageManager,
    dest: &mut Image,
    dma_buf: &DmaBuffer,
    preprocess: Preprocessing,
) -> Result<(), FusionError> {
    if dest.height() as i32 != tensor.shape()[1] {
        return Err(
            "The height of the destination buffer is not equal to the height of the tensor".into(),
        );
    }
    if dest.width() as i32 != tensor.shape()[2] {
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
    let tensor_vol = tensor.volume() as usize;
    trace!("Tensor volume: {}", tensor_vol);
    let tensor_channels = *tensor.shape().last().unwrap_or(&3) as usize;
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
    tensor: &mut Tensor,
    tensor_vol: usize,
    tensor_channels: usize,
    preprocess: Preprocessing,
) -> Result<(), String> {
    match tensor.tensor_type() {
        TensorType::U8 => load_input_u8(dest, data_channels, tensor, tensor_vol, tensor_channels)?,
        TensorType::I16 => todo!(),
        TensorType::U16 => todo!(),
        TensorType::I32 => todo!(),
        TensorType::RAW => todo!(),
        TensorType::STR => todo!(),
        TensorType::I8 => load_input_i8(dest, data_channels, tensor, tensor_vol, tensor_channels)?,
        TensorType::U32 => todo!(),
        TensorType::I64 => todo!(),
        TensorType::U64 => todo!(),
        TensorType::F16 => todo!(),
        TensorType::F32 => load_input_f32(
            dest,
            data_channels,
            tensor,
            tensor_vol,
            tensor_channels,
            preprocess,
        )?,
        TensorType::F64 => todo!(),
    };
    Ok(())
}

#[instrument(skip_all)]
fn load_input_u8(
    dest: &mut Image,
    data_channels: usize,
    tensor: &mut Tensor,
    tensor_vol: usize,
    tensor_channels: usize,
) -> Result<(), String> {
    let mut tensor_mapped = match tensor.maprw() {
        Ok(v) => v,
        Err(e) => return Err(e.to_string()),
    };
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
    tensor: &mut Tensor,
    tensor_vol: usize,
    tensor_channels: usize,
) -> Result<(), String> {
    let mut tensor_mapped = match tensor.maprw() {
        Ok(v) => v,
        Err(e) => return Err(e.to_string()),
    };

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
    tensor: &mut Tensor,
    tensor_vol: usize,
    tensor_channels: usize,
    preprocess: Preprocessing,
) -> Result<(), String> {
    let mut tensor_mapped = match tensor.maprw() {
        Ok(v) => v,
        Err(e) => return Err(e.to_string()),
    };
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
