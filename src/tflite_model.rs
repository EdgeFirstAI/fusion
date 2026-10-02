// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use edgefirst_schemas::edgefirst_msgs::{CameraFrame, Mask, RadarCube};
use log::{debug, error, info, trace, warn};
use std::{fs::read, path::PathBuf, sync::Arc, thread, time::Instant};
use tflitec_sys::{
    delegate::Delegate,
    tensor::{Tensor, TensorMut, TensorType},
    Interpreter, TFLiteLib,
};
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
    camera_ring::{CameraRing, RecycleWindow, SharedCameraRing},
    drain_recv,
    fusion_model::{apply_sigmoid, identify_named_inputs, preprocess_cube, FusionError},
    grid::{GridFrame, GridTracker, SharedGrid},
    image::{Image, ImageManager, Rotation, RGBA},
    stamp::{now_stamp, ns_to_time, time_to_ns, Continuity, StampTimeline, STEP_THRESHOLD_NS},
    stats::PairStats,
    sync::SilenceWatch,
    DrainRecvTimeoutSettings,
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
fn identify_inputs(inputs: &[TensorMut]) -> (Option<usize>, Option<usize>) {
    for (i, inp) in inputs.iter().enumerate() {
        debug!("found input #{i}: {inp:?}");
    }
    identify_named_inputs(inputs.iter().map(|inp| inp.name()))
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
fn open_g2d() -> Result<ImageManager, FusionError> {
    let img_mgr = match ImageManager::new() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open G2D: {e:?}");
            return Err(e.to_string().into());
        }
    };
    info!("Opened G2D with version {}", img_mgr.version());
    Ok(img_mgr)
}

/// RGBA image sized to the camera input tensor (NHWC).
#[instrument(skip_all)]
fn alloc_camera_input_image(camera_input_shape: &[usize]) -> Result<Image, FusionError> {
    match Image::new(
        camera_input_shape[2] as u32,
        camera_input_shape[1] as u32,
        RGBA,
    ) {
        Ok(v) => Ok(v),
        Err(e) => {
            error!("Could not alloc CMA heap: {e:?}");
            Err(e.to_string().into())
        }
    }
}

#[instrument(skip_all)]
pub async fn run_tflite_fusion_model(
    session: Session,
    args: Args,
    grid: SharedGrid,
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

    let radar_input_shape: Vec<_> = get_input_shape(&inputs, radar_input_index)?;

    let camera_input_shape = get_input_shape(&inputs, camera_input_index)?;
    drop(inputs);

    if radar_input_index.is_none() && camera_input_index.is_none() {
        error!("fusion model has no tensor named 'radar' or 'camera'; cannot identify inputs");
        return Err(
            "fusion model has no tensor named 'radar' or 'camera'; cannot identify inputs".into(),
        );
    }

    // warmup the model. Tflite models load on first run, instead of on load.
    if let Err(e) = run_model(&mut backbone, &mut decoder, &input_match) {
        error!("Failed to run model: {e}");
        return Err(e);
    }

    let sub_radarcube = if radar_input_index.is_some() {
        let s = session
            .declare_subscriber(&args.radarcube_topic)
            .await
            .unwrap();
        info!("Declared subscriber on {:?}", args.radarcube_topic);
        Some(s)
    } else {
        None
    };

    let publ_mask = session
        .declare_publisher(args.model_output_topic.clone())
        .await
        .unwrap();
    let publ_tracked = if args.track {
        Some(
            session
                .declare_publisher(format!("{}/tracked", args.model_output_topic))
                .await
                .map_err(|e| FusionError::from(format!("declare tracked publisher: {e}")))?,
        )
    } else {
        None
    };
    let ts_id = crate::stamp::timestamp_id(&session);
    let mut grid_tracker = GridTracker::new(&args);

    // With a radar input, the converter thread owns the camera subscriber.
    let mut sub_camera = None;
    if camera_input_index.is_some() && radar_input_index.is_none() {
        let s = session
            .declare_subscriber(&args.camera_topic)
            .await
            .unwrap();
        info!("Declared subscriber on {:?}", args.camera_topic);
        let _ = sub_camera.insert(s);
    }

    let mut camera = match (camera_input_index, radar_input_index) {
        (None, _) => CameraInput::None,
        (Some(_), None) => CameraInput::Latest {
            img_mgr: open_g2d()?,
            dest: alloc_camera_input_image(&camera_input_shape)?,
            timeout: DrainRecvTimeoutSettings::default(),
        },
        (Some(_), Some(_)) => {
            let slots = (0..args.camera_buffer_size)
                .map(|_| alloc_camera_input_image(&camera_input_shape))
                .collect::<Result<Vec<_>, _>>()?;
            let scratch = alloc_camera_input_image(&camera_input_shape)?;
            let ring = Arc::new(SharedCameraRing::new(CameraRing::new(slots)));
            let converter = spawn_camera_converter(
                session.clone(),
                args.camera_topic.clone(),
                ring.clone(),
                scratch,
            )
            .await?;
            CameraInput::Paired(Box::new(PairedCamera {
                ring,
                _converter: converter,
                cube_silence: SilenceWatch::new(args.radarcube_topic.clone()),
                stats: PairStats::cube_camera(),
                last_stats: Instant::now(),
            }))
        }
    };

    let mut timeout_radarcube = DrainRecvTimeoutSettings::default();
    let mut inputs = FusionInputCtx {
        backbone: &mut backbone,
        args: &args,
        radar_input_index,
        camera_input_index,
        radar_input_shape: &radar_input_shape,
        sub_radarcube: sub_radarcube.as_ref(),
        sub_camera: sub_camera.as_ref(),
        camera: &mut camera,
        timeout_radarcube: &mut timeout_radarcube,
    };
    loop {
        let Some(timestamp) = load_fusion_inputs(&mut inputs).await? else {
            continue;
        };

        if let Err(e) = run_model(inputs.backbone, &mut decoder, &input_match) {
            error!("Failed to run model: {e}");
            return Err(e);
        }

        let output_ctx = match decoder {
            Some(ref v) => v,
            None => inputs.backbone,
        };

        let outputs = output_ctx.outputs()?;

        let (mask, output_shape) = get_model_output(&outputs, args.logits);

        let (mask, buf, enc) = info_span!("publish_output").in_scope(|| {
            let bytes: Vec<u8> = mask
                .iter()
                .flat_map(|v| {
                    [
                        (255.0 * args.model_threshold) as u8,
                        (255.0 * v).min(255.0) as u8,
                    ]
                })
                .collect();
            let msg = Mask::builder()
                .height(output_shape[1] as u32)
                .width(output_shape[2] as u32)
                .length(1)
                .encoding("")
                .mask(&bytes)
                .boxed(false)
                .build()
                .expect("valid Mask");

            let buf = ZBytes::from(msg.into_cdr());
            let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

            (mask, buf, enc)
        });

        let stamp = ns_to_time(timestamp);
        crate::put_stamped(&publ_mask, buf, enc, ts_id, stamp).await;

        let occupied = build_occupancy_grid(&mask, &output_shape);

        let (predictions, tracked_mask) = grid_tracker.update(&occupied, stamp);
        grid.push(stamp, GridFrame { predictions });
        if let (Some(bytes), Some(publ)) = (tracked_mask, publ_tracked.as_ref()) {
            let width = occupied.first().map_or(0, Vec::len);
            let msg = Mask::builder()
                .height(occupied.len() as u32)
                .width(width as u32)
                .length(1)
                .encoding("")
                .mask(&bytes)
                .boxed(false)
                .build()
                .expect("valid Mask");
            crate::put_stamped(
                publ,
                ZBytes::from(msg.into_cdr()),
                Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask"),
                ts_id,
                stamp,
            )
            .await;
        }

        args.tracy.then(|| secondary_frame_mark!("model"));
    }
}

/// Camera input state, by model kind.
enum CameraInput {
    /// Radar-only model.
    None,
    /// Camera-only model: the newest frame is converted when inference runs.
    Latest {
        img_mgr: ImageManager,
        dest: Image,
        timeout: DrainRecvTimeoutSettings,
    },
    /// Radar + camera model: frames are converted on arrival and paired with
    /// each cube by stamp.
    Paired(Box<PairedCamera>),
}

/// Converted camera frames awaiting a radar cube, plus pairing statistics.
struct PairedCamera {
    ring: Arc<SharedCameraRing<Image>>,
    /// Dropped with this thread's state, which stops the converter thread.
    _converter: ConverterHandle,
    cube_silence: SilenceWatch,
    stats: PairStats,
    last_stats: Instant,
}

impl PairedCamera {
    fn maybe_log_stats(&mut self, interval: f32) {
        if interval > 0.0 && self.last_stats.elapsed().as_secs_f32() >= interval {
            self.stats.stale = self.ring.take_stale();
            self.stats.log_and_reset("fusion/model cube↔camera");
            self.last_stats = Instant::now();
        }
    }
}

/// Start the thread that converts camera frames into `ring` as they arrive,
/// so conversion never queues behind inference. Returns once its G2D context
/// and subscriber are ready.
/// Keeps the camera converter thread running; dropping it stops the thread.
struct ConverterHandle {
    _stop: tokio::sync::oneshot::Sender<()>,
}

async fn spawn_camera_converter(
    session: Session,
    topic: String,
    ring: Arc<SharedCameraRing<Image>>,
    scratch: Image,
) -> Result<ConverterHandle, FusionError> {
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
    thread::Builder::new()
        .name("camera".to_string())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("camera converter runtime: {e}")));
                    return;
                }
            };
            rt.block_on(run_camera_converter(
                session, topic, ring, scratch, ready_tx, stop_rx,
            ));
        })
        .map_err(|e| FusionError::from(format!("spawn camera converter: {e}")))?;
    ready_rx
        .await
        .map_err(|_| FusionError::from("camera converter exited during startup"))?
        .map_err(FusionError::from)?;
    Ok(ConverterHandle { _stop: stop_tx })
}

async fn run_camera_converter(
    session: Session,
    topic: String,
    ring: Arc<SharedCameraRing<Image>>,
    mut scratch: Image,
    ready: tokio::sync::oneshot::Sender<Result<(), String>>,
    mut stop: tokio::sync::oneshot::Receiver<()>,
) {
    let img_mgr = match open_g2d() {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    let sub = match session.declare_subscriber(&topic).await {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(format!("declare subscriber on {topic:?}: {e}")));
            return;
        }
    };
    info!("Declared subscriber on {topic:?}");
    let _ = ready.send(Ok(()));

    let mut timeline = StampTimeline::new(STEP_THRESHOLD_NS);
    let mut recycle = RecycleWindow::new();
    let mut silence = SilenceWatch::new(topic.clone());
    loop {
        let received = tokio::select! {
            r = tokio::time::timeout(silence.timeout(), sub.recv_async()) => r,
            _ = &mut stop => {
                info!("camera converter stopped: the fusion model has exited");
                return;
            }
        };
        let sample = match received {
            Ok(Ok(s)) => {
                silence.heard();
                s
            }
            Ok(Err(e)) => {
                error!("camera subscriber {topic} closed: {e}");
                return;
            }
            Err(_) => {
                silence.timed_out();
                continue;
            }
        };
        scratch = convert_into_ring(
            &img_mgr,
            &ring,
            &mut timeline,
            &mut recycle,
            &topic,
            &sample,
            scratch,
        );
    }
}

/// Convert `sample` into `scratch` and swap it into the ring. Returns the image
/// to use as the next scratch buffer. A malformed sample, a frame whose camera
/// buffer may already be overwritten, or a failed conversion leaves the ring
/// unchanged.
#[instrument(skip_all)]
fn convert_into_ring(
    img_mgr: &ImageManager,
    ring: &SharedCameraRing<Image>,
    timeline: &mut StampTimeline,
    recycle: &mut RecycleWindow,
    topic: &str,
    sample: &Sample,
    mut scratch: Image,
) -> Image {
    let frame = match info_span!("camera_deserialize")
        .in_scope(|| CameraFrame::from_cdr(sample.payload().to_bytes().to_vec()))
    {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to deserialize CameraFrame: {e:?}");
            return scratch;
        }
    };
    let stamp_ns = time_to_ns(frame.stamp());
    if matches!(
        timeline.observe(stamp_ns),
        Continuity::SteppedBack(_) | Continuity::Gap(_)
    ) {
        ring.lock().clear();
    }
    let tensor = frame.tensor();
    if let Some(plane) = tensor.plane_at(0) {
        if let Some(pool) = recycle.observe(tensor.pid(), plane.handle, stamp_ns) {
            info!(
                "{topic}: camera cycles {} buffers at {:.1} FPS; frames older than {:.0} ms are skipped",
                pool.buffers,
                1e9 / pool.period_ns as f64,
                pool.window_ns() as f64 * 1e-6
            );
        }
    }
    let now_ns = time_to_ns(now_stamp());
    if recycle.is_stale(stamp_ns, now_ns) {
        if ring.record_stale() {
            warn!(
                "{topic} frame is {:.0} ms old (limit {:.0} ms); skipped, its camera buffer may be overwritten",
                now_ns.saturating_sub(stamp_ns) as f64 * 1e-6,
                recycle.window_ns() as f64 * 1e-6
            );
        }
        return scratch;
    }
    match convert_camera_frame(img_mgr, &frame, &mut scratch) {
        Ok(()) => ring.publish(scratch, stamp_ns, Instant::now()),
        Err(e) => {
            error!("camera frame conversion failed: {e:?}");
            scratch
        }
    }
}

/// Zenoh + G2D state used to fill TFLite inputs for one inference pass.
struct FusionInputCtx<'a, 'b> {
    backbone: &'a mut Interpreter<'b>,
    args: &'a Args,
    radar_input_index: Option<usize>,
    camera_input_index: Option<usize>,
    radar_input_shape: &'a [usize],
    sub_radarcube: Option<&'a Subscriber<FifoChannelHandler<Sample>>>,
    sub_camera: Option<&'a Subscriber<FifoChannelHandler<Sample>>>,
    camera: &'a mut CameraInput,
    timeout_radarcube: &'a mut DrainRecvTimeoutSettings,
}

/// Fill backbone inputs for one iteration and return the output stamp (ns).
/// `None` means skip (no sample, no pairing or a tensor size mismatch); the
/// caller must not run inference on that pass.
#[instrument(skip_all)]
async fn load_fusion_inputs(ctx: &mut FusionInputCtx<'_, '_>) -> Result<Option<u64>, FusionError> {
    match (ctx.sub_radarcube, &*ctx.camera) {
        (Some(_), CameraInput::Paired(_)) => load_paired_inputs(ctx).await,
        (Some(_), _) => load_radar_only_inputs(ctx).await,
        (None, _) => load_camera_only_inputs(ctx).await,
    }
}

fn deserialize_cube(sample: &Sample) -> Option<RadarCube<Vec<u8>>> {
    match info_span!("cube_deserialize")
        .in_scope(|| RadarCube::from_cdr(sample.payload().to_bytes().to_vec()))
    {
        Ok(v) => Some(v),
        Err(e) => {
            error!("Failed to deserialize RadarCube: {e:?}");
            None
        }
    }
}

/// Preprocess `radarcube` into the radar input tensor.
fn fill_radar_input(
    backbone_inputs: &mut [TensorMut],
    radar_input_index: Option<usize>,
    radar_input_shape: &[usize],
    radarcube: &RadarCube<Vec<u8>>,
) -> bool {
    let cube_shape = radarcube
        .shape()
        .iter()
        .map(|v| *v as usize)
        .collect::<Vec<_>>();
    let cube = preprocess_cube(radarcube.cube(), &cube_shape, radar_input_shape);
    match radar_input_index {
        Some(radar_input_index) => load_cube(backbone_inputs, radar_input_index, &cube),
        None => true,
    }
}

#[instrument(skip_all)]
async fn load_radar_only_inputs(
    ctx: &mut FusionInputCtx<'_, '_>,
) -> Result<Option<u64>, FusionError> {
    let sample = {
        let sub_radarcube = ctx.sub_radarcube.expect("radar path has a cube subscriber");
        match drain_recv(sub_radarcube, ctx.timeout_radarcube).await {
            Some(v) => v,
            None => return Ok(None),
        }
    };
    let Some(radarcube) = deserialize_cube(&sample) else {
        return Ok(None);
    };
    let timestamp = time_to_ns(radarcube.stamp());

    let mut backbone_inputs = ctx.backbone.inputs_mut()?;
    if !fill_radar_input(
        &mut backbone_inputs,
        ctx.radar_input_index,
        ctx.radar_input_shape,
        &radarcube,
    ) {
        return Ok(None);
    }
    Ok(Some(timestamp))
}

/// Wait for the next radar cube and pair it with the ring slot nearest its
/// stamp.
#[instrument(skip_all)]
async fn load_paired_inputs(ctx: &mut FusionInputCtx<'_, '_>) -> Result<Option<u64>, FusionError> {
    let CameraInput::Paired(cam) = &mut *ctx.camera else {
        unreachable!("paired path requires a camera ring");
    };
    let sub_radarcube = ctx
        .sub_radarcube
        .expect("paired path has a cube subscriber");
    let camera_input_index = ctx
        .camera_input_index
        .expect("paired path has a camera input");
    cam.maybe_log_stats(ctx.args.stats_interval);

    let mut sample =
        match tokio::time::timeout(cam.cube_silence.timeout(), sub_radarcube.recv_async()).await {
            Ok(Ok(s)) => {
                cam.cube_silence.heard();
                s
            }
            Ok(Err(e)) => {
                return Err(format!(
                    "radar cube subscriber {} closed: {e}",
                    sub_radarcube.key_expr()
                )
                .into())
            }
            Err(_) => {
                cam.cube_silence.timed_out();
                return Ok(None);
            }
        };
    // Newest cube only; older ones are superseded.
    while let Ok(Some(newer)) = sub_radarcube.try_recv() {
        sample = newer;
    }
    let Some(radarcube) = deserialize_cube(&sample) else {
        return Ok(None);
    };
    let cube_ns = time_to_ns(radarcube.stamp());
    let latency_ns = i128::from(time_to_ns(now_stamp())) - i128::from(cube_ns);
    cam.stats
        .latency_ns
        .record_ns(latency_ns.clamp(i64::MIN.into(), i64::MAX.into()) as i64);

    // Frames up to the cube stamp normally arrived long ago; wait briefly only
    // if the ring has nothing at or after it.
    let newest = cam.ring.lock().newest_ns(Instant::now());
    if newest.is_none_or(|n| n < cube_ns) {
        let start = Instant::now();
        cam.ring
            .wait_for_stamp(cube_ns, ctx.args.sync_wait_duration())
            .await;
        cam.stats
            .waited_ns
            .record_ns(i64::try_from(start.elapsed().as_nanos()).unwrap_or(i64::MAX));
    }

    let max = ctx.args.max_temporal_delta_ns();
    let mut backbone_inputs = ctx.backbone.inputs_mut()?;
    {
        let mut ring = cam.ring.lock();
        let Some((idx, delta_ns)) = ring.nearest(cube_ns, Instant::now()) else {
            cam.stats.missing += 1;
            return Ok(None);
        };
        if max != 0 && delta_ns.unsigned_abs() > max {
            cam.stats.too_far += 1;
            if !cam.stats.warned_too_far {
                cam.stats.warned_too_far = true;
                warn!(
                    "{} and {} stamps differ by {:.3} s (limit {:.3} s); \
                     inference skipped. Set MAX_TEMPORAL_DELTA=0 to use the nearest frame.",
                    ctx.args.radarcube_topic,
                    ctx.args.camera_topic,
                    delta_ns as f64 * 1e-9,
                    max as f64 * 1e-9
                );
            }
            return Ok(None);
        }
        cam.stats.paired += 1;
        cam.stats.delta_ns.record_ns(delta_ns);
        if let Err(e) = load_image_into_tensor(
            &mut backbone_inputs[camera_input_index],
            ring.get_mut(idx),
            Preprocessing::UnsignedNorm,
        ) {
            error!("Error loading camera frame into input: {e:?}");
            return Ok(None);
        }
    }
    if !fill_radar_input(
        &mut backbone_inputs,
        ctx.radar_input_index,
        ctx.radar_input_shape,
        &radarcube,
    ) {
        return Ok(None);
    }
    Ok(Some(cube_ns))
}

#[instrument(skip_all)]
async fn load_camera_only_inputs(
    ctx: &mut FusionInputCtx<'_, '_>,
) -> Result<Option<u64>, FusionError> {
    let CameraInput::Latest {
        img_mgr,
        dest,
        timeout,
    } = &mut *ctx.camera
    else {
        unreachable!("camera-only model initializes G2D");
    };
    let camera_input_index = ctx.camera_input_index.expect("camera-only model");
    let sub_camera = ctx
        .sub_camera
        .expect("camera-only model subscribes to camera");
    let mut backbone_inputs = ctx.backbone.inputs_mut()?;
    let camera_input_tensor = &mut backbone_inputs[camera_input_index];
    let loaded = load_camera_frame(camera_input_tensor, sub_camera, timeout, img_mgr, dest).await;
    drop(backbone_inputs);
    Ok(loaded)
}

#[instrument(skip_all)]
fn load_cube(backbone_inputs: &mut [TensorMut], radar_input_index: usize, cube: &[f32]) -> bool {
    let radar_input_tensor = &mut backbone_inputs[radar_input_index];
    let input_tensor_map = match radar_input_tensor.maprw() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not map radar input: {e:?}");
            return false;
        }
    };
    if input_tensor_map.len() != cube.len() {
        error!(
            "radar cube tensor size does not match preprocessed cube; dest={} src={}",
            input_tensor_map.len(),
            cube.len()
        );
        return false;
    }
    input_tensor_map.copy_from_slice(cube);
    true
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
        if output_map.len() < tensor_size || input_map.len() < tensor_size {
            error!(
                "backbone/decoder tensor size mismatch: output={} input={} needed={}",
                output_map.len(),
                input_map.len(),
                tensor_size
            );
            return Err("backbone/decoder tensor size mismatch".into());
        }
        input_map[..tensor_size].copy_from_slice(&output_map[..tensor_size]);
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
) -> Option<u64> {
    let sample = drain_recv(sub_camera, timeout_camera).await?;

    let cam_frame = match info_span!("camera_deserialize")
        .in_scope(|| CameraFrame::from_cdr(sample.payload().to_bytes().to_vec()))
    {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to deserialize CameraFrame: {e:?}");
            return None;
        }
    };
    let timestamp = cam_frame.stamp().to_nanos().unwrap_or(0);

    match convert_camera_frame(img_mgr, &cam_frame, dest).and_then(|()| {
        load_image_into_tensor(camera_input_tensor, dest, Preprocessing::UnsignedNorm)
    }) {
        Ok(_) => Some(timestamp),
        Err(e) => {
            error!("Error loading camera frame into input: {e:?}");
            None
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

/// G2D-convert `frame` into `dest`, which must be RGBA.
#[instrument(skip_all)]
fn convert_camera_frame(
    img_mgr: &ImageManager,
    frame: &CameraFrame<Vec<u8>>,
    dest: &mut Image,
) -> Result<(), FusionError> {
    if dest.format() != RGBA {
        return Err("The format of destination buffer is not RGBA".into());
    }
    let input = Image::try_from(frame)?;
    img_mgr
        .convert(&input, dest, None, Rotation::Rotation0)
        .map_err(|e| format!("Could not g2d convert from {input:?} to {dest:?}: {e:?}"))?;
    trace!("Dest size: {}", dest.size());
    Ok(())
}

/// Copy an RGBA `image` of the tensor's height and width into `tensor`.
#[instrument(skip_all)]
fn load_image_into_tensor(
    tensor: &mut TensorMut,
    image: &mut Image,
    preprocess: Preprocessing,
) -> Result<(), FusionError> {
    if image.height() as usize != tensor.shape()?[1] {
        return Err(
            "The height of the destination buffer is not equal to the height of the tensor".into(),
        );
    }
    if image.width() as usize != tensor.shape()?[2] {
        return Err(
            "The width of the destination buffer is not equal to the width of the tensor".into(),
        );
    }
    if image.format() != RGBA {
        return Err("The format of destination buffer is not RGBA".into());
    }
    const DATA_CHANNELS: usize = 4; // RGBA is 4 channels

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
        image,
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
        if tensor_mapped.len() != tensor_vol || data.len() < tensor_vol {
            return Err(format!(
                "camera tensor size mismatch: dest={}, src={}, needed={}",
                tensor_mapped.len(),
                data.len(),
                tensor_vol
            )
            .into());
        }
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
