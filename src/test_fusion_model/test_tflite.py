from radarexp import get_validation_preprocessor
from time import monotonic_ns as clock_now
from torch.utils.data import DataLoader
from radarexp import RadarDataset
# from draw import draw_bev
from pathlib import Path
import tflite_runtime.interpreter as tflite # type: ignore
from tqdm import tqdm
# import rerun as rr
import numpy as np
import argparse
import torch
import yaml
import os

input_timings = []
inference_timings = []
output_timings = []


def get_device() -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = "cuda"
        print('CUDA Device:', torch.cuda.get_device_name(),
              'Capabilities:', torch.cuda.get_device_capability())
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    return device


def load_dataset(
        params: dict, dataset_path: str, dataset_group: str, sessions: list) -> DataLoader:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset does not exist: {dataset_path}")

    bev_size = (
        params["model"]["bev_size"]["range"],
        params["model"]["bev_size"]["angle"]
    )

    image_cache = Path('cache')
    ds = RadarDataset(
        dataset_path,
        group=dataset_group,
        preproc=get_validation_preprocessor(
            bev_size=bev_size,
            bev_scale=params['model']['bev_scale'],
            target_size=params['model']['target_size'],
            cartesian=params['model']['cartesian'],
            range_bins=params['model']['range_bins'],
            doppler_bins=params['model']['doppler_bins'],
            radar_normalizer=params['model']['radar_normalizer'],
        ),
        sessions=sessions,
        camera_size=(
            params['model']['camera_size']['width'],
            params['model']['camera_size']['height']),
        image_cache=image_cache if image_cache.exists() else None
    )
    return ds
    ds_test = DataLoader(
        ds,
        batch_size=1,  # Can only process one batch at a time.
        pin_memory=True,
        persistent_workers=True,
        num_workers=4,
        shuffle=False,
        drop_last=True
    )
    return ds_test


def inference(
    model: tflite.Interpreter,
    params: dict,
    image_tensor: torch.Tensor,
    radar_tensor: torch.Tensor,
    store_timings: bool = True
):
    
    if image_tensor is not None and isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.cpu().numpy()
    if radar_tensor is not None and isinstance(radar_tensor, torch.Tensor):
        radar_tensor = radar_tensor.cpu().numpy()

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Fusion Model
    if len(input_details) > 1:
        # Measure the input time.
        image_tensor = np.expand_dims(image_tensor, axis=0)
        radar_tensor = np.expand_dims(radar_tensor, axis=0)
        start = clock_now()
        model.set_tensor(
            input_details[0]['index'], image_tensor)
        model.set_tensor(
            input_details[1]['index'], radar_tensor)
        input_ns = clock_now() - start

        # Measure the inference time.
        start = clock_now()
        model.invoke()
        infer_ns = clock_now() - start

    else:
        # Camera Only Model.
        if list(input_details[0]["shape"][1:3]) == [
                params["model"]["camera_size"]["height"],
                params["model"]["camera_size"]["width"]]:

            # Measure the input time.
            start = clock_now()
            model.set_tensor(
                input_details[0]['index'], image_tensor)
            input_ns = clock_now() - start

            # Measure the inference time.
            start = clock_now()
            model.invoke()
            infer_ns = clock_now() - start

        # Radar Only Model.
        else:
            # Measure the input time.
            start = clock_now()
            model.set_tensor(
                input_details[0]['index'], radar_tensor)
            input_ns = clock_now() - start

            # Measure the inference time.
            start = clock_now()
            model.invoke()
            infer_ns = clock_now() - start

    # Single output from the model.
    # Measure the output time.
    start = clock_now()
    output = output_details[0]
    # is TFLite quantized uint8 model.
    int8 = input_details[0]["dtype"] == np.uint8
    x = model.get_tensor(output["index"])
    if int8:
        scale, zero_point = output["quantization"]
        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
    output_ns = clock_now() - start

    if store_timings:
        input_timings.append(input_ns * 1e-6)
        inference_timings.append(infer_ns * 1e-6)
        output_timings.append(output_ns * 1e-6)

    return x


def warmup(
    model: tflite.Interpreter,
    warmup_iter: int,
):
    times = []
    for _ in range(warmup_iter):
        start = clock_now()
        model.invoke()
        stop = clock_now() - start
        times.append(stop*1e-6)
    print("model warmup took %f ms (%f ms avg)" %
          (np.sum(times), np.average(times)))


def main(args):
    # Read the parameters.
    with open(args.params, 'r') as f:
        params = yaml.full_load(f)

    device = get_device()

    # Load the model.
    ext_delegate = tflite.load_delegate('/usr/lib/libvx_delegate.so', {})
    print("Loading Model...")
    model = tflite.Interpreter(
        model_path=args.model, experimental_delegates=[ext_delegate])
    print("Done Loading Model...")
    model.allocate_tensors()

    print("Warmup...")
    warmup(model, args.warmup)
    
    print("Loading dataset...")
    # Load the dataset.
    ds_test = load_dataset(params, args.dataset, args.group, args.sessions)

    # Run the model for inference.
    # rr.init("radarexp", spawn=False)
    # rr.save('radarexp.rrd')
    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frame = 0
    for camera, radar, target, _ in tqdm(iter(ds_test), total=len(ds_test)):
        # rr.set_time_seconds("stable_time", frame)

        frame += 1

        image_tensor = torch.tensor(camera, device=device)
        radar_tensor = torch.tensor(radar, device=device)

        grid = inference(model, params, image_tensor, radar_tensor)
        print(f"{np.unique(grid)=}")
        if output_dir:
            np.save(os.path.join(output_dir, f"{frame}.npy"), grid)

        grid = np.squeeze(grid, axis=(0, 3))
        grid = np.flipud(grid)

        # rr.log("raw", rr.Image(grid))

        grid = (grid > params["model"]["threshold"]).astype(np.float32)
        #target = np.squeeze(target, axis=(0, 3))
        target = np.squeeze(target, axis=2)
        print(f"{target.shape=}")
        target = np.flipud(target)
        #bev = draw_bev(grid, target)
        # rr.log("bev", rr.Image(bev))

        #camera = rr.Image(camera)
        # rr.log("camera", camera.compress(jpeg_quality=90))

        #radar = radar[0, :, :, 0]
        #radar = np.flipud(radar)
        # rr.log("radar", rr.Tensor(radar))

    if len(input_timings):
        print("Input (ms) [min: %.2f max: %.2f avg: %.2f]" % (
            np.min(input_timings),
            np.max(input_timings),
            np.average(input_timings)
        ))

    if len(inference_timings):
        print("Inference (ms) [min: %.2f max: %.2f avg: %.2f]" % (
            np.min(inference_timings),
            np.max(inference_timings),
            np.average(inference_timings)
        ))

    if len(output_timings):
        print("Output (ms) [min: %.2f max: %.2f avg: %.2f]" % (
            np.min(output_timings),
            np.max(output_timings),
            np.average(output_timings)
        ))


if __name__ == '__main__':

    args = argparse.ArgumentParser(
        prog='RadarExp',
        description="Test Keras"
    )
    args.add_argument('-m', '--model',
                      help="The path to the model to validate.",
                      metavar="model.tflite",
                      type=str,
                      default="dvclive/radarexp-last.tflite"
                      )
    args.add_argument('-d', '--dataset',
                      help="The path to the RadarExp dataset.",
                      type=str,
                      default="dataset/train"
                      )
    args.add_argument('-g', '--group',
                      help="The dataset split group to validate",
                      type=str,
                      default="validation"
                      )
    args.add_argument('-p', '--params',
                      help="The path to the params.yaml file",
                      type=str,
                      default="params.yaml"
                      )
    args.add_argument('-s', '--sessions',
                      help="Specify the sessions in the dataset.",
                      nargs="+"
                      )
    args.add_argument('-w', '--warmup',
                      help="Provide warmup iterations",
                      type=int,
                      default=3
                      )
    args.add_argument('-o', '--output',
                      help="The path to store the model outputs.",
                      type=str,
                      )
    args = args.parse_args()

    main(args)
