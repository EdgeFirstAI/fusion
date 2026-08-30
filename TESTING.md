# EdgeFirst Fusion - Testing

## Unit Tests

Run unit tests with:

```bash
cargo test
```

The test suite covers:

- **Kalman filter** (`kalman::tests`) - Validates predict/update convergence and Mahalanobis gating distance
- **ByteTrack tracker** (`tracker::tests`) - Verifies track association and Kalman state estimation
- **3D-to-2D projection** (`transform::projection_test`) - Tests camera matrix projection math
- **Model postprocessing** (`fusion_model::swap_axes_test`) - Validates sigmoid/log1p output transforms

## Integration Testing

Fusion is a pub/sub service that requires live sensor data or recorded Zenoh topics. To test end-to-end:

### Prerequisites

1. A running Zenoh router or peer network
2. Sensor publishers (camera, radar, and/or LiDAR) or recorded data playback
3. A fusion model file (`.tflite` or `.rtm`) if testing ML inference

### Primary Topics

**Inputs to provide:**

| Topic | Type | Description |
|-------|------|-------------|
| `radar/clusters` | `sensor_msgs/PointCloud2` | Radar point cloud |
| `lidar/clusters` | `sensor_msgs/PointCloud2` | LiDAR point cloud |
| `camera/frame` | `edgefirst_msgs/CameraFrame` | Camera frame (tensor + DMA-BUF planes) |
| `radar/cube` | `edgefirst_msgs/RadarCube` | Radar cube for ML model |
| `model/mask` | `edgefirst_msgs/Mask` | Segmentation mask |
| `model/boxes2d` | `edgefirst_msgs/Detect` | 2D detection boxes (optional, for instance-level fusion) |
| `camera/info` | `sensor_msgs/CameraInfo` | Camera calibration |
| `tf_static` | `geometry_msgs/TransformStamped` | Coordinate transforms |

**Outputs to observe:**

| Topic | Type | Description |
|-------|------|-------------|
| `fusion/radar` | `sensor_msgs/PointCloud2` | Classified radar point cloud |
| `fusion/occupancy` | `sensor_msgs/PointCloud2` | Occupancy grid |
| `fusion/boxes3d` | `edgefirst_msgs/Detect` | 3D bounding boxes |
| `fusion/model_output` | `edgefirst_msgs/Mask` | ML model predictions |

### Running with a Model

```bash
# TFLite model on NPU
edgefirst-fusion --model model.tflite --engine npu --track

# DeepView RT model (requires --features deepviewrt build)
edgefirst-fusion --model model.rtm --engine npu --track
```

### Verifying Output

Use Zenoh CLI tools to subscribe to output topics and verify data is being published:

```bash
# Check if fusion is publishing classified point clouds
zenoh-cli subscribe "fusion/radar"

# Check occupancy grid output
zenoh-cli subscribe "fusion/occupancy"

# Check 3D bounding boxes
zenoh-cli subscribe "fusion/boxes3d"
```

### Profiling

Enable Tracy profiling to measure per-stage latency:

```bash
edgefirst-fusion --model model.rtm --track --tracy
```

Connect with the [Tracy profiler](https://github.com/wolfpld/tracy) to visualize fusion loop timing, inference latency, and publishing overhead.
