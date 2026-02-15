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
| `rt/radar/clusters` | `sensor_msgs/PointCloud2` | Radar point cloud |
| `rt/lidar/clusters` | `sensor_msgs/PointCloud2` | LiDAR point cloud |
| `rt/camera/dma` | `edgefirst_msgs/DmaBuffer` | Camera frame (DMA buffer) |
| `rt/radar/cube` | `edgefirst_msgs/RadarCube` | Radar cube for ML model |
| `rt/model/mask` | `edgefirst_msgs/Mask` | Segmentation mask |
| `rt/camera/info` | `sensor_msgs/CameraInfo` | Camera calibration |
| `rt/tf_static` | `geometry_msgs/TransformStamped` | Coordinate transforms |

**Outputs to observe:**

| Topic | Type | Description |
|-------|------|-------------|
| `rt/fusion/radar` | `sensor_msgs/PointCloud2` | Classified radar point cloud |
| `rt/fusion/occupancy` | `sensor_msgs/PointCloud2` | Occupancy grid |
| `rt/fusion/boxes3d` | `edgefirst_msgs/Detect` | 3D bounding boxes |
| `rt/fusion/model_output` | `edgefirst_msgs/Mask` | ML model predictions |

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
zenoh-cli subscribe "rt/fusion/radar"

# Check occupancy grid output
zenoh-cli subscribe "rt/fusion/occupancy"

# Check 3D bounding boxes
zenoh-cli subscribe "rt/fusion/boxes3d"
```

### Profiling

Enable Tracy profiling to measure per-stage latency:

```bash
edgefirst-fusion --model model.rtm --track --tracy
```

Connect with the [Tracy profiler](https://github.com/wolfpld/tracy) to visualize fusion loop timing, inference latency, and publishing overhead.
