# Architecture

## Overview

Maivin Fusion is a ROS 2 node that performs multi-modal sensor fusion for the EdgeFirst Maivin platform. It combines data from multiple sensors (camera, LiDAR, etc.) to provide enhanced perception capabilities.

## System Architecture

### ROS 2 Node

The fusion node operates as a ROS 2 component with the following responsibilities:

- Subscribe to sensor data topics (camera, LiDAR, IMU, etc.)
- Perform sensor calibration and synchronization
- Execute fusion algorithms
- Publish fused perception results

### Key Components

1. **Sensor Input Processing**
   - Image processing for camera data
   - Point cloud processing for LiDAR data
   - Timestamp synchronization

2. **Fusion Models**
   - TFLite model execution
   - RTM (Runtime) model support
   - Custom fusion algorithms

3. **Object Tracking**
   - Kalman filtering for state estimation
   - Multi-object tracking
   - Track association

4. **Output Generation**
   - Fused perception results
   - Tracking data
   - Performance metrics

## Communication

### Zenoh Integration

The fusion node uses Zenoh for distributed communication, enabling:

- Low-latency data distribution
- Zero-copy shared memory transfers
- Efficient network utilization

### Data Flow

```
Sensors → Fusion Node → Perception Results
   ↓           ↓              ↓
Camera    Processing    Tracking Data
LiDAR     + Fusion      + Metrics
IMU       + Tracking
```

## Performance

### Tracy Profiling

The fusion node includes Tracy profiling support for:

- Real-time performance monitoring
- Bottleneck identification
- Algorithm optimization

### Hardware Acceleration

- GPU acceleration for image processing (G2D)
- DMA buffers for zero-copy transfers
- Optimized for ARM platforms

## Configuration

Configuration is managed through command-line arguments and environment variables. See `args.rs` for available options.

## Future Enhancements

- Additional sensor modalities
- Improved fusion algorithms
- Enhanced tracking capabilities
- Multi-camera support
