---
name: Hardware Compatibility Report
about: Report compatibility results for specific hardware platforms or sensors
title: '[HARDWARE] '
labels: hardware, compatibility
assignees: ''
---

## Hardware Platform

**Board/SoM:**
- Manufacturer: [e.g., NXP, Custom]
- Model: [e.g., i.MX8M Plus EVK, Maivin 2.0]
- CPU: [e.g., ARM Cortex-A53, x86_64]
- NPU: [e.g., NXP eIQ, none]
- RAM: [e.g., 2GB, 4GB, 8GB]

**Sensors:**
- Camera: [e.g., MIPI CSI-2 1080p, USB UVC]
- Radar: [e.g., SmartMicro DRVEGRD-171, TI AWR1843]
- LiDAR: [e.g., Ouster OS1-64, Livox Mid-360]

**Operating System:**
- Distribution: [e.g., Yocto Kirkstone, Ubuntu 22.04]
- Kernel version: [e.g., 5.15.52, 6.1.0]
- Rust version: [e.g., 1.90.0]

## Test Results

**edgefirst-fusion version:** [e.g., 0.1.0, commit SHA]

**Command tested:**
```bash
edgefirst-fusion --model model.rtm --track --engine npu
```

**Results:**
- [ ] Builds successfully
- [ ] Runs without errors
- [ ] Camera DMA buffer reception works
- [ ] Radar point cloud processing works
- [ ] LiDAR point cloud processing works
- [ ] ML model inference works (TFLite / DeepView RT)
- [ ] Object tracking works
- [ ] Occupancy grid generation works
- [ ] ⚠️ Partial functionality (see notes)
- [ ] Does not work (see logs)

## Performance Metrics

**Processing Rate:**
- Fusion loop: ___ FPS
- Model inference: ___ ms per frame
- Point cloud processing: ___ ms

**Resource Usage:**
- CPU usage: ___% (single core)
- Memory footprint: ___ MB
- NPU utilization: ___% (if applicable)

## Known Issues

List any issues, workarounds, or limitations discovered on this platform.

**Example:**
> TFLite model inference falls back to CPU when NPU delegate library is not found. Works with DeepView RT engine instead.

## Logs

<details>
<summary>Full logs (click to expand)</summary>

```
# Paste journalctl or stderr output
```

</details>

## Hardware Acceleration

**G2D (NXP platforms):**
- [ ] Available and working
- [ ] ⚠️ Available but issues (see notes)
- [ ] Not available on this platform
- [ ] Not tested

**NPU / ML Accelerator:**
- [ ] Available and working
- [ ] ⚠️ Available but issues (see notes)
- [ ] Not available on this platform
- [ ] Not tested

## Additional Context

Any other details about hardware-specific behavior, configuration requirements, or platform quirks.

## Checklist

- [ ] I have tested with the latest version
- [ ] I have included all hardware details
- [ ] I have provided performance metrics
- [ ] I have attached relevant logs
- [ ] I confirm this report is for EdgeFirst Fusion compatibility (not a bug report)
