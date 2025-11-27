# Maivin Fusion

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://github.com/EdgeFirstAI/fusion/actions/workflows/build.yml/badge.svg)](https://github.com/EdgeFirstAI/fusion/actions/workflows/build.yml)
[![Test Status](https://github.com/EdgeFirstAI/fusion/actions/workflows/test.yml/badge.svg)](https://github.com/EdgeFirstAI/fusion/actions/workflows/test.yml)

Multi-modal sensor fusion for EdgeFirst Maivin platform.

## Overview

Maivin Fusion is a ROS 2 node that performs multi-modal sensor fusion combining camera, LiDAR, and other sensor data for advanced perception capabilities on the EdgeFirst Maivin platform.

## Features

- Multi-modal sensor fusion
- Real-time processing with Tracy profiling support
- Kalman filtering for object tracking
- Point cloud processing
- Zenoh integration for distributed communication

## Requirements

- Rust 1.70 or later
- ROS 2 Humble or later
- EdgeFirst runtime environment

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run --release
```

## Testing

```bash
cargo test
```

## Documentation

For detailed documentation, visit [EdgeFirst Documentation](https://doc.edgefirst.ai/latest/maivin/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

Copyright 2025 Au-Zone Technologies Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md).
