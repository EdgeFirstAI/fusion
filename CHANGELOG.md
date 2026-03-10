# Changelog

All notable changes to EdgeFirst Fusion will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.7.2] - 2026-03-10

### Fixed

- Removed erroneous base_link transform that was modifying point XYZ coordinates — fusion output now preserves original sensor-frame coordinates unchanged
- bbox3d output `frame_id` now uses the source sensor frame_id instead of `base_link_optical`

## [1.7.1] - 2026-03-10

### Fixed

- Output `frame_id` on `rt/fusion/lidar` and `rt/fusion/radar` now preserves the source sensor frame_id instead of incorrectly using `base_link`

### Changed

- Updated ARCHITECTURE.md to clarify that fusion projection does not modify point coordinates

## [1.7.0] - 2026-03-10

### Added

- Dynamic projection using `tf_static` camera transform — camera extrinsics are now resolved from the TF tree instead of hardcoded
- Deterministic FNV-1a `track_id` hash for stable cross-process track identification
- SoA `FusionFrame` layout replacing `ParsedPoint` AoS for cache-friendly processing
- NEON SIMD optimizations for aarch64: vectorized sincos, atan2, magnitude3, and transform+project kernels
- Per-sensor output topics: `rt/fusion/lidar` and `rt/fusion/radar` replacing shared `rt/fusion/classes`
- Dynamic `track_id(u32)` in late-fusion output — detected at runtime from model output, expands PCD layout from 16 to 20 bytes/point
- Unified `--vision-model-topic` (`rt/model/output`) replacing separate `--mask-topic` and `--boxes2d-topic`
- `--model-info-topic` for dynamic label resolution from model service
- Conditional grid publisher — only created when fusion model is configured

### Changed

- Sensor input topics (`--lidar-pcd-topic`, `--radar-pcd-topic`) now default to empty (disabled); set to `rt/lidar/points`, `rt/lidar/clusters`, etc. to enable each fusion pipeline
- Late-fusion output uses compact `vision_class(u16)` + `instance_id(u16)` layout (16 bytes/point, was 24 bytes/point with u8 + u8 + u32 + cluster_id) for natural alignment and reduced bandwidth
- `--max-model-age` replaces `--max-mask-age` for unified vision model staleness check
- Updated `fusion.default` with new topic names and parameters

### Removed

- `--mask-topic`, `--boxes2d-topic` CLI parameters (replaced by `--vision-model-topic`)
- `--classes-topic`, `--tracks-topic` CLI parameters (replaced by per-sensor `--lidar-output-topic`, `--radar-output-topic`)
- `serialize_tracks` output (tracking data now integrated into per-sensor output)

## [1.6.0] - 2026-02-26

### Added

- `instance_id` field (UINT32) in all fusion output PointCloud2 messages for instance-level object identification
- boxes2d detection fusion: subscribe to `rt/model/boxes2d` topic and use 2D bounding boxes for point classification with per-instance tracking
- `--boxes2d-topic` CLI parameter to configure the boxes2d subscription topic
- `--max-mask-age` CLI parameter to control staleness threshold for mask and boxes2d data
- `fusion.default` systemd EnvironmentFile documenting all configuration options
- Behind-camera projection guard: points behind the camera are explicitly set out-of-bounds instead of producing invalid projections
- Monotonic clock staleness warnings for mask and boxes2d data
- `Args::normalize()` for handling empty-string environment variable overrides
- `insert_standard_fields()` helper in pcd.rs to eliminate triplicated field layout code
- `.github/copilot-instructions.md` with project coding conventions
- `check` and `sbom` targets in Makefile
- Unit tests for `parse_box_label`, `box_fusion_clustered` overlapping boxes, and `box_fusion_no_cluster`

### Fixed

- Replaced `.unwrap()` calls with defensive `match`/`if let` in `grid_nearest_cluster` and `late_fusion_clustered`
- `parse_box_label` now logs a warning on invalid input instead of silently returning 0
- Clippy warnings: extracted `LoadedFrame`/`FusionResult` type aliases for complex return types

### Changed

- Upgraded zenoh dependency from 1.5.1 to 1.7.2
- Updated NOTICE file with current dependency versions
- Updated documentation (README.md, ARCHITECTURE.md, TESTING.md) with boxes2d topic, instance_id field, and max_mask_age parameter
- Release workflow now includes `fusion.default` as a release artifact

## [1.5.0] - 2026-02-14

### Added

- Initial open source release
- Complete GitHub Actions CI/CD workflows (test, build, SBOM, release)
- GitHub issue templates (bug report, feature request, hardware compatibility)
- Pull request template with comprehensive checklist
- TESTING.md with unit test and integration testing guidance
- Occupancy grid section in ARCHITECTURE.md
- `rust-version = "1.82"` in Cargo.toml
- Named constants for magic numbers (`MAX_CLASSIFICATION_DISTANCE`, `UNINITIALIZED_COORD`)
- Convergence assertions in Kalman filter tests

### Fixed

- Critical: `draw_point` was overwriting computed x-coordinate with 0.0
- Critical: `munmap` error check using `> 0` instead of `!= 0`
- Critical: Missing `MAP_FAILED` validation after `mmap` calls
- Critical: `min_dist2`/`min_point_ind` not reset per-box in `grid_nearest_point_no_cluster` and `grid_nearest_cluster`
- Critical: PCD deserialization panicking on malformed input instead of returning error
- Critical: Production `assert_eq!` in `get_3d_bbox` replaced with `debug_assert_eq!`
- Critical: Dropped `JoinHandle` for TF static publisher task now logs errors
- Numerically unstable sigmoid in fusion model post-processing
- Unsafe `libc::memcpy` replaced with safe `copy_from_slice` in TFLite and DeepView RT model code
- Flood fill row boundary checks preventing horizontal neighbor wrap-around
- Box2D width/height calculation had erroneous `/ 2.0`
- `clear_bins` resetting `first_marked` to 0 instead of `u128::MAX`
- Missing `j >= 1` and `j >= 2` guards in `mark_grid_one_column` and `mark_cell_three_column`
- Blocking `thread::sleep` in async context replaced with `tokio::time::sleep`
- Release workflow silently skipping missing build artifacts; now fails on missing artifacts
- Typos: `centriods` -> `centroids`, `represetns` -> `represents`, `indicies` -> `indices`

### Changed

- Feature-gated DeepView RT support behind `deepviewrt` Cargo feature; default builds are TFLite-only and no longer require `libdeepview-rt.so`
- Migrated repository from Bitbucket to GitHub
- Updated license to Apache-2.0
- Renamed project from Maivin Fusion to EdgeFirst Fusion
- Switched g2d-sys to crates.io v1.2.0 (removed local crate)
- Updated tflitec-sys metadata for open source compliance
- Updated edgefirst-schemas to v1.5.3 with CDR serialization migration
- Updated NOTICE file format for SBOM validation
- Rust version badge updated from 1.70 to 1.82 in README
- Documentation links updated from `/maivin/` to `/perception`
- RadarCube type corrected in ARCHITECTURE.md
- Idiomatic `entry().or_default().push()` in `get_cluster_ids`
- Return `&[Tracklet]` instead of `&Vec<Tracklet>` from tracker
- Gated test-only types behind `#[cfg(test)]` in Kalman filter
- Removed `workflow_call` trigger from build.yml to prevent redundant builds on tag push

### Removed

- Dead `_process_dmabuffer` functions and unused imports from model code
- ~100 lines of commented-out test code in fusion_model.rs
- Commented-out debug println statements
- `BoolDefaultTrue` type alias from args

### Security

- Added security policy and vulnerability reporting process
