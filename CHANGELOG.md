# Changelog

All notable changes to EdgeFirst Fusion will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
