# Changelog

All notable changes to EdgeFirst Fusion will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial open source release
- Complete GitHub Actions CI/CD workflows (test, build, SBOM, release)
- GitHub issue templates (bug report, feature request, hardware compatibility)
- Pull request template with comprehensive checklist

### Changed

- Feature-gated DeepView RT support behind `deepviewrt` Cargo feature; default builds are TFLite-only and no longer require `libdeepview-rt.so`
- Migrated repository from Bitbucket to GitHub
- Updated license to Apache-2.0
- Renamed project from Maivin Fusion to EdgeFirst Fusion
- Switched g2d-sys to crates.io v1.2.0 (removed local crate)
- Updated tflitec-sys metadata for open source compliance

### Security

- Added security policy and vulnerability reporting process
