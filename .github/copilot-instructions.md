# EdgeFirst Fusion - Development Instructions

These instructions apply to all AI-assisted development in this repository.

## Coding Standards

Follow the **Microsoft Pragmatic Rust Guidelines** for all Rust code:

- Prefer `Result` / `Option` combinators over `.unwrap()` in production paths
- Avoid panics in non-test code; use proper error propagation
- Keep functions focused and avoid excessive argument counts
- Use meaningful names following Rust naming conventions (snake_case for
  functions/variables, CamelCase for types)
- All public APIs must have doc comments
- All new source files must include the SPDX header:
  ```rust
  // Copyright 2025 Au-Zone Technologies Inc.
  // SPDX-License-Identifier: Apache-2.0
  ```

## Pre-Commit Checks

Before every commit, run:

```bash
make format lint check sbom
```

This runs `cargo fmt`, `cargo clippy -- -D warnings`, `cargo check`, and
validates the SBOM/NOTICE file. All four must pass before committing.

## Dependency Management

When modifying `Cargo.toml` (adding, removing, or updating dependencies):

1. Run `cargo update` to refresh the lock file
2. Run `cargo sbom` (or `.github/scripts/generate_sbom.sh`) to regenerate
   the SBOM
3. Verify the `NOTICE` file is up-to-date with first-level dependency
   attributions
4. Commit `Cargo.lock` and `NOTICE` together with the `Cargo.toml` change

Only permissive licenses are allowed: MIT, Apache-2.0, BSD-2-Clause,
BSD-3-Clause, ISC, Unicode-3.0, Zlib. No GPL/AGPL/LGPL dependencies.

## Cross-Compilation

Cross-compilation for ARM64 targets uses `cargo zigbuild`:

```bash
cargo zigbuild --release --target aarch64-unknown-linux-gnu
```

Do **not** use plain `cargo build --target` for cross-compilation as it
requires a full cross-toolchain. `cargo zigbuild` handles the sysroot and
linker automatically.

## Commit Conventions

- All commits must be signed off (`git commit -s`) for DCO compliance
- Use conventional commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`,
  `test:`, `chore:`
- Keep the first line under 50 characters, wrap body at 72

## Testing

- Run `cargo test` before committing
- New functionality requires unit tests (minimum 70% coverage)
- CI runs `cargo clippy -- -D warnings` — all warnings are errors
