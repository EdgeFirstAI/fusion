// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end check that `KEY=""` in the environment behaves as unset.
//!
//! Runs with `harness = false` so this `main` is the only thread in the
//! process when the environment is mutated, which `scrub_empty_env` requires.
#![allow(dead_code)] // args.rs's own #[cfg(test)] unit tests are compiled but never run here

#[path = "../src/args.rs"]
mod args;
use args::{scrub_empty_env, Args, KEEP};
use clap::Parser;

const ARGV: [&str; 1] = ["edgefirst-fusion"];

/// Scrubbed to their defaults: numeric, boolean, and integer arguments.
const SCRUBBED: [&str; 3] = ["MAX_MODEL_AGE", "LOGITS", "THRESHOLD"];

/// In `KEEP`: the empty "leave empty to disable" sentinel must survive.
const KEPT: &str = "LIDAR_OUTPUT_TOPIC";

fn main() {
    for name in SCRUBBED.iter().chain([KEPT].iter()) {
        // SAFETY: single-threaded — this is `main` before any thread is spawned.
        std::env::set_var(name, "");
    }
    let before = Args::try_parse_from(ARGV);
    assert!(
        before.is_err(),
        "empty vars must fail to parse before scrubbing: {before:?}"
    );

    // SAFETY: still single-threaded.
    unsafe { scrub_empty_env::<Args>(KEEP) };
    for name in SCRUBBED {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} should have been removed"
        );
    }
    assert_eq!(
        std::env::var(KEPT).as_deref(),
        Ok(""),
        "{KEPT} is in KEEP and must survive scrubbing"
    );

    let args = Args::try_parse_from(ARGV).expect("defaults must apply after scrubbing");
    assert_eq!(args.max_model_age, 0.5);
    assert!(args.logits);
    assert_eq!(args.threshold, 1);
    assert_eq!(args.lidar_output_topic, "");
    println!("env_scrub: ok");
}
