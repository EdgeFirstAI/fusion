// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end check that `KEY=""` in the environment behaves as unset.
//!
//! Runs with `harness = false` so this `main` is the only thread in the
//! process when the environment is mutated, which `scrub_empty_env` requires.
//!
//! `main` speaks the small subset of the libtest CLI that `cargo test` and
//! `cargo nextest` use to enumerate (`--list --format terse`) and select
//! (`--exact <name>`, `--skip <pattern>`, `--ignored`, positional filters)
//! tests, so the target
//! is discovered and reported like any other test instead of failing
//! nextest's `<name>: test` listing check.
#![allow(dead_code)] // args.rs's own #[cfg(test)] unit tests are compiled but never run here

#[path = "../src/args.rs"]
mod args;
use args::{scrub_empty_env, Args, KEEP};
use clap::Parser;

/// The single test this binary provides, as reported to the harness.
const TEST_NAME: &str = "empty_env_is_treated_as_unset";

const ARGV: [&str; 1] = ["edgefirst-fusion"];

/// Scrubbed to their defaults: numeric, boolean, and integer arguments.
const SCRUBBED: [&str; 3] = ["MAX_MODEL_AGE", "LOGITS", "THRESHOLD"];

/// In `KEEP`: the empty "leave empty to disable" sentinel must survive.
const KEPT: &str = "LIDAR_OUTPUT_TOPIC";

/// libtest flags that consume the following argument, so it is not a filter.
const VALUE_FLAGS: [&str; 5] = [
    "--test-threads",
    "--format",
    "--logfile",
    "--color",
    "--shuffle-seed",
];

/// What the harness asked this binary to do.
struct Request {
    list: bool,
    ignored: bool,
    exact: bool,
    filters: Vec<String>,
    skips: Vec<String>,
}

fn parse_request(argv: impl IntoIterator<Item = String>) -> Request {
    let mut req = Request {
        list: false,
        ignored: false,
        exact: false,
        filters: Vec::new(),
        skips: Vec::new(),
    };
    let mut argv = argv.into_iter();
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--list" => req.list = true,
            "--ignored" => req.ignored = true,
            "--exact" => req.exact = true,
            "--skip" => req.skips.extend(argv.next()),
            flag if VALUE_FLAGS.contains(&flag) => {
                argv.next();
            }
            flag if flag.starts_with('-') => {
                if let Some(pattern) = flag.strip_prefix("--skip=") {
                    req.skips.push(pattern.to_owned());
                }
            }
            filter => req.filters.push(filter.to_owned()),
        }
    }
    req
}

/// libtest matching: substring by default, equality under `--exact`.
fn matches(req: &Request, pattern: &str) -> bool {
    if req.exact {
        pattern == TEST_NAME
    } else {
        TEST_NAME.contains(pattern)
    }
}

/// Selected when it passes the positional filter (if any) and no `--skip`
/// pattern matches.
fn selected(req: &Request) -> bool {
    let wanted = req.filters.is_empty() || req.filters.iter().any(|f| matches(req, f));
    wanted && !req.skips.iter().any(|s| matches(req, s))
}

fn main() {
    let req = parse_request(std::env::args().skip(1));
    // This binary has no #[ignore]d tests, so `--ignored` selects nothing.
    if req.list {
        if !req.ignored && selected(&req) {
            println!("{TEST_NAME}: test");
        }
        return;
    }
    if req.ignored || !selected(&req) {
        return;
    }

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
