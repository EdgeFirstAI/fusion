// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Clone, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// model
    #[arg(short, long, env)]
    pub model: PathBuf,

    #[arg(short, long, env, default_value = "npu")]
    pub engine: String,
}
