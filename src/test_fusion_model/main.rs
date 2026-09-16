// Copyright 2025 Au-Zone Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{fs::read, time::Instant};

use clap::Parser;
use log::{error, info};
use setup::Args;
use tflitec_sys::{delegate::Delegate, TFLiteLib};
mod setup;

const TFLITE_NPU_PATH: &str = "libvx_delegate.so";
fn main() {
    let args = Args::parse();
    let model_data =
        read(args.model.clone()).unwrap_or_else(|_| panic!("Did not find `{:?}` file", args.model));
    info!("Model read from file");

    match args.model.extension() {
        Some(v) if v.eq_ignore_ascii_case("tflite") => {
            run_tflite(&args, model_data).unwrap();
        }
        Some(v) => {
            error!("Unsupported model type {v:?}; fusion models must be TFLite (.tflite)");
        }
        None => {
            error!("No extension found");
        }
    }
}

fn run_tflite(args: &Args, model_data: Vec<u8>) -> Result<(), String> {
    let tflite = match TFLiteLib::new_with_path("libtensorflowlite_c.so") {
        Ok(v) => v,
        Err(e) => {
            return Err(format!("Could not open libtensorflowlite_c.so: {:?}", e));
        }
    };

    let model = tflite.new_model_from_mem(model_data).unwrap();
    let mut builder = tflite.new_interpreter_builder()?;

    if args.engine == "npu" {
        let delegate = Delegate::load_external(TFLITE_NPU_PATH)
            .unwrap_or_else(|_| panic!("Initializing {TFLITE_NPU_PATH} engine failed"));
        builder.add_owned_delegate(delegate);
    }

    let mut interpreter = builder.build(model).unwrap();
    info!(
        "interpreter with {} inputs, {} outputs",
        interpreter.inputs_mut()?.len(),
        interpreter.outputs()?.len()
    );
    let inputs = interpreter.inputs_mut()?;
    for inp in inputs {
        info!("input: {:?}", inp);
    }
    let outputs = interpreter.outputs()?;
    for outp in outputs {
        info!("output: {:?}", outp);
    }

    for _ in 0..10 {
        let start = Instant::now();
        interpreter.invoke().expect("Run model failed");
        let elapsed = start.elapsed();
        info!("Model took {:?}", elapsed);
    }
    Ok(())
}
