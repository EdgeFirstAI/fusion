use std::{fs::read, time::Instant};

use clap::Parser;
use deepviewrt::{context::Context, engine::Engine, model};
use env_logger::Env;
use log::{error, info};
use setup::Args;
use tflitec_sys::{delegate::Delegate, TFLiteLib};
mod setup;

const TFLITE_NPU_PATH: &str = "libvx_delegate.so";
const RTM_NPU_PATH: &str = "deepview-rt-openvx.so";
fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let model_data =
        read(args.model.clone()).unwrap_or_else(|_| panic!("Did not find `{:?}` file", args.model));
    info!("Model read from file");

    match args.model.extension() {
        Some(v) if v.to_ascii_lowercase() == "rtm" => {
            run_rtm(&args, model_data);
        }
        Some(v) if v.to_ascii_lowercase() == "tflite" => {
            run_tflite(&args, model_data).unwrap();
        }
        Some(v) => {
            error!("Unknown extension: {:?}", v);
        }
        None => {
            error!("No extension found");
        }
    }
}

fn run_tflite(args: &Args, model_data: Vec<u8>) -> Result<(), String> {
    let tflite = match TFLiteLib::new("libtensorflowlite_c.so") {
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

    let mut interpreter = builder.build(&model).unwrap();
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

fn run_rtm(args: &Args, model_data: Vec<u8>) {
    let engine = if args.engine == "npu" {
        Some(
            Engine::new(RTM_NPU_PATH)
                .unwrap_or_else(|_| panic!("Initializing {RTM_NPU_PATH} engine failed")),
        )
    } else {
        None
    };

    let mut nn_context = Context::new(engine, model::memory_size(&model_data), 4096 * 1024).expect(
        "NNContext init
    failed",
    );
    info!("NNContext initialized");

    nn_context
        .load_model(model_data)
        .expect("Load model failed");
    info!("Model loaded into NNContext");

    for _ in 0..10 {
        let start = Instant::now();
        nn_context.run().expect("Run model failed");
        let elapsed = start.elapsed();
        info!("Model took {:?}", elapsed);
    }
}
