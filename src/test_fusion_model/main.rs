use std::{fs::read, time::Instant};

use clap::Parser;
use deepviewrt::{context::Context, engine::Engine, model};
use env_logger::Env;
use log::{error, info};
use moonfire_tflite::{delegate::Delegate, Interpreter, Model};
use setup::Args;
mod setup;

const TFLITE_NPU_PATH: &str = "libvx_delegate.so";
const RTM_NPU_PATH: &str = "deepview-rt-openvx.so";
fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let model_data =
        read(args.model.clone()).expect(&format!("Did not find `{:?}` file", args.model));
    info!("Model read from file");

    match args.model.extension() {
        Some(v) if v.to_ascii_lowercase() == "rtm" => {
            run_rtm(&args, model_data);
        }
        Some(v) if v.to_ascii_lowercase() == "tflite" => {
            run_tflite(&args, model_data);
        }
        Some(v) => {
            error!("Unknown extension: {:?}", v);
        }
        None => {
            error!("No extension found");
        }
    }
}

fn run_tflite(args: &Args, model_data: Vec<u8>) {
    let model = Model::from_mem(model_data).unwrap();
    let mut builder = Interpreter::builder();

    if args.engine == "npu" {
        let delegate = Delegate::load_external(TFLITE_NPU_PATH)
            .expect(&format!("Initializing {TFLITE_NPU_PATH} engine failed"));
        builder.add_owned_delegate(delegate);
    }

    let mut interpreter = builder.build(&model).unwrap();
    info!(
        "interpreter with {} inputs, {} outputs",
        interpreter.inputs().len(),
        interpreter.outputs().len()
    );
    let inputs = interpreter.inputs();
    for i in 0..inputs.len() {
        info!("input: {:?}", inputs[i]);
    }
    let outputs = interpreter.outputs();
    for i in 0..outputs.len() {
        info!("output: {:?}", outputs[i]);
    }

    for _ in 0..10 {
        let start = Instant::now();
        interpreter.invoke().expect("Run model failed");
        let elapsed = start.elapsed();
        info!("Model took {:?}", elapsed);
    }
}

fn run_rtm(args: &Args, model_data: Vec<u8>) {
    let engine = if args.engine == "npu" {
        Some(
            Engine::new(RTM_NPU_PATH).expect(&format!("Initializing {RTM_NPU_PATH} engine failed")),
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
