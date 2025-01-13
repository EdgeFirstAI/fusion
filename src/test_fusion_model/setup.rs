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
