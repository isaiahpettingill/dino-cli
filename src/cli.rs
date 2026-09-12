use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
#[derive(Parser, Debug)]
#[command(
    version,
    about = "OpenVINO image embeddings, similarity, search, and classification"
)]
pub struct Args {
    /// Explicit path to libopenvino_c.so / openvino_c.dll
    #[arg(long, global = true)]
    pub ov_library: Option<PathBuf>,
    /// Additional HF files, e.g. ONNX external data; repeat as needed
    #[arg(long, global = true)]
    pub extra_file: Vec<String>,
    #[arg(short = 'm', long, global = true, conflicts_with = "hf")]
    pub model: Option<PathBuf>,
    #[arg(short = 'H', long, global = true)]
    pub hf: Option<String>,
    #[arg(long, global = true, default_value = "main")]
    pub revision: String,
    #[arg(long, global = true)]
    pub model_file: Option<String>,
    #[arg(long, global = true)]
    pub offline: bool,
    #[arg(long, global = true, default_value = "NPU")]
    pub device: String,
    #[arg(long, global = true)]
    pub json: bool,
    /// Explicit Hugging Face-style image processor JSON
    #[arg(long, global = true)]
    pub preprocessor: Option<PathBuf>,
    /// Override spatial size, WIDTHxHEIGHT; also specializes dynamic models
    #[arg(long, global = true)]
    pub size: Option<String>,
    #[arg(long, global = true, value_enum, default_value = "nchw")]
    pub layout: Layout,
    /// Resize directly to input dimensions instead of using processor metadata
    #[arg(long, global = true)]
    pub stretch: bool,
    #[arg(
        long,
        global = true,
        value_delimiter = ',',
        num_args = 1,
        allow_hyphen_values = true
    )]
    pub mean: Option<Vec<f32>>,
    #[arg(
        long,
        global = true,
        value_delimiter = ',',
        num_args = 1,
        allow_hyphen_values = true
    )]
    pub std: Option<Vec<f32>>,
    #[arg(long, global = true)]
    pub scale: Option<f32>,
    #[arg(long, global = true)]
    pub bgr: bool,
    /// Output tensor name (required for ambiguous multi-output models)
    #[arg(long, global = true)]
    pub output_name: Option<String>,
    #[arg(long, global = true, value_enum, default_value = "auto")]
    pub pooling: Pool,
    /// Number of leading tokens to omit with mean pooling (CLS/register tokens)
    #[arg(long, global = true, default_value_t = 0)]
    pub skip_tokens: usize,
    #[arg(long, global = true)]
    pub no_normalize: bool,
    #[command(subcommand)]
    pub command: Command,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Layout {
    Nchw,
    Nhwc,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Pool {
    Auto,
    Cls,
    Mean,
    Spatial,
    Flatten,
}
#[derive(Debug, Subcommand)]
pub enum Command {
    Devices,
    Info,
    Download,
    Compare {
        a: PathBuf,
        b: PathBuf,
    },
    Embed {
        image: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    Batch {
        directory: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    Nearest {
        query: PathBuf,
        directory: PathBuf,
        #[arg(short = 'k', long, default_value_t = 10)]
        top_k: usize,
    },
    Classify {
        image: PathBuf,
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
    },
    Infer {
        image: PathBuf,
    },
}
