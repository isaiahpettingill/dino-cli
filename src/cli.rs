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
    /// Explicit NPU platform (e.g. 3720 for Meteor Lake / Arrow Lake)
    #[arg(long, global = true)]
    pub npu_platform: Option<String>,
    /// Select OpenVINO's plugin compiler or the compiler in the NPU driver
    #[arg(long, global = true, value_enum)]
    pub npu_compiler_type: Option<NpuCompilerType>,
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

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum NpuCompilerType {
    Driver,
    Plugin,
    PreferPlugin,
}
impl NpuCompilerType {
    pub fn property_value(self) -> &'static str {
        match self {
            Self::Driver => "DRIVER",
            Self::Plugin => "PLUGIN",
            Self::PreferPlugin => "PREFER_PLUGIN",
        }
    }
}
impl Args {
    pub fn npu_properties(&self) -> anyhow::Result<Vec<(&'static str, String)>> {
        let mut properties = Vec::new();
        if self.npu_platform.is_some() || self.npu_compiler_type.is_some() {
            anyhow::ensure!(
                self.device == "NPU" || self.device.starts_with("NPU."),
                "--npu-platform and --npu-compiler-type require --device NPU or NPU.<index>"
            );
        }
        if let Some(platform) = &self.npu_platform {
            anyhow::ensure!(
                !platform.is_empty() && platform.bytes().all(|b| b.is_ascii_digit()),
                "--npu-platform must be a numeric platform ID, such as 3720"
            );
            properties.push(("NPU_PLATFORM", platform.clone()));
        }
        if let Some(compiler) = self.npu_compiler_type {
            properties.push(("NPU_COMPILER_TYPE", compiler.property_value().to_string()));
        }
        Ok(properties)
    }
}
#[cfg(test)]
mod npu_tests {
    use super::*;
    #[test]
    fn explicit_npu_configuration() {
        let a = Args::try_parse_from([
            "dino-cli",
            "--npu-platform",
            "3720",
            "--npu-compiler-type",
            "driver",
            "devices",
        ])
        .unwrap();
        assert_eq!(
            a.npu_properties().unwrap(),
            vec![
                ("NPU_PLATFORM", "3720".into()),
                ("NPU_COMPILER_TYPE", "DRIVER".into())
            ]
        );
    }
    #[test]
    fn defaults_do_not_override_openvino() {
        let a = Args::try_parse_from(["dino-cli", "devices"]).unwrap();
        assert!(a.npu_properties().unwrap().is_empty());
    }
    #[test]
    fn rejects_invalid_target() {
        let a = Args::try_parse_from([
            "dino-cli",
            "--device",
            "CPU",
            "--npu-platform",
            "3720",
            "devices",
        ])
        .unwrap();
        assert!(a.npu_properties().is_err());
        let a = Args::try_parse_from(["dino-cli", "--npu-platform", "invalid", "devices"]).unwrap();
        assert!(a.npu_properties().is_err());
    }
}
