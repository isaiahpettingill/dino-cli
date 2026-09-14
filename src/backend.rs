use crate::{
    cli::{Args, Layout},
    math,
    preprocess::Processor,
    resolve::Files,
};
use anyhow::{ensure, Context, Result};
use openvino::{Core, ElementType, InferRequest, PartialShape, RwPropertyKey, Shape, Tensor};
use serde::Serialize;
use std::path::Path;
#[derive(Serialize)]
pub struct Output {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}
pub struct Encoder {
    request: InferRequest,
    processor: Processor,
    names: Vec<String>,
    shape: Vec<i64>,
}
pub fn core() -> Result<Core> {
    Core::new().context("Cannot load OpenVINO C runtime. Source OpenVINO setupvars.sh or configure its library path (libopenvino_c).")
}
pub fn model(c: &mut Core, f: &Files) -> Result<openvino::Model> {
    let weights = if f.model.extension().and_then(|s| s.to_str()) == Some("xml") {
        f.model.with_extension("bin")
    } else {
        Default::default()
    };
    c.read_model_from_file(
        f.model.to_str().context("Non UTF-8 model path")?,
        weights.to_str().context("Non UTF-8 weights path")?,
    )
    .context("Loading IR/ONNX model; ONNX external data must exist beside the model")
}
impl Encoder {
    pub fn new(a: &Args, f: &Files) -> Result<Self> {
        let mut c = core()?;
        let mut m = model(&mut c, f)?;
        ensure!(m.get_inputs_len()?==1,"This version accepts one image input. Export the vision-only encoder for multimodal models.");
        let node = m.get_input_by_index(0)?;
        ensure!(
            node.get_element_type()? == ElementType::F32,
            "Image input must be f32; export an f32 interface (internal weights can be FP16/INT8)"
        );
        if let Some(size) = &a.size {
            let (w, h) = size
                .split_once('x')
                .context("--size requires WIDTHxHEIGHT")?;
            let (w, h): (i64, i64) = (w.parse()?, h.parse()?);
            ensure!(w > 0 && h > 0 && w <= 16384 && h <= 16384, "Invalid size");
            let dims = match a.layout {
                Layout::Nchw => vec![1, 3, h, w],
                Layout::Nhwc => vec![1, h, w, 3],
            };
            m.reshape_single_input(&PartialShape::new_static(4, &dims)?)?;
        }
        let shape = m
            .get_input_by_index(0)?
            .get_shape()
            .context("Dynamic input: supply --size WIDTHxHEIGHT")?
            .get_dimensions()
            .to_vec();
        ensure!(
            shape.len() == 4 && shape[0] == 1,
            "Expected rank-4 image input with batch 1, got {shape:?}"
        );
        let (ch, h, w) = match a.layout {
            Layout::Nchw => (shape[1], shape[2], shape[3]),
            Layout::Nhwc => (shape[3], shape[1], shape[2]),
        };
        ensure!(ch == 3, "Expected 3 image channels; check --layout");
        let processor = Processor::new(a, f.processor.as_deref(), w.try_into()?, h.try_into()?)?;
        let names = (0..m.get_outputs_len()?)
            .map(|i| {
                Ok(m.get_output_by_index(i)?
                    .get_name()
                    .unwrap_or_else(|_| format!("output_{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        for (key, value) in a.npu_properties()? {
            c.set_property(
                &a.device.as_str().into(),
                &RwPropertyKey::Other(key.into()),
                &value,
            )
            .with_context(|| format!("Setting {key}={value} on {}", a.device))?;
        }
        let mut compiled=c.compile_model(&m,a.device.as_str().into()).with_context(||format!("Cannot compile on {}. Check `devices`; use --device CPU to test CPU explicitly.",a.device))?;
        let request = compiled.create_infer_request()?;
        Ok(Self {
            request,
            processor,
            names,
            shape,
        })
    }
    pub fn infer(&mut self, path: &Path) -> Result<Vec<Output>> {
        let pixels = self.processor.pixels(path)?;
        let mut tensor = Tensor::new(ElementType::F32, &Shape::new(&self.shape)?)?;
        tensor.get_data_mut::<f32>()?.copy_from_slice(&pixels);
        self.request.set_input_tensor(&tensor)?;
        self.request.infer()?;
        self.names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let t = self.request.get_output_tensor_by_index(i)?;
                ensure!(
                    t.get_element_type()? == ElementType::F32,
                    "Output {name} must have an f32 interface"
                );
                let data = t.get_data::<f32>()?.to_vec();
                ensure!(
                    data.iter().all(|v| v.is_finite()),
                    "Nonfinite output {name}"
                );
                Ok(Output {
                    name: name.clone(),
                    shape: t.get_shape()?.get_dimensions().to_vec(),
                    data,
                })
            })
            .collect()
    }
    pub fn selected(&mut self, path: &Path, a: &Args) -> Result<Output> {
        let mut outputs = self.infer(path)?;
        let index = if let Some(name) = &a.output_name {
            outputs
                .iter()
                .position(|o| &o.name == name)
                .with_context(|| format!("Unknown output {name}; inspect info"))?
        } else {
            ensure!(
                outputs.len() == 1,
                "Multiple outputs; select --output-name using info"
            );
            0
        };
        Ok(outputs.swap_remove(index))
    }
    pub fn embed(&mut self, path: &Path, a: &Args) -> Result<Vec<f32>> {
        let o = self.selected(path, a)?;
        let mut v = math::pool(&o.data, &o.shape, a.pooling, a.skip_tokens)?;
        if !a.no_normalize {
            math::normalize(&mut v)?;
        }
        Ok(v)
    }
}
