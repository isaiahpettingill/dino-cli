mod backend;
mod cli;
mod math;
mod preprocess;
mod resolve;
use anyhow::{ensure, Context, Result};
use clap::Parser;
use cli::{Args, Command};
use serde_json::json;
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
};
fn images(dir: &Path) -> Result<Vec<PathBuf>> {
    ensure!(dir.is_dir(), "Not a directory: {}", dir.display());
    let mut v = Vec::new();
    for e in walkdir::WalkDir::new(dir) {
        let e = e?;
        if e.file_type().is_file()
            && e.path()
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| {
                    matches!(
                        s.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tif" | "tiff"
                    )
                })
        {
            v.push(e.into_path());
        }
    }
    v.sort();
    ensure!(!v.is_empty(), "No supported images found");
    Ok(v)
}
fn emit(v: &impl serde::Serialize) -> Result<()> {
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, v)?;
    writeln!(out)?;
    Ok(())
}
fn run(a: Args) -> Result<()> {
    a.npu_properties()?;
    if let Some(path) = &a.ov_library {
        openvino_sys::library::load_from(path)
            .map_err(|e| anyhow::anyhow!("Loading OpenVINO runtime: {e}"))?;
    }
    if matches!(a.command, Command::Devices) {
        let c = backend::core()?;
        let devices = c
            .available_devices()?
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        return if a.json {
            emit(&devices)
        } else {
            for d in devices {
                println!("{d}");
            }
            Ok(())
        };
    }
    let f = resolve::resolve(&a)?;
    if matches!(a.command, Command::Download) {
        return emit(&json!({"model":f.model,"preprocessor":f.processor,"config":f.config}));
    }
    if matches!(a.command, Command::Info) {
        let mut c = backend::core()?;
        let m = backend::model(&mut c, &f)?;
        let describe = |node: openvino::Node| -> Result<serde_json::Value> {
            Ok(
                json!({"name":node.get_name().ok(),"element_type":format!("{:?}",node.get_element_type()?),"shape":node.get_partial_shape()?.get_dimensions().iter().map(|d| json!({"min":d.get_min(),"max":d.get_max()})).collect::<Vec<_>>()}),
            )
        };
        let inputs = (0..m.get_inputs_len()?)
            .map(|i| describe(m.get_input_by_index(i)?))
            .collect::<Result<Vec<_>>>()?;
        let outputs = (0..m.get_outputs_len()?)
            .map(|i| describe(m.get_output_by_index(i)?))
            .collect::<Result<Vec<_>>>()?;
        return emit(
            &json!({"model":f.model,"inputs":inputs,"outputs":outputs,"preprocessor":f.processor}),
        );
    }
    let mut e = backend::Encoder::new(&a, &f)?;
    match &a.command {
        Command::Compare { a: left, b } => {
            let x = e.embed(left, &a)?;
            let y = e.embed(b, &a)?;
            let score = math::cosine(&x, &y)?;
            if a.json {
                emit(&json!({"similarity":score}))?
            } else {
                println!("similarity: {score:.8}");
            }
        }
        Command::Embed { image, output } => {
            let v = e.embed(image, &a)?;
            if let Some(p) = output {
                let mut f = std::fs::File::create(p)?;
                if p.extension().and_then(|s| s.to_str()) == Some("f32") {
                    for x in v {
                        f.write_all(&x.to_le_bytes())?;
                    }
                } else {
                    serde_json::to_writer(&mut f, &json!({"image":image,"embedding":v}))?;
                }
            } else {
                emit(&json!({"image":image,"embedding":v}))?;
            }
        }
        Command::Batch { directory, output } => {
            let mut out: Box<dyn Write> = match output {
                Some(p) => Box::new(std::fs::File::create(p)?),
                None => Box::new(io::stdout().lock()),
            };
            for image in images(directory)? {
                let embedding = e
                    .embed(&image, &a)
                    .with_context(|| format!("Embedding {}", image.display()))?;
                serde_json::to_writer(&mut out, &json!({"image":image,"embedding":embedding}))?;
                writeln!(out)?;
            }
        }
        Command::Nearest {
            query,
            directory,
            top_k,
        } => {
            ensure!(*top_k > 0, "top-k must be positive");
            let q = e.embed(query, &a)?;
            let mut scores = Vec::new();
            for image in images(directory)? {
                let v = e.embed(&image, &a)?;
                scores.push((image, math::cosine(&q, &v)?));
            }
            scores.sort_by(|x, y| y.1.total_cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
            scores.truncate(*top_k);
            if a.json {
                emit(
                    &scores
                        .iter()
                        .map(|(p, s)| json!({"image":p,"similarity":s}))
                        .collect::<Vec<_>>(),
                )?
            } else {
                for (p, s) in scores {
                    println!("{s:.8}\t{}", p.display());
                }
            }
        }
        Command::Infer { image } => emit(&e.infer(image)?)?,
        Command::Classify { image, top_k } => {
            ensure!(*top_k > 0, "top-k must be positive");
            let o = e.selected(image, &a)?;
            ensure!(
                o.shape.len() == 2 && o.shape[0] == 1 && !o.data.is_empty(),
                "Classification requires [1,classes] logits"
            );
            let max = o.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp = o
                .data
                .iter()
                .map(|x| ((*x - max) as f64).exp())
                .collect::<Vec<_>>();
            let sum = exp.iter().sum::<f64>();
            let cfg: serde_json::Value = match &f.config {
                Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
                None => serde_json::Value::Null,
            };
            let mut scores = exp
                .iter()
                .enumerate()
                .map(|(i, v)| (i, v / sum))
                .collect::<Vec<_>>();
            scores.sort_by(|x, y| y.1.total_cmp(&x.1));
            scores.truncate(*top_k);
            emit(&scores.iter().map(|(i,p)|json!({"index":i,"label":cfg["id2label"][i.to_string()],"probability":p})).collect::<Vec<_>>())?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
fn main() {
    let args = std::env::args_os().map(|s| if s == "-hf" { "--hf".into() } else { s });
    if let Err(e) = run(Args::parse_from(args)) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
