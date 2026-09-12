use crate::cli::Args;
use anyhow::{bail, ensure, Context, Result};
use hf_hub::{api::sync::ApiBuilder, Cache, Repo, RepoType};
use std::path::PathBuf;
pub struct Files {
    pub model: PathBuf,
    pub processor: Option<PathBuf>,
    pub config: Option<PathBuf>,
}
pub fn resolve(a: &Args) -> Result<Files> {
    if let Some(path) = &a.model {
        let model = if path.is_dir() {
            path.join(a.model_file.as_deref().unwrap_or("openvino_model.xml"))
        } else {
            path.clone()
        };
        ensure!(model.is_file(), "Model does not exist: {}", model.display());
        let dir = model.parent().context("No model parent")?;
        let optional = |name: &str| {
            let p = dir.join(name);
            p.is_file().then_some(p)
        };
        return Ok(Files {
            processor: a
                .preprocessor
                .clone()
                .or_else(|| optional("preprocessor_config.json")),
            config: optional("config.json"),
            model,
        });
    }
    let id =
        a.hf.as_ref()
            .context("Specify -m MODEL or -hf REPO / -H REPO")?;
    let repo = Repo::with_revision(id.clone(), RepoType::Model, a.revision.clone());
    let cache = Cache::from_env().repo(repo.clone());
    let api = if a.offline {
        None
    } else {
        Some({
            let mut builder = ApiBuilder::from_env();
            if let Ok(token) = std::env::var("HF_TOKEN") {
                builder = builder.with_token(Some(token));
            }
            builder.build()?.repo(repo)
        })
    };
    let get = |name: &str| -> Result<PathBuf> {
        ensure!(
            !name.starts_with('/') && !name.split('/').any(|x| x == ".."),
            "Unsafe model filename"
        );
        if let Some(api) = &api {
            Ok(api.get(name)?)
        } else {
            cache
                .get(name)
                .with_context(|| format!("Not cached: {name}"))
        }
    };
    let name = if let Some(n) = &a.model_file {
        n.clone()
    } else {
        let names = if let Some(api) = &api {
            api.info()?
                .siblings
                .into_iter()
                .map(|s| s.rfilename)
                .collect::<Vec<_>>()
        } else {
            vec!["openvino_model.xml".into(), "model.onnx".into()]
        };
        if names.iter().any(|n| n == "openvino_model.xml")
            && (!a.offline || cache.get("openvino_model.xml").is_some())
        {
            "openvino_model.xml".into()
        } else {
            let candidates = names
                .into_iter()
                .filter(|n| n.ends_with(".xml") || n.ends_with(".onnx"))
                .filter(|n| !a.offline || cache.get(n).is_some())
                .collect::<Vec<_>>();
            if candidates.len() != 1 {
                bail!("Expected one IR/ONNX model; found {candidates:?}. Use --model-file. PyTorch/Safetensors require a separate export.");
            }
            candidates[0].clone()
        }
    };
    let model = get(&name)?;
    for name in &a.extra_file {
        get(name)?;
    }
    if name.ends_with(".xml") {
        get(&format!("{}.bin", name.trim_end_matches(".xml")))?;
    }
    // Optional metadata is never downloaded in offline mode. Missing metadata is reported by preprocessing.
    let processor = a
        .preprocessor
        .clone()
        .or_else(|| get("preprocessor_config.json").ok());
    let config = get("config.json").ok();
    Ok(Files {
        model,
        processor,
        config,
    })
}
