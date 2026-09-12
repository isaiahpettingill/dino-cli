use crate::cli::{Args, Layout};
use anyhow::{ensure, Context, Result};
use image::{
    imageops::{self, FilterType},
    RgbImage,
};
use serde_json::Value;
use std::path::Path;
pub struct Processor {
    pub width: u32,
    pub height: u32,
    pub metadata: Value,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub scale: f32,
    pub layout: Layout,
    pub stretch: bool,
    pub bgr: bool,
}
impl Processor {
    pub fn new(a: &Args, path: Option<&Path>, width: u32, height: u32) -> Result<Self> {
        let metadata: Value = match path {
            Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
            None => Value::Null,
        };
        ensure!(!metadata.is_null() || (a.mean.is_some() && a.std.is_some() && a.scale.is_some() && a.stretch), "No processor metadata: supply --preprocessor or explicit --mean R,G,B --std R,G,B --scale N --stretch");
        let triple = |key: &str, ov: &Option<Vec<f32>>, default: f32| -> Result<[f32; 3]> {
            let v = ov
                .clone()
                .or_else(|| {
                    metadata[key].as_array().map(|v| {
                        v.iter()
                            .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                            .collect()
                    })
                })
                .unwrap_or(vec![default; 3]);
            ensure!(
                v.len() == 3 && v.iter().all(|x| x.is_finite()),
                "Invalid {key}"
            );
            Ok([v[0], v[1], v[2]])
        };
        let normalize = metadata["do_normalize"].as_bool().unwrap_or(true);
        let mean = if normalize || a.mean.is_some() {
            triple("image_mean", &a.mean, 0.)?
        } else {
            [0.; 3]
        };
        let std = if normalize || a.std.is_some() {
            triple("image_std", &a.std, 1.)?
        } else {
            [1.; 3]
        };
        ensure!(
            std.iter().all(|x| *x > 0.),
            "Standard deviations must be positive"
        );
        let scale = a
            .scale
            .unwrap_or(if metadata["do_rescale"].as_bool() == Some(false) {
                1.
            } else {
                metadata["rescale_factor"].as_f64().unwrap_or(1. / 255.) as f32
            });
        ensure!(
            scale.is_finite() && width > 0 && height > 0 && width <= 16384 && height <= 16384,
            "Invalid scale or image dimensions"
        );
        Ok(Self {
            width,
            height,
            metadata,
            mean,
            std,
            scale,
            layout: a.layout,
            stretch: a.stretch,
            bgr: a.bgr,
        })
    }
    pub fn pixels(&self, path: &Path) -> Result<Vec<f32>> {
        let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
        reader.limits(image::Limits::default());
        let mut im = reader
            .decode()
            .with_context(|| format!("Decoding {}", path.display()))?
            .to_rgb8();
        let filter = match self.metadata["resample"].as_u64().unwrap_or(3) {
            0 => FilterType::Nearest,
            1 => FilterType::Lanczos3,
            2 => FilterType::Triangle,
            3 => FilterType::CatmullRom,
            n => anyhow::bail!("Unsupported PIL resample code {n}"),
        };
        if self.stretch {
            im = imageops::resize(&im, self.width, self.height, filter);
        } else {
            if self.metadata["do_resize"].as_bool().unwrap_or(true) {
                let size = &self.metadata["size"];
                let short = size["shortest_edge"].as_u64().or_else(|| size.as_u64());
                let (w, h) = if let Some(s) = short {
                    let ratio = s as f64 / im.width().min(im.height()) as f64;
                    (
                        (im.width() as f64 * ratio).floor() as u32,
                        (im.height() as f64 * ratio).floor() as u32,
                    )
                } else {
                    (
                        size["width"]
                            .as_u64()
                            .context("Processor size.width missing; use --stretch")?
                            as u32,
                        size["height"]
                            .as_u64()
                            .context("Processor size.height missing")?
                            as u32,
                    )
                };
                ensure!(
                    w > 0 && h > 0 && w <= 16384 && h <= 16384,
                    "Invalid resize dimensions"
                );
                im = imageops::resize(&im, w, h, filter);
            }
            if self.metadata["do_center_crop"].as_bool().unwrap_or(false) {
                let crop = &self.metadata["crop_size"];
                let w = crop["width"]
                    .as_u64()
                    .or_else(|| crop.as_u64())
                    .unwrap_or(self.width as u64) as u32;
                let h = crop["height"]
                    .as_u64()
                    .or_else(|| crop.as_u64())
                    .unwrap_or(self.height as u64) as u32;
                ensure!(
                    w > 0 && h > 0 && w <= 16384 && h <= 16384,
                    "Invalid crop size"
                );
                if im.width() < w || im.height() < h {
                    let mut padded = RgbImage::new(w.max(im.width()), h.max(im.height()));
                    let x = (padded.width() - im.width()) / 2;
                    let y = (padded.height() - im.height()) / 2;
                    imageops::replace(&mut padded, &im, x as i64, y as i64);
                    im = padded;
                }
                let x = (im.width() - w) / 2;
                let y = (im.height() - h) / 2;
                im = imageops::crop_imm(&im, x, y, w, h).to_image();
            }
        }
        ensure!(
            im.dimensions() == (self.width, self.height),
            "Processor produced {:?}, model expects {}x{}; check metadata or use --stretch",
            im.dimensions(),
            self.width,
            self.height
        );
        let area = (self.width * self.height) as usize;
        let mut out = vec![0.; area * 3];
        for (i, p) in im.pixels().enumerate() {
            for c in 0..3 {
                let channel = if self.bgr { 2 - c } else { c };
                let v = (p[channel] as f32 * self.scale - self.mean[c]) / self.std[c];
                let j = match self.layout {
                    Layout::Nchw => c * area + i,
                    Layout::Nhwc => i * 3 + c,
                };
                out[j] = v;
            }
        }
        Ok(out)
    }
}
