# dino-cli

A Rust CLI for simple vision tasks, with **OpenVINO as its only inference backend**. Inspired by llama-cli's local-model and Hugging Face workflow: `-m`, `-hf` (also `-H`/`--hf`), explicit devices, and machine-readable output. No Python, Node, automatic export, or background service is required at runtime.

## Build and install

```sh
cargo install --path . --locked
```

Or build in the checkout:

```sh
cargo build --release --locked
```

Install the native OpenVINO runtime and the drivers/plugins for your target device. The Rust binary loads OpenVINO at runtime; building and displaying help do not require an OpenVINO installation. Your existing OpenVINO build is usable if it includes the **C API library** (`libopenvino_c.so` on Linux, `openvino_c.dll` on Windows), alongside the runtime and plugins.

For an Intel archive installation:

```sh
source /opt/intel/openvino/setupvars.sh
```

For a custom build, set the shared-library search path or supply the exact C library path:

```sh
dino-cli --ov-library /path/to/libopenvino_c.so devices
```

Versioned names such as `libopenvino_c.so.2631` also work with this flag. Dependencies of that library must also be discoverable by the OS loader. Windows users can run the OpenVINO setup script or put the runtime DLL directory on `PATH`. Windows and GPU/NPU execution have not yet been tested in this project.

## Commands

```sh
dino-cli devices
```

```sh
dino-cli -m ./model/openvino_model.xml info
```

```sh
dino-cli -m ./model/openvino_model.xml --device NPU compare a.png b.png
```

```sh
dino-cli -hf OWNER/EXPORTED-MODEL --model-file openvino_model.xml --device NPU --size 224x224 --pooling cls compare a.png b.png
```

`OWNER/EXPORTED-MODEL` is a placeholder for a repository containing an exported image encoder, not a claim that a particular hosted model exists. Run `info` first to identify input dimensions and output names. Use `--output-name last_hidden_state` only if that is the exported tensor's actual name. An already pooled `[1,D]` embedding uses the default `--pooling auto`, without `--pooling cls`.

```sh
dino-cli -H OWNER/EXPORTED-MODEL --revision COMMIT download
```

```sh
dino-cli -H OWNER/EXPORTED-MODEL --revision COMMIT --offline --model-file openvino_model.xml --device CPU embed cat.jpg --output cat.f32
```

```sh
dino-cli -m ./model --device GPU batch ./photos --output embeddings.jsonl
```

```sh
dino-cli -m ./model --device CPU --json nearest query.png ./photos -k 10
```

```sh
dino-cli -m ./classifier --device CPU classify cat.png -k 5
```

```sh
dino-cli -m ./model --device CPU infer cat.png
```

| Command | Output |
| --- | --- |
| `devices` | Runtime-visible devices; JSON with `--json` |
| `info` | JSON input/output names, types and dimension bounds; no device compilation |
| `download` | Resolve/cache model and metadata; JSON paths; no runtime required |
| `compare` | Cosine similarity, or a JSON object with `--json` |
| `embed` | JSON embedding, or raw little-endian float32 for an output ending in `.f32` |
| `batch` | Recursive, sorted, sequential image embeddings as JSONL |
| `nearest` | Exact cosine ranking over recursively discovered images; JSON with `--json` |
| `classify` | Top-k softmax probabilities from **logits**, with `config.json` labels if available |
| `infer` | All float32 output tensors with their names, shapes and values, as JSON |

JSON-based commands always produce JSON regardless of `--json`. Batch stops on the first invalid image and may leave a partial JSONL file. Nearest recomputes embeddings each invocation; this version does not persist/search an index. Directory traversal does not follow symlinks. Supported image formats: JPEG, PNG, WebP, BMP and TIFF. `classify` is single-label classification; do not use it on already-softmaxed probabilities or multilabel logits. Similarity is not a calibrated probability or an identity decision.

## Model compatibility

OpenVINO IR (`.xml` + `.bin`) and ONNX are accepted. Hugging Face resolution prefers root `openvino_model.xml`, otherwise a unique IR/ONNX candidate; use `--model-file` for ambiguous or nested exports. `HF_HOME`, `HF_TOKEN`, revisions and the standard Hub cache are handled by `hf-hub`. Pin a commit with `--revision` for reproducibility. `--offline` performs no Hub network calls and requires cached files; nested models should specify `--model-file` explicitly.

IR weights are downloaded automatically. For ONNX external data, explicitly download the repository-relative filenames with repeated `--extra-file` options. Local ONNX data files must be in the locations referenced by the graph. No Safetensors/PyTorch conversion occurs.

The common image interface is **one float32 RGB/BGR image tensor**, batch 1, rank 4, NCHW or NHWC. Model internals may use FP16 or quantization; input/output interfaces currently must be float32. `--size WIDTHxHEIGHT` specializes dynamic input shapes before device compilation. NPU operator and shape support depends on the installed OpenVINO version and driver. Selecting NPU does not silently fall back to CPU; select `--device CPU` yourself when desired. OpenVINO device strings such as `GPU.0` are passed through.

This covers many exported DINO, ViT, CNN, CLIP **vision-only**, and SigLIP **vision-only** graphs. It is not universal support for every checkpoint in those families: export interfaces and processor semantics must match. Full multimodal graphs needing text/token inputs, generative VLMs, variable-resolution packing, detection decoding/NMS, and segmentation postprocessing are not implemented. `infer` can expose raw floating-point detection/segmentation outputs from compatible single-image graphs.

## Preprocessing and pooling

The CLI reads `preprocessor_config.json` beside a local model or from the Hub repository root. `--preprocessor PATH` overrides it. Supported processor operations are RGB conversion, direct or shortest-edge resize, center crop with zero padding, rescaling and channel normalization. Supported PIL resampling codes: nearest (0), Lanczos (1), bilinear (2), bicubic (3). Rust image resampling is not bit-identical to Pillow/Transformers, so expect small numerical differences. EXIF orientation is not applied automatically. Alpha is discarded. Processors with additional/custom operations need adaptation before claiming equivalent embeddings.

Metadata must produce the model's input dimensions. To explicitly override resize/crop and normalization:

```sh
dino-cli -m model.xml --device CPU --size 224x224 --stretch --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225 --scale 0.003921568627 --pooling cls embed image.png
```

The values above are an ImageNet-style example, not universal defaults. Missing metadata requires all four explicit overrides (`--stretch`, `--mean`, `--std`, `--scale`) to avoid silently guessing the model's input convention. `--layout nhwc` changes tensor layout; `--bgr` swaps channel order before normalization.

| Pooling | Input output-shape | Meaning |
| --- | --- | --- |
| `auto` | `[1,D]` | Already pooled vector; other ranks require an explicit choice |
| `cls` | `[1,T,D]` | First token, for encoders exporting a CLS token |
| `mean` | `[1,T,D]` | Average tokens; `--skip-tokens N` omits CLS/register prefixes |
| `spatial` | `[1,C,H,W]` | Average each channel's spatial map |
| `flatten` | Any nonempty shape | Explicitly flatten all values |

Embedding commands L2-normalize by default; `--no-normalize` preserves the pooled scale. Compare/search always calculate cosine similarity. Multi-output models require an explicit `--output-name` for embedding/classification; `infer` returns all outputs. Do not apply CLS pooling to models without a CLS token, or treat class logits as semantic image embeddings.

## Development and verification

```sh
cargo fmt --check
```

```sh
cargo clippy --locked --all-targets -- -D warnings
```

```sh
cargo test --locked
```

```sh
cargo build --locked
```

```sh
uv run --with openvino --with pillow python tests/smoke.py
```

Python is used only to generate a tiny deterministic OpenVINO test graph and pixel fixtures. The smoke test invokes the actual Rust binary and CPU runtime, checking embeddings, comparison, search, raw inference, classification, metadata, binary output and error paths. CI runs the same checks. These tests do not validate accuracy of downloaded pretrained models or accelerator compatibility.

API references: [Intel OpenVINO Rust bindings](https://github.com/intel/openvino-rs), [Hugging Face Hub Rust client](https://github.com/huggingface/hf-hub).

## NPU compiler troubleshooting

If compilation fails with `Unsupported platform: 'AUTO_DETECT'`, pass the hardware platform explicitly:

```sh
dino-cli -hf Xenova/dinov2-small --model-file onnx/model.onnx --device NPU --npu-platform 3720 --size 224x224 --output-name last_hidden_state --pooling cls compare a.jpg b.jpg
```

`3720` is the target for Meteor Lake (including Core Ultra 7 155H) and Arrow Lake. Other generations use other IDs; consult [Intel's platform table](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md). Setting a target fixes compiler target selection; it does not install drivers or guarantee that the compiled model can execute.

`--npu-compiler-type driver` selects the compiler provided by the NPU driver. `--npu-compiler-type plugin` selects OpenVINO's bundled compiler. `prefer-plugin` requests OpenVINO's preference/fallback policy when supported by the installed version. Neither option is forced by default. These flags set OpenVINO properties before compilation and require an NPU device selection; they do not rely on developer-build environment variables.

The option parsing/property mapping is tested without hardware. NPU compilation and execution still require validation on the target machine.
