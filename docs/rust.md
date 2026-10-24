---
title: "Step 3: High speed execution with Ultralytics Inference"
---

# Step 3: High speed execution with Ultralytics Inference :crab:

[![Crates.io](https://img.shields.io/crates/v/ultralytics-inference?logo=rust&logoColor=white&label=crates.io&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![docs.rs](https://img.shields.io/docsrs/ultralytics-inference?logo=docs.rs&logoColor=white&label=docs.rs&color=CE422B)](https://docs.rs/ultralytics-inference)
[![Downloads](https://img.shields.io/crates/d/ultralytics-inference?logo=rust&logoColor=white&label=downloads&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![MSRV](https://img.shields.io/crates/msrv/ultralytics-inference?logo=rust&logoColor=white&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![CI](https://github.com/ultralytics/inference/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/inference/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/ultralytics/inference/branch/main/graph/badge.svg)](https://app.codecov.io/github/ultralytics/inference)
[![License](https://img.shields.io/crates/l/ultralytics-inference?label=license&color=blue)](https://github.com/ultralytics/inference/blob/main/LICENSE)
[![dependency status](https://deps.rs/repo/github/ultralytics/inference/status.svg)](https://deps.rs/repo/github/ultralytics/inference)
[![arXiv](https://img.shields.io/badge/arXiv-2606.03748-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.03748)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)

Ultralytics Inference is a Rust library and CLI for Ultralytics YOLO. It reads ONNX and OpenVINO exports, handles all seven tasks, and runs on CPU, CUDA, TensorRT, CoreML, OpenVINO and more. One small binary, no Python at runtime.

## 3.1 The CLI

```bash
# any machine, CPU only
cargo install ultralytics-inference

# NVIDIA GPU with CUDA and TensorRT
cargo install ultralytics-inference --features "nvidia,video"

# Intel CPU, GPU and NPU with OpenVINO, which needs to find the runtime from the pip package
cargo install ultralytics-inference --features "openvino-native,video"
export LD_LIBRARY_PATH=$(python -c "import openvino, os; print(os.path.dirname(openvino.__file__))")/libs

# single image on the CPU
ultralytics-inference predict --model yolo26n.onnx --source bus.jpg

# a folder
ultralytics-inference predict --model yolo26n.onnx --source images/ --batch 4

# a video on an NVIDIA GPU
ultralytics-inference predict --model yolo26n.onnx --device cuda:0 --source video.mp4

# a video on an Intel GPU
ultralytics-inference predict --model yolo26n_openvino_model --device intel:gpu --source video.mp4

# the webcam on an Intel NPU
ultralytics-inference predict --model yolo26n_openvino_model --device intel:npu --source 0 --show

# any task, model auto downloads
ultralytics-inference predict --task pose
```

Or let the workshop script drive it:

```bash
# image, batch, video, all tasks, Intel devices and the Rust app
./local_inference/run_cli.sh all
CAMERA=50 ./local_inference/run_cli.sh webcam
```

## 3.2 The crate in your own app

`local_inference/` is a complete Cargo project with four modes in about 200 lines.

```toml
[dependencies]
ultralytics-inference = { version = "0.0.54", features = ["openvino-native", "video"] }
```

```rust
use ultralytics_inference::{Device, InferenceConfig, Source, SourceIterator, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = InferenceConfig::new().with_confidence(0.25).with_device(Device::IntelGpu);
    let mut model = YOLOModel::load_with_config("yolo26n_openvino_model", config)?;

    // One frame in memory at a time, so a 10 hour stream costs the same RAM as a 10 second clip
    for item in SourceIterator::new(Source::from("video.mp4"))? {
        let (frame, meta) = item?;
        for result in model.predict_image(&frame, meta.path)? {
            println!("{}", result.detection_summary());
        }
    }
    Ok(())
}
```

```bash
# pick the features for your machine: none for CPU, "nvidia,video" or "openvino,video"
cargo build --release --manifest-path local_inference/Cargo.toml --features "openvino,video"
BIN=local_inference/target/release/gdg-inference

# single image
$BIN image yolo26n.onnx bus.jpg

# folder in batches
$BIN batch yolo26n.onnx datasets/coco8/images/val 4

# video, webcam or RTSP
DEVICE=intel:gpu $BIN video yolo26n_openvino_model video.mp4

# latency, DEVICE can be cpu, cuda:0, tensorrt:0, intel:cpu, intel:gpu or intel:npu
DEVICE=intel:npu $BIN bench yolo26n_openvino_model bus.jpg 300
```

## 3.3 All seven tasks in Rust

Pass `--task` and the CLI downloads the matching YOLO26 nano model, runs it on the sample images and saves the annotated results under `runs/<task>/`. Every picture below was drawn by the Rust CLI on the workshop laptop, with no Python involved.

```bash
# all seven tasks in one go
./local_inference/run_cli.sh tasks

# one task on your own image
ultralytics-inference predict --task segment --source bus.jpg
```

<div class="gallery" markdown>
<figure markdown>
![Rust detect](assets/img/rust_detect.jpg)
<figcaption>detect, the same boxes as Python</figcaption>
</figure>
<figure markdown>
![Rust segment](assets/img/rust_segment.jpg)
<figcaption>segment</figcaption>
</figure>
<figure markdown>
![Rust semantic](assets/img/rust_semantic.jpg)
<figcaption>semantic</figcaption>
</figure>
<figure markdown>
![Rust depth](assets/img/rust_depth.jpg)
<figcaption>depth</figcaption>
</figure>
<figure markdown>
![Rust classify](assets/img/rust_classify.jpg)
<figcaption>classify (minibus 0.53)</figcaption>
</figure>
<figure markdown>
![Rust pose](assets/img/rust_pose.jpg)
<figcaption>pose</figcaption>
</figure>
<figure markdown>
![Rust obb](assets/img/rust_obb.jpg)
<figcaption>obb, every ship and harbor in the port</figcaption>
</figure>
</div>

## 3.4 NVIDIA TensorRT FP16 straight from ONNX

You do not need the `.engine` file from [Step 2](export.md) to use TensorRT in Rust. Ultralytics Inference hands the ONNX model to the ONNX Runtime TensorRT execution provider, which builds an FP16 engine on the first run and caches it in `.trt_cache/` next to the model. The first load takes about a minute, and every load after that starts in seconds.

```bash
cargo install ultralytics-inference --features "nvidia,video"
ultralytics-inference predict --model yolo26n.onnx --source bus.jpg --device tensorrt:0 --quantize 16

cargo build --release --manifest-path local_inference/Cargo.toml --features "nvidia,video"

# FP16 by default
DEVICE=tensorrt:0 local_inference/target/release/gdg-inference bench yolo26n.onnx bus.jpg 300
```

```rust
let config = InferenceConfig::new().with_device(Device::TensorRt(0)).with_quantize(Quantization::Fp16);
let mut model = YOLOModel::load_with_config("yolo26n.onnx", config)?;
```

!!! info "What TensorRT needs"

    An NVIDIA driver plus the CUDA 13 runtime, cuDNN 9 and TensorRT 10 on the library path. The prebuilt ONNX Runtime is built for CUDA 13 only. Pass `--rect false` to the CLI so every frame has the same shape and only one engine gets built. Want plain CUDA without TensorRT? Use `--device cuda:0`.

!!! tip "Hybrid laptops and CPU threads"

    On Intel hybrid CPUs, OpenVINO in Rust picks its own thread count by default. Pass `THREADS=16` (all cores) to the workshop app or call `.with_threads(16)` and the CPU path gets faster.

[:material-arrow-right: Next: Wrap up](wrap-up.md){ .md-button .md-button--primary .next-step }
