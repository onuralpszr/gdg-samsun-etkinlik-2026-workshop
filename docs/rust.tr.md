---
title: "Adım 3: Ultralytics Inference ile yüksek hızda çalıştırma"
---

# Adım 3: Ultralytics Inference ile yüksek hızda çalıştırma :crab:

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

Ultralytics Inference, Ultralytics YOLO için geliştirilmiş bir Rust kütüphanesi ve CLI aracıdır. ONNX ve OpenVINO formatındaki modelleri okur, yedi görevin tamamını destekler ve CPU, CUDA, TensorRT, CoreML, OpenVINO gibi pek çok altyapıda çalışır. Tek ve küçük bir uygulama dosyasıdır, çalışırken Python gerektirmez.

## 3.1 CLI aracı

```bash
# her bilgisayar, sadece CPU
cargo install ultralytics-inference

# CUDA ve TensorRT ile NVIDIA GPU
cargo install ultralytics-inference --features "nvidia,video"

# OpenVINO ile Intel CPU, GPU ve NPU, çalışma kütüphanesini pip paketinden bulması gerekir
cargo install ultralytics-inference --features "openvino-native,video"
export LD_LIBRARY_PATH=$(python -c "import openvino, os; print(os.path.dirname(openvino.__file__))")/libs

# CPU üzerinde tek görsel
ultralytics-inference predict --model yolo26n.onnx --source bus.jpg

# bir klasör
ultralytics-inference predict --model yolo26n.onnx --source images/ --batch 4

# NVIDIA GPU üzerinde bir video
ultralytics-inference predict --model yolo26n.onnx --device cuda:0 --source video.mp4

# Intel GPU üzerinde bir video
ultralytics-inference predict --model yolo26n_openvino_model --device intel:gpu --source video.mp4

# Intel NPU üzerinde webcam
ultralytics-inference predict --model yolo26n_openvino_model --device intel:npu --source 0 --show

# istediğiniz görev, model otomatik iner
ultralytics-inference predict --task pose
```

İsterseniz tüm adımları atölye betiğine bırakabilirsiniz:

```bash
# görsel, klasör, video, tüm görevler, Intel cihazları ve Rust uygulaması
./local_inference/run_cli.sh all
CAMERA=50 ./local_inference/run_cli.sh webcam
```

## 3.2 Kütüphaneyi kendi uygulamanızda kullanın

`local_inference/` klasörü, yaklaşık 200 satırda dört farklı modu olan eksiksiz bir Cargo projesidir.

```toml
[dependencies]
ultralytics-inference = { version = "0.0.54", features = ["openvino-native", "video"] }
```

```rust
use ultralytics_inference::{Device, InferenceConfig, Source, SourceIterator, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = InferenceConfig::new().with_confidence(0.25).with_device(Device::IntelGpu);
    let mut model = YOLOModel::load_with_config("yolo26n_openvino_model", config)?;

    // Bellekte her an tek bir kare bulunur, 10 saatlik bir yayın da 10 saniyelik bir klip kadar RAM kullanır
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
# bilgisayarınıza uygun özellikleri seçin: CPU için hiçbiri, "nvidia,video" ya da "openvino,video"
cargo build --release --manifest-path local_inference/Cargo.toml --features "openvino,video"
BIN=local_inference/target/release/gdg-inference

# tek görsel
$BIN image yolo26n.onnx bus.jpg

# klasördeki görselleri gruplar halinde işler
$BIN batch yolo26n.onnx datasets/coco8/images/val 4

# video, webcam ya da RTSP yayını
DEVICE=intel:gpu $BIN video yolo26n_openvino_model video.mp4

# gecikme ölçümü, DEVICE değeri cpu, cuda:0, tensorrt:0, intel:cpu, intel:gpu ya da intel:npu olabilir
DEVICE=intel:npu $BIN bench yolo26n_openvino_model bus.jpg 300
```

## 3.3 Rust ile yedi görevin tamamı

`--task` parametresini verdiğinizde CLI ilgili YOLO26 nano modelini indirir, örnek görseller üzerinde çalıştırır ve sonuçları `runs/<task>/` klasörüne kaydeder. Aşağıdaki tüm görseller atölye bilgisayarında Rust CLI ile, hiç Python kullanılmadan üretildi.

```bash
# yedi görevi tek seferde çalıştırır
./local_inference/run_cli.sh tasks

# tek bir görevi kendi görselinizde çalıştırır
ultralytics-inference predict --task segment --source bus.jpg
```

<div class="gallery" markdown>
<figure markdown>
![Rust ile nesne tespiti](assets/img/rust_detect.jpg)
<figcaption>detect, Python ile aynı kutular</figcaption>
</figure>
<figure markdown>
![Rust ile örnek segmentasyonu](assets/img/rust_segment.jpg)
<figcaption>segment</figcaption>
</figure>
<figure markdown>
![Rust ile anlamsal segmentasyon](assets/img/rust_semantic.jpg)
<figcaption>semantic</figcaption>
</figure>
<figure markdown>
![Rust ile derinlik tahmini](assets/img/rust_depth.jpg)
<figcaption>depth</figcaption>
</figure>
<figure markdown>
![Rust ile sınıflandırma](assets/img/rust_classify.jpg)
<figcaption>classify (minibus 0.53)</figcaption>
</figure>
<figure markdown>
![Rust ile poz tahmini](assets/img/rust_pose.jpg)
<figcaption>pose</figcaption>
</figure>
<figure markdown>
![Rust ile yönlü nesne tespiti](assets/img/rust_obb.jpg)
<figcaption>obb, limandaki tüm gemiler ve rıhtımlar tek tek bulunuyor</figcaption>
</figure>
</div>

## 3.4 ONNX üzerinden doğrudan NVIDIA TensorRT FP16

Rust tarafında TensorRT kullanmak için [Adım 2](export.md) bölümündeki `.engine` dosyasına ihtiyacınız yok. Ultralytics Inference, ONNX modelini ONNX Runtime TensorRT execution provider katmanına verir. Bu katman ilk çalıştırmada bir FP16 motoru derler ve modelin yanındaki `.trt_cache/` klasörüne kaydeder. İlk yükleme yaklaşık bir dakika sürer, sonraki yüklemeler ise saniyeler içinde açılır.

```bash
cargo install ultralytics-inference --features "nvidia,video"
ultralytics-inference predict --model yolo26n.onnx --source bus.jpg --device tensorrt:0 --quantize 16

cargo build --release --manifest-path local_inference/Cargo.toml --features "nvidia,video"

# varsayılan olarak FP16
DEVICE=tensorrt:0 local_inference/target/release/gdg-inference bench yolo26n.onnx bus.jpg 300
```

```rust
let config = InferenceConfig::new().with_device(Device::TensorRt(0)).with_quantize(Quantization::Fp16);
let mut model = YOLOModel::load_with_config("yolo26n.onnx", config)?;
```

!!! info "TensorRT için gerekenler"

    Bir NVIDIA sürücüsünün yanında CUDA 13 runtime, cuDNN 9 ve TensorRT 10 kütüphanelerinin de kütüphane yolunda bulunması gerekir. Hazır ONNX Runtime yalnızca CUDA 13 için derlenmiştir. CLI'a `--rect false` verirseniz her kare aynı boyutta olur ve yalnızca tek bir motor derlenir. TensorRT olmadan sadece CUDA kullanmak isterseniz `--device cuda:0` yazmanız yeterlidir.

!!! tip "Hibrit işlemcilerde CPU thread sayısı"

    Intel hibrit işlemcilerde Rust tarafındaki OpenVINO, kullanacağı thread sayısını varsayılan olarak kendisi belirler. Atölye uygulamasına `THREADS=16` (tüm çekirdekler) verirseniz ya da kodda `.with_threads(16)` kullanırsanız CPU üzerindeki çıkarım hızlanır.

[:material-arrow-right: Sonraki: İleri seviye](advanced.md){ .md-button .md-button--primary .next-step }
