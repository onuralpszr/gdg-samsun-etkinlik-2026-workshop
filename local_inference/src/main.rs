// No GPU Left Behind: fast Ultralytics YOLO inference with the `ultralytics-inference` Rust crate.
//
// Four modes, one tiny binary:
//   image  <model> <image>             single image, prints detections and saves an annotated copy
//   batch  <model> <folder> [size]     every image in a folder, processed in small batches
//   video  <model> <source>            video file, webcam index or RTSP stream, one frame in memory at a time
//   bench  <model> <image> [runs]      median latency so you can compare with Python
//
// Pick the hardware with the DEVICE environment variable: cpu, intel:cpu, intel:gpu, intel:npu, cuda:0, coreml
// Cap CPU threads with THREADS, for example THREADS=8 (0 or unset uses every core)

use std::error::Error;
use std::path::Path;
use std::time::Instant;

use image::DynamicImage;
use ultralytics_inference::{Device, InferenceConfig, Results, Source, SourceIterator, YOLOModel};

type AnyResult<T> = Result<T, Box<dyn Error>>;

const USAGE: &str = "Usage:
  gdg-inference image <model> <image>
  gdg-inference batch <model> <folder> [batch_size]
  gdg-inference video <model> <video|webcam_index|rtsp_url>
  gdg-inference bench <model> <image> [runs]

Set DEVICE to cpu, intel:gpu, intel:npu, cuda:0 or coreml (default: cpu)
Set THREADS to cap CPU threads (default: every core)";

fn main() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("{USAGE}");
        std::process::exit(1);
    }
    let (mode, model_path, source) = (args[0].as_str(), &args[1], &args[2]);
    let extra = args.get(3).and_then(|v| v.parse::<usize>().ok());
    let mut model = load(model_path)?;

    match mode {
        "image" => image(&mut model, source),
        "batch" => batch(&mut model, source, extra.unwrap_or(4)),
        "video" => video(&mut model, source),
        "bench" => bench(&mut model, source, extra.unwrap_or(100)),
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(1);
        }
    }
}

/// Load an ONNX file or OpenVINO export folder on the device picked by the DEVICE variable.
fn load(model_path: &str) -> AnyResult<YOLOModel> {
    let device: Device = std::env::var("DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()?;
    // THREADS caps CPU threads, handy on hybrid laptops where the low power cores slow things down
    let threads = std::env::var("THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let config = InferenceConfig::new()
        .with_confidence(0.25)
        .with_device(device)
        .with_threads(threads);
    let mut model = YOLOModel::load_with_config(model_path, config)?;
    model.warmup()?;
    println!(
        "Loaded {model_path} on {} with {} classes at {:?}",
        model.execution_provider(),
        model.num_classes(),
        model.imgsz()
    );
    Ok(model)
}

/// Print one line per object and return how many objects were found.
fn report(result: &Results) -> usize {
    let Some(boxes) = &result.boxes else {
        return 0;
    };
    let (cls, conf) = (boxes.cls(), boxes.conf());
    for i in 0..boxes.len() {
        let id = cls[i] as usize;
        let name = result.names.get(&id).map_or("unknown", String::as_str);
        println!("  {name:<12} {:.2}", conf[i]);
    }
    boxes.len()
}

/// Single image: predict, print and save the annotated picture.
fn image(model: &mut YOLOModel, path: &str) -> AnyResult<()> {
    let results = model.predict(path)?;
    std::fs::create_dir_all("runs/rust")?;
    for result in &results {
        let count = report(result);
        let out = Path::new("runs/rust").join(format!(
            "{}_annotated.jpg",
            Path::new(path)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        ));
        result.save(&out)?;
        println!(
            "{count} objects in {:.1} ms, saved {}",
            result.speed.total(),
            out.display()
        );
    }
    Ok(())
}

/// Folder of images: stream them from disk and run small batches so memory stays tiny.
fn batch(model: &mut YOLOModel, folder: &str, batch_size: usize) -> AnyResult<()> {
    let start = Instant::now();
    let (mut images, mut paths, mut total) = (Vec::with_capacity(batch_size), Vec::new(), 0usize);
    for item in SourceIterator::new(Source::from(folder))? {
        let (img, meta) = item?;
        images.push(img);
        paths.push(meta.path);
        if images.len() == batch_size {
            total += run_batch(model, &images, &paths)?;
            images.clear();
            paths.clear();
        }
    }
    if !images.is_empty() {
        total += run_batch(model, &images, &paths)?;
    }
    println!(
        "Batch done: {total} objects in {:.2} s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Run one batch, save every annotated image and return the object count.
fn run_batch(model: &mut YOLOModel, images: &[DynamicImage], paths: &[String]) -> AnyResult<usize> {
    std::fs::create_dir_all("runs/rust/batch")?;
    let mut count = 0;
    for (results, path) in model.predict_batch(images, paths)?.iter().zip(paths) {
        for result in results {
            let objects = result.boxes.as_ref().map_or(0, |b| b.len());
            let name = Path::new(path).file_name().unwrap_or_default();
            result.save(Path::new("runs/rust/batch").join(name))?;
            println!(
                "{path}: {objects} objects in {:.1} ms",
                result.speed.total()
            );
            count += objects;
        }
    }
    Ok(count)
}

/// Video, webcam or stream: decode one frame, predict, drop it, repeat. Memory never grows.
fn video(model: &mut YOLOModel, source: &str) -> AnyResult<()> {
    let start = Instant::now();
    let (mut frames, mut objects) = (0usize, 0usize);
    for item in SourceIterator::new(Source::from(source))? {
        let (frame, meta) = item?;
        for result in model.predict_image(&frame, meta.path.clone())? {
            objects += result.boxes.as_ref().map_or(0, |b| b.len());
        }
        frames += 1;
        if frames % 30 == 0 {
            let fps = frames as f64 / start.elapsed().as_secs_f64();
            println!("frame {frames:>5}  {fps:6.1} FPS  {objects} objects so far");
        }
    }
    let fps = frames as f64 / start.elapsed().as_secs_f64();
    println!("Video done: {frames} frames, {objects} objects, {fps:.1} FPS end to end");
    Ok(())
}

/// Median latency over many runs on the same image.
fn bench(model: &mut YOLOModel, path: &str, runs: usize) -> AnyResult<()> {
    let (frame, _) = SourceIterator::new(Source::from(path))?
        .next()
        .ok_or("could not read image")??;
    let mut inference = Vec::with_capacity(runs);
    let mut total = Vec::with_capacity(runs);
    for _ in 0..runs {
        let results = model.predict_image(&frame, path.to_string())?;
        let speed = &results[0].speed;
        inference.push(speed.inference.unwrap_or_default());
        total.push(speed.total());
    }
    println!(
        "{runs} runs: inference median {:.2} ms, end to end median {:.2} ms ({:.0} FPS)",
        median(&mut inference),
        median(&mut total),
        1000.0 / median(&mut total)
    );
    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
