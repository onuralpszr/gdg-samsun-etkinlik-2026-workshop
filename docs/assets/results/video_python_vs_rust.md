| Runtime | Engine and device | Frames | FPS | Peak memory |
|---|---|---:|---:|---:|
| Python Ultralytics YOLO `predict(stream=True)` | OpenVINO on intel:gpu | 620 | 283 | 942 MB |
| Rust `gdg-inference video` (Ultralytics Inference crate) | OpenVINO on intel:gpu | 620 | 400 | 284 MB |
