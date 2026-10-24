| Format | Device | Size (MB) | Inference (ms) | End to end (ms) | FPS | Speedup | Objects |
|---|---|---:|---:|---:|---:|---:|---:|
| PyTorch | cpu | 5.5 | 14.6 | 15.7 | 69 | 1.0x | 5 |
| PyTorch | xpu | 5.5 | 3.2 | 5.3 | 312 | 4.6x | 5 |
| ONNX | cpu | 9.9 | 42.4 | 83.6 | 24 | 0.3x | 5 |
| OpenVINO | intel:cpu | 10.1 | 15.2 | 16.7 | 66 | 1.0x | 5 |
| OpenVINO | intel:gpu | 10.1 | 1.9 | 3.0 | 532 | 7.8x | 5 |
| OpenVINO | intel:npu | 10.1 | 3.2 | 4.7 | 316 | 4.6x | 5 |
