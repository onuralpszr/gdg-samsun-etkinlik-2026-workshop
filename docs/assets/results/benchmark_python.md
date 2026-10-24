| Format | Device | Inference (ms) | End to end (ms) | FPS |
|---|---|---:|---:|---:|
| PyTorch | cpu | 14.39 | 15.50 | 65 |
| PyTorch | xpu | 3.53 | 5.74 | 174 |
| ONNX | cpu | 28.38 | 67.14 | 15 |
| OpenVINO | intel:cpu | 14.31 | 15.71 | 64 |
| OpenVINO | intel:gpu | 1.87 | 2.97 | 337 |
| OpenVINO | intel:npu | 3.09 | 4.54 | 220 |
