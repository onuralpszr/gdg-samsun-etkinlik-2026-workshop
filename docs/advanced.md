---
title: "Advanced: INT8 and every chip at once"
---

# Advanced: INT8 and every chip at once :rocket:

Finished early? These two extras squeeze more out of the laptop you already have.

## INT8: a smaller and faster model

INT8 export stores the weights as 8 bit integers instead of 32 bit floats. Ultralytics YOLO runs a few images from `data` through the model to calibrate it, so pass a dataset that looks like what you will predict on. The OpenVINO model drops from about 10 MB to about 3 MB and the mAP stays within a few points.

```bash
# Intel CPU, GPU and NPU with OpenVINO
yolo export model=yolo26n.pt format=openvino int8=True data=coco8.yaml

# NVIDIA GPU with TensorRT
yolo export model=yolo26n.pt format=engine int8=True data=coco8.yaml

# Apple silicon with CoreML
yolo export model=yolo26n.pt format=coreml int8=True
```

Now check what INT8 costs. Run `yolo val` on the full precision model and on the INT8 model, then compare the mAP50-95 column and the inference time in the `Speed` line.

```bash
# full precision vs INT8 on the same device
yolo val model=yolo26n_openvino_model data=coco8.yaml device=intel:cpu
yolo val model=yolo26n_int8_openvino_model data=coco8.yaml device=intel:cpu

# then try the NPU
yolo val model=yolo26n_int8_openvino_model data=coco8.yaml device=intel:npu

# on NVIDIA
yolo val model=yolo26n.engine data=coco8.yaml device=0
```

The INT8 OpenVINO model runs in Rust as well:

```bash
ultralytics-inference predict --model yolo26n_int8_openvino_model --device intel:npu --source bus.jpg
```

!!! tip "Calibrate on your own data"

    `coco8.yaml` holds only 8 images, which is fine for the demo. For your own model, pass your own dataset YAML as `data` so the calibration sees your real images.

## Every chip at once

A laptop has more than one engine. Open three terminals and give each one its own model and its own device. They all run at the same time, so nothing sits idle.

```bash
# terminal 1: people on the webcam
yolo track model=yolo26n.pt source=0 device=0 show=True

# terminal 2: poses in a video
yolo track model=yolo26n-pose.pt source=video.mp4 device=cpu show=True

# terminal 3: Rust on the CPU, no Python at all
ultralytics-inference predict --model yolo26n.onnx --source video.mp4 --show
```

Swap the models and `device` values to match your machine:

| Machine | Terminal 1 | Terminal 2 | Terminal 3 |
|---|---|---|---|
| NVIDIA | `yolo26n.pt` with `device=0` | `yolo26n-pose.pt` with `device=cpu` | Rust on the CPU |
| Intel Core Ultra | `yolo26n_int8_openvino_model` with `device=intel:npu` | `yolo26n-pose_openvino_model` with `device=intel:gpu` | Rust on the CPU |
| Apple silicon | `yolo26n.pt` with `device=mps` | `yolo26n-pose.pt` with `device=cpu` | Rust on the CPU |
| CPU only | `yolo26n.pt` with `device=cpu` | `yolo26n-pose_openvino_model` with `device=intel:cpu` | Rust on the CPU |

!!! info "One camera, one terminal"

    Most webcams open in only one program at a time. Give the webcam to one terminal and a video file to the others. The `solutions_ci_demo.mp4` sample that `04_track.py` downloads works well.

## Ideas to keep going :bulb:

Some ideas to try after the workshop:

- [ ] Label ten of your own photos with Smart Annotation, train on the Ultralytics Platform, export to INT8 and run the model in Rust.
- [ ] Serve predictions from a small Rust web server built on the `ultralytics-inference` crate.
- [ ] Read an RTSP camera, or several cameras at once, from the Rust app.
- [ ] Put INT8 on the smallest computer you own, such as a Raspberry Pi or an old laptop.
- [ ] Export a segment or pose model to INT8 and check how much accuracy it keeps.

[:material-arrow-right: Next: Wrap up](wrap-up.md){ .md-button .md-button--primary .next-step }
