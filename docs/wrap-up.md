---
title: "Wrap up"
---

# Wrap up :tada:

## Mini challenges :trophy:

- [ ] Make the NPU count people walking past the camera with `yolo solutions count model=yolo26n_openvino_model device=intel:npu source=0 show=True`
- [ ] Find the fastest device on your own laptop with `02_export_model.py` and share the table
- [ ] Auto annotate ten photos from your phone and upload them as a dataset on the Ultralytics Platform
- [ ] Track your friends with `yolo track model=yolo26n-pose.pt source=0 show=True`
- [ ] Build the Rust app and beat the Python FPS on your machine

## Troubleshooting :wrench:

??? question "Webcam index 0 does not open"

    Many new laptops expose the camera through an Intel IPU with a v4l2loopback device. List devices with `v4l2-ctl --list-devices` and use the loopback index, for example `CAMERA=50` for `05_webcam.py` and `run_cli.sh`, or `source=50` with the `yolo` CLI. If the `yolo` CLI window freezes after a few seconds on these cameras, use `05_webcam.py`, which reads frames directly with OpenCV.

??? question "OpenVINO does not list intel:gpu or intel:npu"

    Run `python -c "import openvino as ov; print(ov.Core().available_devices)"`. If GPU or NPU is not in the list, install the Intel GPU compute runtime and the NPU driver for your distro and reboot.

??? question "The Rust CLI cannot find libopenvino_c"

    Export `LD_LIBRARY_PATH` to the `libs` folder of the `openvino` pip package as shown in [Step 3.1](rust.md#31-the-cli). `run_cli.sh` does it for you.

??? question "PyTorch does not see my GPU"

    Run `python local_python/00_check_env.py` to see what PyTorch finds. NVIDIA cards need the CUDA build from `./setup_env.sh cuda` and a recent driver. Intel GPUs need the XPU build from `./setup_env.sh` and an up to date Intel GPU driver. Apple silicon gets MPS from the `./setup_env.sh cpu` build.

??? question "Training does not show up on the Ultralytics Platform"

    You need Python 3.11 or newer, `ultralytics` 8.4.120 or newer, a valid key from `yolo login`, and a `project` in the `username/project` form.

## Stay in touch :handshake:

<div class="qr" markdown>
<div markdown>
![Workshop page QR](assets/img/qr-workshop.svg)

**This workshop page**
</div>
<div markdown>
![Ultralytics Platform QR](assets/img/qr-platform.svg)

**Join the Ultralytics Platform**
</div>
</div>

- :rocket: [Ultralytics Platform](https://platform.ultralytics.com) and its [docs](https://docs.ultralytics.com/platform/)
- :crab: [Ultralytics Inference on GitHub](https://github.com/ultralytics/inference) and [crates.io](https://crates.io/crates/ultralytics-inference)
- :books: [Ultralytics YOLO docs](https://docs.ultralytics.com)
- :speech_balloon: [Ultralytics Discord](https://discord.com/invite/ultralytics)

No GPU was left behind today. Teşekkürler Samsun! :tada:

[:material-home: Back to the start](index.md){ .md-button .md-button--primary .next-step }
