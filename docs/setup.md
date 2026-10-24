---
title: "Step 0: Prerequisites check"
---

# Step 0: Prerequisites check :white_check_mark:

=== "Linux and macOS"

    ```bash
    git clone https://github.com/onuralpszr/gdg-samsun-etkinlik-2026-workshop.git
    cd gdg-samsun-etkinlik-2026-workshop

    # Intel GPU and NPU laptops (PyTorch XPU plus OpenVINO)
    ./setup_env.sh

    # NVIDIA machines
    ./setup_env.sh cuda

    # Apple silicon (MPS) and any other machine
    ./setup_env.sh cpu

    source .venv/bin/activate
    ```

=== "Windows PowerShell"

    ```powershell
    git clone https://github.com/onuralpszr/gdg-samsun-etkinlik-2026-workshop.git
    cd gdg-samsun-etkinlik-2026-workshop
    uv venv --python 3.13 .venv
    .venv\Scripts\activate

    # or whl/cu130 for NVIDIA, whl/cpu for CPU only
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
    uv pip install -r requirements.txt
    python local_python/00_check_env.py
    ```

=== "Google Colab"

    Open the [workshop notebook in Colab](https://colab.research.google.com/github/onuralpszr/gdg-samsun-etkinlik-2026-workshop/blob/main/notebooks/hybrid_yolo_workshop.ipynb) and pick **Runtime > Change runtime type > T4 GPU**. Nothing to install on your laptop.

`setup_env.sh` ends with `local_python/00_check_env.py`, a short check that prints your PyTorch device, the OpenVINO devices, the Rust CLI and your Ultralytics Platform key. For the full system report run `yolo checks`.

!!! tip "Checklist before we start"

    - [x] Python 3.11 or newer (the Ultralytics Platform integration needs it)
    - [x] `ultralytics` 8.4.120 or newer
    - [x] An Ultralytics Platform account and API key
    - [x] Rust 1.92 or newer for Step 3 (`curl https://sh.rustup.rs -sSf | sh`)
    - [x] FFmpeg development libraries if you want video and webcam in Rust

[:material-arrow-right: Next: Step 1 Train](train.md){ .md-button .md-button--primary .next-step }
