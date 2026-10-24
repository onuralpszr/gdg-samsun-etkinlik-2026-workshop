---
title: "Step 0: Prerequisites check"
---

# Step 0: Prerequisites check :white_check_mark:

!!! warning "Do this at home before the workshop"

    The PyTorch download is a few GB and the Rust CLI builds from source, so neither fits in the venue wifi or in a few minutes. At the workshop you only run the check.

    - [ ] Python 3.11 or newer (the Ultralytics Platform integration needs it)
    - [ ] Clone the repo and run `./setup_env.sh` for your machine, as shown below
    - [ ] An Ultralytics Platform account and API key
    - [ ] Rust 1.92 or newer (`curl https://sh.rustup.rs -sSf | sh`)
    - [ ] `cargo install ultralytics-inference` with the features for your machine from [Step 3](rust.md#31-the-cli)
    - [ ] FFmpeg development libraries if you want video and webcam in Rust

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

[:material-arrow-right: Next: Step 1 Train](train.md){ .md-button .md-button--primary .next-step }
