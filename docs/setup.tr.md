---
title: "Adım 0: Ön koşul kontrolü"
hide:
  - navigation
---

# Adım 0: Ön koşul kontrolü :white_check_mark:

!!! warning "Atölyeden önce evde yapın"

    PyTorch indirmesi birkaç GB tutar ve Rust CLI aracı kaynak koddan derlenir. Bu yüzden ikisi de etkinlik alanındaki internetle birkaç dakikada bitmez. Atölyede yalnızca kontrolü çalıştırırsınız.

    - [ ] Python 3.11 ya da üzeri (Ultralytics Platform entegrasyonu için gerekli)
    - [ ] Depoyu klonlayın ve aşağıda gösterildiği gibi bilgisayarınıza uygun `./setup_env.sh` komutunu çalıştırın
    - [ ] Bir Ultralytics Platform hesabı ve API anahtarı
    - [ ] Rust 1.92 ya da üzeri (`curl https://sh.rustup.rs -sSf | sh`)
    - [ ] [Adım 3](rust.md#31-cli-arac) bölümündeki özelliklerle `cargo install ultralytics-inference`
    - [ ] Rust ile video ve webcam kullanacaksanız FFmpeg geliştirme kütüphaneleri

=== "Linux ve macOS"

    ```bash
    git clone https://github.com/onuralpszr/gdg-samsun-etkinlik-2026-workshop.git
    cd gdg-samsun-etkinlik-2026-workshop

    # Intel GPU ve NPU bulunan dizüstüler (PyTorch XPU ve OpenVINO)
    ./setup_env.sh

    # NVIDIA ekran kartlı bilgisayarlar
    ./setup_env.sh cuda

    # Apple silicon (MPS) ve diğer tüm bilgisayarlar
    ./setup_env.sh cpu

    source .venv/bin/activate
    ```

=== "Windows PowerShell"

    ```powershell
    git clone https://github.com/onuralpszr/gdg-samsun-etkinlik-2026-workshop.git
    cd gdg-samsun-etkinlik-2026-workshop
    uv venv --python 3.13 .venv
    .venv\Scripts\activate

    # NVIDIA için whl/cu130, sadece CPU için whl/cpu
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
    uv pip install -r requirements.txt
    python local_python/00_check_env.py
    ```

=== "Google Colab"

    [Atölye notebook'unu Colab'da açın](https://colab.research.google.com/github/onuralpszr/gdg-samsun-etkinlik-2026-workshop/blob/main/notebooks/hybrid_yolo_workshop.ipynb) ve **Runtime > Change runtime type > T4 GPU** seçeneğini seçin. Bu yolda bilgisayarınıza hiçbir şey kurmanız gerekmez.

`setup_env.sh` en sonda `local_python/00_check_env.py` betiğini çalıştırır. Bu kısa kontrol PyTorch cihazınızı, OpenVINO cihazlarını, Rust CLI aracını ve Ultralytics Platform anahtarınızı gösterir. Ayrıntılı sistem raporu için `yolo checks` komutunu çalıştırabilirsiniz.

[:material-arrow-right: Sonraki: Adım 1 Eğitim](train.md){ .md-button .md-button--primary .next-step }
