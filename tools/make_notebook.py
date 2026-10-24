"""Build notebooks/hybrid_yolo_workshop.ipynb from the cells below.

Usage:
    python tools/make_notebook.py
"""

import sys

import nbformat as nbf

REPO = "onuralpszr/gdg-samsun-etkinlik-2026-workshop"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/hybrid_yolo_workshop.ipynb"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(f"""
<a href="{COLAB}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# No GPU Left Behind 🚀
## Hybrid Ultralytics YOLO Training and Fast Execution

**GDG Samsun DevFest 2026 workshop notebook**

Welcome! In the next hour you will:

1. 🔑 Connect to the **Ultralytics Platform** with an API key
2. 🏋️ Train `yolo26n.pt` on `coco8.yaml` while the Platform shows your metrics live
3. 🎨 Try all seven Ultralytics YOLO26 tasks plus tracking and auto annotation
4. 📦 Export to **ONNX**, **OpenVINO** and **TensorRT**
5. 🏎️ Race PyTorch against every exported format
6. 🦀 Run the same model from Rust with **Ultralytics Inference** (`ultralytics-inference`)

> 💡 In Colab pick **Runtime > Change runtime type > T4 GPU** for TensorRT. A CPU runtime works for everything else.
""")

md("""
## 0. Install 📥

One `pip install` gives you Ultralytics YOLO plus the export toolchain. The Ultralytics Platform client ships inside the `ultralytics` package.
""")
code("""
!pip install -q ultralytics openvino onnx onnxslim onnxruntime
""")
code("""
import ultralytics

ultralytics.checks()
""")

md("""
## 1. Connect to the Ultralytics Platform 🔑

1. Sign up at [platform.ultralytics.com](https://platform.ultralytics.com) (new accounts get free credits)
2. Open **Settings > API Keys** and create a key with the training scope
3. In Colab click the 🔑 **Secrets** icon on the left, add `ULTRALYTICS_API_KEY` and enable notebook access

No key? No stress. Everything below still runs locally and the training cell just skips the live dashboard.
""")
code("""
import os

from ultralytics import Platform
from ultralytics.utils import PLATFORM_URL

try:
    from google.colab import userdata

    os.environ["ULTRALYTICS_API_KEY"] = userdata.get("ULTRALYTICS_API_KEY") or ""
except Exception:
    pass  # outside Colab we read ULTRALYTICS_API_KEY from the environment

USERNAME = None
if os.getenv("ULTRALYTICS_API_KEY"):
    try:
        with Platform(api_key=os.environ["ULTRALYTICS_API_KEY"], base_url=PLATFORM_URL) as platform:
            me = platform.account.summary()
        USERNAME = me["username"]
        print(f"✅ Logged in as {USERNAME} on the {me['plan']} plan with ${me['creditsCents'] / 100:.2f} credits")
    except Exception as error:
        os.environ.pop("ULTRALYTICS_API_KEY")
        print(f"⚠️ Login failed ({error}), continuing offline")
else:
    print(f"ℹ️ No API key found. Get one at {PLATFORM_URL}/settings?tab=api-keys, continuing offline")
""")
code("""
# Cloud GPU availability is public, no key needed
with Platform(base_url=PLATFORM_URL) as platform:
    gpus = platform.training.gpu_availability()
print({gpu: level for gpu, level in gpus.items() if level})
""")

md("""
## 2. Hybrid training 🏋️

`project="username/project"` is the magic switch. Training runs on **this** machine and the Ultralytics Platform receives live metrics, plots, console output and the final checkpoint.

Want the Platform GPUs instead? Open your project on the Platform, click **New Model**, choose `yolo26n` plus a dataset, pick a GPU and press **Start Training**. Same model, zero setup.
""")
code("""
import torch

from ultralytics import YOLO

if torch.cuda.is_available():
    DEVICE = "0"  # NVIDIA CUDA, the Colab T4 lands here
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = "xpu"  # Intel GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple silicon
else:
    DEVICE = "cpu"
PROJECT = f"{USERNAME}/gdg-samsun" if USERNAME else "gdg-samsun"
print(f"Training on {DEVICE}, project {PROJECT}")

model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=10, imgsz=640, device=DEVICE, project=PROJECT, name="colab-run", exist_ok=True)
BEST = str(model.trainer.best)
print(f"Best weights: {BEST}")
if USERNAME:
    print(f"Watch it on the Platform: {PLATFORM_URL}/{PROJECT}")
""")
code("""
from IPython.display import Image, display

display(Image(filename=f"{model.trainer.save_dir}/results.png", width=900))
""")

md("""
## 3. Seven tasks, one API 🎨

Swap the model name and you swap the superpower.
""")
code("""
BUS, ZIDANE, BOATS = (f"https://ultralytics.com/images/{name}.jpg" for name in ("bus", "zidane", "boats"))
TASKS = {
    "detect": ("yolo26n.pt", BUS),
    "segment": ("yolo26n-seg.pt", BUS),
    "semantic": ("yolo26n-sem.pt", BUS),
    "depth": ("yolo26n-depth.pt", BUS),
    "classify": ("yolo26n-cls.pt", BUS),
    "pose": ("yolo26n-pose.pt", ZIDANE),
    "obb": ("yolo26n-obb.pt", BOATS),
}
os.makedirs("showcase", exist_ok=True)
for task, (weights, source) in TASKS.items():
    result = YOLO(weights).predict(source, device=DEVICE, verbose=False)[0]
    busy = task == "obb"
    result.save(filename=f"showcase/{task}.jpg", labels=not busy, conf=not busy)
    print(f"{task:<9} {result.speed['inference']:6.1f} ms  {result.verbose().strip(', ')}")
""")
code("""
for task in ("segment", "pose", "depth"):
    display(Image(filename=f"showcase/{task}.jpg", width=420))
""")

md("""
## 4. Tracking and Solutions 🏃

Every person keeps the same ID across frames. `stream=True` keeps memory flat on long videos. Swap `botsort.yaml` for `bytetrack.yaml`, `ocsort.yaml`, `deepocsort.yaml`, `fasttrack.yaml` or `tracktrack.yaml` to try the other trackers.
""")
code("""
from ultralytics.utils.downloads import safe_download

VIDEO = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
safe_download(VIDEO, file="crowd.mp4")  # a local file is read frame by frame, a URL like a live stream
ids = set()
for result in YOLO("yolo26n.pt").track("crowd.mp4", tracker="botsort.yaml", device=DEVICE, classes=[0], stream=True, persist=True, verbose=False):
    if result.boxes.id is not None:
        ids.update(result.boxes.id.int().tolist())
print(f"Tracked {len(ids)} unique people")
""")
code("""
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("crowd.mp4")
heatmap = solutions.Heatmap(model="yolo26n.pt", device=DEVICE, classes=[0], show=False, verbose=False)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    last = heatmap(frame).plot_im
cv2.imwrite("showcase/heatmap.jpg", last)
display(Image(filename="showcase/heatmap.jpg", width=640))
""")

md("""
## 5. Automatic annotation ✍️

Ultralytics YOLO finds the objects, SAM draws the masks, and you get YOLO segmentation labels without clicking a single polygon. The Ultralytics Platform goes further with **Smart Annotation** in the browser: SAM 3.1 clicks, pretrained or your own Ultralytics YOLO models, and vision language models that label a whole dataset from class names, either hosted open source models (Moondream 3.1 by default) or API models from OpenAI, Anthropic, Google and more with your own provider key.
""")
code("""
from ultralytics.data.annotator import auto_annotate

os.makedirs("auto/images", exist_ok=True)
safe_download(BUS, dir="auto/images")
auto_annotate(data="auto/images", det_model="yolo26n.pt", sam_model="sam2.1_b.pt", device=DEVICE, output_dir="auto/labels")
print(open("auto/labels/bus.txt").read()[:300], "...")
""")

md("""
## 6. Export 📦

* **ONNX** with `simplify=True` runs everywhere
* **OpenVINO** squeezes Intel CPUs, GPUs and NPUs
* **TensorRT** with `half=True` is the NVIDIA speed king (needs a CUDA GPU runtime)
""")
code("""
trained = YOLO(BEST)
exports = {"PyTorch": BEST}
exports["ONNX"] = trained.export(format="onnx", simplify=True)
exports["OpenVINO"] = trained.export(format="openvino")
if torch.cuda.is_available():
    exports["TensorRT FP16"] = trained.export(format="engine", half=True, device=0)
exports
""")

md("""
## 7. The great speed race 🏎️

Same image, same model, different engines. Median of 50 runs after warmup.
""")
code("""
import statistics
from pathlib import Path


def size_mb(path):
    path = Path(path)
    files = path.rglob("*") if path.is_dir() else [path]
    return sum(f.stat().st_size for f in files if f.is_file()) / 1e6


def race(weights, device, runs=50):
    m = YOLO(weights, task="detect")
    for _ in range(5):
        m.predict(BUS, device=device, verbose=False)
    times = [m.predict(BUS, device=device, verbose=False)[0].speed["inference"] for _ in range(runs)]
    return statistics.median(times)


rows = [("PyTorch", BEST, "cpu"), ("ONNX", exports["ONNX"], "cpu"), ("OpenVINO", exports["OpenVINO"], "intel:cpu")]
if DEVICE != "cpu":
    rows.insert(1, ("PyTorch", BEST, DEVICE))
if "TensorRT FP16" in exports:
    rows.append(("TensorRT FP16", exports["TensorRT FP16"], "0"))

baseline = None
print(f"{'Format':<14}{'Device':<11}{'Size MB':>8}{'ms':>8}{'FPS':>7}{'Speedup':>9}")
for name, path, device in rows:
    ms = race(path, device)
    baseline = baseline or ms
    print(f"{name:<14}{device:<11}{size_mb(path):>8.1f}{ms:>8.1f}{1000 / ms:>7.0f}{baseline / ms:>8.1f}x")
""")

md("""
## 8. Rust time 🦀 Ultralytics Inference

The `ultralytics-inference` crate gives you a tiny native binary with no Python at runtime. Perfect for edge boxes, containers and services. First build takes a few minutes, so start it and grab a çay ☕.
""")
code("""
!command -v cargo >/dev/null || (curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal)
import os
os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"
!cargo --version
""")
code("""
# Colab has no screen, so we skip the live window feature and keep image annotation
!command -v apt-get >/dev/null && (apt-get -qq install -y pkg-config libssl-dev > /dev/null)
!command -v ultralytics-inference >/dev/null || cargo install ultralytics-inference --no-default-features --features annotate
!ultralytics-inference --version
""")
code("""
safe_download(BUS, file="bus.jpg")
!ultralytics-inference predict --model {exports["ONNX"]} --source bus.jpg
""")
code("""
# Every task works from Rust too, the matching nano model downloads automatically
!ultralytics-inference predict --task pose --source https://ultralytics.com/images/zidane.jpg --save false
!ultralytics-inference predict --task segment --source bus.jpg --save false
""")
code("""
# A whole folder in batches of 4 with a stopwatch, compare it with the Python race above
for name in ("bus", "zidane"):
    safe_download(f"https://ultralytics.com/images/{name}.jpg", dir="rust_inputs")
!time ultralytics-inference predict --model {exports["ONNX"]} --source rust_inputs --batch 4 --save false
""")

md("""
> 🎬 Video files, webcams and RTSP streams need the `video` feature with FFmpeg: `cargo install ultralytics-inference --features video`. On Intel laptops add `openvino-native` and use `--device intel:gpu` or `--device intel:npu`. The workshop page shows both running on a laptop.
""")

md("""
## 🎉 You made it!

* ⭐ Star the repo and grab the slides on the [workshop page](https://onuralpszr.github.io/gdg-samsun-etkinlik-2026-workshop/)
* 🌐 Keep training on the [Ultralytics Platform](https://platform.ultralytics.com)
* 🦀 Read the [Ultralytics Inference](https://github.com/ultralytics/inference) docs for CUDA, TensorRT, CoreML and OpenVINO builds

No GPU was left behind today. Teşekkürler Samsun! 🇹🇷
""")

nb = nbf.v4.new_notebook()
for i, cell in enumerate(cells):
    cell["id"] = f"cell{i:02d}"  # fixed ids keep regenerated notebooks diff friendly
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4", "name": "hybrid_yolo_workshop.ipynb"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.validate(nb)
out = sys.argv[1] if len(sys.argv) > 1 else "notebooks/hybrid_yolo_workshop.ipynb"
nbf.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")
