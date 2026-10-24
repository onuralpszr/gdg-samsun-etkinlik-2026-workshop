---
title: No GPU Left Behind
description: Hybrid Ultralytics YOLO training and fast execution, a hands on GDG Samsun DevFest 2026 workshop
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

<div class="logos">
  <img src="assets/logos/gdg-samsun.svg#only-light" alt="GDG Samsun DevFest 2026">
  <img src="assets/logos/gdg-samsun-white.svg#only-dark" alt="GDG Samsun DevFest 2026">
  <img src="assets/logos/ultralytics-logo.svg#only-light" alt="Ultralytics">
  <img src="assets/logos/ultralytics-logo-white.svg#only-dark" alt="Ultralytics">
</div>

<h1><span class="b">No</span> <span class="r">GPU</span> <span class="y">Left</span> <span class="g">Behind</span></h1>

<p class="tagline">Hybrid Ultralytics YOLO training and fast execution on any laptop, from the cloud to Rust in 60 minutes</p>

[:simple-googlecolab: Open in Colab](https://colab.research.google.com/github/onuralpszr/gdg-samsun-etkinlik-2026-workshop/blob/main/notebooks/hybrid_yolo_workshop.ipynb){ .md-button .md-button--primary }
[:fontawesome-brands-github: Get the code](https://github.com/onuralpszr/gdg-samsun-etkinlik-2026-workshop){ .md-button }
[:material-rocket-launch: Ultralytics Platform](https://platform.ultralytics.com){ .md-button }

</div>

## What we build today :sparkles:

You do not need a giant GPU to do serious computer vision. Today we train an Ultralytics YOLO model on whatever you brought (an NVIDIA card, an Intel laptop with Arc graphics and an NPU, a MacBook or a plain CPU), stream the run to the **Ultralytics Platform**, export it to **ONNX**, **OpenVINO** and **TensorRT**, and finally ship it as a tiny Rust binary with **Ultralytics Inference**.

<div class="steps" markdown>
<div class="card" markdown>
<strong>[:key: Step 0 Setup](setup.md)</strong>
One script builds a `.venv` with the right PyTorch for your hardware plus OpenVINO and checks every accelerator you have.
</div>
<div class="card" markdown>
<strong>[:weight_lifter: Step 1 Train](train.md)</strong>
Train `yolo26n.pt` locally while the Ultralytics Platform shows live metrics, or click once to train on a cloud GPU.
</div>
<div class="card" markdown>
<strong>[:art: Playground](playground.md)</strong>
Seven tasks, tracking, Solutions, auto annotation and a live webcam, a few lines each.
</div>
<div class="card" markdown>
<strong>[:package: Step 2 Export](export.md)</strong>
Export to ONNX, OpenVINO and TensorRT, then race every format on every device you own.
</div>
<div class="card" markdown>
<strong>[:crab: Step 3 Ship](rust.md)</strong>
Run the same model from the `ultralytics-inference` CLI and Rust crate with flat memory and no Python at all.
</div>
<div class="card" markdown>
<strong>[:rocket: Advanced](advanced.md)</strong>
Export to INT8 for a smaller and faster model, then run every chip in your laptop at the same time.
</div>
<div class="card" markdown>
<strong>[:tada: Wrap up](wrap-up.md)</strong>
Mini challenges, troubleshooting tips and the links to keep going after today.
</div>
</div>

[:material-play: Start with Step 0](setup.md){ .md-button .md-button--primary }

## Agenda for the hour :alarm_clock:

| Time | Block | What happens |
|---|---|---|
| 00:00 | Warm up | Who has a GPU? Who has an NPU? Hands up, nobody gets left behind |
| 00:03 | [Step 0](setup.md) | Check the setup you installed at home and log in to the Ultralytics Platform |
| 00:07 | [Step 1](train.md) | Hybrid training on your laptop with live Platform metrics, plus a cloud training tour |
| 00:20 | [Playground](playground.md) | Seven tasks, tracking, Solutions and auto annotation in a few lines each |
| 00:30 | [Step 2](export.md) | Export to ONNX, OpenVINO and TensorRT and run the great speed race |
| 00:38 | [Step 3](rust.md) | Ultralytics Inference in Rust, the CLI, batch, video and webcam |
| 00:48 | [Advanced](advanced.md) | INT8 export and every chip at once |
| 00:55 | [Wrap up](wrap-up.md) | Live webcam showdown, follow up links and questions |
