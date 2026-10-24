---
title: "Playground: everything Ultralytics YOLO can do"
---

# Playground: everything Ultralytics YOLO can do :art:

[![Ultralytics CI](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)
[![Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)
[![Open Ultralytics In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)
[![Open Ultralytics In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/models/ultralytics/yolo26)

## Seven tasks, one API

```bash
# all seven, saved in runs/tasks
python local_python/03_all_tasks.py

# one task from the CLI
yolo predict model=yolo26n-pose.pt source=https://ultralytics.com/images/zidane.jpg
```

```python
from ultralytics import YOLO

YOLO("yolo26n-pose.pt").predict("https://ultralytics.com/images/zidane.jpg", save=True)
```

<div class="gallery" markdown>
<figure markdown>
![Detect](assets/img/task_detect.jpg)
<figcaption>detect with <code>yolo26n.pt</code></figcaption>
</figure>
<figure markdown>
![Segment](assets/img/task_segment.jpg)
<figcaption>segment with <code>yolo26n-seg.pt</code></figcaption>
</figure>
<figure markdown>
![Semantic](assets/img/task_semantic.jpg)
<figcaption>semantic with <code>yolo26n-sem.pt</code></figcaption>
</figure>
<figure markdown>
![Depth](assets/img/task_depth.jpg)
<figcaption>depth with <code>yolo26n-depth.pt</code></figcaption>
</figure>
<figure markdown>
![Classify](assets/img/task_classify.jpg)
<figcaption>classify with <code>yolo26n-cls.pt</code> (minibus 0.52)</figcaption>
</figure>
<figure markdown>
![Pose](assets/img/task_pose.jpg)
<figcaption>pose with <code>yolo26n-pose.pt</code></figcaption>
</figure>
<figure markdown>
![OBB](assets/img/task_obb.jpg)
<figcaption>obb with <code>yolo26n-obb.pt</code> (167 ships, 5 harbors)</figcaption>
</figure>
</div>

## Tracking

Every person keeps the same ID across frames. Pick from six trackers (`botsort`, `bytetrack`, `ocsort`, `deepocsort`, `fasttrack` and `tracktrack`) with any detect, segment, pose or obb model and with exported formats too.

```bash
# people in the sample video
python local_python/04_track.py

# another tracker
yolo track model=yolo26n.pt source=video.mp4 tracker=bytetrack.yaml

# pick the device: cpu, 0 for NVIDIA, mps for Apple silicon
yolo track model=yolo26n.pt source=video.mp4 device=0

# OpenVINO on an Intel GPU
yolo track model=yolo26n_openvino_model source=video.mp4 device=intel:gpu

# your webcam
yolo track model=yolo26n.pt source=0 show=True
```

<div class="wide" markdown>
![Tracking demo](assets/img/tracking.gif)
</div>

## Real world use cases with Solutions

```bash
# people crossing a line
python local_python/06_solutions.py count

# where people spend time
python local_python/06_solutions.py heatmap

# privacy blur
python local_python/06_solutions.py blur

# speed estimation
python local_python/06_solutions.py speed

# any solution straight from the CLI
yolo solutions heatmap source=video.mp4 show=True
```

<div class="wide" markdown>
![Counting, heatmap, privacy blur and speed estimation](assets/img/solutions_grid.jpg)
</div>

## Automatic annotation

Labeling by hand is slow. On your own machine, Ultralytics YOLO finds the objects and SAM draws the masks, so you get YOLO segmentation labels for free.

The Ultralytics Platform goes further with **Smart Annotation** right in the browser. Click objects with SAM 3.1, run pretrained or your own fine tuned Ultralytics YOLO models, or type the class names and let a vision language model label the whole dataset. Pick one of the eight hosted open source models (Moondream 3.1 by default) or add your own provider key to use vision models from OpenAI, Anthropic, Google and more.

```bash
# set IMAGES at the top to label your own photos
python local_python/07_auto_annotate.py
```

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolo26n.pt", sam_model="sam2.1_b.pt")
```

<div class="gallery" markdown>
<figure markdown>
![Auto annotation on bus](assets/img/auto_annotate_bus.jpg)
<figcaption>5 polygons, zero clicks</figcaption>
</figure>
<figure markdown>
![Auto annotation on zidane](assets/img/auto_annotate_zidane.jpg)
<figcaption>Even the tie gets a mask</figcaption>
</figure>
</div>

## Live webcam

The crowd favorite. Point the camera at the room and switch engines live.

```bash
# FPS counter on the best device
python local_python/05_webcam.py

# another camera
CAMERA=1 python local_python/05_webcam.py

# switch engines live: CPU, NVIDIA GPU or OpenVINO on an Intel GPU
yolo predict model=yolo26n.pt source=0 device=cpu show=True
yolo predict model=yolo26n.pt source=0 device=0 show=True
yolo predict model=yolo26n_openvino_model source=0 device=intel:gpu show=True

# pose with tracking IDs
yolo track model=yolo26n-pose.pt source=0 show=True
```

[:material-arrow-right: Next: Step 2 Export](export.md){ .md-button .md-button--primary .next-step }
