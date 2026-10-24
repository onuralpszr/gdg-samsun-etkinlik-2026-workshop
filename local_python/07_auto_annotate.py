"""Auto annotation: Ultralytics YOLO finds the objects, SAM draws the masks, you get YOLO segmentation labels.

Point IMAGES at your own photos, then upload the images and labels folders to the Ultralytics Platform to review.
"""

from ultralytics.data.annotator import auto_annotate
from ultralytics.utils.downloads import safe_download

from device_utils import DEVICE

IMAGES = "datasets/auto_annotate_demo/images"
for url in ("https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/zidane.jpg"):
    safe_download(url, dir=IMAGES)

auto_annotate(
    data=IMAGES,
    det_model="yolo26n.pt",
    sam_model="sam2.1_b.pt",
    device=DEVICE,
    output_dir="datasets/auto_annotate_demo/labels",
)
print("Labels saved in datasets/auto_annotate_demo/labels")
