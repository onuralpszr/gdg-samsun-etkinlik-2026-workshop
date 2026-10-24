"""Track every person across frames with Ultralytics YOLO.

Other trackers from the CLI: yolo track model=yolo26n.pt source=video.mp4 tracker=bytetrack.yaml
On your webcam: yolo track model=yolo26n.pt source=0 show=True
"""

from ultralytics import YOLO
from ultralytics.utils.downloads import safe_download

from device_utils import DEVICE

VIDEO = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"

video = safe_download(VIDEO, dir="videos")

ids = set()
model = YOLO("yolo26n.pt")
for result in model.track(
    video, tracker="botsort.yaml", classes=[0], device=DEVICE, stream=True, save=True, verbose=False
):
    if result.boxes.id is not None:
        ids.update(result.boxes.id.int().tolist())
print(f"Tracked {len(ids)} people, the annotated video is in {result.save_dir}")
