"""A quick tour of the Ultralytics Platform Python client that ships inside the ultralytics package.

status    live cloud GPU availability (works without an API key)
whoami    your plan, credits and resource counts
projects  your projects and the models inside them
predict   run a model hosted on the Platform against a local image

Usage:
    python local_python/08_platform_tour.py status
    yolo login ul_xxx && python local_python/08_platform_tour.py whoami
    python local_python/08_platform_tour.py projects
    python local_python/08_platform_tour.py predict --model your_username/gdg-samsun/laptop-run --image bus.jpg
"""

import argparse
import os

from ultralytics import Platform
from ultralytics.utils import ASSETS, PLATFORM_URL, SETTINGS


def client(need_key=True):
    """Build a Platform client from ULTRALYTICS_API_KEY or the key saved by yolo login."""
    key = os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    if need_key and not key:
        raise SystemExit(f"Log in first: yolo login YOUR_API_KEY (get one at {PLATFORM_URL}/settings?tab=api-keys)")
    return Platform(api_key=key or None, base_url=PLATFORM_URL, timeout=60)


def status():
    with client(need_key=False) as p:
        gpus = p.training.gpu_availability()
    print("Cloud GPU availability right now:")
    for gpu, level in gpus.items():
        print(f"  {gpu:<16} {level or 'busy'}")


def whoami():
    with client() as p:
        me = p.account.summary()
    print(f"User {me['username']} on the {me['plan']} plan with ${me['creditsCents'] / 100:.2f} credits")
    print(f"Counts: {me['counts']}")


def projects():
    with client() as p:
        me = p.account.summary()["username"]
        for project in p.projects.list(me)["projects"]:
            print(f"{me}/{project['project']}")
            for model in p.models.list(me, project["project"])["models"]:
                print(f"   {model['model']:<24} {model.get('task', '')} {model.get('status', '')}")


def predict(model, image):
    owner, project, name = model.split("/")
    with client() as p, open(image, "rb") as f:
        response = p.models.predict(owner, project, name, body={"file": f, "conf": 0.25})
    for item in response["images"]:
        for det in item["results"]:
            print(f"  {det['name']} {det.get('confidence', 0):.2f}")
    print(f"Raw metadata: {response['metadata']}")


def main():
    parser = argparse.ArgumentParser(description="Ultralytics Platform client tour")
    parser.add_argument("action", choices=["status", "whoami", "projects", "predict"])
    parser.add_argument("--model", help="owner/project/model on the Platform, used by predict")
    parser.add_argument("--image", default=str(ASSETS / "bus.jpg"), help="local image for predict")
    args = parser.parse_args()

    if args.action == "predict":
        if not args.model:
            raise SystemExit("predict needs --model owner/project/model")
        predict(args.model, args.image)
    else:
        {"status": status, "whoami": whoami, "projects": projects}[args.action]()


if __name__ == "__main__":
    main()
