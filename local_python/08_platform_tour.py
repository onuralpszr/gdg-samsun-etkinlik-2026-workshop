"""A quick tour of the Ultralytics Platform Python client that ships inside the ultralytics package.

Cloud GPU availability works without a key. Run `yolo login YOUR_API_KEY` to also see your account and projects.
"""

import os

from ultralytics import Platform
from ultralytics.utils import SETTINGS

with Platform() as platform:
    print("Cloud GPU availability right now:")
    for gpu, level in platform.training.gpu_availability().items():
        print(f"  {gpu:<16} {level or 'busy'}")
    if SETTINGS.get("api_key") or os.getenv("ULTRALYTICS_API_KEY"):
        me = platform.account.summary()
        print(f"{me['username']} on the {me['plan']} plan with ${me['creditsCents'] / 100:.2f} credits")
        for project in platform.projects.list(me["username"])["projects"]:
            print(f"  project {project['project']}")
