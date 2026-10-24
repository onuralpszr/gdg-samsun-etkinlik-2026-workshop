---
title: "Step 1: Ultralytics Platform and hybrid training"
hide:
  - navigation
---

# Step 1: Ultralytics Platform and hybrid training :weight_lifter:

## 1.1 Get your API key

1. Sign up at [platform.ultralytics.com](https://platform.ultralytics.com). New accounts get free credits.
2. Open **Settings > API Keys** and create a key with the training scope. Copy it right away, it is shown once.
3. Log in from your terminal. The key is saved in your Ultralytics settings.

```bash
yolo login ul_your_api_key

# or for one shell session
export ULTRALYTICS_API_KEY=ul_your_api_key
```

## 1.2 Train locally, watch it in the cloud

The magic switch is `project="username/project"`. Training runs on **your** hardware and the Ultralytics Platform receives live metrics, plots, console output and the final checkpoint.

=== "CLI"

    ```bash
    # device=0 for NVIDIA CUDA, device=xpu for Intel GPU, device=mps for Apple silicon, device=cpu for any machine

    yolo train model=yolo26n.pt data=coco8.yaml epochs=10 device=0 project=your_username/gdg-samsun name=laptop-run
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    # device: 0 for NVIDIA CUDA, "xpu" for Intel GPU, "mps" for Apple silicon, "cpu" for any machine
    model.train(data="coco8.yaml", epochs=10, device=0, project="your_username/gdg-samsun", name="laptop-run")
    ```

=== "Workshop script"

    ```bash
    python local_python/01_hybrid_train.py
    ```

    The same Python code in a file. Put your username in `PROJECT` at the top and it picks the best device on its own (NVIDIA CUDA, Intel XPU, Apple MPS or CPU). Without `yolo login` it simply trains offline.

=== "Platform dataset"

    ```bash
    yolo train model=yolo26n.pt data=ul://your_username/datasets/your_dataset epochs=50 project=your_username/gdg-samsun
    ```

Ten epochs of `yolo26n.pt` on `coco8.yaml` reach **mAP50 0.941**.

<div class="gallery" markdown>
<figure markdown>
![Training curves](assets/results/training_results.png)
<figcaption>Loss and mAP curves during training</figcaption>
</figure>
<figure markdown>
![Training mosaic](assets/results/train_batch0.jpg)
<figcaption>A training batch with mosaic augmentation</figcaption>
</figure>
<figure markdown>
![Validation predictions](assets/results/val_batch0_pred.jpg)
<figcaption>Validation predictions after 10 epochs</figcaption>
</figure>
</div>

## 1.3 Or train on a cloud GPU with one click

1. Open your project on the Ultralytics Platform and click **New Model**.
2. Pick `YOLO26n` under **Official**, pick a dataset in `ready` status.
3. Set epochs, batch and image size, then choose a GPU. The RTX 2000 Ada is perfect for a quick test.
4. Press **Start Training** and watch the same dashboard fill up.

See which cloud GPUs are free right now with the Platform client that ships inside `ultralytics`. After `yolo login` it also lists your plan, credits and projects.

```bash
python local_python/08_platform_tour.py
```

[:material-arrow-right: Next: Playground](playground.md){ .md-button .md-button--primary .next-step }
