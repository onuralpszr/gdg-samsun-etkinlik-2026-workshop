---
title: "Deneme alanı: Ultralytics YOLO ile neler yapılabilir"
hide:
  - navigation
---

# Deneme alanı: Ultralytics YOLO ile neler yapılabilir :art:

[![Ultralytics CI](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)
[![Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)
[![Open Ultralytics In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)
[![Open Ultralytics In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/models/ultralytics/yolo26)

## Yedi görev, tek API

```bash
# yedi görev, runs/tasks klasörüne kaydedilir
python local_python/03_all_tasks.py

# CLI ile tek bir görev
yolo predict model=yolo26n-pose.pt source=https://ultralytics.com/images/zidane.jpg
```

```python
from ultralytics import YOLO

YOLO("yolo26n-pose.pt").predict("https://ultralytics.com/images/zidane.jpg", save=True)
```

<div class="gallery" markdown>
<figure markdown>
![Nesne tespiti](assets/img/task_detect.jpg)
<figcaption>detect, <code>yolo26n.pt</code> modeli</figcaption>
</figure>
<figure markdown>
![Örnek segmentasyonu](assets/img/task_segment.jpg)
<figcaption>segment, <code>yolo26n-seg.pt</code> modeli</figcaption>
</figure>
<figure markdown>
![Anlamsal segmentasyon](assets/img/task_semantic.jpg)
<figcaption>semantic, <code>yolo26n-sem.pt</code> modeli</figcaption>
</figure>
<figure markdown>
![Derinlik tahmini](assets/img/task_depth.jpg)
<figcaption>depth, <code>yolo26n-depth.pt</code> modeli</figcaption>
</figure>
<figure markdown>
![Sınıflandırma](assets/img/task_classify.jpg)
<figcaption>classify, <code>yolo26n-cls.pt</code> modeli (minibus 0.52)</figcaption>
</figure>
<figure markdown>
![Poz tahmini](assets/img/task_pose.jpg)
<figcaption>pose, <code>yolo26n-pose.pt</code> modeli</figcaption>
</figure>
<figure markdown>
![Yönlü nesne tespiti](assets/img/task_obb.jpg)
<figcaption>obb, <code>yolo26n-obb.pt</code> modeli (167 gemi, 5 liman)</figcaption>
</figure>
</div>

## Nesne takibi

Her kişi video boyunca aynı ID numarasını korur. Altı takip algoritmasından birini seçebilirsiniz: `botsort`, `bytetrack`, `ocsort`, `deepocsort`, `fasttrack` ve `tracktrack`. Hepsi detect, segment, pose ve obb modelleriyle, dışa aktarılmış formatlar dahil, birlikte çalışır.

```bash
# örnek videodaki kişiler
python local_python/04_track.py

# başka bir takip algoritması
yolo track model=yolo26n.pt source=video.mp4 tracker=bytetrack.yaml

# cihazı seçin: cpu, NVIDIA için 0, Apple silicon için mps
yolo track model=yolo26n.pt source=video.mp4 device=0

# Intel GPU üzerinde OpenVINO
yolo track model=yolo26n_openvino_model source=video.mp4 device=intel:gpu

# kendi webcam'iniz
yolo track model=yolo26n.pt source=0 show=True
```

<div class="wide" markdown>
![Nesne takibi örneği](assets/img/tracking.gif)
</div>

## Solutions ile gerçek hayattan örnekler

```bash
# çizgiyi geçen kişileri sayar
python local_python/06_solutions.py count

# kişilerin en çok vakit geçirdiği alanları gösterir
python local_python/06_solutions.py heatmap

# gizlilik için bulanıklaştırır
python local_python/06_solutions.py blur

# hız tahmini yapar
python local_python/06_solutions.py speed

# her çözüm doğrudan CLI ile de çalışır
yolo solutions heatmap source=video.mp4 show=True
```

<div class="wide" markdown>
![Sayma, ısı haritası, gizlilik bulanıklaştırma ve hız tahmini](assets/img/solutions_grid.jpg)
</div>

## Otomatik etiketleme

Elle etiketleme çok zaman alır. Kendi bilgisayarınızda Ultralytics YOLO nesneleri bulur, SAM maskeleri çizer ve YOLO segmentasyon etiketleri kendiliğinden hazırlanır.

Ultralytics Platform ise tarayıcıdaki **Smart Annotation** özelliğiyle bir adım daha ileri gider. Nesnelere SAM 3.1 ile tıklayarak etiketleyebilir, hazır ya da kendi eğittiğiniz Ultralytics YOLO modellerini çalıştırabilir veya sınıf adlarını yazıp tüm veri setini bir görsel dil modeline etiketletebilirsiniz. Platform'da barındırılan sekiz açık kaynak modelden birini (varsayılan olarak Moondream 3.1) seçebilir ya da kendi sağlayıcı anahtarınızı ekleyerek OpenAI, Anthropic, Google ve diğer sağlayıcıların görsel modellerini kullanabilirsiniz.

```bash
# kendi fotoğraflarınız için en üstteki IMAGES satırını değiştirin
python local_python/07_auto_annotate.py
```

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolo26n.pt", sam_model="sam2.1_b.pt")
```

<div class="gallery" markdown>
<figure markdown>
![Otobüs görselinde otomatik etiketleme](assets/img/auto_annotate_bus.jpg)
<figcaption>Tek tıklama yapmadan 5 poligon</figcaption>
</figure>
<figure markdown>
![Zidane görselinde otomatik etiketleme](assets/img/auto_annotate_zidane.jpg)
<figcaption>Kravat bile kendi maskesini alıyor</figcaption>
</figure>
</div>

## Canlı webcam

Atölyenin en sevilen bölümü burası. Kamerayı salona çevirin ve çıkarım motorlarını canlı olarak değiştirin.

```bash
# en uygun cihazda FPS sayacı
python local_python/05_webcam.py

# başka bir kamera
CAMERA=1 python local_python/05_webcam.py

# motoru canlı değiştirin: CPU, NVIDIA GPU ya da Intel GPU üzerinde OpenVINO
yolo predict model=yolo26n.pt source=0 device=cpu show=True
yolo predict model=yolo26n.pt source=0 device=0 show=True
yolo predict model=yolo26n_openvino_model source=0 device=intel:gpu show=True

# takip numaralarıyla poz tahmini
yolo track model=yolo26n-pose.pt source=0 show=True
```

[:material-arrow-right: Sonraki: Adım 2 Dışa aktarma](export.md){ .md-button .md-button--primary .next-step }
