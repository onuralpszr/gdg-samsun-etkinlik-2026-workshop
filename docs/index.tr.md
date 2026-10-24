---
title: Hiçbir GPU Geride Kalmasın
description: Hibrit Ultralytics YOLO eğitimi ve hızlı çalıştırma üzerine GDG Samsun DevFest 2026 uygulamalı atölyesi
hide:
  - navigation
---

<div class="hero" markdown>

<div class="logos">
  <img src="../assets/logos/gdg-samsun.svg#only-light" alt="GDG Samsun DevFest 2026">
  <img src="../assets/logos/gdg-samsun-white.svg#only-dark" alt="GDG Samsun DevFest 2026">
  <img src="../assets/logos/ultralytics-logo.svg#only-light" alt="Ultralytics">
  <img src="../assets/logos/ultralytics-logo-white.svg#only-dark" alt="Ultralytics">
</div>

<h1><span class="b">Hiçbir</span> <span class="r">GPU</span> <span class="y">Geride</span> <span class="g">Kalmasın</span></h1>

<p class="tagline">Her dizüstü bilgisayarda hibrit Ultralytics YOLO eğitimi ve hızlı çalıştırma. Buluttan Rust'a 60 dakikada.</p>

[:simple-googlecolab: Colab'da aç](https://colab.research.google.com/github/onuralpszr/gdg-samsun-etkinlik-2026-workshop/blob/main/notebooks/hybrid_yolo_workshop.ipynb){ .md-button .md-button--primary }
[:fontawesome-brands-github: Kodu indir](https://github.com/onuralpszr/gdg-samsun-etkinlik-2026-workshop){ .md-button }
[:material-rocket-launch: Ultralytics Platform](https://platform.ultralytics.com){ .md-button }

</div>

## Bugün neler yapacağız :sparkles:

Ciddi bir bilgisayarlı görü projesi için dev bir GPU'ya ihtiyacınız yok. Bugün elinizde ne varsa onunla çalışacağız. Bu bir NVIDIA ekran kartı, Arc grafikli ve NPU'lu bir Intel dizüstü, bir MacBook ya da sadece bir CPU olabilir. Önce bir Ultralytics YOLO modeli eğiteceğiz ve eğitimi canlı olarak **Ultralytics Platform** üzerinden izleyeceğiz. Ardından modeli **ONNX**, **OpenVINO** ve **TensorRT** formatlarına dönüştüreceğiz. Son olarak da **Ultralytics Inference** ile küçük bir Rust uygulaması olarak çalıştıracağız.

<div class="steps" markdown>
<div class="card" markdown>
<strong>:key: Adım 0 Kurulum</strong>
Tek bir betik, donanımınıza uygun PyTorch sürümünü ve OpenVINO'yu içeren bir `.venv` ortamı kurar ve bilgisayarınızdaki tüm hızlandırıcıları kontrol eder.
</div>
<div class="card" markdown>
<strong>:weight_lifter: Adım 1 Eğitim</strong>
`yolo26n.pt` modelini kendi bilgisayarınızda eğitirken metrikleri Ultralytics Platform üzerinde canlı izleyin. Dilerseniz tek tıkla bulut GPU'sunda da eğitebilirsiniz.
</div>
<div class="card" markdown>
<strong>:package: Adım 2 Dışa aktarma</strong>
Modeli ONNX, OpenVINO ve TensorRT formatlarına aktarın ve her formatı sahip olduğunuz tüm cihazlarda yarıştırın.
</div>
<div class="card" markdown>
<strong>:crab: Adım 3 Çalıştırma</strong>
Aynı modeli `ultralytics-inference` CLI aracı ve Rust kütüphanesi ile çalıştırın. Bellek kullanımı sabit kalır ve Python'a hiç gerek kalmaz.
</div>
</div>

## Atölye programı :alarm_clock:

| Zaman | Bölüm | İçerik |
|---|---|---|
| 00:00 | Isınma | Kimde GPU var, kimde NPU var? Herkes katılıyor, kimse geride kalmıyor. |
| 00:05 | Adım 0 | Ön koşul kontrolünü çalıştırıp Ultralytics Platform'a giriş yapıyoruz. |
| 00:12 | Adım 1 | Dizüstü bilgisayarda hibrit eğitim yapıyor, metrikleri Platform'da canlı izliyor ve bulut eğitimini tanıyoruz. |
| 00:25 | Deneme alanı | Yedi görev, nesne takibi, Solutions ve otomatik etiketlemeyi birkaç satır kodla deniyoruz. |
| 00:35 | Adım 2 | Modeli ONNX, OpenVINO ve TensorRT formatlarına aktarıp büyük hız yarışını başlatıyoruz. |
| 00:45 | Adım 3 | Rust ile Ultralytics Inference, CLI, toplu işleme, video ve webcam kullanımını görüyoruz. |
| 00:55 | Kapanış | Canlı webcam gösterisi, faydalı bağlantılar ve soru cevap ile bitiriyoruz. |

## Mimari :world_map:

```mermaid
flowchart TB
    subgraph Data["Veri"]
        direction LR
        A[Görselleriniz] --> B[YOLO, SAM ya da görsel dil modelleriyle otomatik etiketleme]
        B --> C[(Ultralytics Platform veri seti)]
    end
    subgraph Train["Adım 1 Eğitim"]
        direction LR
        C --> D{Nerede eğitilecek?}
        D -->|Bulut GPU| E[Ultralytics Platform bulut eğitimi]
        D -->|Dizüstü bilgisayar| F[CUDA, XPU, MPS ya da CPU ile yerel eğitim]
        F -.->|canlı metrikler ve model dosyaları| G[Ultralytics Platform paneli]
        E --> G
    end
    subgraph Export["Adım 2 Dışa aktarma"]
        direction LR
        G --> H[best.pt]
        F --> H
        H --> I[ONNX]
        H --> J[OpenVINO]
        H --> K[TensorRT FP16]
    end
    subgraph Run["Adım 3 Çalıştırma"]
        direction LR
        I --> L[ultralytics-inference CLI]
        J --> L
        I --> M[Rust uygulaması]
        J --> M
        K --> N[NVIDIA üzerinde Python]
        I --> P[NVIDIA üzerinde Rust ile TensorRT FP16]
        J --> O[Intel CPU, GPU ve NPU]
    end
```

## Adım 0: Ön koşul kontrolü :white_check_mark:

=== "Linux ve macOS"

    ```bash
    git clone https://github.com/onuralpszr/gdg-samsun-etkinlik-2026-workshop.git
    cd gdg-samsun-etkinlik-2026-workshop

    # Intel GPU ve NPU bulunan dizüstüler (PyTorch XPU ve OpenVINO)
    ./setup_env.sh
    ./setup_env.sh cuda     # NVIDIA ekran kartlı bilgisayarlar
    ./setup_env.sh cpu      # Apple silicon (MPS) ve diğer tüm bilgisayarlar
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

    Sayfanın en üstündeki **Colab'da aç** düğmesine basın ve **Runtime > Change runtime type > T4 GPU** seçeneğini seçin. Bu yolda bilgisayarınıza hiçbir şey kurmanız gerekmez.

`setup_env.sh` en sonda `local_python/00_check_env.py` betiğini çalıştırır. Bu kısa kontrol PyTorch cihazınızı, OpenVINO cihazlarını, Rust CLI aracını ve Ultralytics Platform anahtarınızı gösterir. Ayrıntılı sistem raporu için `yolo checks` komutunu çalıştırabilirsiniz.

!!! tip "Başlamadan önce kontrol listesi"

    - [x] Python 3.11 ya da üzeri (Ultralytics Platform entegrasyonu için gerekli)
    - [x] `ultralytics` 8.4.120 ya da üzeri
    - [x] Bir Ultralytics Platform hesabı ve API anahtarı
    - [x] Adım 3 için Rust 1.92 ya da üzeri (`curl https://sh.rustup.rs -sSf | sh`)
    - [x] Rust ile video ve webcam kullanacaksanız FFmpeg geliştirme kütüphaneleri

## Adım 1: Ultralytics Platform ve hibrit eğitim :weight_lifter:

### 1.1 API anahtarınızı alın

1. [platform.ultralytics.com](https://platform.ultralytics.com) adresinden ücretsiz hesap açın. Yeni hesaplara başlangıç kredisi tanımlanır.
2. **Settings > API Keys** sayfasına gidin ve eğitim yetkisine sahip bir anahtar oluşturun. Anahtar yalnızca bir kez gösterildiği için hemen kopyalayın.
3. Terminalden giriş yapın. Anahtarınız Ultralytics ayarlarına kaydedilir.

```bash
yolo login ul_your_api_key

# ya da yalnızca bu terminal oturumu için
export ULTRALYTICS_API_KEY=ul_your_api_key
```

### 1.2 Yerelde eğitin, bulutta izleyin

İşin sırrı `project="username/project"` parametresinde. Eğitim **sizin** donanımınızda çalışırken Ultralytics Platform canlı metrikleri, grafikleri, konsol çıktısını ve eğitilen son modeli alır.

=== "CLI"

    ```bash
    # NVIDIA CUDA için device=0, Intel GPU için device=xpu, Apple silicon için device=mps, her bilgisayarda device=cpu

    yolo train model=yolo26n.pt data=coco8.yaml epochs=10 device=0 project=your_username/gdg-samsun name=laptop-run
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    # device: NVIDIA CUDA için 0, Intel GPU için "xpu", Apple silicon için "mps", her bilgisayarda "cpu"
    model.train(data="coco8.yaml", epochs=10, device=0, project="your_username/gdg-samsun", name="laptop-run")
    ```

=== "Atölye betiği"

    ```bash
    python local_python/01_hybrid_train.py
    ```

    Aynı Python kodu, bu kez bir dosyada. En üstteki `PROJECT` satırına kullanıcı adınızı yazmanız yeterli. Betik en uygun cihazı (NVIDIA CUDA, Intel XPU, Apple MPS ya da CPU) kendisi seçer. `yolo login` yapmadıysanız eğitimi çevrimdışı tamamlar.

=== "Platform veri seti"

    ```bash
    yolo train model=yolo26n.pt data=ul://your_username/datasets/your_dataset epochs=50 project=your_username/gdg-samsun
    ```

`yolo26n.pt` modelinin `coco8.yaml` üzerinde on epoch eğitimi, bir dizüstü bilgisayar GPU'sunda yaklaşık **18 saniye** sürdü ve **0.941 mAP50** değerine ulaştı.

<div class="gallery" markdown>
<figure markdown>
![Eğitim grafikleri](assets/results/training_results.png)
<figcaption>Eğitim boyunca kayıp ve mAP grafikleri</figcaption>
</figure>
<figure markdown>
![Eğitim mozaiği](assets/results/train_batch0.jpg)
<figcaption>Mozaik veri artırma uygulanmış bir eğitim grubu</figcaption>
</figure>
<figure markdown>
![Doğrulama tahminleri](assets/results/val_batch0_pred.jpg)
<figcaption>10 epoch sonunda doğrulama tahminleri</figcaption>
</figure>
</div>

### 1.3 Ya da tek tıkla bulut GPU'sunda eğitin

1. Ultralytics Platform'da projenizi açın ve **New Model** düğmesine tıklayın.
2. **Official** başlığı altından `YOLO26n` modelini, ardından durumu `ready` olan bir veri setini seçin.
3. Epoch sayısını, batch boyutunu ve görsel boyutunu ayarlayıp bir GPU seçin. Hızlı bir deneme için RTX 2000 Ada yeterlidir.
4. **Start Training** düğmesine basın ve aynı panelin dolmasını izleyin.

Hangi bulut GPU'larının şu anda boşta olduğunu `ultralytics` paketiyle gelen Platform istemcisiyle görebilirsiniz. `yolo login` yaptıysanız planınızı, kredinizi ve projelerinizi de listeler.

```bash
python local_python/08_platform_tour.py
```

## Deneme alanı: Ultralytics YOLO ile neler yapılabilir :art:

[![Ultralytics CI](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml)
[![PyPI Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)
[![Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)
[![Open Ultralytics In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)
[![Open Ultralytics In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/models/ultralytics/yolo26)

### Yedi görev, tek API

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

### Nesne takibi

Her kişi video boyunca aynı ID numarasını korur. Altı takip algoritmasından birini seçebilirsiniz: `botsort`, `bytetrack`, `ocsort`, `deepocsort`, `fasttrack` ve `tracktrack`. Hepsi detect, segment, pose ve obb modelleriyle, dışa aktarılmış formatlar dahil, birlikte çalışır.

```bash
# örnek videodaki kişiler
python local_python/04_track.py

# başka bir takip algoritması
yolo track model=yolo26n.pt source=video.mp4 tracker=bytetrack.yaml

# Intel GPU üzerinde OpenVINO
yolo track model=yolo26n_openvino_model source=video.mp4 device=intel:gpu

# kendi webcam'iniz
yolo track model=yolo26n.pt source=0 show=True
```

<div class="wide" markdown>
![Nesne takibi örneği](assets/img/tracking.gif)
</div>

### Solutions ile gerçek hayattan örnekler

```bash
python local_python/06_solutions.py count        # çizgiyi geçen kişileri sayar

# kişilerin en çok vakit geçirdiği alanları gösterir
python local_python/06_solutions.py heatmap
python local_python/06_solutions.py blur         # gizlilik için bulanıklaştırır
python local_python/06_solutions.py speed        # hız tahmini yapar

# her çözüm doğrudan CLI ile de çalışır
yolo solutions heatmap source=video.mp4 show=True
```

<div class="wide" markdown>
![Sayma, ısı haritası, gizlilik bulanıklaştırma ve hız tahmini](assets/img/solutions_grid.jpg)
</div>

### Otomatik etiketleme

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

### Canlı webcam

Atölyenin en sevilen bölümü burası. Kamerayı salona çevirin ve çıkarım motorlarını canlı olarak değiştirin.

```bash
# en uygun cihazda FPS sayacı
python local_python/05_webcam.py

# başka bir kamera
CAMERA=1 python local_python/05_webcam.py

# motoru canlı değiştirin
yolo predict model=yolo26n_openvino_model source=0 device=intel:gpu show=True

# takip numaralarıyla poz tahmini
yolo track model=yolo26n-pose.pt source=0 show=True
```

## Adım 2: Dışa aktarma ve büyük hız yarışı :package:

```bash
yolo export model=yolo26n.pt format=onnx simplify=True      # her yerde çalışır

# Intel CPU, GPU ve NPU için
yolo export model=yolo26n.pt format=openvino
yolo export model=yolo26n.pt format=engine half=True        # NVIDIA TensorRT FP16

# dışa aktarır, sonra her formatı her cihazda yarıştırır
python local_python/02_export_model.py
```

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", simplify=True)  # her yerde çalışır
model.export(format="openvino")  # Intel CPU, GPU ve NPU için
model.export(format="engine", half=True)  # NVIDIA TensorRT FP16

YOLO("yolo26n_openvino_model").predict("bus.jpg", device="intel:npu")  # evet, NPU üzerinde de çalışır
```

### Atölye bilgisayarındaki sonuçlar

Bu tablo, Arc B390 grafik birimi ve dahili NPU bulunan Intel Core Ultra X7 358H işlemcide `02_export_model.py` betiğinin çıktısıdır. `yolo26n.pt` modeli 640 boyutunda, `bus.jpg` üzerinde 100 kez çalıştırıldı ve medyan değerler alındı. Uçtan uca süre, çıkarım süresine ön işleme ve son işlemeyi de ekler. FPS ise 1000'in uçtan uca süreye bölümüdür.

| Format | Cihaz | Çıkarım (ms) | Uçtan uca (ms) | FPS |
|---|---|---:|---:|---:|
| PyTorch | cpu | 14.39 | 15.50 | 65 |
| PyTorch | xpu | 3.53 | 5.74 | 174 |
| ONNX | cpu | 28.38 | 67.14 | 15 |
| OpenVINO | intel:cpu | 14.31 | 15.71 | 64 |
| OpenVINO | intel:gpu | 1.87 | 2.97 | 337 |
| OpenVINO | intel:npu | 3.09 | 4.54 | 220 |

!!! success "Öne çıkanlar"

    Dahili Arc GPU üzerinde çalışan OpenVINO, CPU üzerindeki PyTorch'un **5 katı FPS** veriyor (65 yerine 337). NPU ise çok az enerji harcarken dizüstü GPU'sundaki PyTorch'u bile geçiyor (174 yerine 220 FPS). Bu yüzden sürekli açık kalan kameralar için ideal bir seçenek. Model ve ağırlıklar aynı, değişen tek şey çalışma ortamı.

!!! info "NVIDIA GPU'nuz varsa"

    Yarış CUDA bulduğunda tabloya kendiliğinden bir TensorRT FP16 satırı ekler. Colab'da T4 çalışma ortamını seçerseniz notebook bu adımı sizin için yapar. Rust da ONNX dosyasından doğrudan TensorRT FP16 çalıştırabilir, ayrıntılar Adım 3.4'te.

## Adım 3: Ultralytics Inference ile yüksek hızda çalıştırma :crab:

[![Crates.io](https://img.shields.io/crates/v/ultralytics-inference?logo=rust&logoColor=white&label=crates.io&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![docs.rs](https://img.shields.io/docsrs/ultralytics-inference?logo=docs.rs&logoColor=white&label=docs.rs&color=CE422B)](https://docs.rs/ultralytics-inference)
[![Downloads](https://img.shields.io/crates/d/ultralytics-inference?logo=rust&logoColor=white&label=downloads&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![MSRV](https://img.shields.io/crates/msrv/ultralytics-inference?logo=rust&logoColor=white&color=CE422B)](https://crates.io/crates/ultralytics-inference)
[![CI](https://github.com/ultralytics/inference/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/inference/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/ultralytics/inference/branch/main/graph/badge.svg)](https://app.codecov.io/github/ultralytics/inference)
[![License](https://img.shields.io/crates/l/ultralytics-inference?label=license&color=blue)](https://github.com/ultralytics/inference/blob/main/LICENSE)
[![dependency status](https://deps.rs/repo/github/ultralytics/inference/status.svg)](https://deps.rs/repo/github/ultralytics/inference)
[![arXiv](https://img.shields.io/badge/arXiv-2606.03748-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.03748)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)

Ultralytics Inference, Ultralytics YOLO için geliştirilmiş bir Rust kütüphanesi ve CLI aracıdır. ONNX ve OpenVINO formatındaki modelleri okur, yedi görevin tamamını destekler ve CPU, CUDA, TensorRT, CoreML, OpenVINO gibi pek çok altyapıda çalışır. Tek ve küçük bir uygulama dosyasıdır, çalışırken Python gerektirmez.

### 3.1 CLI aracı

```bash
cargo install ultralytics-inference --features "openvino-native,video"

# OpenVINO native altyapısının çalışma kütüphanesini bulması gerekir, pip paketi bunun için yeterlidir
export LD_LIBRARY_PATH=$(python -c "import openvino, os; print(os.path.dirname(openvino.__file__))")/libs

# tek görsel
ultralytics-inference predict --model yolo26n.onnx --source bus.jpg

# bir klasör
ultralytics-inference predict --model yolo26n.onnx --source images/ --batch 4

# Intel GPU üzerinde bir video
ultralytics-inference predict --model yolo26n_openvino_model --device intel:gpu --source video.mp4

# NPU üzerinde webcam
ultralytics-inference predict --model yolo26n_openvino_model --device intel:npu --source 0 --show

# istediğiniz görev, model otomatik iner
ultralytics-inference predict --task pose
```

İsterseniz tüm adımları atölye betiğine bırakabilirsiniz:

```bash
# görsel, klasör, video, tüm görevler, Intel cihazları ve Rust uygulaması
./local_inference/run_cli.sh all
CAMERA=50 ./local_inference/run_cli.sh webcam
```

### 3.2 Kütüphaneyi kendi uygulamanızda kullanın

`local_inference/` klasörü, yaklaşık 200 satırda dört farklı modu olan eksiksiz bir Cargo projesidir.

```toml
[dependencies]
ultralytics-inference = { version = "0.0.54", features = ["openvino-native", "video"] }
```

```rust
use ultralytics_inference::{Device, InferenceConfig, Source, SourceIterator, YOLOModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = InferenceConfig::new().with_confidence(0.25).with_device(Device::IntelGpu);
    let mut model = YOLOModel::load_with_config("yolo26n_openvino_model", config)?;

    // Bellekte her an tek bir kare bulunur, 10 saatlik bir yayın da 10 saniyelik bir klip kadar RAM kullanır
    for item in SourceIterator::new(Source::from("video.mp4"))? {
        let (frame, meta) = item?;
        for result in model.predict_image(&frame, meta.path)? {
            println!("{}", result.detection_summary());
        }
    }
    Ok(())
}
```

```bash
cargo build --release --manifest-path local_inference/Cargo.toml --features "openvino,video"
BIN=local_inference/target/release/gdg-inference
$BIN image yolo26n.onnx bus.jpg                                       # tek görsel

# klasördeki görselleri gruplar halinde işler
$BIN batch yolo26n.onnx datasets/coco8/images/val 4

# video, webcam ya da RTSP yayını
DEVICE=intel:gpu $BIN video yolo26n_openvino_model video.mp4

# gecikme ölçümü
DEVICE=intel:npu $BIN bench yolo26n_openvino_model bus.jpg 300
```

### 3.3 Rust ile yedi görevin tamamı

`--task` parametresini verdiğinizde CLI ilgili YOLO26 nano modelini indirir, örnek görseller üzerinde çalıştırır ve sonuçları `runs/<task>/` klasörüne kaydeder. Aşağıdaki tüm görseller atölye bilgisayarında Rust CLI ile, hiç Python kullanılmadan üretildi.

```bash
# yedi görevi tek seferde çalıştırır
./local_inference/run_cli.sh tasks

# tek bir görevi kendi görselinizde çalıştırır
ultralytics-inference predict --task segment --source bus.jpg
```

<div class="gallery" markdown>
<figure markdown>
![Rust ile nesne tespiti](assets/img/rust_detect.jpg)
<figcaption>detect, Python ile aynı kutular</figcaption>
</figure>
<figure markdown>
![Rust ile örnek segmentasyonu](assets/img/rust_segment.jpg)
<figcaption>segment</figcaption>
</figure>
<figure markdown>
![Rust ile anlamsal segmentasyon](assets/img/rust_semantic.jpg)
<figcaption>semantic</figcaption>
</figure>
<figure markdown>
![Rust ile derinlik tahmini](assets/img/rust_depth.jpg)
<figcaption>depth</figcaption>
</figure>
<figure markdown>
![Rust ile sınıflandırma](assets/img/rust_classify.jpg)
<figcaption>classify (minibus 0.53)</figcaption>
</figure>
<figure markdown>
![Rust ile poz tahmini](assets/img/rust_pose.jpg)
<figcaption>pose</figcaption>
</figure>
<figure markdown>
![Rust ile yönlü nesne tespiti](assets/img/rust_obb.jpg)
<figcaption>obb, limandaki tüm gemiler ve rıhtımlar tek tek bulunuyor</figcaption>
</figure>
</div>

### 3.4 ONNX üzerinden doğrudan NVIDIA TensorRT FP16

Rust tarafında TensorRT kullanmak için Adım 2'deki `.engine` dosyasına ihtiyacınız yok. Ultralytics Inference, ONNX modelini ONNX Runtime TensorRT execution provider katmanına verir. Bu katman ilk çalıştırmada bir FP16 motoru derler ve modelin yanındaki `.trt_cache/` klasörüne kaydeder. İlk yükleme yaklaşık bir dakika sürer, sonraki yüklemeler ise saniyeler içinde açılır.

```bash
cargo install ultralytics-inference --features "nvidia,video"
ultralytics-inference predict --model yolo26n.onnx --source bus.jpg --device tensorrt:0 --quantize 16

cargo build --release --manifest-path local_inference/Cargo.toml --features "nvidia,video"

# varsayılan olarak FP16
DEVICE=tensorrt:0 local_inference/target/release/gdg-inference bench yolo26n.onnx bus.jpg 300
```

```rust
let config = InferenceConfig::new().with_device(Device::TensorRt(0)).with_quantize(Quantization::Fp16);
let mut model = YOLOModel::load_with_config("yolo26n.onnx", config)?;
```

!!! info "TensorRT için gerekenler"

    Bir NVIDIA sürücüsünün yanında CUDA 13 runtime, cuDNN 9 ve TensorRT 10 kütüphanelerinin de kütüphane yolunda bulunması gerekir. Hazır ONNX Runtime yalnızca CUDA 13 için derlenmiştir. CLI'a `--rect false` verirseniz her kare aynı boyutta olur ve yalnızca tek bir motor derlenir. TensorRT olmadan sadece CUDA kullanmak isterseniz `--device cuda:0` yazmanız yeterlidir.

### Sonuçlar: Python ve Rust karşılaştırması

Aynı bilgisayar ve aynı `yolo26n` modelleri kullanıldı. Her ölçüm 300 kez çalıştırıldı ve üç denemenin en iyi medyan değeri alındı.

<div class="wide" markdown>
![Her motor ve cihazda Python ile Rust saniyedeki kare sayısı](assets/img/fps_tr_light.png#only-light)
![Her motor ve cihazda Python ile Rust saniyedeki kare sayısı](assets/img/fps_tr_dark.png#only-dark)
</div>

| Motor | Cihaz | Çıkarım (ms) | Uçtan uca (ms) | FPS |
|---|---|---:|---:|---:|
| ONNX Runtime | cpu | 15.65 | 16.33 | 61 |
| OpenVINO | intel:cpu | 24.63 | 25.30 | 40 |
| OpenVINO, THREADS=16 | intel:cpu | 15.04 | 15.80 | 63 |
| OpenVINO | intel:gpu | 1.91 | 2.40 | 417 |
| OpenVINO | intel:npu | 2.19 | 2.73 | 366 |

620 karelik bir kalabalık videosunda işleme hızı ve bellek kullanımı:

| Çalışma ortamı | Motor ve cihaz | Kare | FPS | En yüksek bellek |
|---|---|---:|---:|---:|
| Python Ultralytics YOLO `predict(stream=True)` | intel:gpu üzerinde OpenVINO | 620 | 283 | 942 MB |
| Rust `gdg-inference video` (Ultralytics Inference kütüphanesi) | intel:gpu üzerinde OpenVINO | 620 | 400 | 284 MB |

!!! success "Öne çıkanlar"

    Rust uygulaması yalnızca **37 MB** boyutunda ve Python ortamına ihtiyaç duymuyor. Karşılaştırma için, atölyenin Python `.venv` klasörü 7.9 GB yer kaplıyor. Uçtan uca ölçümlerde Rust, Python'u GPU, NPU ve ONNX Runtime üzerinde geride bırakıyor. CPU'da ise thread sayısı ayarlandığında Python ile aynı seviyeye geliyor. Videoyu **1.4 kat daha hızlı** işliyor ve **3.3 kat daha az bellek** kullanıyor. CPU üzerindeki ONNX Runtime süresi Python'da 67.14 ms iken Rust'ta 16.33 ms'ye düşüyor.

!!! tip "Hibrit işlemcilerde CPU thread sayısı"

    Intel hibrit işlemcilerde Rust tarafındaki OpenVINO, kullanacağı thread sayısını varsayılan olarak kendisi belirler. Atölye uygulamasına `THREADS=16` (tüm çekirdekler) verirseniz ya da kodda `.with_threads(16)` kullanırsanız CPU üzerindeki çıkarım yaklaşık 1.6 kat hızlanır.

## Mini görevler :trophy:

- [ ] `yolo solutions count model=yolo26n_openvino_model device=intel:npu source=0 show=True` komutuyla kameranın önünden geçen kişileri NPU üzerinde sayın.
- [ ] `02_export_model.py` ile kendi bilgisayarınızdaki en hızlı cihazı bulun ve sonuç tablosunu paylaşın.
- [ ] Telefonunuzdan on fotoğraf seçip otomatik etiketleyin ve Ultralytics Platform'a veri seti olarak yükleyin.
- [ ] `yolo track model=yolo26n-pose.pt source=0 show=True` komutuyla webcam üzerinden arkadaşlarınızı takip edin.
- [ ] Rust uygulamasını derleyin ve kendi bilgisayarınızda Python'un FPS değerini geçin.

## Sorun giderme :wrench:

??? question "Webcam 0 numaralı cihazla açılmıyor"

    Yeni dizüstü bilgisayarların çoğu kamerayı bir Intel IPU ve v4l2loopback cihazı üzerinden sunar. `v4l2-ctl --list-devices` komutuyla cihazları listeleyin ve loopback cihazının numarasını kullanın. Örneğin `05_webcam.py` ve `run_cli.sh` için `CAMERA=50`, `yolo` CLI için `source=50` yazabilirsiniz. Bu kameralarda `yolo` CLI penceresi birkaç saniye sonra donarsa, kareleri doğrudan OpenCV ile okuyan `05_webcam.py` betiğini kullanın.

??? question "OpenVINO intel:gpu ya da intel:npu cihazını listelemiyor"

    `python -c "import openvino as ov; print(ov.Core().available_devices)"` komutunu çalıştırın. Listede GPU ya da NPU yoksa dağıtımınız için Intel GPU compute runtime paketini ve NPU sürücüsünü kurun, ardından bilgisayarı yeniden başlatın.

??? question "Rust CLI libopenvino_c kütüphanesini bulamıyor"

    Adım 3.1'de gösterildiği gibi `LD_LIBRARY_PATH` değişkenini `openvino` pip paketinin `libs` klasörüne yönlendirin. `run_cli.sh` bu ayarı sizin için otomatik yapar.

??? question "PyTorch GPU'mu görmüyor"

    PyTorch'un neleri gördüğünü öğrenmek için `python local_python/00_check_env.py` komutunu çalıştırın. NVIDIA kartlar için `./setup_env.sh cuda` ile kurulan CUDA sürümü ve güncel bir sürücü gerekir. Intel GPU'lar için `./setup_env.sh` ile kurulan XPU sürümü ve güncel bir Intel GPU sürücüsü gerekir. Apple silicon ise `./setup_env.sh cpu` ile kurulan sürümde MPS üzerinden çalışır.

??? question "Eğitim Ultralytics Platform'da görünmüyor"

    Python 3.11 ya da üzeri, `ultralytics` 8.4.120 ya da üzeri, `yolo login` ile kaydedilmiş geçerli bir anahtar ve `username/project` biçiminde bir `project` değeri gerekir.

## Depo haritası :file_folder:

| Yol | Açıklama |
|---|---|
| `setup_env.sh` | CUDA, Intel XPU ya da CPU için PyTorch ile OpenVINO içeren `.venv` ortamını kurar |
| `local_python/00_check_env.py` | PyTorch cihazınızı, OpenVINO cihazlarını, Rust CLI aracını ve Platform anahtarınızı gösterir |
| `local_python/01_hybrid_train.py` | Canlı Ultralytics Platform metrikleriyle yerel eğitim yapar |
| `local_python/02_export_model.py` | ONNX ve OpenVINO'ya aktarır, sonra her formatı her cihazda yarıştırır |
| `local_python/03_all_tasks.py` | detect, segment, semantic, depth, classify, pose ve obb görevlerini çalıştırır |
| `local_python/04_track.py` | Videodaki her kişiyi takip eder |
| `local_python/05_webcam.py` | FPS sayacıyla canlı webcam görüntüsü gösterir |
| `local_python/06_solutions.py` | Sayma, ısı haritası, gizlilik bulanıklaştırma ve hız tahmini yapar |
| `local_python/07_auto_annotate.py` | Ultralytics YOLO ve SAM ile otomatik etiketleme yapar |
| `local_python/08_platform_tour.py` | Bulut GPU durumunu, hesabınızı ve projelerinizi gösterir |
| `local_python/device_utils.py` | NVIDIA CUDA, Intel XPU, Apple MPS ya da CPU arasından seçim yapar |
| `local_inference/` | `ultralytics-inference` için Rust Cargo projesi ve `run_cli.sh` betiği |
| `notebooks/hybrid_yolo_workshop.ipynb` | Atölyenin tamamını Google Colab üzerinde sunar |
| `tools/` | Notebook üretici, sayfa grafikleri ve GIF dosyaları ile yazım stili kontrol aracı |

## İletişimde kalalım :handshake:

<div class="qr" markdown>
<div markdown>
![Atölye sayfası QR kodu](assets/img/qr-workshop.svg)

**Bu atölye sayfası**
</div>
<div markdown>
![Ultralytics Platform QR kodu](assets/img/qr-platform.svg)

**Ultralytics Platform'a katılın**
</div>
</div>

- :rocket: [Ultralytics Platform](https://platform.ultralytics.com) ve [Platform dokümantasyonu](https://docs.ultralytics.com/platform/)
- :crab: [GitHub'da Ultralytics Inference](https://github.com/ultralytics/inference) ve [crates.io sayfası](https://crates.io/crates/ultralytics-inference)
- :books: [Ultralytics YOLO dokümantasyonu](https://docs.ultralytics.com)
- :speech_balloon: [Ultralytics Discord topluluğu](https://discord.com/invite/ultralytics)

Bugün hiçbir GPU geride kalmadı. Katıldığınız için teşekkürler Samsun! :tada:
