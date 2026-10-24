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
<strong>[:key: Adım 0 Kurulum](setup.md)</strong>
Tek bir betik, donanımınıza uygun PyTorch sürümünü ve OpenVINO'yu içeren bir `.venv` ortamı kurar ve bilgisayarınızdaki tüm hızlandırıcıları kontrol eder.
</div>
<div class="card" markdown>
<strong>[:weight_lifter: Adım 1 Eğitim](train.md)</strong>
`yolo26n.pt` modelini kendi bilgisayarınızda eğitirken metrikleri Ultralytics Platform üzerinde canlı izleyin. Dilerseniz tek tıkla bulut GPU'sunda da eğitebilirsiniz.
</div>
<div class="card" markdown>
<strong>[:art: Deneme alanı](playground.md)</strong>
Yedi görev, nesne takibi, Solutions, otomatik etiketleme ve canlı webcam, her biri birkaç satır kodla.
</div>
<div class="card" markdown>
<strong>[:package: Adım 2 Dışa aktarma](export.md)</strong>
Modeli ONNX, OpenVINO ve TensorRT formatlarına aktarın ve her formatı sahip olduğunuz tüm cihazlarda yarıştırın.
</div>
<div class="card" markdown>
<strong>[:crab: Adım 3 Çalıştırma](rust.md)</strong>
Aynı modeli `ultralytics-inference` CLI aracı ve Rust kütüphanesi ile çalıştırın. Bellek kullanımı sabit kalır ve Python'a hiç gerek kalmaz.
</div>
<div class="card" markdown>
<strong>[:rocket: İleri seviye](advanced.md)</strong>
Modeli INT8 olarak dışa aktarıp küçültün ve hızlandırın, ardından bilgisayarınızdaki bütün çipleri aynı anda çalıştırın.
</div>
<div class="card" markdown>
<strong>[:tada: Kapanış](wrap-up.md)</strong>
Mini görevler, sorun giderme ipuçları ve bugünden sonra devam etmek için bağlantılar.
</div>
</div>

[:material-play: Adım 0 ile başlayın](setup.md){ .md-button .md-button--primary }

## Atölye programı :alarm_clock:

| Zaman | Bölüm | İçerik |
|---|---|---|
| 00:00 | Isınma | Kimde GPU var, kimde NPU var? Herkes katılıyor, kimse geride kalmıyor. |
| 00:03 | [Adım 0](setup.md) | Evde kurduğunuz ortamı kontrol edip Ultralytics Platform'a giriş yapıyoruz. |
| 00:07 | [Adım 1](train.md) | Dizüstü bilgisayarda hibrit eğitim yapıyor, metrikleri Platform'da canlı izliyor ve bulut eğitimini tanıyoruz. |
| 00:20 | [Deneme alanı](playground.md) | Yedi görev, nesne takibi, Solutions ve otomatik etiketlemeyi birkaç satır kodla deniyoruz. |
| 00:30 | [Adım 2](export.md) | Modeli ONNX, OpenVINO ve TensorRT formatlarına aktarıp büyük hız yarışını başlatıyoruz. |
| 00:38 | [Adım 3](rust.md) | Rust ile Ultralytics Inference, CLI, toplu işleme, video ve webcam kullanımını görüyoruz. |
| 00:48 | [İleri seviye](advanced.md) | INT8 dışa aktarma yapıyor ve bütün çipleri aynı anda çalıştırıyoruz. |
| 00:55 | [Kapanış](wrap-up.md) | Canlı webcam gösterisi, faydalı bağlantılar ve soru cevap ile bitiriyoruz. |
