---
title: "İleri seviye: INT8 ve bütün çipler aynı anda"
hide:
  - navigation
---

# İleri seviye: INT8 ve bütün çipler aynı anda :rocket:

Erken mi bitirdiniz? Bu iki ek bölüm, elinizdeki bilgisayardan daha fazla performans almanızı sağlar.

## INT8: daha küçük ve daha hızlı bir model

INT8 dışa aktarma, ağırlıkları 32 bit ondalıklı sayılar yerine 8 bit tam sayılar olarak saklar. Ultralytics YOLO, modeli kalibre etmek için `data` içindeki birkaç görseli modelden geçirir. Bu yüzden tahmin yapacağınız görsellere benzeyen bir veri seti verin. OpenVINO modeli yaklaşık 10 MB'tan yaklaşık 3 MB'a iner ve mAP değeri yalnızca birkaç puan değişir.

```bash
# OpenVINO ile Intel CPU, GPU ve NPU
yolo export model=yolo26n.pt format=openvino int8=True data=coco8.yaml

# TensorRT ile NVIDIA GPU
yolo export model=yolo26n.pt format=engine int8=True data=coco8.yaml

# CoreML ile Apple silicon
yolo export model=yolo26n.pt format=coreml int8=True
```

Şimdi INT8'in neye mal olduğuna bakalım. `yolo val` komutunu önce tam hassasiyetli modelde, sonra INT8 modelde çalıştırın. Ardından mAP50-95 sütununu ve `Speed` satırındaki çıkarım süresini karşılaştırın.

```bash
# aynı cihazda tam hassasiyet ile INT8 karşılaştırması
yolo val model=yolo26n_openvino_model data=coco8.yaml device=intel:cpu
yolo val model=yolo26n_int8_openvino_model data=coco8.yaml device=intel:cpu

# sonra NPU'yu deneyin
yolo val model=yolo26n_int8_openvino_model data=coco8.yaml device=intel:npu

# NVIDIA üzerinde
yolo val model=yolo26n.engine data=coco8.yaml device=0
```

INT8 OpenVINO modeli Rust ile de çalışır:

```bash
ultralytics-inference predict --model yolo26n_int8_openvino_model --device intel:npu --source bus.jpg
```

!!! tip "Kendi verinizle kalibre edin"

    `coco8.yaml` yalnızca 8 görsel içerir ve gösterim için yeterlidir. Kendi modeliniz için `data` değerine kendi veri setinizin YAML dosyasını verin, böylece kalibrasyon gerçek görsellerinizi görür.

## Bütün çipler aynı anda

Bir dizüstü bilgisayarda birden fazla motor bulunur. Üç terminal açın ve her birine kendi modelini ve kendi cihazını verin. Hepsi aynı anda çalışır, böylece hiçbir çip boşta kalmaz.

```bash
# terminal 1: webcam'deki kişiler
yolo track model=yolo26n.pt source=0 device=0 show=True

# terminal 2: bir videodaki pozlar
yolo track model=yolo26n-pose.pt source=video.mp4 device=cpu show=True

# terminal 3: CPU üzerinde Rust, hiç Python yok
ultralytics-inference predict --model yolo26n.onnx --source video.mp4 --show
```

Modelleri ve `device` değerlerini bilgisayarınıza göre değiştirin:

| Bilgisayar | Terminal 1 | Terminal 2 | Terminal 3 |
|---|---|---|---|
| NVIDIA | `device=0` ile `yolo26n.pt` | `device=cpu` ile `yolo26n-pose.pt` | CPU üzerinde Rust |
| Intel Core Ultra | `device=intel:npu` ile `yolo26n_int8_openvino_model` | `device=intel:gpu` ile `yolo26n-pose_openvino_model` | CPU üzerinde Rust |
| Apple silicon | `device=mps` ile `yolo26n.pt` | `device=cpu` ile `yolo26n-pose.pt` | CPU üzerinde Rust |
| Sadece CPU | `device=cpu` ile `yolo26n.pt` | `device=intel:cpu` ile `yolo26n-pose_openvino_model` | CPU üzerinde Rust |

!!! info "Bir kamera, bir terminal"

    Çoğu webcam aynı anda yalnızca bir programda açılabilir. Webcam'i bir terminale, video dosyasını diğerlerine verin. `04_track.py` betiğinin indirdiği `solutions_ci_demo.mp4` örnek videosu bu iş için uygundur.

## Devam etmek için fikirler :bulb:

Atölyeden sonra deneyebileceğiniz birkaç fikir:

- [ ] Kendi fotoğraflarınızdan onunu Smart Annotation ile etiketleyin, Ultralytics Platform'da eğitin, INT8 olarak dışa aktarın ve modeli Rust ile çalıştırın.
- [ ] `ultralytics-inference` kütüphanesiyle tahmin sonuçlarını sunan küçük bir Rust web sunucusu yazın.
- [ ] Rust uygulamasıyla bir RTSP kamerasını ya da aynı anda birkaç kamerayı okuyun.
- [ ] INT8 modeli elinizdeki en küçük bilgisayarda çalıştırın, örneğin bir Raspberry Pi ya da eski bir dizüstü bilgisayarda.
- [ ] Bir segment ya da pose modelini INT8 olarak dışa aktarın ve doğruluğun ne kadarının korunduğuna bakın.

[:material-arrow-right: Sonraki: Kapanış](wrap-up.md){ .md-button .md-button--primary .next-step }
