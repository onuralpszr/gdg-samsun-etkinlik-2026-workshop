---
title: "Kapanış"
---

# Kapanış :tada:

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

    [Adım 3.1](rust.md#31-cli-arac) bölümünde gösterildiği gibi `LD_LIBRARY_PATH` değişkenini `openvino` pip paketinin `libs` klasörüne yönlendirin. `run_cli.sh` bu ayarı sizin için otomatik yapar.

??? question "PyTorch GPU'mu görmüyor"

    PyTorch'un neleri gördüğünü öğrenmek için `python local_python/00_check_env.py` komutunu çalıştırın. NVIDIA kartlar için `./setup_env.sh cuda` ile kurulan CUDA sürümü ve güncel bir sürücü gerekir. Intel GPU'lar için `./setup_env.sh` ile kurulan XPU sürümü ve güncel bir Intel GPU sürücüsü gerekir. Apple silicon ise `./setup_env.sh cpu` ile kurulan sürümde MPS üzerinden çalışır.

??? question "Eğitim Ultralytics Platform'da görünmüyor"

    Python 3.11 ya da üzeri, `ultralytics` 8.4.120 ya da üzeri, `yolo login` ile kaydedilmiş geçerli bir anahtar ve `username/project` biçiminde bir `project` değeri gerekir.

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

[:material-home: Başa dön](index.md){ .md-button .md-button--primary .next-step }
