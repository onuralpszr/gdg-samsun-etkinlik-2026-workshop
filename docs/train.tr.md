---
title: "Adım 1: Ultralytics Platform ve hibrit eğitim"
hide:
  - navigation
---

# Adım 1: Ultralytics Platform ve hibrit eğitim :weight_lifter:

## 1.1 API anahtarınızı alın

1. [platform.ultralytics.com](https://platform.ultralytics.com) adresinden ücretsiz hesap açın. Yeni hesaplara başlangıç kredisi tanımlanır.
2. **Settings > API Keys** sayfasına gidin ve eğitim yetkisine sahip bir anahtar oluşturun. Anahtar yalnızca bir kez gösterildiği için hemen kopyalayın.
3. Terminalden giriş yapın. Anahtarınız Ultralytics ayarlarına kaydedilir.

```bash
yolo login ul_your_api_key

# ya da yalnızca bu terminal oturumu için
export ULTRALYTICS_API_KEY=ul_your_api_key
```

## 1.2 Yerelde eğitin, bulutta izleyin

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

`yolo26n.pt` modelinin `coco8.yaml` üzerinde on epoch eğitimi **0.941 mAP50** değerine ulaştı.

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

## 1.3 Ya da tek tıkla bulut GPU'sunda eğitin

1. Ultralytics Platform'da projenizi açın ve **New Model** düğmesine tıklayın.
2. **Official** başlığı altından `YOLO26n` modelini, ardından durumu `ready` olan bir veri setini seçin.
3. Epoch sayısını, batch boyutunu ve görsel boyutunu ayarlayıp bir GPU seçin. Hızlı bir deneme için RTX 2000 Ada yeterlidir.
4. **Start Training** düğmesine basın ve aynı panelin dolmasını izleyin.

Hangi bulut GPU'larının şu anda boşta olduğunu `ultralytics` paketiyle gelen Platform istemcisiyle görebilirsiniz. `yolo login` yaptıysanız planınızı, kredinizi ve projelerinizi de listeler.

```bash
python local_python/08_platform_tour.py
```

[:material-arrow-right: Sonraki: Deneme alanı](playground.md){ .md-button .md-button--primary .next-step }
