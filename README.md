# Fundus Görüntü Standardizasyonu ve Diyabetik Retinopati Sınıflandırması

Bu proje, APTOS 2019 Blindness Detection veri setindeki fundus/retina görüntülerinden diyabetik retinopati seviyesini sınıflandırmak için hazırlanmış akademik bir derin öğrenme çalışmasıdır.



## Amaç

Proje iki ana deneyin ve bir ek photo style transfer deneyinin sonucunu karşılaştırır:

1. Deney 1: Orijinal görüntülerle eğitim
   - Görüntüler yalnızca model giriş boyutuna resize/padding yapılır.

2. Deney 2: Normalize/preprocessed görüntülerle eğitim
   - Siyah kenar temizleme/crop
   - Resize/padding
   - Işık normalizasyonu
   - Kontrast artırma
   - CLAHE
   - Renk normalizasyonu
   - LAB tabanlı style normalization benzeri dönüşüm

3. Ek deney: Photo style transfer
   - Referans bir fundus görüntüsünün LAB renk/ışık istatistikleri diğer görüntülere aktarılır.
   - Retina yapısı ve damar/lezyon konumu korunur; bu klasik neural artistic style transfer değildir.

Sonuçta şu soruya cevap aranır:

> Görüntü standardizasyonu diyabetik retinopati sınıflandırma başarısını artırdı mı?

Ana karşılaştırma metriği olarak `macro F1-score` kullanılır. Çünkü APTOS sınıfları dengesizdir.

Terminoloji notu: Bu projede "iritasyon" adlı bir işlem yoktur. Kastedilen büyük ihtimalle `augmentasyon` ise, evet: eğitim sırasında hafif fundus-safe augmentation kullanılmıştır. Kastedilen `iterasyon/epoch` ise, eğitim döngüsü epoch bazlıdır ve her epoch içinde batch iterasyonları yapılır. Kastedilen `standardizasyon` ise, evet: normalize/style-standardized preprocessing hattı uygulanmıştır.

## Sınıflar

| Etiket | Sınıf |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Klasör Yapısı

```text
derma_retina_style_project/
│
├── data/
│   ├── train_images/
│   ├── test_images/
│   └── train.csv
│
├── outputs/
│   ├── processed_samples/
│   ├── models/
│   ├── plots/
│   └── results/
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── train_torch.py
│   ├── evaluate.py
│   └── visualize.py
│
├── requirements.txt
└── README.md
```

## Veri Setini Yerleştirme

APTOS 2019 dosyalarını şu şekilde yerleştirin:

```text
data/train.csv
data/train_images/*.png
data/test_images/*.png
```

`train.csv` dosyasında Kaggle formatındaki şu kolonlar beklenir:

```text
id_code,diagnosis
```

Kaggle `test_images` klasöründe etiket bulunmadığı için accuracy/F1 gibi metrikler `train.csv` içinden ayrılan hold-out test bölümüyle hesaplanır.

## Kurulum

Yerel ortamda:

```bash
cd derma_retina_style_project
pip install -r requirements.txt
```

Google Colab’da:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Sonra proje klasörüne geçip bağımlılıkları kurabilirsiniz:

```bash
%cd /content/drive/MyDrive/derma_retina_style_project
!pip install -r requirements.txt
```

## Eğitim

### Güncel Önerilen PyTorch/CUDA Hattı

Bu projedeki en iyi kayıtlı sonuçlar PyTorch hattından gelmiştir:

```bash
python src/train_torch.py --experiment baseline_original --model-type efficientnet_b0 --epochs 3 --batch-size 32 --weights imagenet --fine-tune --no-class-weights --image-size 320 --loss focal --run-tag 320_focal_nocw
```

GPU varsa varsayılan `--device auto` CUDA kullanır; GPU yoksa CPU'ya düşer. Üç deneyin tamamını birlikte çalıştırmak için:

```bash
python src/train_torch.py --experiment all --model-type resnet18 --epochs 3 --batch-size 32 --weights imagenet --fine-tune
```

### TensorFlow/Keras Hattı

İki deneyi EfficientNetB0 ile çalıştırmak için:

```bash
python src/train.py --experiment both --model-type efficientnetb0 --epochs 3
```

Daha hızlı test için küçük örneklem:

```bash
python src/train.py --experiment both --model-type baseline_cnn --epochs 2 --limit 300
```

ResNet50 kullanmak için:

```bash
python src/train.py --experiment both --model-type resnet50 --epochs 3
```

ImageNet ağırlıkları indirilmesin, rastgele ağırlıkla başlasın:

```bash
python src/train.py --experiment both --model-type efficientnetb0 --weights none
```

## Çıktılar

Eğitim tamamlandığında şu çıktılar üretilir:

```text
outputs/processed_samples/
  *_stages.png

outputs/models/
  baseline_original_efficientnetb0.keras
  normalized_style_efficientnetb0.keras
  torch_baseline_original_efficientnet_b0_320_focal_nocw.pt

outputs/plots/
  *_history.png
  *_confusion_matrix.png
  experiment_comparison.png
  torch_experiment_comparison.png

outputs/results/
  *_metrics.json
  *_classification_report.json
  *_predictions.csv
  *_history.csv
  split_distribution.csv
  experiment_comparison.csv
  torch_experiment_comparison.csv
  model_selection_summary.csv
  standardization_answer.txt
  torch_standardization_answer.txt
```

`standardization_answer.txt` dosyası proje sorusuna doğrudan cevap verir:

```text
Görüntü standardizasyonu diyabetik retinopati sınıflandırma başarısını artırdı mı?
```

Karar, normalize/style-standardized deneyin `macro F1-score` değerinin baseline deneyden yüksek olup olmamasına göre yazılır.

## Kayıtlı En İyi Sonuçlar

Mevcut `outputs/results/model_selection_summary.csv` dosyasına göre:

| Model | Preprocess | Accuracy | Macro F1 |
|---|---|---:|---:|
| EfficientNet-B0 320px + focal loss | Original | 0.8182 | 0.6465 |
| EfficientNet-B0 320px | Original | 0.8145 | 0.6710 |
| EfficientNet-B1 320px | Original | 0.8164 | 0.6613 |
| ResNet18 | Normalize/style standardized | 0.7727 | 0.6064 |
| ResNet18 | Photo style transfer | 0.7564 | 0.5974 |

Accuracy için seçilecek kayıtlı model:

```text
outputs/models/torch_baseline_original_efficientnet_b0_320_focal_nocw.pt
```

Macro F1 için seçilecek kayıtlı model:

```text
outputs/models/torch_baseline_original_efficientnet_b0_320_nocw.pt
```

Bu sonuçlara göre mevcut koşulda en başarılı eğitim orijinal görüntülerle yapılmıştır; normalize/style ve photo style transfer deneyleri bu kayıtlı koşuda orijinal görüntü eğitimini geçmemiştir.

## Dosyaların Görevi

`src/config.py`
: Veri yolları, eğitim parametreleri, sınıf adları, deney tanımları ve output klasörleri.

`src/preprocessing.py`
: Siyah kenar kırpma, resize, ışık normalizasyonu, kontrast artırma, CLAHE, renk normalizasyonu ve style-normalization benzeri LAB dönüşümü.

`src/dataset.py`
: `train.csv` okuma, görüntü yollarını bulma, train/validation/test ayrımı, `tf.data.Dataset` üretimi ve class weight hesaplama.

`src/model.py`
: Baseline CNN, EfficientNetB0 ve ResNet50 modelleri.

`src/train.py`
: İki deneyi eğitir, modelleri kaydeder, metrikleri hesaplar ve karşılaştırma sonucunu üretir.

`src/train_torch.py`
: PyTorch/CUDA eğitim hattıdır. ResNet18, ResNet50, EfficientNet-B0/B1, focal loss, label smoothing, early stopping, ReduceLROnPlateau ve mixed precision desteği içerir.

`src/evaluate.py`
: Accuracy, precision, recall, F1-score, confusion matrix ve classification report hesaplar. Tek başına da çalıştırılabilir.

`src/visualize.py`
: Örnek preprocessing görselleri, eğitim grafikleri, confusion matrix ve deney karşılaştırma grafiği üretir.

## Tek Başına Değerlendirme

Kaydedilmiş bir modeli tekrar değerlendirmek için:

```bash
python src/evaluate.py \
  --model-path outputs/models/normalized_style_efficientnetb0.keras \
  --preprocess-mode standardized \
  --experiment-name normalized_style_eval
```

Baseline model için:

```bash
python src/evaluate.py \
  --model-path outputs/models/baseline_original_efficientnetb0.keras \
  --preprocess-mode none \
  --experiment-name baseline_original_eval
```

## Rapor İçin Kısa Açıklamalar

Bu çalışmada APTOS 2019 fundus görüntüleri kullanılarak diyabetik retinopati seviyeleri beş sınıfta sınıflandırılmıştır. Temel amaç, görüntü standardizasyonunun sınıflandırma başarısına etkisini incelemektir.

İlk deneyde görüntüler doğrudan modele verilmeden önce yalnızca ortak giriş boyutuna getirilmiştir. İkinci deneyde ise siyah kenar kırpma, ışık/renk normalizasyonu, CLAHE ve LAB tabanlı style-normalization benzeri dönüşüm uygulanmıştır.

Her iki deney aynı train/validation/test ayrımıyla eğitilmiştir. Böylece performans farkının veri bölünmesinden değil, görüntü standardizasyonu adımından kaynaklanması hedeflenmiştir.

Performans accuracy, precision, recall, F1-score ve confusion matrix ile değerlendirilmiştir. Veri setinde sınıf dağılımı dengesiz olduğu için karşılaştırmada macro F1-score özellikle dikkate alınmıştır.

## Notlar

- Varsayılan model `EfficientNetB0` ve varsayılan epoch sayısı `3` olarak ayarlanmıştır.
- GPU varsa TensorFlow otomatik kullanır. GPU yoksa CPU ile de çalışır.
- Dosya yolları `src/config.py` içinden veya environment variable ile değiştirilebilir.
- Eğitim süresini azaltmak için transfer learning tabanı varsayılan olarak dondurulmuştur.
