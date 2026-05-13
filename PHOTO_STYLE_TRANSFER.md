# Photo Style Transfer Bolumu

Bu projede photo style transfer, retina goruntulerinin anatomik yapisini
bozmadan bir referans fundus goruntusunun fotografik renk ve isik stilini
diger goruntulere aktarmak icin kullanilir.

## Yontem

Uygulanan yontem reference-based photorealistic photo style transfer olarak
tasarlanmistir:

1. Icerik goruntusu okunur.
2. Siyah kenarlar kirpilir ve goruntu model boyutuna getirilir.
3. Egitim bolumunden secilen referans fundus goruntusu ayni sekilde hazirlanir.
4. Goruntuler LAB renk uzayina cevrilir.
5. Icerik goruntusunun L, A ve B kanal istatistikleri, referans goruntunun kanal
   ortalama ve standart sapmalarina gore yeniden olceklenir.
6. Siyah arka plan korunur, retina yapisi ve damar dokusu piksel konumu olarak
   korunur.

Bu yontem klasik neural artistic style transfer degildir. Tibbi goruntulerde
sekil ve lezyon bilgisini bozmamak icin fotogercekci renk/aydinlatma stili
aktarimi kullanilmistir.

## Koddaki Karsiligi

Ana fonksiyonlar:

- `src/preprocessing.py`: `preprocess_photo_style_transfer`
- `src/preprocessing.py`: `transfer_lab_photo_style`
- `src/train_torch.py`: `--experiment photo_style_transfer`

Deney adi:

```powershell
--experiment photo_style_transfer
```

Tum deneyleri birlikte calistirma:

```powershell
$env:TORCH_HOME = "$PWD\outputs\torch_cache"
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" src\train_torch.py --experiment all --model-type resnet18 --epochs 3 --batch-size 32 --weights imagenet --fine-tune
```

## Son GPU Egitim Sonucu

Model: ResNet18
Donanim: NVIDIA GeForce RTX 4060 Laptop GPU
Epoch: 3

| Deney | Accuracy | Macro F1 |
|---|---:|---:|
| Orijinal goruntuler | 0.7818 | 0.6267 |
| Normalize/style standardized | 0.7727 | 0.6064 |
| Photo style transfer | 0.7564 | 0.5974 |

Bu kosuda photo style transfer konusu projeye deney olarak dahil edilmistir,
ancak en yuksek siniflandirma basarisi orijinal goruntulerde elde edilmistir.
