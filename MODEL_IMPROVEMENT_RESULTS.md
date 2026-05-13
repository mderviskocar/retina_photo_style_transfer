# Model Improvement Results

GPU: NVIDIA GeForce RTX 4060 Laptop GPU

Dataset split:

- Train: 2562 images
- Validation: 550 images
- Test: 550 images

## Yapilan Iyilestirmeler

- PyTorch/CUDA egitim hatti kullanildi.
- ImageNet agirliklariyla transfer learning uygulandi.
- Hafif fundus-safe augmentation eklendi.
- Label smoothing eklendi.
- ReduceLROnPlateau scheduler eklendi.
- Validation macro F1 uzerinden best checkpoint kaydi ve early stopping eklendi.
- ResNet18, ResNet50 ve EfficientNet-B0 denendi.
- 320x320 goruntu cozunurlugu denendi.
- Focal loss denendi.
- EfficientNet-B1 denendi.

## En Iyi Accuracy Sonucu

| Model | Preprocess | Class Weight | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| EfficientNet-B0, 320px, focal loss | Original | No | 0.8182 | 0.6465 |
| EfficientNet-B1, 320px | Original | No | 0.8164 | 0.6613 |
| EfficientNet-B0, 320px | Original | No | 0.8145 | 0.6710 |
| EfficientNet-B0 | Original | No | 0.8091 | 0.6360 |
| EfficientNet-B0 | Original | Yes | 0.7873 | 0.6375 |
| ResNet50 | Original | Yes | 0.7836 | 0.6147 |
| ResNet18 | Original | Yes | 0.7818 | 0.6267 |
| ResNet18 | Photo style transfer | Yes | 0.7564 | 0.5974 |

Accuracy hedefi icin secilecek model:

```text
outputs/models/torch_baseline_original_efficientnet_b0_320_focal_nocw.pt
```

Macro F1 hedefi icin secilecek model:

```text
outputs/models/torch_baseline_original_efficientnet_b0_320_nocw.pt
```

Bu veri seti dengesiz oldugu icin raporda accuracy ile birlikte macro F1 de
verilmelidir.
