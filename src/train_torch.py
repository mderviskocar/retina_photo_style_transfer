"""
Train diabetic retinopathy classifiers with PyTorch/CUDA.

This script mirrors the TensorFlow experiment flow, but uses PyTorch so native
Windows can train on an NVIDIA GPU.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms

try:
    from . import config
    from . import preprocessing
    from . import visualize
except ImportError:  # Allows: python src/train_torch.py
    import config
    import preprocessing
    import visualize


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retina classifiers with PyTorch/CUDA.")
    parser.add_argument(
        "--experiment",
        choices=["both", "all"] + list(config.EXPERIMENTS.keys()),
        default="both",
    )
    parser.add_argument(
        "--model-type",
        choices=["small_cnn", "resnet18", "resnet50", "efficientnet_b0", "efficientnet_b1"],
        default="resnet18",
    )
    parser.add_argument("--weights", choices=["none", "imagenet"], default="none")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-csv", type=Path, default=config.TRAIN_CSV)
    parser.add_argument("--image-dir", type=Path, default=config.TRAIN_IMAGE_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. auto uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--fine-tune", action="store_true", help="Train pretrained base layers too.")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=config.IMG_SIZE[0])
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--run-tag", type=str, default="")
    return parser.parse_args()


def find_image_path(image_id: str, image_dir: Path) -> Path:
    image_id = str(image_id)
    direct_path = image_dir / image_id
    if direct_path.suffix and direct_path.exists():
        return direct_path

    for extension in SUPPORTED_EXTENSIONS:
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate

    return image_dir / f"{image_id}.png"


def load_dataframe(csv_path: Path, image_dir: Path, strict: bool = True) -> pd.DataFrame:
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"train_images folder not found: {image_dir}")

    df = pd.read_csv(csv_path)
    missing_columns = {"id_code", "diagnosis"}.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {sorted(missing_columns)}")

    df = df[["id_code", "diagnosis"]].copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df["image_path"] = df["id_code"].apply(lambda image_id: find_image_path(image_id, image_dir))

    missing_files = [path for path in df["image_path"] if not Path(path).exists()]
    if strict and missing_files:
        preview = "\n".join(str(path) for path in missing_files[:10])
        raise FileNotFoundError(
            f"{len(missing_files)} image files were not found. First examples:\n{preview}"
        )

    df["image_path"] = df["image_path"].astype(str)
    return df


def _can_stratify(labels: Iterable[int]) -> bool:
    counts = pd.Series(labels).value_counts()
    return len(counts) > 1 and bool((counts >= 2).all())


def _safe_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify = df["diagnosis"] if _can_stratify(df["diagnosis"]) else None
    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(df, test_size=test_size, random_state=seed, stratify=None)


def create_splits(
    df: pd.DataFrame,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = _safe_train_test_split(df, test_size=test_size, seed=seed)
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = _safe_train_test_split(
        train_val_df,
        test_size=relative_val_size,
        seed=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def maybe_limit_dataframe(df: pd.DataFrame, limit: Optional[int], seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(df):
        return df
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def selected_experiments(selection: str) -> List[str]:
    if selection == "both":
        return ["baseline_original", "normalized_style"]
    if selection == "all":
        return list(config.EXPERIMENTS.keys())
    return [selection]


def configure_photo_style_reference(train_df: pd.DataFrame) -> Path:
    """Choose a train-split reference image for photo style transfer."""
    configured_reference = getattr(config, "PHOTO_STYLE_REFERENCE_IMAGE", None)
    if configured_reference:
        return Path(configured_reference)

    candidates = train_df.loc[train_df["diagnosis"] == 0]
    if candidates.empty:
        candidates = train_df
    reference_path = Path(str(candidates.iloc[0]["image_path"])).resolve()
    config.PHOTO_STYLE_REFERENCE_IMAGE = reference_path
    return reference_path


def class_distribution(df: pd.DataFrame) -> Dict[int, int]:
    counts = df["diagnosis"].value_counts().sort_index()
    return {class_id: int(counts.get(class_id, 0)) for class_id in range(config.NUM_CLASSES)}


def compute_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    y = train_df["diagnosis"].astype(int).to_numpy()
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weights = np.ones(config.NUM_CLASSES, dtype=np.float32)
    weights[classes] = class_weights.astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_train_augmentation() -> transforms.Compose:
    """Light geometric augmentation that preserves fundus anatomy."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=8,
                        translate=(0.02, 0.02),
                        scale=(0.95, 1.05),
                        fill=0,
                    )
                ],
                p=0.75,
            ),
        ]
    )


class RetinaTorchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        preprocess_mode: str,
        image_size: Tuple[int, int],
        training: bool = False,
        use_augmentation: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.preprocess_mode = preprocess_mode
        self.image_size = image_size
        self.training = training
        self.augmentation = build_train_augmentation() if training and use_augmentation else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image = preprocessing.preprocess_for_model(
            row["image_path"],
            mode=self.preprocess_mode,
            image_size=self.image_size,
        )
        image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        if self.augmentation is not None:
            image = self.augmentation(image)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        label = torch.tensor(int(row["diagnosis"]), dtype=torch.long)
        return image, label


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image))


class FocalLoss(nn.Module):
    """Multiclass focal loss for class-imbalanced datasets."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_model(model_type: str, weights: str, fine_tune: bool) -> nn.Module:
    model_type = model_type.lower()
    use_imagenet = weights == "imagenet"

    if model_type == "small_cnn":
        return SmallCNN()

    if model_type == "resnet18":
        model_weights = models.ResNet18_Weights.DEFAULT if use_imagenet else None
        model = models.resnet18(weights=model_weights)
        if use_imagenet and not fine_tune:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
        return model

    if model_type == "resnet50":
        model_weights = models.ResNet50_Weights.DEFAULT if use_imagenet else None
        model = models.resnet50(weights=model_weights)
        if use_imagenet and not fine_tune:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
        return model

    if model_type == "efficientnet_b0":
        model_weights = models.EfficientNet_B0_Weights.DEFAULT if use_imagenet else None
        model = models.efficientnet_b0(weights=model_weights)
        if use_imagenet and not fine_tune:
            for parameter in model.parameters():
                parameter.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, config.NUM_CLASSES)
        return model

    if model_type == "efficientnet_b1":
        model_weights = models.EfficientNet_B1_Weights.DEFAULT if use_imagenet else None
        model = models.efficientnet_b1(weights=model_weights)
        if use_imagenet and not fine_tune:
            for parameter in model.parameters():
                parameter.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, config.NUM_CLASSES)
        return model

    raise ValueError(f"Unknown model type: {model_type}")


def make_loader(
    df: pd.DataFrame,
    preprocess_mode: str,
    batch_size: int,
    training: bool,
    num_workers: int,
    device: torch.device,
    use_augmentation: bool,
    image_size: Tuple[int, int],
) -> DataLoader:
    return DataLoader(
        RetinaTorchDataset(
            df,
            preprocess_mode=preprocess_mode,
            image_size=image_size,
            training=training,
            use_augmentation=use_augmentation,
        ),
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += batch_size

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities = []
    labels = []
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, batch_labels)

        batch_probs = torch.softmax(logits, dim=1)
        probabilities.append(batch_probs.cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
        total_loss += float(loss.detach()) * batch_labels.size(0)
        total += batch_labels.size(0)

    return np.vstack(probabilities), np.concatenate(labels), total_loss / max(total, 1)


def metrics_from_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    experiment_name: str,
) -> Tuple[Dict[str, float], np.ndarray, Dict]:
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(config.NUM_CLASSES))
    target_names = [config.CLASS_NAMES[label] for label in labels]
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "experiment": experiment_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return metrics, cm, report


def save_prediction_outputs(
    run_name: str,
    sample_df: pd.DataFrame,
    probabilities: np.ndarray,
    metrics: Dict[str, float],
    cm: np.ndarray,
    report: Dict,
) -> None:
    result_dir = config.RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)

    with (result_dir / f"{run_name}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with (result_dir / f"{run_name}_classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    predictions = sample_df[["id_code", "image_path", "diagnosis"]].copy()
    predictions["prediction"] = probabilities.argmax(axis=1)
    predictions["prediction_name"] = [
        config.CLASS_NAMES[int(class_id)] for class_id in predictions["prediction"]
    ]
    for class_id in range(config.NUM_CLASSES):
        predictions[f"prob_class_{class_id}"] = probabilities[:, class_id]
    predictions.to_csv(result_dir / f"{run_name}_predictions.csv", index=False)

    visualize.plot_confusion_matrix(
        cm,
        config.PLOT_DIR / f"{run_name}_confusion_matrix.png",
        title=f"{run_name} - Confusion Matrix",
    )


def run_experiment(
    experiment_key: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_weights: Optional[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    experiment_cfg = config.EXPERIMENTS[experiment_key]
    preprocess_mode = experiment_cfg["preprocess_mode"]
    run_name = f"torch_{experiment_key}_{args.model_type}"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    use_amp = device.type == "cuda" and not args.no_amp
    image_size = (args.image_size, args.image_size)

    print(f"\n=== {experiment_cfg['title']} | model={args.model_type} | device={device} ===")
    print(f"Preprocess mode: {preprocess_mode}")
    print(f"Image size: {image_size[0]}x{image_size[1]} | loss={args.loss}")

    train_loader = make_loader(
        train_df,
        preprocess_mode=preprocess_mode,
        batch_size=args.batch_size,
        training=True,
        num_workers=args.num_workers,
        device=device,
        use_augmentation=not args.no_augmentation,
        image_size=image_size,
    )
    val_loader = make_loader(
        val_df,
        preprocess_mode=preprocess_mode,
        batch_size=args.batch_size,
        training=False,
        num_workers=args.num_workers,
        device=device,
        use_augmentation=False,
        image_size=image_size,
    )
    test_loader = make_loader(
        test_df,
        preprocess_mode=preprocess_mode,
        batch_size=args.batch_size,
        training=False,
        num_workers=args.num_workers,
        device=device,
        use_augmentation=False,
        image_size=image_size,
    )

    model = build_model(args.model_type, weights=args.weights, fine_tune=args.fine_tune).to(device)
    if args.loss == "focal":
        criterion = FocalLoss(
            weight=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    history = []
    best_val_f1 = -1.0
    best_path = config.MODEL_DIR / f"{run_name}.pt"
    start_time = time.time()
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_probs, val_labels, val_loss = predict(model, val_loader, device=device, use_amp=use_amp)
        val_metrics, _, _ = metrics_from_predictions(
            val_labels,
            val_probs,
            experiment_name=f"{run_name}_val",
        )
        row = {
            "epoch": epoch,
            "loss": train_stats["loss"],
            "accuracy": train_stats["accuracy"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"loss={row['loss']:.4f} acc={row['accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_f1={row['val_f1_macro']:.4f} lr={row['learning_rate']:.2e}"
        )
        scheduler.step(row["val_f1_macro"])

        if row["val_f1_macro"] > best_val_f1 + args.min_delta:
            best_val_f1 = row["val_f1_macro"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": args.model_type,
                    "weights": args.weights,
                    "experiment_key": experiment_key,
                    "preprocess_mode": preprocess_mode,
                    "epoch": epoch,
                    "val_f1_macro": best_val_f1,
                    "class_names": config.CLASS_NAMES,
                    "photo_style_reference_image": str(getattr(config, "PHOTO_STYLE_REFERENCE_IMAGE", "")),
                    "image_size": image_size,
                    "loss": args.loss,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs; best val_f1={best_val_f1:.4f}")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    test_probs, y_true, test_loss = predict(model, test_loader, device=device, use_amp=use_amp)
    metrics, cm, report = metrics_from_predictions(y_true, test_probs, experiment_name=run_name)
    metrics["experiment_key"] = experiment_key
    metrics["preprocess_mode"] = preprocess_mode
    metrics["model_type"] = args.model_type
    metrics["device"] = str(device)
    metrics["test_loss"] = float(test_loss)
    metrics["seconds"] = float(time.time() - start_time)
    metrics["best_val_f1_macro"] = float(best_val_f1)
    metrics["photo_style_reference_image"] = str(getattr(config, "PHOTO_STYLE_REFERENCE_IMAGE", ""))
    metrics["image_size"] = int(args.image_size)
    metrics["loss"] = args.loss
    metrics["focal_gamma"] = float(args.focal_gamma)

    history_df = pd.DataFrame(history)
    history_df.to_csv(config.RESULT_DIR / f"{run_name}_history.csv", index=False)
    visualize.plot_history(
        {
            "loss": history_df["loss"].tolist(),
            "accuracy": history_df["accuracy"].tolist(),
            "val_loss": history_df["val_loss"].tolist(),
            "val_accuracy": history_df["val_accuracy"].tolist(),
        },
        config.PLOT_DIR / f"{run_name}_history.png",
        title=experiment_cfg["title"],
    )
    save_prediction_outputs(run_name, test_df, test_probs, metrics, cm, report)
    print(f"Saved best model: {best_path}")
    print(f"Test accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['f1_macro']:.4f}")
    return metrics


def save_comparison(metrics_list: List[Dict[str, float]]) -> None:
    metrics_df = pd.DataFrame(metrics_list)
    comparison_path = config.RESULT_DIR / "torch_experiment_comparison.csv"
    metrics_df.to_csv(comparison_path, index=False)
    visualize.plot_metrics_comparison(
        metrics_df,
        output_path=config.PLOT_DIR / "torch_experiment_comparison.png",
    )

    answer = "No PyTorch experiment result is available yet."
    experiment_keys = set(metrics_df["experiment_key"]) if "experiment_key" in metrics_df else set()
    if {"baseline_original", "normalized_style"}.issubset(experiment_keys):
        baseline = metrics_df.loc[metrics_df["experiment_key"] == "baseline_original"].iloc[0]
        normalized = metrics_df.loc[metrics_df["experiment_key"] == "normalized_style"].iloc[0]
        f1_delta = normalized["f1_macro"] - baseline["f1_macro"]
        verdict = "Evet" if f1_delta > 0 else "Hayir" if f1_delta < 0 else "Belirsiz"
        answer = (
            "Soru: Goruntu standardizasyonu diyabetik retinopati siniflandirma "
            "basarisini artirdi mi?\n"
            f"Cevap: {verdict}.\n"
            f"Baseline macro F1: {baseline['f1_macro']:.4f}, "
            f"Normalized macro F1: {normalized['f1_macro']:.4f}, "
            f"Delta F1: {f1_delta:+.4f}\n"
        )
    elif not metrics_df.empty:
        row = metrics_df.iloc[0]
        answer = (
            "Tek PyTorch deneyi calistirildi; baseline-normalized karsilastirmasi yok.\n"
            f"Deney: {row['experiment']}\n"
            f"Preprocess: {row['preprocess_mode']}, model: {row['model_type']}, "
            f"accuracy: {row['accuracy']:.4f}, macro F1: {row['f1_macro']:.4f}\n"
            "Standardizasyon sorusunu cevaplamak icin --experiment both veya "
            "--experiment all ile en az baseline_original ve normalized_style birlikte "
            "calistirilmalidir.\n"
        )

    if "photo_style_transfer" in experiment_keys:
        photo_style = metrics_df.loc[metrics_df["experiment_key"] == "photo_style_transfer"].iloc[0]
        reference_image = photo_style.get("photo_style_reference_image", "")
        answer += (
            f"Photo style transfer macro F1: {photo_style['f1_macro']:.4f}, "
            f"accuracy: {photo_style['accuracy']:.4f}\n"
            f"Photo style reference image: {reference_image}\n"
        )

    with (config.RESULT_DIR / "torch_standardization_answer.txt").open("w", encoding="utf-8") as file:
        file.write(answer)

    print(f"\nPyTorch comparison saved to: {comparison_path}")


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU.")

    df = load_dataframe(args.train_csv, args.image_dir)
    df = maybe_limit_dataframe(df, args.limit, seed=args.seed)
    train_df, val_df, test_df = create_splits(df, seed=args.seed)
    reference_path = configure_photo_style_reference(train_df)
    distributions = {
        "train": class_distribution(train_df),
        "validation": class_distribution(val_df),
        "test": class_distribution(test_df),
    }
    visualize.save_split_distribution(distributions, config.RESULT_DIR / "torch_split_distribution.csv")
    print("Class distributions:")
    print(json.dumps(distributions, indent=2))
    print(f"Photo style reference image: {reference_path}")

    class_weights = None if args.no_class_weights else compute_weights(train_df, device=device)
    if class_weights is not None:
        print(f"Class weights: {[round(float(weight), 4) for weight in class_weights.cpu()]}")

    metrics_list = []
    for experiment_key in selected_experiments(args.experiment):
        metrics = run_experiment(
            experiment_key=experiment_key,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_weights=class_weights,
            args=args,
            device=device,
        )
        metrics_list.append(metrics)

    save_comparison(metrics_list)


if __name__ == "__main__":
    main()
