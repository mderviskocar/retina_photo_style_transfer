"""
Dataset loading and tf.data helpers for APTOS 2019.

The Kaggle test folder has no labels, so train/validation/test splits are
created from train.csv for supervised evaluation.
"""

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    from . import config
    from . import preprocessing
except ImportError:  # Allows: python src/dataset.py
    import config
    import preprocessing


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")
AUTOTUNE = tf.data.AUTOTUNE


def find_image_path(image_id: str, image_dir: Path) -> Path:
    """Find an image file by id_code, supporting common image extensions."""
    image_id = str(image_id)
    direct_path = image_dir / image_id
    if direct_path.suffix and direct_path.exists():
        return direct_path

    for extension in SUPPORTED_EXTENSIONS:
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate

    # Return the expected APTOS path so the later error message is clear.
    return image_dir / f"{image_id}.png"


def load_aptos_dataframe(
    csv_path: Path = config.TRAIN_CSV,
    image_dir: Path = config.TRAIN_IMAGE_DIR,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Load APTOS train.csv and attach absolute image paths.

    Expected CSV columns:
        id_code, diagnosis
    """
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"train_images folder not found: {image_dir}")

    df = pd.read_csv(csv_path)
    required_columns = {"id_code", "diagnosis"}
    missing_columns = required_columns.difference(df.columns)
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
    """Check if each class has enough samples for stratified splitting."""
    counts = pd.Series(labels).value_counts()
    return len(counts) > 1 and bool((counts >= 2).all())


def _safe_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use stratification when possible, otherwise fall back to random split."""
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
    """Create train/validation/test splits from labeled training metadata."""
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


def _load_image_py(path_tensor: tf.Tensor, preprocess_mode: str, image_size: Tuple[int, int]) -> np.ndarray:
    """Bridge OpenCV preprocessing into tf.data through tf.py_function."""
    path = path_tensor.numpy().decode("utf-8")
    image = preprocessing.preprocess_for_model(path, mode=preprocess_mode, image_size=image_size)
    return image.astype(np.float32)


def make_tf_dataset(
    df: pd.DataFrame,
    preprocess_mode: str,
    batch_size: int = config.BATCH_SIZE,
    image_size: Tuple[int, int] = config.IMG_SIZE,
    training: bool = False,
    seed: int = config.SEED,
) -> tf.data.Dataset:
    """
    Build a TensorFlow dataset that outputs (image, label).

    Images are float32 in 0-255 range. Model-specific preprocessing is handled
    inside model.py, which keeps CNN and transfer models consistent.
    """
    paths = df["image_path"].astype(str).to_numpy()
    labels = df["diagnosis"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    def map_fn(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.py_function(
            func=lambda p: _load_image_py(p, preprocess_mode, image_size),
            inp=[path],
            Tout=tf.float32,
        )
        image.set_shape((image_size[0], image_size[1], 3))
        label.set_shape(())
        return image, label

    return (
        dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )


def make_class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    """Compute class weights to reduce the effect of APTOS class imbalance."""
    y = train_df["diagnosis"].astype(int).to_numpy()
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def class_distribution(df: pd.DataFrame) -> Dict[int, int]:
    """Return class counts in label order."""
    counts = df["diagnosis"].value_counts().sort_index()
    return {int(class_id): int(counts.get(class_id, 0)) for class_id in range(config.NUM_CLASSES)}


def list_unlabeled_test_images(test_image_dir: Path = config.TEST_IMAGE_DIR) -> pd.DataFrame:
    """
    List Kaggle test images. These have no labels and are not used for metrics.
    """
    test_image_dir = Path(test_image_dir)
    image_paths = []
    for extension in SUPPORTED_EXTENSIONS:
        image_paths.extend(test_image_dir.glob(f"*{extension}"))

    return pd.DataFrame(
        {
            "id_code": [path.stem for path in sorted(image_paths)],
            "image_path": [str(path) for path in sorted(image_paths)],
        }
    )
