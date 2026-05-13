"""
Evaluation utilities for trained diabetic retinopathy classifiers.

Metrics:
accuracy, precision, recall, F1-score, confusion matrix, and classification
report. This module can be imported by train.py or run separately.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    from . import config
    from . import dataset as dataset_utils
except ImportError:  # Allows: python src/evaluate.py
    import config
    import dataset as dataset_utils


def load_keras_model(model_path: Path) -> tf.keras.Model:
    """Load a saved Keras model for inference."""
    model_path = Path(model_path)
    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment_name: str,
) -> Tuple[Dict[str, float], np.ndarray, Dict]:
    """Compute main classification metrics and confusion matrix."""
    labels = list(range(config.NUM_CLASSES))
    target_names = [config.CLASS_NAMES[label] for label in labels]

    accuracy = accuracy_score(y_true, y_pred)
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

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "experiment": experiment_name,
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }
    return metrics, cm, report


def evaluate_model_on_dataset(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    y_true: np.ndarray,
    experiment_name: str,
    sample_df: Optional[pd.DataFrame] = None,
    result_dir: Path = config.RESULT_DIR,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Run prediction, compute metrics, and save tabular results."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.asarray(y_true).astype(int)

    metrics, cm, report = evaluate_predictions(y_true, y_pred, experiment_name)

    metrics_path = result_dir / f"{experiment_name}_metrics.json"
    report_path = result_dir / f"{experiment_name}_classification_report.json"
    prediction_path = result_dir / f"{experiment_name}_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    if sample_df is not None:
        prediction_df = sample_df[["id_code", "image_path", "diagnosis"]].copy()
    else:
        prediction_df = pd.DataFrame({"diagnosis": y_true})

    prediction_df["prediction"] = y_pred
    prediction_df["prediction_name"] = [
        config.CLASS_NAMES[int(class_id)] for class_id in y_pred
    ]
    for class_id in range(config.NUM_CLASSES):
        prediction_df[f"prob_class_{class_id}"] = probabilities[:, class_id]
    prediction_df.to_csv(prediction_path, index=False)

    return metrics, cm, y_pred


def parse_args() -> argparse.Namespace:
    """CLI arguments for standalone evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a saved APTOS DR model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--preprocess-mode",
        choices=[
            "none",
            "preprocessed",
            "without_style",
            "standardized",
            "normalized",
            "style",
            "photo_style_transfer",
            "photo_style",
            "style_transfer",
        ],
        default="standardized",
    )
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--train-csv", type=Path, default=config.TRAIN_CSV)
    parser.add_argument("--image-dir", type=Path, default=config.TRAIN_IMAGE_DIR)
    parser.add_argument("--experiment-name", type=str, default="standalone_eval")
    return parser.parse_args()


def main() -> None:
    """Evaluate a saved model on the held-out split from train.csv."""
    args = parse_args()
    config.ensure_output_dirs()

    df = dataset_utils.load_aptos_dataframe(args.train_csv, args.image_dir)
    _, _, test_df = dataset_utils.create_splits(df)

    test_ds = dataset_utils.make_tf_dataset(
        test_df,
        preprocess_mode=args.preprocess_mode,
        batch_size=args.batch_size,
        training=False,
    )

    model = load_keras_model(args.model_path)
    metrics, cm, _ = evaluate_model_on_dataset(
        model=model,
        test_ds=test_ds,
        y_true=test_df["diagnosis"].to_numpy(),
        experiment_name=args.experiment_name,
        sample_df=test_df,
    )

    print(pd.Series(metrics).to_string())
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
