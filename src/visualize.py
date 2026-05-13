"""
Visualization helpers for samples, training curves, confusion matrices, and
experiment comparison plots.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from . import config
    from . import preprocessing
except ImportError:  # Allows: python src/visualize.py
    import config
    import preprocessing


def _class_labels() -> Iterable[str]:
    """Return class names in label order."""
    return [config.CLASS_NAMES[index] for index in range(config.NUM_CLASSES)]


def save_preprocessing_samples(
    df: pd.DataFrame,
    output_dir: Path = config.PROCESSED_SAMPLE_DIR,
    max_samples: int = 5,
) -> None:
    """
    Save sample grids: original, preprocessed, and style-normalized.

    These images can be used directly in the project report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = min(max_samples, len(df))
    if sample_count == 0:
        return

    samples = df.sample(n=sample_count, random_state=config.SEED)
    for _, row in samples.iterrows():
        stages = preprocessing.build_visualization_stages(row["image_path"])

        stage_order = [
            ("original", "Original"),
            ("preprocessed", "Preprocessing"),
            ("standardized", "Normalize/Style"),
            ("photo_style_transfer", "Photo Style Transfer"),
        ]
        available_stages = [(key, title) for key, title in stage_order if key in stages]

        fig, axes = plt.subplots(1, len(available_stages), figsize=(3.5 * len(available_stages), 4))
        axes = np.atleast_1d(axes)

        for axis, (key, title) in zip(axes, available_stages):
            axis.imshow(stages[key])
            axis.set_title(title)
            axis.axis("off")

        label_name = config.CLASS_NAMES[int(row["diagnosis"])]
        fig.suptitle(f"{row['id_code']} | Label: {label_name}", fontsize=11)
        fig.tight_layout()

        output_path = output_dir / f"{row['id_code']}_stages.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_history(history, output_path: Path, title: str) -> None:
    """Plot training/validation accuracy and loss."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history_dict = history.history if hasattr(history, "history") else history
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history_dict.get("accuracy", []), marker="o", label="Train")
    axes[0].plot(epochs, history_dict.get("val_accuracy", []), marker="o", label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history_dict.get("loss", []), marker="o", label="Train")
    axes[1].plot(epochs, history_dict.get("val_loss", []), marker="o", label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str,
    normalize: bool = False,
) -> None:
    """Save a confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "d"
    matrix = cm
    if normalize:
        matrix = cm.astype(np.float32)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        fmt = ".2f"

    fig, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=_class_labels(),
        yticklabels=_class_labels(),
        cbar=False,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path = config.PLOT_DIR / "experiment_comparison.png",
) -> None:
    """Plot the main metrics for both experiments."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    available_columns = [column for column in metric_columns if column in metrics_df.columns]
    if not available_columns:
        return

    plot_df = metrics_df[["experiment"] + available_columns].melt(
        id_vars="experiment",
        var_name="metric",
        value_name="score",
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="score",
        hue="experiment",
        palette=["#2f6f9f", "#c45a3c", "#4f8f5d"],
        ax=axis,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Metric")
    axis.set_ylabel("Score")
    axis.set_title("Original vs Normalized/Style-Standardized")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Experiment", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_split_distribution(
    distributions: Dict[str, Dict[int, int]],
    output_path: Optional[Path] = None,
) -> None:
    """Save a class distribution table for train/validation/test splits."""
    output_path = output_path or (config.RESULT_DIR / "split_distribution.csv")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_name, distribution in distributions.items():
        for class_id, count in distribution.items():
            rows.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": config.CLASS_NAMES[class_id],
                    "count": count,
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)
