"""
Train and compare two experiments:

Deney 1: Original images with only resize/padding.
Deney 2: Normalized/style-standardized images.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import tensorflow as tf
from tensorflow import keras

try:
    from . import config
    from . import dataset as dataset_utils
    from . import evaluate
    from . import model as model_utils
    from . import visualize
except ImportError:  # Allows: python src/train.py
    import config
    import dataset as dataset_utils
    import evaluate
    import model as model_utils
    import visualize


def setup_runtime() -> None:
    """Use GPU automatically when TensorFlow can see one."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"GPU detected: {len(gpus)} device(s)")
    else:
        print("No GPU detected. Training will run on CPU.")


def parse_weights(weights: str) -> Optional[str]:
    """Convert CLI weight argument to Keras-compatible value."""
    if weights.lower() in {"none", "null", "random"}:
        return None
    return weights


def parse_args() -> argparse.Namespace:
    """Training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train APTOS DR experiments.")
    parser.add_argument(
        "--experiment",
        choices=["both"] + list(config.EXPERIMENTS.keys()),
        default="both",
        help="Run both experiments or a single experiment.",
    )
    parser.add_argument(
        "--model-type",
        choices=["baseline_cnn", "efficientnetb0", "resnet50"],
        default="efficientnetb0",
        help="Classifier architecture used for each image experiment.",
    )
    parser.add_argument("--weights", type=str, default="imagenet", help="imagenet or none")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--train-csv", type=Path, default=config.TRAIN_CSV)
    parser.add_argument("--image-dir", type=Path, default=config.TRAIN_IMAGE_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional small-sample debug limit.")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze transfer base model.")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    return parser.parse_args()


def selected_experiments(selection: str) -> List[str]:
    """Return experiment keys in the requested order."""
    if selection == "both":
        return ["baseline_original", "normalized_style"]
    return [selection]


def maybe_limit_dataframe(df: pd.DataFrame, limit: Optional[int]) -> pd.DataFrame:
    """Optionally use a smaller dataset for quick code tests."""
    if limit is None or limit >= len(df):
        return df
    return df.sample(n=limit, random_state=config.SEED).reset_index(drop=True)


def callbacks_for(run_name: str) -> List[keras.callbacks.Callback]:
    """Create callbacks for short, reproducible training runs."""
    model_path = config.MODEL_DIR / f"{run_name}.keras"
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def run_experiment(
    experiment_key: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_weights: Optional[Dict[int, float]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Train one experiment and save its outputs."""
    experiment_cfg = config.EXPERIMENTS[experiment_key]
    preprocess_mode = experiment_cfg["preprocess_mode"]
    run_name = f"{experiment_key}_{args.model_type}"

    print(f"\n=== {experiment_cfg['title']} | model={args.model_type} ===")
    print(f"Preprocess mode: {preprocess_mode}")

    train_ds = dataset_utils.make_tf_dataset(
        train_df,
        preprocess_mode=preprocess_mode,
        batch_size=args.batch_size,
        training=True,
    )
    val_ds = dataset_utils.make_tf_dataset(
        val_df,
        preprocess_mode=preprocess_mode,
        batch_size=args.batch_size,
        training=False,
    )
    test_ds = dataset_utils.make_tf_dataset(
        test_df,
        preprocess_mode=preprocess_mode,
        batch_size=args.batch_size,
        training=False,
    )

    keras_model = model_utils.build_model(
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        weights=parse_weights(args.weights),
        train_base=args.fine_tune,
        use_augmentation=not args.no_augmentation,
    )

    model_utils.save_model_summary(
        keras_model,
        config.RESULT_DIR / f"{run_name}_model_summary.txt",
    )

    history = keras_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks_for(run_name),
        verbose=1,
    )

    # Save the restored best model for standalone evaluation.
    keras_model.save(str(config.MODEL_DIR / f"{run_name}.keras"))

    pd.DataFrame(history.history).to_csv(
        config.RESULT_DIR / f"{run_name}_history.csv",
        index=False,
    )
    visualize.plot_history(
        history,
        config.PLOT_DIR / f"{run_name}_history.png",
        title=experiment_cfg["title"],
    )

    metrics, cm, _ = evaluate.evaluate_model_on_dataset(
        model=keras_model,
        test_ds=test_ds,
        y_true=test_df["diagnosis"].to_numpy(),
        experiment_name=run_name,
        sample_df=test_df,
    )
    metrics["experiment_key"] = experiment_key
    metrics["preprocess_mode"] = preprocess_mode
    metrics["model_type"] = args.model_type

    visualize.plot_confusion_matrix(
        cm,
        config.PLOT_DIR / f"{run_name}_confusion_matrix.png",
        title=f"{experiment_cfg['title']} - Confusion Matrix",
    )

    return metrics


def save_comparison(metrics_list: List[Dict[str, float]]) -> None:
    """Save comparison table, plot, and a direct answer to the research question."""
    metrics_df = pd.DataFrame(metrics_list)
    comparison_path = config.RESULT_DIR / "experiment_comparison.csv"
    metrics_df.to_csv(comparison_path, index=False)
    visualize.plot_metrics_comparison(metrics_df)

    answer_path = config.RESULT_DIR / "standardization_answer.txt"
    answer = "No experiment result is available yet."

    experiment_keys = set(metrics_df["experiment_key"]) if "experiment_key" in metrics_df else set()
    if {"baseline_original", "normalized_style"}.issubset(experiment_keys):
        baseline = metrics_df.loc[metrics_df["experiment_key"] == "baseline_original"].iloc[0]
        normalized = metrics_df.loc[metrics_df["experiment_key"] == "normalized_style"].iloc[0]

        f1_delta = normalized["f1_macro"] - baseline["f1_macro"]
        acc_delta = normalized["accuracy"] - baseline["accuracy"]

        if f1_delta > 0:
            verdict = "Evet"
            detail = "normalize/style-standardized görüntüler macro F1 skorunu artırdı."
        elif f1_delta < 0:
            verdict = "Hayır"
            detail = "normalize/style-standardized görüntüler macro F1 skorunu artırmadı."
        else:
            verdict = "Belirsiz"
            detail = "macro F1 skoru iki deneyde aynı çıktı."

        answer = (
            f"Soru: Görüntü standardizasyonu diyabetik retinopati sınıflandırma "
            f"başarısını artırdı mı?\n"
            f"Cevap: {verdict}. {detail}\n"
            f"Baseline macro F1: {baseline['f1_macro']:.4f}, "
            f"Normalized macro F1: {normalized['f1_macro']:.4f}, "
            f"Delta F1: {f1_delta:+.4f}\n"
            f"Baseline accuracy: {baseline['accuracy']:.4f}, "
            f"Normalized accuracy: {normalized['accuracy']:.4f}, "
            f"Delta accuracy: {acc_delta:+.4f}\n"
            "Yorum: Ana karar metriği olarak macro F1 kullanıldı; çünkü APTOS "
            "sınıfları dengesizdir."
        )
    elif not metrics_df.empty:
        row = metrics_df.iloc[0]
        answer = (
            "Tek deney çalıştırıldı; baseline-normalized karşılaştırması yok.\n"
            f"Deney: {row['experiment']}\n"
            f"Preprocess: {row['preprocess_mode']}, model: {row['model_type']}, "
            f"accuracy: {row['accuracy']:.4f}, macro F1: {row['f1_macro']:.4f}\n"
            "Standardizasyon sorusunu cevaplamak için --experiment both ile "
            "baseline_original ve normalized_style birlikte çalıştırılmalıdır."
        )

    with answer_path.open("w", encoding="utf-8") as file:
        file.write(answer)

    print(f"\nComparison table saved to: {comparison_path}")
    print(f"Research-question answer saved to: {answer_path}")


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    config.ensure_output_dirs()
    setup_runtime()

    df = dataset_utils.load_aptos_dataframe(args.train_csv, args.image_dir)
    df = maybe_limit_dataframe(df, args.limit)

    train_df, val_df, test_df = dataset_utils.create_splits(df)
    distributions = {
        "train": dataset_utils.class_distribution(train_df),
        "validation": dataset_utils.class_distribution(val_df),
        "test": dataset_utils.class_distribution(test_df),
    }
    visualize.save_split_distribution(distributions)

    print("Class distributions:")
    print(json.dumps(distributions, indent=2))

    # Save sample images once. They illustrate the transformation pipeline.
    visualize.save_preprocessing_samples(train_df)

    class_weights = None if args.no_class_weights else dataset_utils.make_class_weights(train_df)
    if class_weights is not None:
        print(f"Class weights: {class_weights}")

    metrics_list = []
    for experiment_key in selected_experiments(args.experiment):
        metrics = run_experiment(
            experiment_key=experiment_key,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_weights=class_weights,
            args=args,
        )
        metrics_list.append(metrics)

    save_comparison(metrics_list)


if __name__ == "__main__":
    main()
