"""
Keras model builders.

Includes:
1. A small baseline CNN.
2. Transfer learning with EfficientNetB0 or ResNet50.
"""

from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from . import config
except ImportError:  # Allows: python src/model.py
    import config


def compile_model(
    model: keras.Model,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """Compile a multiclass classifier."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_augmentation() -> keras.Sequential:
    """Light data augmentation that is safe for fundus classification."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.08),
        ],
        name="augmentation",
    )


def build_baseline_cnn(
    input_shape=config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    learning_rate: float = config.LEARNING_RATE,
    use_augmentation: bool = True,
) -> keras.Model:
    """Build a compact CNN for a quick baseline."""
    inputs = keras.Input(shape=input_shape, name="image")
    x = build_augmentation()(inputs) if use_augmentation else inputs
    x = layers.Rescaling(1.0 / 255.0)(x)

    # Three convolution blocks keep the model fast enough for CPU/Colab tests.
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="diagnosis")(x)

    model = keras.Model(inputs, outputs, name="baseline_cnn")
    return compile_model(model, learning_rate=learning_rate)


def build_transfer_model(
    model_type: str = "efficientnetb0",
    input_shape=config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    learning_rate: float = config.LEARNING_RATE,
    weights: Optional[str] = "imagenet",
    train_base: bool = False,
    use_augmentation: bool = True,
) -> keras.Model:
    """
    Build a transfer-learning classifier.

    The convolutional base is frozen by default for shorter training time.
    """
    model_type = model_type.lower()
    inputs = keras.Input(shape=input_shape, name="image")
    x = build_augmentation()(inputs) if use_augmentation else inputs

    if model_type == "efficientnetb0":
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
        base_input = x
    elif model_type == "resnet50":
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
        base_input = keras.applications.resnet50.preprocess_input(x)
    else:
        raise ValueError("model_type must be one of: baseline_cnn, efficientnetb0, resnet50")

    base_model.trainable = train_base

    # training=False keeps BatchNorm stable when the base is frozen.
    features = base_model(base_input, training=train_base)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.35)(features)
    features = layers.Dense(256, activation="relu")(features)
    features = layers.Dropout(0.25)(features)
    outputs = layers.Dense(num_classes, activation="softmax", name="diagnosis")(features)

    model = keras.Model(inputs, outputs, name=f"{model_type}_classifier")
    return compile_model(model, learning_rate=learning_rate)


def build_model(
    model_type: str = "efficientnetb0",
    input_shape=config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    learning_rate: float = config.LEARNING_RATE,
    weights: Optional[str] = "imagenet",
    train_base: bool = False,
    use_augmentation: bool = True,
) -> keras.Model:
    """Factory function used by train.py."""
    model_type = model_type.lower()
    if model_type == "baseline_cnn":
        return build_baseline_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
            use_augmentation=use_augmentation,
        )

    return build_transfer_model(
        model_type=model_type,
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weights=weights,
        train_base=train_base,
        use_augmentation=use_augmentation,
    )


def save_model_summary(model: keras.Model, output_path: Path) -> None:
    """Save model.summary() to a text file for the report folder."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        model.summary(print_fn=lambda line: file.write(line + "\n"))
