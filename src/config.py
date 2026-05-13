"""
Project-wide configuration.

Paths can be changed here or overridden with environment variables in Colab.
Example:
    RETINA_DATA_DIR=/content/aptos python src/train.py
"""

from pathlib import Path
import os


# Root folder of the project. In Colab, set RETINA_PROJECT_ROOT if needed.
PROJECT_ROOT = Path(
    os.getenv("RETINA_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

# Dataset paths. APTOS 2019 uses train.csv and train_images/*.png.
DATA_DIR = Path(os.getenv("RETINA_DATA_DIR", PROJECT_ROOT / "data")).resolve()
TRAIN_CSV = Path(os.getenv("RETINA_TRAIN_CSV", DATA_DIR / "train.csv")).resolve()
TRAIN_IMAGE_DIR = Path(
    os.getenv("RETINA_TRAIN_IMAGE_DIR", DATA_DIR / "train_images")
).resolve()
TEST_IMAGE_DIR = Path(
    os.getenv("RETINA_TEST_IMAGE_DIR", DATA_DIR / "test_images")
).resolve()
_STYLE_REFERENCE_ENV = os.getenv("RETINA_STYLE_REFERENCE_IMAGE")
PHOTO_STYLE_REFERENCE_IMAGE = (
    Path(_STYLE_REFERENCE_ENV).resolve() if _STYLE_REFERENCE_ENV else None
)

# Output folders.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_SAMPLE_DIR = OUTPUT_DIR / "processed_samples"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULT_DIR = OUTPUT_DIR / "results"

# Model and training defaults. Keep these small enough for quick Colab tests.
IMG_SIZE = (224, 224)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-4
SEED = 42

# Split ratios are taken from train.csv because Kaggle test_images has no labels.
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Class labels for APTOS diabetic retinopathy levels.
CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# Two main experiments requested for the project.
EXPERIMENTS = {
    "baseline_original": {
        "title": "Deney 1 - Orijinal Görüntüler",
        "preprocess_mode": "none",
    },
    "normalized_style": {
        "title": "Deney 2 - Normalize/Style Standardized Görüntüler",
        "preprocess_mode": "standardized",
    },
    "photo_style_transfer": {
        "title": "Deney 3 - Reference Photo Style Transfer",
        "preprocess_mode": "photo_style_transfer",
    },
}

# Target LAB statistics for a simple style-normalization-like transform.
# These values create a consistent fundus color/illumination style without
# requiring a separate reference-style image set.
STYLE_TARGET_LAB_MEAN = (138.0, 145.0, 142.0)
STYLE_TARGET_LAB_STD = (42.0, 14.0, 18.0)


def ensure_output_dirs() -> None:
    """Create all output folders used by training/evaluation scripts."""
    for folder in [OUTPUT_DIR, PROCESSED_SAMPLE_DIR, MODEL_DIR, PLOT_DIR, RESULT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
