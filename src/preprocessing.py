"""
Fundus image preprocessing and style-standardization utilities.

The baseline experiment uses only resizing. The normalized experiment uses:
black-border crop, resize, illumination correction, contrast enhancement,
CLAHE, color normalization, and LAB-statistics style normalization.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np

try:
    from . import config
except ImportError:  # Allows: python src/preprocessing.py
    import config


ImageInput = Union[str, Path, np.ndarray]
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")
_REFERENCE_CACHE: Dict[Tuple[str, Tuple[int, int]], np.ndarray] = {}


def read_rgb(image_path: Union[str, Path]) -> np.ndarray:
    """Read an image from disk as RGB uint8."""
    image_path = Path(image_path)
    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"Image could not be read: {image_path}") from exc

    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image safely to uint8 range."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def get_image(image_or_path: ImageInput) -> np.ndarray:
    """Accept either a path or an already loaded RGB image."""
    if isinstance(image_or_path, (str, Path)):
        return read_rgb(image_or_path)
    return ensure_uint8(image_or_path.copy())


def crop_black_borders(
    image: np.ndarray,
    threshold: int = 10,
    margin: int = 8,
) -> np.ndarray:
    """
    Remove mostly black borders around the retinal disk.

    APTOS images often contain large black corners. This crop finds pixels
    brighter than a small threshold and keeps the bounding box around them.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold

    if not np.any(mask):
        return image

    y_coords, x_coords = np.where(mask)
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, image.shape[0] - 1)
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, image.shape[1] - 1)

    return image[y_min : y_max + 1, x_min : x_max + 1]


def resize_with_padding(
    image: np.ndarray,
    size: Tuple[int, int] = config.IMG_SIZE,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Resize an image to the target size while keeping aspect ratio.

    Padding avoids geometric distortion of the retinal disk.
    """
    target_h, target_w = size
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Cannot resize an empty image.")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def normalize_illumination(image: np.ndarray, sigma: int = 30) -> np.ndarray:
    """
    Reduce uneven lighting by dividing each channel by a blurred background.

    This approximates shade correction without learning a separate model.
    """
    image_float = image.astype(np.float32)
    normalized = np.zeros_like(image_float)

    for channel in range(3):
        channel_image = image_float[:, :, channel]
        background = cv2.GaussianBlur(channel_image, (0, 0), sigmaX=sigma)
        background_mean = float(np.mean(background))
        normalized[:, :, channel] = channel_image / (background + 1.0) * background_mean

    return ensure_uint8(normalized)


def enhance_contrast(image: np.ndarray, alpha: float = 1.08, beta: int = 2) -> np.ndarray:
    """Apply a mild global contrast/brightness adjustment."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE on the L channel in LAB color space.

    CLAHE improves local contrast while limiting over-amplification of noise.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def gray_world_balance(image: np.ndarray) -> np.ndarray:
    """
    Normalize color cast with the gray-world assumption.

    The average red, green, and blue responses are scaled toward a common mean.
    """
    image_float = image.astype(np.float32)
    foreground_mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) > 8
    if not np.any(foreground_mask):
        return image

    # Estimate color cast from the retinal foreground, not from black padding.
    channel_means = image_float[foreground_mask].mean(axis=0)
    gray_mean = channel_means.mean()
    scale = gray_mean / (channel_means + 1e-6)
    balanced = image_float * scale
    balanced[~foreground_mask] = image_float[~foreground_mask]
    return ensure_uint8(balanced)


def style_normalize_lab(
    image: np.ndarray,
    target_mean: Tuple[float, float, float] = config.STYLE_TARGET_LAB_MEAN,
    target_std: Tuple[float, float, float] = config.STYLE_TARGET_LAB_STD,
) -> np.ndarray:
    """
    Apply a simple style-normalization-like transform in LAB space.

    This is not neural style transfer. It standardizes color statistics toward
    a canonical fundus appearance, which is lightweight and Colab-friendly.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    foreground_mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) > 8
    if not np.any(foreground_mask):
        return image

    for channel in range(3):
        current = lab[:, :, channel]
        foreground_values = current[foreground_mask]
        mean = foreground_values.mean()
        std = foreground_values.std() + 1e-6
        transformed = (current - mean) / std * target_std[channel] + target_mean[channel]
        current[foreground_mask] = transformed[foreground_mask]
        lab[:, :, channel] = current

    lab = ensure_uint8(lab)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def find_default_photo_style_reference() -> Path:
    """Return the configured reference image or the first available train image."""
    configured_reference = getattr(config, "PHOTO_STYLE_REFERENCE_IMAGE", None)
    if configured_reference:
        reference_path = Path(configured_reference)
        if reference_path.exists():
            return reference_path

    candidates = []
    for extension in SUPPORTED_EXTENSIONS:
        candidates.extend(Path(config.TRAIN_IMAGE_DIR).glob(f"*{extension}"))

    if not candidates:
        raise FileNotFoundError(
            "No photo style reference image was found. Set RETINA_STYLE_REFERENCE_IMAGE "
            "or place images in data/train_images."
        )
    return sorted(candidates)[0]


def load_photo_style_reference(image_size: Tuple[int, int] = config.IMG_SIZE) -> np.ndarray:
    """Load and cache the reference image used for photo style transfer."""
    reference_path = find_default_photo_style_reference()
    cache_key = (str(reference_path), image_size)
    if cache_key not in _REFERENCE_CACHE:
        reference = read_rgb(reference_path)
        reference = crop_black_borders(reference)
        reference = resize_with_padding(reference, size=image_size)
        _REFERENCE_CACHE[cache_key] = reference
    return _REFERENCE_CACHE[cache_key]


def transfer_lab_photo_style(
    content: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Transfer photographic color/illumination style from a reference image.

    This is a reference-based photo style transfer in LAB space. It keeps the
    retinal structure of the content image and transfers channel statistics
    from the reference foreground.
    """
    content_lab = cv2.cvtColor(content, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    content_mask = cv2.cvtColor(content, cv2.COLOR_RGB2GRAY) > 8
    reference_mask = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY) > 8

    if not np.any(content_mask) or not np.any(reference_mask):
        return content

    transferred = content_lab.copy()
    for channel in range(3):
        content_values = content_lab[:, :, channel][content_mask]
        reference_values = reference_lab[:, :, channel][reference_mask]
        content_mean = content_values.mean()
        content_std = content_values.std() + 1e-6
        reference_mean = reference_values.mean()
        reference_std = reference_values.std() + 1e-6
        channel_values = transferred[:, :, channel]
        channel_values[content_mask] = (
            (channel_values[content_mask] - content_mean)
            / content_std
            * reference_std
            + reference_mean
        )
        transferred[:, :, channel] = channel_values

    transferred[~content_mask] = 0
    transferred = ensure_uint8(transferred)
    return cv2.cvtColor(transferred, cv2.COLOR_LAB2RGB)


def preprocess_photo_style_transfer(
    image_or_path: ImageInput,
    image_size: Tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    """Apply reference-based photorealistic style transfer."""
    content = get_image(image_or_path)
    content = crop_black_borders(content)
    content = resize_with_padding(content, size=image_size)
    reference = load_photo_style_reference(image_size=image_size)
    return transfer_lab_photo_style(content, reference)


def basic_resize(image_or_path: ImageInput, image_size: Tuple[int, int] = config.IMG_SIZE) -> np.ndarray:
    """Baseline preprocessing: keep the original style and only resize/pad."""
    image = get_image(image_or_path)
    return resize_with_padding(image, size=image_size)


def preprocess_without_style(
    image_or_path: ImageInput,
    image_size: Tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    """Run the preprocessing pipeline before the final style normalization step."""
    image = get_image(image_or_path)
    image = crop_black_borders(image)
    image = resize_with_padding(image, size=image_size)
    foreground_mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) > 8
    image = normalize_illumination(image)
    image = enhance_contrast(image)
    image = apply_clahe(image)
    image = gray_world_balance(image)
    image[~foreground_mask] = 0
    return image


def preprocess_standardized(
    image_or_path: ImageInput,
    image_size: Tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    """Full normalized/style-standardized preprocessing pipeline."""
    image = preprocess_without_style(image_or_path, image_size=image_size)
    image = style_normalize_lab(image)
    return image


def preprocess_for_model(
    image_or_path: ImageInput,
    mode: str = "none",
    image_size: Tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    """
    Return a model-ready RGB image according to the selected experiment mode.

    mode="none" keeps the original image style.
    mode="standardized" applies the full standardization pipeline.
    """
    if mode == "none":
        return basic_resize(image_or_path, image_size=image_size)
    if mode in {"preprocessed", "without_style"}:
        return preprocess_without_style(image_or_path, image_size=image_size)
    if mode in {"standardized", "normalized", "style"}:
        return preprocess_standardized(image_or_path, image_size=image_size)
    if mode in {"photo_style_transfer", "photo_style", "style_transfer"}:
        return preprocess_photo_style_transfer(image_or_path, image_size=image_size)
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def build_visualization_stages(
    image_or_path: ImageInput,
    image_size: Tuple[int, int] = config.IMG_SIZE,
) -> Dict[str, np.ndarray]:
    """Create original, preprocessed, and style-normalized stages for reports."""
    original = get_image(image_or_path)
    original_resized = basic_resize(original, image_size=image_size)
    preprocessed = preprocess_without_style(original, image_size=image_size)
    standardized = style_normalize_lab(preprocessed)
    photo_style = preprocess_photo_style_transfer(original, image_size=image_size)

    return {
        "original": original_resized,
        "preprocessed": preprocessed,
        "standardized": standardized,
        "photo_style_transfer": photo_style,
    }
