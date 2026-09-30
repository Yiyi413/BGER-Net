"""Albumentations preset pipelines for owl datasets.

Presets are convenience functions that construct standard Albumentations
pipelines. Users may use these presets directly or provide their own
``albumentations.Compose`` instances to ``OwlDataset``.
"""

import albumentations as A


ImageSize = tuple[int, int]


def train(size: ImageSize) -> A.Compose:
    """Build the default training augmentation pipeline.

    The preset applies common spatial and image-only augmentations before
    resizing the sample to the requested output size. Albumentations keeps the
    input image and all mask targets synchronized for spatial transforms.

    Args:
        size:
            Output image size expressed as ``(height, width)``.

    Returns:
        An Albumentations composition operating on NumPy arrays.
    """
    height, width = _validate_size(size)

    return A.Compose(
        [
            A.RandomScale(
                scale_limit=0.2,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.2,
            ),
            A.ImageCompression(
                quality_range=(70, 100),
                p=0.2,
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),
                p=0.2,
            ),
            A.Resize(
                height=height,
                width=width,
            ),
        ]
    )


def infer(size: ImageSize) -> A.Compose:
    """Build the default inference augmentation pipeline.

    The inference preset only performs deterministic spatial adaptation and
    does not apply random augmentation, normalization, or tensor conversion.

    Args:
        size:
            Output image size expressed as ``(height, width)``.

    Returns:
        An Albumentations composition operating on NumPy arrays.
    """
    height, width = _validate_size(size)

    return A.Compose(
        [
            A.Resize(
                height=height,
                width=width,
            ),
        ]
    )


def _validate_size(size: ImageSize) -> ImageSize:
    """Validate and return an image size expressed as height and width."""
    if len(size) != 2:
        raise ValueError("size must contain exactly two values: (height, width)")

    height, width = size

    if height <= 0 or width <= 0:
        raise ValueError(
            f"size values must be positive, got height={height}, width={width}"
        )

    return height, width