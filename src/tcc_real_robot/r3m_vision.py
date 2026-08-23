"""Minimal deterministic image preprocessing for R3M downstream policies."""

from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_r3m_transform(image_size: int) -> transforms.Compose:
    """Resize RGB observations and apply ImageNet normalization."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=False,
            ),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_r3m_train_transform(
    image_size: int,
    *,
    brightness: float = 0.15,
    contrast: float = 0.15,
    saturation: float = 0.10,
    hue: float = 0.02,
    blur_probability: float = 0.10,
) -> transforms.Compose:
    """Photometric RealSense augmentation followed by runtime preprocessing.

    Spatial transforms are intentionally excluded: flipping, rotating, or cropping
    an image without transforming its absolute joint-action label breaks the
    visuomotor supervision contract.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    strengths = (brightness, contrast, saturation, hue)
    if any(value < 0.0 for value in strengths):
        raise ValueError("Color-jitter strengths cannot be negative")
    if not 0.0 <= blur_probability <= 1.0:
        raise ValueError("Blur probability must be in [0, 1]")
    return transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=5,
                        sigma=(0.1, 1.0),
                    )
                ],
                p=blur_probability,
            ),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=False,
            ),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
