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
