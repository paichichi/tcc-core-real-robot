"""Shared image preprocessing and augmentation for HRP training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def _pair(settings: Mapping[str, Any], key: str, default: tuple[float, float]):
    values = settings.get(key, default)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"augmentation.{key} must contain two values")
    return float(values[0]), float(values[1])


def build_hrp_image_transform(
    image_size: int,
    *,
    training: bool,
    augmentation: Mapping[str, Any] | None = None,
) -> transforms.Compose:
    """Build train/eval transforms with one shared inference contract.

    Geometry-changing augmentation is deliberately small. Horizontal flips and
    rotations are excluded because they would change the robot-frame meaning of
    an action while leaving its label unchanged.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    settings: Mapping[str, Any] = augmentation or {}
    operations: list[torch.nn.Module] = [
        transforms.ConvertImageDtype(torch.float32)
    ]
    if training:
        operations.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=_pair(
                    settings, "random_resized_crop_scale", (0.9, 1.0)
                ),
                ratio=_pair(
                    settings, "random_resized_crop_ratio", (0.95, 1.05)
                ),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        )

        jitter = settings.get("color_jitter", {})
        jitter_probability = float(settings.get("color_jitter_probability", 0.0))
        if jitter_probability > 0.0:
            if not isinstance(jitter, Mapping):
                raise ValueError("augmentation.color_jitter must be a mapping")
            operations.append(
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=float(jitter.get("brightness", 0.0)),
                            contrast=float(jitter.get("contrast", 0.0)),
                            saturation=float(jitter.get("saturation", 0.0)),
                            hue=float(jitter.get("hue", 0.0)),
                        )
                    ],
                    p=jitter_probability,
                )
            )

        if bool(settings.get("gaussian_blur", False)):
            kernel = int(0.05 * image_size)
            kernel += 1 - kernel % 2
            operations.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=kernel)],
                    p=float(settings.get("gaussian_blur_probability", 0.2)),
                )
            )
    else:
        operations.append(
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        )
    operations.append(
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    )
    return transforms.Compose(operations)
