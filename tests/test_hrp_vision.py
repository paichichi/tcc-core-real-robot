from __future__ import annotations

import torch
from torchvision import transforms

from tcc_real_robot.hrp_vision import build_hrp_image_transform


def test_hrp_eval_transform_uses_antialiased_resize() -> None:
    transform = build_hrp_image_transform(224, training=False)
    resize = next(
        item for item in transform.transforms if isinstance(item, transforms.Resize)
    )
    assert resize.antialias is True
    output = transform(torch.zeros((3, 480, 640), dtype=torch.uint8))
    assert output.shape == (3, 224, 224)
    assert torch.isfinite(output).all()


def test_hrp_training_transform_applies_configured_robustness() -> None:
    transform = build_hrp_image_transform(
        64,
        training=True,
        augmentation={
            "random_resized_crop_scale": [0.9, 1.0],
            "random_resized_crop_ratio": [1.2, 4 / 3],
            "color_jitter": {
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.1,
                "hue": 0.01,
            },
            "color_jitter_probability": 1.0,
            "gaussian_blur": True,
            "gaussian_blur_probability": 1.0,
        },
    )
    output = transform(torch.full((3, 80, 100), 128, dtype=torch.uint8))
    assert output.shape == (3, 64, 64)
    assert torch.isfinite(output).all()
    crop = next(
        item
        for item in transform.transforms
        if isinstance(item, transforms.RandomResizedCrop)
    )
    assert crop.ratio == (1.2, 4 / 3)


def test_hrp_training_transform_never_mirrors_robot_coordinates() -> None:
    transform = build_hrp_image_transform(64, training=True)
    assert not any(
        isinstance(item, (transforms.RandomHorizontalFlip, transforms.RandomRotation))
        for item in transform.transforms
    )
