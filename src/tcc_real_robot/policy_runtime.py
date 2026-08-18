"""Restore and run the frozen-backbone MLP policy for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as vision_f

from tcc_real_robot.policy import ActionNormalizer, TCCMLPPolicy

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


@dataclass(frozen=True)
class PolicyBundle:
    """A restored policy head and the action normalizer saved with it."""

    model: TCCMLPPolicy
    normalizer: ActionNormalizer
    config: dict[str, Any]
    step: int


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` without silently accepting an unavailable accelerator."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} requested but CUDA is unavailable"
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def preprocess_rgb_frames(frames: list[np.ndarray], image_size: int) -> torch.Tensor:
    """Apply the exact RGB preprocessing used to build the training cache."""
    if not frames:
        raise ValueError("At least one RGB frame is required")
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Frames must be uint8 RGB arrays with shape [H, W, 3]")
    images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    images.div_(255.0)
    images = vision_f.resize(images, [image_size, image_size], antialias=True)
    return (images - IMAGENET_MEAN) / IMAGENET_STD


def load_policy_bundle(
    checkpoint_path: str | Path,
    *,
    expected_feature_dim: int,
    device: torch.device,
) -> PolicyBundle:
    """Restore the MLP and action statistics from a training checkpoint."""
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Policy checkpoint must contain a dictionary")
    required = {"model", "action_mean", "action_std", "feature_dim", "config"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Policy checkpoint is missing keys: {sorted(missing)}")
    feature_dim = int(checkpoint["feature_dim"])
    if feature_dim != expected_feature_dim:
        raise ValueError(
            f"Backbone/policy feature mismatch: {expected_feature_dim} != {feature_dim}"
        )
    config = checkpoint["config"]
    if not isinstance(config, dict) or not isinstance(config.get("policy"), dict):
        raise TypeError("Policy checkpoint contains an invalid config")
    policy_config = config["policy"]
    if policy_config.get("proprioception") is not False:
        raise ValueError("This runner only supports the no-proprioception v0 policy")
    if int(policy_config.get("action_chunk_size", -1)) != 1:
        raise ValueError("This runner requires a single-step policy checkpoint")
    model = TCCMLPPolicy(
        feature_dim=feature_dim,
        num_tasks=int(policy_config["number_of_tasks"]),
        action_dim=int(policy_config["action_dim"]),
        hidden_dims=tuple(policy_config["hidden_dimensions"]),
        proprio_dim=0,
        input_batch_norm=bool(policy_config["input_batch_norm"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    normalizer = (
        ActionNormalizer(
            torch.as_tensor(checkpoint["action_mean"]),
            torch.as_tensor(checkpoint["action_std"]),
        )
        .eval()
        .to(device)
    )
    return PolicyBundle(
        model=model,
        normalizer=normalizer,
        config=config,
        step=int(checkpoint.get("step", 0)),
    )


@torch.inference_mode()
def predict_action(
    backbone: nn.Module,
    bundle: PolicyBundle,
    cam_main_rgb: np.ndarray,
    cam_wrist_rgb: np.ndarray,
    task_index: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Predict one denormalized 7-D absolute action on CPU."""
    number_of_tasks = int(bundle.config["policy"]["number_of_tasks"])
    if not 0 <= task_index < number_of_tasks:
        raise ValueError(f"Task index {task_index} is outside [0, {number_of_tasks})")
    images = preprocess_rgb_frames([cam_main_rgb, cam_wrist_rgb], image_size).to(
        device, non_blocking=True
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        features = backbone(images).float()
        normalized_action = bundle.model(
            features[0:1],
            features[1:2],
            torch.tensor([task_index], device=device),
        )
        action = bundle.normalizer.denormalize(normalized_action)
    result = action[0].float().cpu()
    if result.shape != (7,) or not torch.isfinite(result).all():
        raise RuntimeError(f"Policy produced an invalid action: {result}")
    return result
