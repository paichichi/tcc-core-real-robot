"""Small frozen-feature behavior-cloning policies for real-robot baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ActionNormalizer(nn.Module):
    """Normalize actions using statistics computed from training episodes only."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError("Action mean/std must be matching one-dimensional tensors")
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-6))

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.mean) / self.std

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.std + self.mean


class TCCMLPPolicy(nn.Module):
    """Predict one action from two frozen features, task ID, and optional state."""

    def __init__(
        self,
        feature_dim: int,
        num_tasks: int,
        action_dim: int = 7,
        hidden_dims: Sequence[int] = (256, 256),
        proprio_dim: int = 0,
        progress_dim: int = 0,
        input_batch_norm: bool = True,
        input_layer_norm: bool = False,
        output_layer_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if proprio_dim < 0 or progress_dim < 0:
            raise ValueError("Conditioning dimensions cannot be negative")
        if feature_dim < 1 or num_tasks < 1 or action_dim < 1:
            raise ValueError("Feature, task, and action dimensions must be positive")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("At least one positive hidden dimension is required")
        if input_batch_norm and input_layer_norm:
            raise ValueError("Choose at most one input normalization layer")
        if output_layer_scale <= 0:
            raise ValueError("Output-layer scale must be positive")

        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.progress_dim = progress_dim
        input_dim = 2 * feature_dim + num_tasks + proprio_dim + progress_dim

        layers: list[nn.Module] = []
        if input_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        elif input_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        output_layer = nn.Linear(previous, action_dim)
        with torch.no_grad():
            output_layer.weight.mul_(output_layer_scale)
            if output_layer.bias is not None:
                output_layer.bias.mul_(output_layer_scale)
        layers.append(output_layer)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cam_main.shape != cam_wrist.shape:
            raise ValueError("Camera feature tensors must have identical shapes")
        if cam_main.ndim != 2 or cam_main.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected camera features [B, {self.feature_dim}], "
                f"got {tuple(cam_main.shape)}"
            )
        task = F.one_hot(task_index.long(), self.num_tasks).to(cam_main.dtype)
        inputs = [cam_main, cam_wrist, task]
        if self.proprio_dim:
            if proprioception is None:
                raise ValueError("This policy requires proprioception")
            if proprioception.shape != (cam_main.shape[0], self.proprio_dim):
                raise ValueError("Unexpected proprioception shape")
            inputs.append(proprioception)
        elif proprioception is not None:
            raise ValueError("This policy was configured without proprioception")
        if self.progress_dim:
            if progress is None:
                raise ValueError("This policy requires episode progress")
            if progress.shape != (cam_main.shape[0], self.progress_dim):
                raise ValueError("Unexpected episode progress shape")
            inputs.append(progress)
        elif progress is not None:
            raise ValueError("This policy was configured without episode progress")
        return self.mlp(torch.cat(inputs, dim=-1))
