"""Small frozen-feature behavior-cloning policy used for the first baseline."""

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
    """Predict one action from two frozen TCC camera features and a task ID."""

    def __init__(
        self,
        feature_dim: int,
        num_tasks: int,
        action_dim: int = 7,
        hidden_dims: Sequence[int] = (256, 256),
        proprio_dim: int = 0,
        input_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or num_tasks < 1 or action_dim < 1:
            raise ValueError("Feature, task, and action dimensions must be positive")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("At least one positive hidden dimension is required")

        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        input_dim = 2 * feature_dim + num_tasks + proprio_dim

        layers: list[nn.Module] = []
        if input_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
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
        return self.mlp(torch.cat(inputs, dim=-1))
