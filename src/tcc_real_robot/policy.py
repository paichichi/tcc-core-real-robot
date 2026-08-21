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
    """Predict one action from configured frozen features and optional state."""

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
        camera_names: Sequence[str] = ("cam_main", "cam_wrist"),
        camera_fusion: str = "raw_concat",
        camera_projection_dim: int = 0,
        camera_gate_hidden_dim: int = 0,
        dropout: float = 0.0,
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
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")
        if camera_projection_dim < 0:
            raise ValueError("Camera projection dimension cannot be negative")
        if camera_gate_hidden_dim < 0:
            raise ValueError("Camera gate hidden dimension cannot be negative")
        camera_names = tuple(camera_names)
        if camera_names not in (("cam_main",), ("cam_main", "cam_wrist")):
            raise ValueError(
                "Camera inputs must be cam_main alone or cam_main plus cam_wrist"
            )
        if camera_fusion not in {
            "raw_concat",
            "project_then_concat",
            "gated_residual",
        }:
            raise ValueError(f"Unsupported camera fusion: {camera_fusion}")
        if camera_fusion == "raw_concat" and camera_projection_dim:
            raise ValueError("raw_concat cannot use camera projections")
        if camera_fusion == "project_then_concat" and not camera_projection_dim:
            raise ValueError("project_then_concat requires camera projections")
        if camera_fusion == "gated_residual" and (
            camera_names != ("cam_main", "cam_wrist")
            or not camera_projection_dim
            or not camera_gate_hidden_dim
        ):
            raise ValueError(
                "gated_residual requires both cameras and positive projection/gate "
                "dimensions"
            )

        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.progress_dim = progress_dim
        self.camera_names = camera_names
        self.camera_fusion = camera_fusion
        self.camera_projection_dim = camera_projection_dim
        self.camera_gate_hidden_dim = camera_gate_hidden_dim
        self.dropout = dropout
        projected_feature_dim = camera_projection_dim or feature_dim
        if camera_projection_dim:
            self.cam_main_projection: nn.Module = self._make_projection(
                feature_dim,
                camera_projection_dim,
                layer_norm=camera_fusion == "gated_residual",
            )
            self.cam_wrist_projection: nn.Module | None = (
                self._make_projection(
                    feature_dim,
                    camera_projection_dim,
                    layer_norm=camera_fusion == "gated_residual",
                )
                if "cam_wrist" in camera_names
                else None
            )
        else:
            self.cam_main_projection = nn.Identity()
            self.cam_wrist_projection = (
                nn.Identity() if "cam_wrist" in camera_names else None
            )
        conditioning_dim = num_tasks + proprio_dim + progress_dim
        self.camera_gate: nn.Module | None = None
        if camera_fusion == "gated_residual":
            gate_input_dim = 2 * projected_feature_dim + conditioning_dim
            self.camera_gate = nn.Sequential(
                nn.Linear(gate_input_dim, camera_gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(camera_gate_hidden_dim, projected_feature_dim),
                nn.Sigmoid(),
            )
            visual_input_dim = projected_feature_dim
        else:
            visual_input_dim = len(camera_names) * projected_feature_dim
        input_dim = visual_input_dim + conditioning_dim

        layers: list[nn.Module] = []
        if input_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        elif input_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = width
        output_layer = nn.Linear(previous, action_dim)
        with torch.no_grad():
            output_layer.weight.mul_(output_layer_scale)
            if output_layer.bias is not None:
                output_layer.bias.mul_(output_layer_scale)
        layers.append(output_layer)
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _make_projection(
        feature_dim: int, projection_dim: int, *, layer_norm: bool
    ) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(feature_dim, projection_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(projection_dim))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(
        self,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor | None,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mlp(
            self._policy_inputs(
                cam_main,
                cam_wrist,
                task_index,
                proprioception,
                progress,
            )
        )

    def _policy_inputs(
        self,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor | None,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the shared visual/state input consumed by a policy head."""
        if cam_main.ndim != 2 or cam_main.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected camera features [B, {self.feature_dim}], "
                f"got {tuple(cam_main.shape)}"
            )
        task = F.one_hot(task_index.long(), self.num_tasks).to(cam_main.dtype)
        main_embedding = self.cam_main_projection(cam_main)
        wrist_embedding: torch.Tensor | None = None
        if "cam_wrist" in self.camera_names:
            if cam_wrist is None or cam_main.shape != cam_wrist.shape:
                raise ValueError("Camera feature tensors must have identical shapes")
            if self.cam_wrist_projection is None:
                raise RuntimeError("Wrist projection was not initialized")
            wrist_embedding = self.cam_wrist_projection(cam_wrist)
        conditioning = [task]
        if self.proprio_dim:
            if proprioception is None:
                raise ValueError("This policy requires proprioception")
            if proprioception.shape != (cam_main.shape[0], self.proprio_dim):
                raise ValueError("Unexpected proprioception shape")
            conditioning.append(proprioception)
        elif proprioception is not None:
            raise ValueError("This policy was configured without proprioception")
        if self.progress_dim:
            if progress is None:
                raise ValueError("This policy requires episode progress")
            if progress.shape != (cam_main.shape[0], self.progress_dim):
                raise ValueError("Unexpected episode progress shape")
            conditioning.append(progress)
        elif progress is not None:
            raise ValueError("This policy was configured without episode progress")
        if self.camera_fusion == "gated_residual":
            if wrist_embedding is None or self.camera_gate is None:
                raise RuntimeError("Gated camera fusion was not initialized")
            gate = self.camera_gate(
                torch.cat([main_embedding, wrist_embedding, *conditioning], dim=-1)
            )
            visual_inputs = [main_embedding + gate * wrist_embedding]
        else:
            visual_inputs = [main_embedding]
            if wrist_embedding is not None:
                visual_inputs.append(wrist_embedding)
        return torch.cat([*visual_inputs, *conditioning], dim=-1)


class TCCMLPGaussianMixturePolicy(TCCMLPPolicy):
    """HRP-style MLP policy with a Gaussian-mixture action head."""

    def __init__(
        self,
        *args: object,
        num_modes: int = 5,
        min_std: float = 1e-4,
        **kwargs: object,
    ) -> None:
        if num_modes < 2:
            raise ValueError("A Gaussian-mixture policy requires at least two modes")
        if min_std <= 0:
            raise ValueError("Minimum standard deviation must be positive")
        super().__init__(*args, **kwargs)
        output_layer = self.mlp[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("Expected the base policy to end in a linear layer")
        hidden_dim = output_layer.in_features
        self.mlp = nn.Sequential(*list(self.mlp.children())[:-1])
        self.num_modes = num_modes
        self.min_std = min_std
        self.mixture_means = nn.Linear(hidden_dim, num_modes * self.action_dim)
        self.mixture_scales = nn.Linear(hidden_dim, num_modes * self.action_dim)
        self.mixture_logits = nn.Linear(hidden_dim, num_modes)

    def mixture_parameters(
        self,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor | None,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._policy_inputs(
            cam_main,
            cam_wrist,
            task_index,
            proprioception,
            progress,
        )
        hidden = self.mlp(inputs)
        batch = hidden.shape[0]
        means = self.mixture_means(hidden).reshape(
            batch, self.num_modes, self.action_dim
        )
        scales = F.softplus(self.mixture_scales(hidden)).reshape(
            batch, self.num_modes, self.action_dim
        )
        scales = scales + self.min_std
        logits = self.mixture_logits(hidden)
        return means, scales, logits

    def forward(
        self,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor | None,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the mean of the most probable mode for deterministic control."""
        means, _, logits = self.mixture_parameters(
            cam_main,
            cam_wrist,
            task_index,
            proprioception,
            progress,
        )
        modes = logits.argmax(dim=-1)
        batch = torch.arange(means.shape[0], device=means.device)
        return means[batch, modes]

    def negative_log_likelihood(
        self,
        target: torch.Tensor,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor | None,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return mean negative log likelihood of normalized expert actions."""
        if target.ndim != 2 or target.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected action targets [B, {self.action_dim}], got {target.shape}"
            )
        means, scales, logits = self.mixture_parameters(
            cam_main,
            cam_wrist,
            task_index,
            proprioception,
            progress,
        )
        components = torch.distributions.Independent(
            torch.distributions.Normal(means, scales), 1
        )
        mixture = torch.distributions.Categorical(logits=logits)
        distribution = torch.distributions.MixtureSameFamily(mixture, components)
        return -distribution.log_prob(target).mean()


class HRPSingleViewGaussianMixturePolicy(nn.Module):
    """Single-view state-token MLP-GMM used by the official HRP BC pipeline."""

    camera_names = ("cam_main",)
    camera_fusion = "raw_concat"
    camera_projection_dim = 0
    camera_gate_hidden_dim = 0
    progress_dim = 0

    def __init__(
        self,
        feature_dim: int,
        action_dim: int = 7,
        state_dim: int = 7,
        hidden_dims: Sequence[int] = (512, 512),
        num_modes: int = 5,
        dropout: float = 0.2,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or action_dim < 1 or state_dim < 1:
            raise ValueError("HRP dimensions must be positive")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("HRP requires positive hidden dimensions")
        if num_modes < 2 or not 0.0 <= dropout < 1.0 or min_std <= 0:
            raise ValueError("Invalid HRP mixture/dropout configuration")
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.proprio_dim = state_dim
        self.num_modes = num_modes
        self.min_std = min_std
        self.state_token = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(state_dim, feature_dim),
        )
        self.token_batch_norm = nn.BatchNorm1d(feature_dim)
        self.token_dropout = nn.Dropout(dropout)
        layers: list[nn.Module] = []
        previous = 2 * feature_dim
        for width in hidden_dims:
            layers.extend(
                (nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout))
            )
            previous = width
        self.mlp = nn.Sequential(*layers)
        self.mixture_means = nn.Linear(previous, num_modes * action_dim)
        self.mixture_scales = nn.Linear(previous, num_modes * action_dim)
        self.mixture_logits = nn.Linear(previous, num_modes)

    def _hidden(
        self,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor | None,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None,
        progress: torch.Tensor | None,
    ) -> torch.Tensor:
        if cam_main.ndim != 2 or cam_main.shape[1] != self.feature_dim:
            raise ValueError("Unexpected HRP visual feature shape")
        if proprioception is None or proprioception.shape != (
            cam_main.shape[0],
            self.proprio_dim,
        ):
            raise ValueError("HRP policy requires normalized robot state")
        if cam_wrist is not None or progress is not None:
            raise ValueError("Official single-view HRP accepts no wrist/progress input")
        if task_index.shape != (cam_main.shape[0],) or bool(
            torch.any(task_index != 0)
        ):
            raise ValueError("Official HRP policy is trained separately per task")
        state = self.state_token(proprioception)
        tokens = torch.stack((cam_main, state), dim=1)
        tokens = self.token_batch_norm(tokens.transpose(1, 2)).transpose(1, 2)
        return self.mlp(self.token_dropout(tokens).flatten(1))

    def mixture_parameters(
        self,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor | None,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self._hidden(
            cam_main, cam_wrist, task_index, proprioception, progress
        )
        batch = hidden.shape[0]
        means = self.mixture_means(hidden).reshape(
            batch, self.num_modes, self.action_dim
        )
        scales = F.softplus(self.mixture_scales(hidden)).reshape(
            batch, self.num_modes, self.action_dim
        )
        logits = self.mixture_logits(hidden)
        return means, scales + self.min_std, logits

    def forward(
        self,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor | None,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        means, _, logits = self.mixture_parameters(
            cam_main, cam_wrist, task_index, proprioception, progress
        )
        modes = logits.argmax(dim=-1)
        batch = torch.arange(means.shape[0], device=means.device)
        return means[batch, modes]

    def negative_log_likelihood(
        self,
        target: torch.Tensor,
        cam_main: torch.Tensor,
        cam_wrist: torch.Tensor | None,
        task_index: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        means, scales, logits = self.mixture_parameters(
            cam_main, cam_wrist, task_index, proprioception, progress
        )
        components = torch.distributions.Independent(
            torch.distributions.Normal(means, scales), 1
        )
        distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logits), components
        )
        return -distribution.log_prob(target).mean()
