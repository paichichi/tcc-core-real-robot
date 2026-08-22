"""Restore and run the frozen-backbone MLP policy for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as vision_f

from tcc_real_robot.policy import (
    ActionNormalizer,
    HRPSingleViewGaussianMixturePolicy,
    R3MRobomimicPolicy,
    TCCMLPGaussianMixturePolicy,
    TCCMLPPolicy,
)
from tcc_real_robot.tcc_backbone import IndependentCameraBackbones

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
_NORMALIZATION_CACHE: dict[
    tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


@dataclass(frozen=True)
class PolicyBundle:
    """A restored policy head and the action normalizer saved with it."""

    model: TCCMLPPolicy | HRPSingleViewGaussianMixturePolicy | R3MRobomimicPolicy
    normalizer: ActionNormalizer
    state_normalizer: ActionNormalizer | None
    config: dict[str, Any]
    step: int
    backbone_state: dict[str, torch.Tensor] | None = None


def validate_policy_contract(
    runtime_config: dict[str, Any], bundle: PolicyBundle
) -> None:
    """Reject a checkpoint trained for different policy input/output semantics."""
    expected = runtime_config["policy"]
    actual = bundle.config["policy"]
    fields = (
        "cameras",
        "shared_camera_backbone",
        "camera_fusion",
        "camera_projection_dim",
        "camera_gate_hidden_dim",
        "proprioception",
        "proprioception_dim",
        "action_representation",
        "action_adapter",
        "action_space",
        "action_frame",
        "state_representation",
        "action_source",
        "action_leads_measured_state_frames",
        "driver_pose_representation",
        "rotation_velocity",
        "gripper_action",
        "action_distribution",
        "inference",
        "normalize_state",
        "normalize_actions",
        "architecture",
        "precision",
        "num_modes",
        "dropout",
        "progress_conditioning",
        "progress_dim",
    )
    mismatches = {
        field: (expected.get(field), actual.get(field))
        for field in fields
        if expected.get(field) != actual.get(field)
    }
    if mismatches:
        raise RuntimeError(
            "Runtime config/checkpoint policy contract mismatch: "
            f"{mismatches}"
        )


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


def preprocess_rgb_frames(
    frames: list[np.ndarray],
    image_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Apply the exact RGB preprocessing used to build the training cache."""
    if not frames:
        raise ValueError("At least one RGB frame is required")
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Frames must be uint8 RGB arrays with shape [H, W, 3]")
    images = torch.from_numpy(np.stack(frames))
    if device is not None:
        # Transfer compact uint8 data instead of a 4x larger float32 tensor.
        images = images.to(device, non_blocking=device.type == "cuda")
    images = images.permute(0, 3, 1, 2).float()
    images.div_(255.0)
    images = vision_f.resize(images, [image_size, image_size], antialias=False)
    cache_key = (str(images.device), images.dtype)
    statistics = _NORMALIZATION_CACHE.get(cache_key)
    if statistics is None:
        statistics = (
            IMAGENET_MEAN.to(device=images.device, dtype=images.dtype),
            IMAGENET_STD.to(device=images.device, dtype=images.dtype),
        )
        _NORMALIZATION_CACHE[cache_key] = statistics
    mean, std = statistics
    images.sub_(mean).div_(std)
    if images.device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


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
    camera_fusion = policy_config.get("camera_fusion", "raw_concat")
    camera_projection_dim = int(policy_config.get("camera_projection_dim", 0))
    camera_gate_hidden_dim = int(policy_config.get("camera_gate_hidden_dim", 0))
    if camera_fusion not in {
        "raw_concat",
        "project_then_concat",
        "gated_residual",
    }:
        raise ValueError(f"Unsupported camera fusion: {camera_fusion}")
    if camera_fusion == "raw_concat" and camera_projection_dim:
        raise ValueError("raw_concat cannot use camera projections")
    if camera_fusion == "project_then_concat" and not camera_projection_dim:
        raise ValueError(
            "project_then_concat requires a positive camera_projection_dim"
        )
    camera_names = tuple(
        policy_config.get("cameras", ("cam_main", "cam_wrist"))
    )
    if camera_fusion == "gated_residual" and (
        camera_names != ("cam_main", "cam_wrist")
        or camera_projection_dim <= 0
        or camera_gate_hidden_dim <= 0
    ):
        raise ValueError(
            "gated_residual requires both cameras and positive projection/gate "
            "dimensions"
        )
    if int(policy_config.get("action_chunk_size", -1)) != 1:
        raise ValueError("This runner requires a single-step policy checkpoint")
    uses_proprioception = policy_config.get("proprioception") is True
    progress_conditioning = policy_config.get("progress_conditioning")
    if progress_conditioning not in (None, "normalized_episode_time"):
        raise ValueError(
            f"Unsupported progress conditioning: {progress_conditioning}"
        )
    action_representation = policy_config.get("action_representation", "absolute")
    if action_representation not in {
        "absolute",
        "future_delta",
        "current_delta",
        "cartesian_velocity",
    }:
        raise ValueError(f"Unsupported action representation: {action_representation}")
    if action_representation in {"future_delta", "current_delta"} and not uses_proprioception:
        raise ValueError(f"{action_representation} checkpoints require proprioception")
    if action_representation == "future_delta":
        lookahead_frames = int(policy_config.get("lookahead_frames", 1))
        default_gain = 1.0 / lookahead_frames
        execution_delta_gain = float(
            policy_config.get("execution_delta_gain", default_gain)
        )
        if not 0.0 < execution_delta_gain <= 1.0:
            raise ValueError("execution_delta_gain must be in (0, 1]")
    proprio_dim = (
        int(policy_config.get("proprioception_dim", 7))
        if uses_proprioception
        else 0
    )
    progress_dim = 1 if progress_conditioning == "normalized_episode_time" else 0
    if int(policy_config.get("progress_dim", progress_dim)) != progress_dim:
        raise ValueError("Checkpoint progress_dim disagrees with progress_conditioning")
    architecture = str(policy_config.get("architecture", "pooled_feature_mlp"))
    if architecture == "r3m_deterministic_mlp_dual_independent_encoder":
        if (
            camera_names != ("cam_main", "cam_wrist")
            or uses_proprioception
            or progress_dim
            or camera_fusion != "raw_concat"
        ):
            raise ValueError("Minimal R3M policy requires two raw camera features")
        model = R3MRobomimicPolicy(
            feature_dim=feature_dim,
            action_dim=int(policy_config["action_dim"]),
            hidden_dims=tuple(policy_config["hidden_dimensions"]),
            output_layer_scale=float(policy_config.get("output_layer_scale", 0.01)),
        )
    elif architecture == "hrp_state_token_gmm":
        if camera_names != ("cam_main",) or progress_dim or not uses_proprioception:
            raise ValueError("HRP state-token policy requires one camera and state")
        model: TCCMLPPolicy | HRPSingleViewGaussianMixturePolicy = (
            HRPSingleViewGaussianMixturePolicy(
                feature_dim=feature_dim,
                action_dim=int(policy_config["action_dim"]),
                state_dim=proprio_dim,
                hidden_dims=tuple(policy_config["hidden_dimensions"]),
                num_modes=int(policy_config.get("num_modes", 5)),
                dropout=float(policy_config.get("dropout", 0.2)),
                min_std=float(policy_config.get("min_std", 1e-4)),
            )
        )
    else:
        action_distribution = str(
            policy_config.get("action_distribution", "deterministic")
        )
        if action_distribution == "deterministic":
            model_class = TCCMLPPolicy
            model_kwargs: dict[str, object] = {}
        elif action_distribution == "gaussian_mixture":
            model_class = TCCMLPGaussianMixturePolicy
            model_kwargs = {
                "num_modes": int(policy_config.get("num_modes", 5)),
                "min_std": float(policy_config.get("min_std", 1e-4)),
            }
        else:
            raise ValueError(
                f"Unsupported action distribution: {action_distribution}"
            )
        model = model_class(
            feature_dim=feature_dim,
            num_tasks=int(policy_config["number_of_tasks"]),
            action_dim=int(policy_config["action_dim"]),
            hidden_dims=tuple(policy_config["hidden_dimensions"]),
            proprio_dim=proprio_dim,
            progress_dim=progress_dim,
            input_batch_norm=bool(policy_config["input_batch_norm"]),
            input_layer_norm=bool(policy_config.get("input_layer_norm", False)),
            output_layer_scale=float(policy_config.get("output_layer_scale", 1.0)),
            camera_names=camera_names,
            camera_fusion=camera_fusion,
            camera_projection_dim=camera_projection_dim,
            camera_gate_hidden_dim=camera_gate_hidden_dim,
            dropout=float(policy_config.get("dropout", 0.0)),
            **model_kwargs,
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
    state_normalizer: ActionNormalizer | None = None
    if uses_proprioception:
        if "state_mean" not in checkpoint or "state_std" not in checkpoint:
            raise ValueError("Proprioceptive checkpoint lacks state normalization")
        state_normalizer = (
            ActionNormalizer(
                torch.as_tensor(checkpoint["state_mean"]),
                torch.as_tensor(checkpoint["state_std"]),
            )
            .eval()
            .to(device)
        )
    return PolicyBundle(
        model=model,
        normalizer=normalizer,
        state_normalizer=state_normalizer,
        config=config,
        step=int(checkpoint.get("step", 0)),
        backbone_state=checkpoint.get("backbone_model"),
    )


def restore_policy_backbone(backbone: nn.Module, bundle: PolicyBundle) -> bool:
    """Restore an end-to-end fine-tuned backbone embedded in a policy checkpoint."""
    if bundle.backbone_state is None:
        return False
    missing, unexpected = backbone.load_state_dict(bundle.backbone_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Fine-tuned backbone mismatch: missing={missing}, unexpected={unexpected}"
        )
    backbone.eval()
    return True


@torch.inference_mode()
def predict_action(
    backbone: nn.Module,
    bundle: PolicyBundle,
    cam_main_rgb: np.ndarray,
    cam_wrist_rgb: np.ndarray,
    task_index: int,
    image_size: int,
    device: torch.device,
    observation_state: list[float] | np.ndarray | torch.Tensor | None = None,
    execution_delta_gain_override: float | None = None,
    episode_progress: float | torch.Tensor | None = None,
    gmm_inference_override: str | None = None,
) -> torch.Tensor:
    """Predict one denormalized 7-D absolute action on CPU."""
    number_of_tasks = int(bundle.config["policy"]["number_of_tasks"])
    if not 0 <= task_index < number_of_tasks:
        raise ValueError(f"Task index {task_index} is outside [0, {number_of_tasks})")
    frames = [cam_main_rgb]
    if "cam_wrist" in bundle.model.camera_names:
        frames.append(cam_wrist_rgb)
    images = preprocess_rgb_frames(frames, image_size, device=device)
    policy_config = bundle.config["policy"]
    strict_float32 = policy_config.get("precision") == "float32"
    if strict_float32 and device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and not strict_float32,
    ):
        if isinstance(backbone, IndependentCameraBackbones):
            if "cam_wrist" not in bundle.model.camera_names:
                raise ValueError("Independent camera backbones require cam_wrist")
            cam_main_features, cam_wrist_features = backbone(
                images[0:1], images[1:2]
            )
            cam_main_features = cam_main_features.float()
            cam_wrist_features = cam_wrist_features.float()
        else:
            features = backbone(images).float()
            cam_main_features = features[0:1]
            cam_wrist_features = (
                features[1:2]
                if "cam_wrist" in bundle.model.camera_names
                else None
            )
        proprioception: torch.Tensor | None = None
        if bundle.model.proprio_dim:
            if observation_state is None or bundle.state_normalizer is None:
                raise ValueError("This policy checkpoint requires observation_state")
            proprioception = torch.as_tensor(
                observation_state, dtype=torch.float32, device=device
            ).reshape(1, -1)
            if proprioception.shape != (1, bundle.model.proprio_dim):
                raise ValueError(
                    f"Expected proprioception shape [1, {bundle.model.proprio_dim}]"
                )
            if not torch.isfinite(proprioception).all():
                raise ValueError("observation_state must be finite")
            normalized_state = bundle.state_normalizer.normalize(proprioception)
        else:
            normalized_state = None
        progress: torch.Tensor | None = None
        if bundle.model.progress_dim:
            if episode_progress is None:
                raise ValueError("This policy checkpoint requires episode_progress")
            progress = torch.as_tensor(
                episode_progress, dtype=torch.float32, device=device
            ).reshape(1, -1)
            if progress.shape != (1, bundle.model.progress_dim):
                raise ValueError(
                    f"Expected episode progress shape [1, {bundle.model.progress_dim}]"
                )
            valid_progress = torch.isfinite(progress) & (progress >= 0.0) & (
                progress <= 1.0
            )
            if not bool(valid_progress.all().item()):
                raise ValueError("episode_progress must be finite and within [0, 1]")
        task_tensor = torch.tensor([task_index], device=device)
        if gmm_inference_override not in (None, "highest-probability-mode"):
            raise ValueError(
                "gmm_inference_override must be None or "
                "'highest-probability-mode'"
            )
        deterministic_checkpoint_inference = policy_config.get(
            "deterministic_inference"
        ) in {"highest_probability_mode_mean", "highest-probability-mode"}
        if isinstance(bundle.model, R3MRobomimicPolicy):
            if cam_wrist_features is None:
                raise RuntimeError("R3M multi-view policy requires cam_wrist")
            normalized_action = bundle.model(
                cam_main_features, cam_wrist_features
            )
        elif (
            (
                gmm_inference_override == "highest-probability-mode"
                or (
                    gmm_inference_override is None
                    and deterministic_checkpoint_inference
                )
            )
            and isinstance(bundle.model, HRPSingleViewGaussianMixturePolicy)
        ):
            normalized_action = bundle.model.highest_probability_mean(
                cam_main_features,
                cam_wrist_features,
                task_tensor,
                normalized_state,
                progress,
            )
        else:
            normalized_action = bundle.model(
                cam_main_features,
                cam_wrist_features,
                task_tensor,
                normalized_state,
                progress,
            )
        action = bundle.normalizer.denormalize(normalized_action)
        policy_config = bundle.config["policy"]
        action_representation = policy_config.get(
            "action_representation", "absolute"
        )
        if action_representation in {"future_delta", "current_delta"}:
            if proprioception is None:
                raise RuntimeError(f"{action_representation} policy has no proprioception")
            if action_representation == "future_delta":
                lookahead_frames = int(policy_config.get("lookahead_frames", 1))
                default_gain = 1.0 / lookahead_frames
            else:
                default_gain = 1.0
            gain = float(
                execution_delta_gain_override
                if execution_delta_gain_override is not None
                else policy_config.get("execution_delta_gain", default_gain)
            )
            if not 0.0 < gain <= 1.0:
                raise ValueError("execution_delta_gain must be in (0, 1]")
            action = proprioception + gain * action
    result = action[0].float().cpu()
    if result.shape != (7,) or not torch.isfinite(result).all():
        raise RuntimeError(f"Policy produced an invalid action: {result}")
    return result
