#!/usr/bin/env python3
"""Train V10: shared RN50, main policy, and gated wrist action residual."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset

from tcc_real_robot.model_assets import resolve_backbone_asset
from tcc_real_robot.policy import ActionNormalizer, MainWristResidualPolicy
from tcc_real_robot.policy_runtime import resolve_device
from tcc_real_robot.r3m_vision import (
    build_r3m_train_transform,
    build_r3m_transform,
)
from tcc_real_robot.tcc_backbone import load_trainable_tcc_backbone
from tcc_real_robot.trossen_image_data import (
    TrossenMultiViewDataset,
    episode_split_indices,
    require_complete_trossen_buffer,
)

ARCHITECTURE = (
    "r3m_deterministic_mlp_shared_rn50_main_gated_wrist_residual_proprio"
)
CAMERA_FUSION = "main_policy_with_gated_wrist_action_residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiment_v10_shared_rn50_main_wrist_residual_100.yaml"
        ),
    )
    parser.add_argument("--image-buffer", type=Path, required=True)
    parser.add_argument("--backbone", default="ours_rn50")
    parser.add_argument("--hub-cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tcc-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def train_with_frozen_batch_norm_statistics(backbone: nn.Module) -> None:
    """Train convolution weights while preserving pretrained BN running stats."""
    backbone.train()
    for module in backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def validate_config(config: dict, backbone_name: str) -> None:
    policy = config["policy"]
    expected = {
        "architecture": ARCHITECTURE,
        "cameras": ["cam_main", "cam_wrist"],
        "shared_camera_backbone": True,
        "camera_fusion": CAMERA_FUSION,
        "action_representation": "absolute",
        "action_adapter": "trossen_joint_position_passthrough",
        "action_distribution": "deterministic",
        "action_chunk_size": 1,
        "proprioception": True,
        "proprioception_dim": 7,
        "normalize_state": True,
    }
    mismatches = {
        key: (policy.get(key), value)
        for key, value in expected.items()
        if policy.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Invalid V10 contract: {mismatches}")
    if backbone_name != "ours_rn50":
        raise ValueError("V10 intentionally supports only ours_rn50")
    if config["backbone"].get("fine_tuning") != (
        "shared_full_end_to_end_freeze_batch_norm_statistics"
    ):
        raise ValueError("V10 must fine-tune one shared backbone with frozen BN stats")
    augmentation = config.get("augmentation", {})
    if augmentation.get("scope") != "train_only_independent_per_camera":
        raise ValueError("V10 augmentation must be train-only and per-camera")
    if augmentation.get("spatial_transforms") is not False:
        raise ValueError("V10 forbids spatial transforms for absolute-action labels")


def policy_loss(
    model: MainWristResidualPolicy,
    main_features: torch.Tensor,
    wrist_features: torch.Tensor,
    normalized_state: torch.Tensor,
    normalized_target: torch.Tensor,
    *,
    main_weight: float,
    residual_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction, main_action, correction, gate = model.forward_components(
        main_features, wrist_features, normalized_state
    )
    fused_mse = nn.functional.mse_loss(prediction, normalized_target)
    main_mse = nn.functional.mse_loss(main_action, normalized_target)
    residual_penalty = correction.square().mean()
    total = fused_mse + main_weight * main_mse + residual_weight * residual_penalty
    return total, {
        "fused_mse": fused_mse,
        "main_mse": main_mse,
        "residual_penalty": residual_penalty,
        "gate_mean": gate.mean(),
        "correction_abs_mean": correction.abs().mean(),
        "prediction": prediction,
    }


@torch.inference_mode()
def evaluate(
    backbone: nn.Module,
    model: MainWristResidualPolicy,
    loader: DataLoader,
    normalizer: ActionNormalizer,
    state_normalizer: ActionNormalizer,
    device: torch.device,
    *,
    main_weight: float,
    residual_weight: float,
) -> dict[str, object]:
    backbone.eval()
    model.eval()
    totals = {
        "loss": 0.0,
        "fused_mse": 0.0,
        "main_mse": 0.0,
        "residual_penalty": 0.0,
        "gate_mean": 0.0,
        "correction_abs_mean": 0.0,
    }
    absolute_error = torch.zeros(model.action_dim, device=device)
    count = 0
    for main_images, wrist_images, states, actions in loader:
        main_images = main_images.to(device, non_blocking=True)
        wrist_images = wrist_images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        stacked = torch.cat([main_images, wrist_images], dim=0)
        features = backbone(stacked).float()
        main_features, wrist_features = features.chunk(2, dim=0)
        loss, metrics = policy_loss(
            model,
            main_features,
            wrist_features,
            state_normalizer.normalize(states),
            normalizer.normalize(actions),
            main_weight=main_weight,
            residual_weight=residual_weight,
        )
        batch = actions.shape[0]
        totals["loss"] += float(loss) * batch
        for key in totals:
            if key != "loss":
                totals[key] += float(metrics[key]) * batch
        prediction = normalizer.denormalize(metrics["prediction"])
        absolute_error += torch.abs(prediction - actions).sum(0)
        count += batch
    if count == 0:
        raise RuntimeError("Evaluation split is empty")
    result: dict[str, object] = {
        key: value / count for key, value in totals.items()
    }
    result["action_mae"] = float(absolute_error.sum() / (count * model.action_dim))
    result["action_mae_per_dimension"] = (absolute_error / count).cpu().tolist()
    return result


def save_checkpoint(
    path: Path,
    *,
    backbone: nn.Module,
    model: MainWristResidualPolicy,
    normalizer: ActionNormalizer,
    state_normalizer: ActionNormalizer,
    config: dict,
    metadata: dict,
    step: int,
    evaluation_metrics: dict[str, object] | None = None,
) -> None:
    payload = {
        "backbone_model": cpu_state(backbone),
        "model": cpu_state(model),
        "action_mean": normalizer.mean.detach().cpu(),
        "action_std": normalizer.std.detach().cpu(),
        "state_mean": state_normalizer.mean.detach().cpu(),
        "state_std": state_normalizer.std.detach().cpu(),
        "feature_dim": int(metadata["feature_dim"]),
        "backbone_metadata": metadata,
        "config": config,
        "step": step,
        "loss": config["policy"]["loss"],
        "actuation_enabled": False,
    }
    if evaluation_metrics is not None:
        payload["evaluation_metrics"] = evaluation_metrics
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    policy = config["policy"]
    validate_config(config, args.backbone)
    require_complete_trossen_buffer(
        args.image_buffer,
        dataset_revision=str(config["dataset"]["revision"]),
        tasks=[str(task) for task in config["dataset"]["tasks"]],
        episodes_per_task=int(config["dataset"]["demonstrations_per_task"]),
        frames_per_episode=int(config["evaluation"]["max_rollout_steps"]),
    )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(args.device)
    asset, _ = resolve_backbone_asset(
        config,
        args.backbone,
        cache_dir=args.hub_cache_dir,
        local_files_only=args.offline,
    )
    backbone, metadata = load_trainable_tcc_backbone(
        asset, args.tcc_source_root, device
    )
    metadata = {
        **metadata,
        "camera_backbones": "shared",
        "camera_names": ["cam_main", "cam_wrist"],
        "batch_norm_statistics": "frozen_pretrained",
    }
    train_with_frozen_batch_norm_statistics(backbone)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    augmentation = config["augmentation"]
    train_dataset = TrossenMultiViewDataset(
        args.image_buffer,
        build_r3m_train_transform(
            int(metadata["image_size"]),
            brightness=float(augmentation["brightness"]),
            contrast=float(augmentation["contrast"]),
            saturation=float(augmentation["saturation"]),
            hue=float(augmentation["hue"]),
            blur_probability=float(augmentation["blur_probability"]),
        ),
        include_state=True,
    )
    evaluation_dataset = TrossenMultiViewDataset(
        args.image_buffer,
        build_r3m_transform(int(metadata["image_size"])),
        include_state=True,
    )
    split = config["split"]
    train_indices, validation_indices, test_indices = episode_split_indices(
        args.image_buffer,
        train_episodes=int(split["train_episodes_per_task"]),
        validation_episodes=int(split["validation_episodes_per_task"]),
        test_episodes=int(split["test_episodes_per_task"]),
        seed=int(split["shuffle_seed"]),
    )
    if validation_indices:
        raise ValueError("V10 train/test protocol requires an empty validation split")
    action_mean, action_std = train_dataset.action_statistics(train_indices)
    normalizer = ActionNormalizer(action_mean, action_std).to(device)
    state_mean, state_std = train_dataset.state_statistics(train_indices)
    state_normalizer = ActionNormalizer(state_mean, state_std).to(device)
    model = MainWristResidualPolicy(
        feature_dim=int(metadata["feature_dim"]),
        action_dim=int(policy["action_dim"]),
        hidden_dims=tuple(policy["hidden_dimensions"]),
        projection_dim=int(policy["camera_projection_dim"]),
        gate_hidden_dim=int(policy["camera_gate_hidden_dim"]),
        proprio_dim=int(policy["proprioception_dim"]),
        wrist_dropout=float(policy["wrist_dropout"]),
        wrist_residual_scale=float(policy["wrist_residual_scale"]),
        gate_initial_bias=float(policy["gate_initial_bias"]),
        output_layer_scale=float(policy["output_layer_scale"]),
    ).to(device)

    iterations = args.iterations or int(policy["training_steps"])
    batch_size = int(policy["batch_size"])
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(policy["num_workers"])
    )
    train_subset = Subset(train_dataset, train_indices)
    sampler = RandomSampler(
        train_subset,
        replacement=True,
        num_samples=iterations * batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=True,
    )
    test_loader = DataLoader(
        Subset(evaluation_dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(workers, 4),
    )
    fusion_parameters = [
        *model.cam_main_projection.parameters(),
        *model.cam_wrist_projection.parameters(),
        *model.camera_gate.parameters(),
    ]
    head_parameters = [
        *model.main_policy.parameters(),
        *model.wrist_residual_policy.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": float(policy["learning_rate"])},
            {
                "params": fusion_parameters,
                "lr": float(policy["fusion_learning_rate"]),
            },
            {
                "params": backbone.parameters(),
                "lr": float(policy["backbone_learning_rate"]),
            },
        ],
        weight_decay=float(policy["weight_decay"]),
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    main_weight = float(policy["main_loss_weight"])
    residual_weight = float(policy["residual_regularization_weight"])
    checkpoint_every = int(policy["checkpoint_every"])
    history: list[dict[str, object]] = []
    iterator = iter(train_loader)
    for step in range(1, iterations + 1):
        train_with_frozen_batch_norm_statistics(backbone)
        model.train()
        main_images, wrist_images, states, actions = next(iterator)
        main_images = main_images.to(device, non_blocking=True)
        wrist_images = wrist_images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        features = backbone(torch.cat([main_images, wrist_images], dim=0)).float()
        main_features, wrist_features = features.chunk(2, dim=0)
        loss, train_metrics = policy_loss(
            model,
            main_features,
            wrist_features,
            state_normalizer.normalize(states),
            normalizer.normalize(actions),
            main_weight=main_weight,
            residual_weight=residual_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            [*model.parameters(), *backbone.parameters()],
            float(policy["gradient_clip_norm"]),
        )
        optimizer.step()
        row = {
            "step": step,
            "train_loss": float(loss.detach()),
            "train_fused_mse": float(train_metrics["fused_mse"]),
            "train_main_mse": float(train_metrics["main_mse"]),
            "train_gate_mean": float(train_metrics["gate_mean"]),
            "gradient_norm": float(gradient_norm),
        }
        if step == 1 or step % 100 == 0:
            print(json.dumps(row))
        if step % checkpoint_every == 0 or step == iterations:
            history.append(row)
            save_checkpoint(
                output_dir / f"checkpoint_{step:06d}.pt",
                backbone=backbone,
                model=model,
                normalizer=normalizer,
                state_normalizer=state_normalizer,
                config=config,
                metadata=metadata,
                step=step,
            )

    shutil.copyfile(
        output_dir / f"checkpoint_{iterations:06d}.pt",
        output_dir / "checkpoint_last.pt",
    )
    test_metrics = evaluate(
        backbone,
        model,
        test_loader,
        normalizer,
        state_normalizer,
        device,
        main_weight=main_weight,
        residual_weight=residual_weight,
    )
    metrics = {
        "history": history,
        "test": test_metrics,
        "train_transitions": len(train_indices),
        "test_transitions": len(test_indices),
        "training_iterations": iterations,
        "selected_checkpoint": f"checkpoint_{iterations:06d}.pt",
        "checkpoint_selection": "fixed_final_iteration_without_test_selection",
        "test_evaluations": 1,
        "action_representation": "absolute",
        "camera_backbones": "shared",
        "camera_fusion": CAMERA_FUSION,
        "proprioception": True,
        "batch_norm_statistics": "frozen_pretrained",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
