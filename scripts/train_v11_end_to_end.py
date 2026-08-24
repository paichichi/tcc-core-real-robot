#!/usr/bin/env python3
"""Train single-view V11 while fine-tuning ours RN50 end to end."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset

from tcc_real_robot.model_assets import resolve_backbone_asset
from tcc_real_robot.policy import ActionNormalizer, TCCMLPPolicy
from tcc_real_robot.policy_runtime import resolve_device
from tcc_real_robot.r3m_vision import build_r3m_transform
from tcc_real_robot.tcc_backbone import load_trainable_tcc_backbone
from tcc_real_robot.trossen_image_data import (
    TrossenSingleViewChunkDataset,
    episode_split_indices,
    require_complete_trossen_buffer,
)

ARCHITECTURE = "pooled_feature_mlp"
IMPLEMENTATION = "tcc_mlp_bc_v11_end_to_end_chunked_absolute"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_v11_end_to_end_rn50_100.yaml"),
    )
    parser.add_argument("--image-buffer", type=Path, required=True)
    parser.add_argument("--backbone", default="ours_rn50")
    parser.add_argument("--hub-cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tcc-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def train_with_frozen_batch_norm_statistics(backbone: nn.Module) -> None:
    """Train convolution weights while preserving pretrained BN statistics."""
    backbone.train()
    for module in backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def validate_config(config: dict, backbone_name: str) -> None:
    policy = config["policy"]
    expected = {
        "implementation": IMPLEMENTATION,
        "architecture": ARCHITECTURE,
        "cameras": ["cam_main"],
        "camera_fusion": "raw_concat",
        "action_representation": "absolute",
        "action_space": "joint_position",
        "action_adapter": "trossen_joint_position_passthrough",
        "action_distribution": "deterministic",
        "action_chunk_size": 40,
        "proprioception": True,
        "proprioception_dim": 7,
        "normalize_state": True,
        "normalize_actions": True,
        "action_leads_measured_state_frames": 2,
    }
    mismatches = {
        key: (policy.get(key), value)
        for key, value in expected.items()
        if policy.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Invalid end-to-end V11 contract: {mismatches}")
    if backbone_name != "ours_rn50":
        raise ValueError("End-to-end V11 intentionally supports only ours_rn50")
    if config["backbone"].get("frozen") is not False:
        raise ValueError("End-to-end V11 requires backbone.frozen=false")
    if config["backbone"].get("fine_tuning") != (
        "full_end_to_end_freeze_batch_norm_statistics"
    ):
        raise ValueError("End-to-end V11 must fine-tune RN50 with frozen BN stats")
    split = config["split"]
    expected_split = {
        "protocol": "act_train_validation_episode_holdout",
        "train_episodes_per_task": 80,
        "validation_episodes_per_task": 20,
        "test_episodes_per_task": 0,
        "unused_episodes_per_task": 0,
    }
    split_mismatches = {
        key: (split.get(key), value)
        for key, value in expected_split.items()
        if split.get(key) != value
    }
    if split_mismatches:
        raise ValueError(f"V11 requires the ACT 80/20 split: {split_mismatches}")
    if config["dataset"].get("action_leads_measured_state_frames") != policy.get(
        "action_leads_measured_state_frames"
    ):
        raise ValueError("Dataset and policy action lead disagree")


@torch.inference_mode()
def evaluate(
    backbone: nn.Module,
    model: TCCMLPPolicy,
    loader: DataLoader,
    action_normalizer: ActionNormalizer,
    state_normalizer: ActionNormalizer,
    device: torch.device,
) -> dict[str, object]:
    backbone.eval()
    model.eval()
    loss_total = 0.0
    absolute_error = torch.zeros(model.action_dim, device=device)
    count = 0
    for images, states, actions, task_indices in loader:
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        task_indices = task_indices.to(device, non_blocking=True)
        features = backbone(images).float()
        prediction = model(
            features,
            None,
            task_indices,
            state_normalizer.normalize(states),
            None,
        )
        normalized_actions = action_normalizer.normalize(actions)
        loss = nn.functional.mse_loss(prediction, normalized_actions)
        batch = actions.shape[0]
        loss_total += float(loss) * batch
        absolute_error += torch.abs(
            action_normalizer.denormalize(prediction) - actions
        ).sum(0)
        count += batch
    if count == 0:
        raise RuntimeError("Validation split is empty")
    per_dimension = (absolute_error / count).cpu().tolist()
    return {
        "normalized_mse": loss_total / count,
        "action_mae": float(absolute_error.sum() / (count * model.action_dim)),
        "action_mae_per_dimension": per_dimension,
    }


def checkpoint_payload(
    *,
    backbone_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    action_normalizer: ActionNormalizer,
    state_normalizer: ActionNormalizer,
    config: dict,
    metadata: dict,
    step: int,
    best_validation_step: int,
    best_validation_loss: float,
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "backbone_model": backbone_state,
        "model": model_state,
        "action_mean": action_normalizer.mean.detach().cpu(),
        "action_std": action_normalizer.std.detach().cpu(),
        "state_mean": state_normalizer.mean.detach().cpu(),
        "state_std": state_normalizer.std.detach().cpu(),
        "feature_dim": int(metadata["feature_dim"]),
        "backbone_metadata": metadata,
        "config": config,
        "step": step,
        "best_validation_step": best_validation_step,
        "best_validation_normalized_loss": best_validation_loss,
        "loss": "mse",
        "history": history,
        "actuation_enabled": False,
    }


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
        action_leads_measured_state_frames=int(
            config["dataset"]["action_leads_measured_state_frames"]
        ),
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
        "camera_names": ["cam_main"],
        "camera_backbones": "single_trainable",
        "batch_norm_statistics": "frozen_pretrained",
    }
    train_with_frozen_batch_norm_statistics(backbone)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    transform = build_r3m_transform(int(metadata["image_size"]))
    train_dataset = TrossenSingleViewChunkDataset(
        args.image_buffer,
        transform,
        action_chunk_size=int(policy["action_chunk_size"]),
    )
    validation_dataset = TrossenSingleViewChunkDataset(
        args.image_buffer,
        transform,
        action_chunk_size=int(policy["action_chunk_size"]),
    )
    split = config["split"]
    train_indices, validation_indices, test_indices = episode_split_indices(
        args.image_buffer,
        train_episodes=int(split["train_episodes_per_task"]),
        validation_episodes=int(split["validation_episodes_per_task"]),
        test_episodes=int(split["test_episodes_per_task"]),
        seed=int(split["shuffle_seed"]),
    )
    if test_indices:
        raise RuntimeError("ACT-style V11 must not construct an offline test split")
    action_mean, action_std = train_dataset.action_statistics(train_indices)
    state_mean, state_std = train_dataset.state_statistics(train_indices)
    action_normalizer = ActionNormalizer(action_mean, action_std).to(device)
    state_normalizer = ActionNormalizer(state_mean, state_std).to(device)
    model = TCCMLPPolicy(
        feature_dim=int(metadata["feature_dim"]),
        num_tasks=int(policy["number_of_tasks"]),
        action_dim=int(policy["action_dim"]) * int(policy["action_chunk_size"]),
        hidden_dims=tuple(policy["hidden_dimensions"]),
        proprio_dim=int(policy["proprioception_dim"]),
        progress_dim=0,
        input_batch_norm=bool(policy["input_batch_norm"]),
        input_layer_norm=bool(policy["input_layer_norm"]),
        output_layer_scale=float(policy["output_layer_scale"]),
        camera_names=("cam_main",),
        camera_fusion="raw_concat",
        dropout=float(policy["dropout"]),
    ).to(device)

    steps = args.steps or int(policy["training_steps"])
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
        num_samples=steps * batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(policy["prefetch_factor"]) if workers > 0 else None,
        drop_last=True,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=min(workers, 4),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": float(policy["head_learning_rate"]),
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
    validation_every = int(policy["validation_every"])
    checkpoint_every = int(policy["checkpoint_every"])
    if validation_every <= 0 or checkpoint_every <= 0:
        raise ValueError("Validation and checkpoint intervals must be positive")
    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_step = 0
    best_backbone_state: dict[str, torch.Tensor] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    iterator = iter(train_loader)
    for step in range(1, steps + 1):
        train_with_frozen_batch_norm_statistics(backbone)
        model.train()
        images, states, actions, task_indices = next(iterator)
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        task_indices = task_indices.to(device, non_blocking=True)
        features = backbone(images).float()
        prediction = model(
            features,
            None,
            task_indices,
            state_normalizer.normalize(states),
            None,
        )
        loss = nn.functional.mse_loss(
            prediction,
            action_normalizer.normalize(actions),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            [*backbone.parameters(), *model.parameters()],
            float(policy["gradient_clip_norm"]),
        )
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_normalized_mse": float(loss.detach()),
                        "gradient_norm": float(gradient_norm),
                    },
                    sort_keys=True,
                )
            )
        should_validate = step % validation_every == 0 or step == steps
        if should_validate:
            validation = evaluate(
                backbone,
                model,
                validation_loader,
                action_normalizer,
                state_normalizer,
                device,
            )
            row = {
                "step": step,
                "train_normalized_mse_last": float(loss.detach()),
                "validation_normalized_mse": validation["normalized_mse"],
                "validation_action_mae": validation["action_mae"],
                "validation_action_mae_per_dimension": validation[
                    "action_mae_per_dimension"
                ],
            }
            history.append(row)
            print(json.dumps({"validation": row}, sort_keys=True))
            if float(validation["normalized_mse"]) < best_validation_loss:
                best_validation_loss = float(validation["normalized_mse"])
                best_step = step
                best_backbone_state = cpu_state(backbone)
                best_model_state = cpu_state(model)
        if step % checkpoint_every == 0 and step != steps:
            torch.save(
                checkpoint_payload(
                    backbone_state=cpu_state(backbone),
                    model_state=cpu_state(model),
                    action_normalizer=action_normalizer,
                    state_normalizer=state_normalizer,
                    config=config,
                    metadata=metadata,
                    step=step,
                    best_validation_step=best_step,
                    best_validation_loss=best_validation_loss,
                    history=history,
                ),
                output_dir / f"checkpoint_{step:06d}.pt",
            )

    if best_backbone_state is None or best_model_state is None:
        raise RuntimeError("No validation-selected V11 checkpoint was produced")
    last_payload = checkpoint_payload(
        backbone_state=cpu_state(backbone),
        model_state=cpu_state(model),
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        config=config,
        metadata=metadata,
        step=steps,
        best_validation_step=best_step,
        best_validation_loss=best_validation_loss,
        history=history,
    )
    best_payload = checkpoint_payload(
        backbone_state=best_backbone_state,
        model_state=best_model_state,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        config=config,
        metadata=metadata,
        step=best_step,
        best_validation_step=best_step,
        best_validation_loss=best_validation_loss,
        history=history,
    )
    torch.save(last_payload, output_dir / "checkpoint_last.pt")
    torch.save(best_payload, output_dir / "checkpoint_best.pt")
    torch.save(best_payload, output_dir / f"checkpoint_{steps:06d}.pt")
    metrics = {
        "history": history,
        "best_validation_step": best_step,
        "best_validation_normalized_mse": best_validation_loss,
        "selected_checkpoint": f"checkpoint_{steps:06d}.pt",
        "checkpoint_selection": "lowest_held_out_episode_validation_mse",
        "train_episodes": int(split["train_episodes_per_task"]),
        "validation_episodes": int(split["validation_episodes_per_task"]),
        "test_episodes": 0,
        "train_transitions": len(train_indices),
        "validation_transitions": len(validation_indices),
        "training_steps": steps,
        "camera_names": ["cam_main"],
        "backbone_fine_tuning": "full_end_to_end",
        "batch_norm_statistics": "frozen_pretrained",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
