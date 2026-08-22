#!/usr/bin/env python3
"""Train V9 R3M with independent cameras and closed-loop robot state."""

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
from tcc_real_robot.policy import ActionNormalizer, R3MRobomimicPolicy
from tcc_real_robot.policy_runtime import resolve_device
from tcc_real_robot.r3m_vision import build_r3m_transform
from tcc_real_robot.tcc_backbone import load_independent_camera_backbones
from tcc_real_robot.trossen_image_data import (
    TrossenMultiViewDataset,
    episode_split_indices,
    require_complete_trossen_buffer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_v9_r3m_robomimic_proprio_100.yaml"),
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


def save_checkpoint(
    path: Path,
    *,
    backbone: nn.Module,
    model: R3MRobomimicPolicy,
    normalizer: ActionNormalizer,
    state_normalizer: ActionNormalizer,
    config: dict,
    metadata: dict,
    step: int,
    validation_mse: float,
) -> None:
    torch.save(
        {
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
            "validation_normalized_mse": validation_mse,
            "loss": "mse",
            "actuation_enabled": False,
        },
        path,
    )


@torch.inference_mode()
def evaluate(
    backbone: nn.Module,
    model: R3MRobomimicPolicy,
    loader: DataLoader,
    normalizer: ActionNormalizer,
    state_normalizer: ActionNormalizer,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    backbone.eval()
    model.eval()
    total_mse = 0.0
    total_absolute_error = torch.zeros(model.action_dim, device=device)
    count = 0
    for main_images, wrist_images, states, actions in loader:
        main_images = main_images.to(device, non_blocking=True)
        wrist_images = wrist_images.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        main_features, wrist_features = backbone(main_images, wrist_images)
        prediction = model(
            main_features.float(),
            wrist_features.float(),
            state_normalizer.normalize(states),
        )
        target = normalizer.normalize(actions)
        batch = actions.shape[0]
        total_mse += float(nn.functional.mse_loss(prediction, target)) * batch
        denormalized = normalizer.denormalize(prediction)
        total_absolute_error += torch.abs(denormalized - actions).sum(0)
        count += batch
    per_dimension = (total_absolute_error / count).cpu().tolist()
    return total_mse / count, float(total_absolute_error.sum() / (count * 7)), per_dimension


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    policy = config["policy"]
    expected_contract = {
        "architecture": "r3m_deterministic_mlp_dual_independent_encoder_proprio",
        "cameras": ["cam_main", "cam_wrist"],
        "shared_camera_backbone": False,
        "camera_fusion": "raw_concat",
        "action_representation": "absolute",
        "action_adapter": "trossen_joint_position_passthrough",
        "action_distribution": "deterministic",
        "loss": "mse",
        "proprioception": True,
        "proprioception_dim": 7,
        "normalize_state": True,
    }
    mismatches = {
        key: (policy.get(key), value)
        for key, value in expected_contract.items()
        if policy.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V9 is not the minimal R3M contract: {mismatches}")
    if policy.get("hidden_dimensions") != [256, 256]:
        raise ValueError("V9 R3M MLP must use hidden dimensions [256, 256]")
    if policy.get("input_batch_norm") is not True:
        raise ValueError("V9 R3M MLP requires input BatchNorm")
    forbidden = {
        "camera_projection_dim",
        "camera_gate_hidden_dim",
        "task_conditioning",
        "progress_conditioning",
        "progress_dim",
        "dropout",
        "input_layer_norm",
        "num_modes",
        "min_std",
    }
    present = sorted(forbidden.intersection(policy))
    if present or "augmentation" in config or "sampling" in config:
        raise ValueError(
            "V9 contains components outside the minimal R3M design: "
            f"policy={present}, augmentation={'augmentation' in config}, "
            f"sampling={'sampling' in config}"
        )
    if policy.get("state_representation") != (
        "measured_joint_position_6_plus_gripper"
    ):
        raise ValueError("V9 proprioception must use measured 7-D joint state")
    proprioception_dropout = float(policy.get("proprioception_dropout", 0.0))
    if not 0.0 <= proprioception_dropout < 1.0:
        raise ValueError("proprioception_dropout must be in [0, 1)")

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
    backbone, metadata = load_independent_camera_backbones(
        asset, args.tcc_source_root, device, trainable=True
    )
    backbone.train()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    dataset = TrossenMultiViewDataset(
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
    action_mean, action_std = dataset.action_statistics(train_indices)
    normalizer = ActionNormalizer(action_mean, action_std).to(device)
    state_mean, state_std = dataset.state_statistics(train_indices)
    state_normalizer = ActionNormalizer(state_mean, state_std).to(device)
    model = R3MRobomimicPolicy(
        feature_dim=int(metadata["feature_dim"]),
        action_dim=int(policy["action_dim"]),
        hidden_dims=tuple(policy["hidden_dimensions"]),
        output_layer_scale=float(policy["output_layer_scale"]),
        proprio_dim=int(policy["proprioception_dim"]),
        proprio_dropout=proprioception_dropout,
    ).to(device)

    iterations = args.iterations or int(policy["training_steps"])
    batch_size = int(policy["batch_size"])
    workers = args.num_workers if args.num_workers is not None else int(policy["num_workers"])
    sampler = RandomSampler(
        Subset(dataset, train_indices),
        replacement=True,
        num_samples=iterations * batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(workers, 4),
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(workers, 4),
    )
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": float(policy["learning_rate"])},
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
    eval_every = int(policy["eval_every"])
    history: list[dict[str, object]] = []
    best_mse = float("inf")
    best_step = 0
    best_backbone_state: dict[str, torch.Tensor] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    iterator = iter(train_loader)
    for step in range(1, iterations + 1):
        backbone.train()
        model.train()
        main_images, wrist_images, states, actions = next(iterator)
        main_images = main_images.to(device, non_blocking=True)
        wrist_images = wrist_images.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        main_features, wrist_features = backbone(main_images, wrist_images)
        prediction = model(
            main_features.float(),
            wrist_features.float(),
            state_normalizer.normalize(states),
        )
        loss = nn.functional.mse_loss(prediction, normalizer.normalize(actions))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_normalized_mse": float(loss.detach()),
                    }
                )
            )
        if step % eval_every == 0 or step == iterations:
            validation_mse, validation_mae, validation_mae_per_dim = evaluate(
                backbone,
                model,
                validation_loader,
                normalizer,
                state_normalizer,
                device,
            )
            row = {
                "step": step,
                "validation_normalized_mse": validation_mse,
                "validation_action_mae": validation_mae,
                "validation_action_mae_per_dimension": validation_mae_per_dim,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True))
            save_checkpoint(
                output_dir / f"checkpoint_{step:06d}.pt",
                backbone=backbone,
                model=model,
                normalizer=normalizer,
                state_normalizer=state_normalizer,
                config=config,
                metadata=metadata,
                step=step,
                validation_mse=validation_mse,
            )
            if validation_mse < best_mse:
                best_mse = validation_mse
                best_step = step
                best_backbone_state = cpu_state(backbone)
                best_model_state = cpu_state(model)

    shutil.copyfile(
        output_dir / f"checkpoint_{iterations:06d}.pt",
        output_dir / "checkpoint_last.pt",
    )
    if best_backbone_state is None or best_model_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    backbone.load_state_dict(best_backbone_state, strict=True)
    model.load_state_dict(best_model_state, strict=True)
    save_checkpoint(
        output_dir / "checkpoint_best.pt",
        backbone=backbone,
        model=model,
        normalizer=normalizer,
        state_normalizer=state_normalizer,
        config=config,
        metadata=metadata,
        step=best_step,
        validation_mse=best_mse,
    )
    test_mse, test_mae, test_mae_per_dim = evaluate(
        backbone,
        model,
        test_loader,
        normalizer,
        state_normalizer,
        device,
    )
    metrics = {
        "history": history,
        "best_step": best_step,
        "best_validation_normalized_mse": best_mse,
        "test_normalized_mse": test_mse,
        "test_action_mae": test_mae,
        "test_action_mae_per_dimension": test_mae_per_dim,
        "train_transitions": len(train_indices),
        "validation_transitions": len(validation_indices),
        "test_transitions": len(test_indices),
        "training_iterations": iterations,
        "action_representation": "absolute",
        "proprioception": True,
        "proprioception_dim": 7,
        "proprioception_dropout": proprioception_dropout,
        "action_distribution": "deterministic",
        "camera_backbones": "independent",
        "camera_fusion": "raw_concat",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
