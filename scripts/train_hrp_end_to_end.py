#!/usr/bin/env python3
"""Train the single-view HRP MLP-GMM policy end to end from JPEG observations."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, Subset

from tcc_real_robot.hrp_image_data import (
    HRPImageDataset,
    official_hrp_transition_split,
    require_complete_joint_position_buffer,
)
from tcc_real_robot.hrp_vision import build_hrp_image_transform
from tcc_real_robot.model_assets import resolve_backbone_asset
from tcc_real_robot.policy import (
    ActionNormalizer,
    HRPSingleViewGaussianMixturePolicy,
)
from tcc_real_robot.policy_runtime import resolve_device
from tcc_real_robot.tcc_backbone import load_trainable_tcc_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_v8_hrp_official_single_view_60.yaml"),
    )
    parser.add_argument("--image-buffer", type=Path, required=True)
    parser.add_argument("--backbone", default="hrp_imagenet")
    parser.add_argument("--hub-cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tcc-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Keep the visual backbone fixed and train only the HRP MLP-GMM head.",
    )
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    *,
    backbone: torch.nn.Module,
    model: HRPSingleViewGaussianMixturePolicy,
    config: dict,
    metadata: dict,
    state_normalizer: ActionNormalizer,
    action_normalizer: ActionNormalizer,
    step: int,
    heldout_loss: float,
) -> None:
    torch.save(
        {
            "backbone_model": {
                key: value.detach().cpu()
                for key, value in backbone.state_dict().items()
            },
            "model": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "state_mean": state_normalizer.mean.detach().cpu(),
            "state_std": state_normalizer.std.detach().cpu(),
            "action_mean": action_normalizer.mean.detach().cpu(),
            "action_std": action_normalizer.std.detach().cpu(),
            "feature_dim": int(metadata["feature_dim"]),
            "backbone_metadata": metadata,
            "config": config,
            "step": step,
            "heldout_normalized_gmm_nll": heldout_loss,
            "loss": "gmm_nll",
            "actuation_enabled": False,
        },
        path,
    )


@torch.inference_mode()
def evaluate(
    backbone: torch.nn.Module,
    model: HRPSingleViewGaussianMixturePolicy,
    loader: DataLoader,
    state_normalizer: ActionNormalizer,
    action_normalizer: ActionNormalizer,
    device: torch.device,
    max_samples: int,
    *,
    train_backbone: bool,
) -> tuple[float, float, float, float]:
    backbone.eval()
    model.eval()
    loss_total = 0.0
    error_total = 0.0
    normalized_l2_total = 0.0
    sign_disagreement_total = 0.0
    count = 0
    for images, states, actions, tasks in loader:
        remaining = max_samples - count
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=True)
        states = states[:remaining].to(device, non_blocking=True)
        actions = actions[:remaining].to(device, non_blocking=True)
        tasks = tasks[:remaining].to(device, non_blocking=True)
        normalized_state = state_normalizer.normalize(states)
        target = action_normalizer.normalize(actions)
        features = backbone(images).float()
        loss = model.negative_log_likelihood(
            target, features, None, tasks, normalized_state
        )
        normalized_prediction = model(features, None, tasks, normalized_state)
        prediction = action_normalizer.denormalize(normalized_prediction)
        batch_count = images.shape[0]
        loss_total += float(loss) * batch_count
        error_total += float(torch.abs(prediction - actions).mean()) * batch_count
        normalized_l2_total += (
            float(torch.square(normalized_prediction - target).mean()) * batch_count
        )
        sign_disagreement_total += (
            float(
                torch.logical_xor(target > 0, normalized_prediction > 0).float().mean()
            )
            * batch_count
        )
        count += batch_count
    if train_backbone:
        backbone.train()
    model.train()
    return (
        loss_total / count,
        normalized_l2_total / count,
        sign_disagreement_total / count,
        error_total / count,
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    policy_config = config["policy"]
    require_complete_joint_position_buffer(
        args.image_buffer,
        dataset_revision=str(config["dataset"]["revision"]),
        tasks=[str(task) for task in config["dataset"]["tasks"]],
        episodes_per_task=int(config["dataset"]["demonstrations_per_task"]),
        frames_per_episode=int(config["evaluation"]["max_rollout_steps"]),
    )
    required_action_contract = {
        "action_representation": "absolute",
        "action_space": "joint_position",
        "action_adapter": "trossen_joint_position_passthrough",
        "state_representation": "measured_joint_position",
        "action_source": "original_lerobot_action",
        "gripper_action": "position",
    }
    action_contract_mismatches = {
        key: (policy_config.get(key), value)
        for key, value in required_action_contract.items()
        if policy_config.get(key) != value
    }
    if action_contract_mismatches:
        raise ValueError(
            "Training config is not the original joint-position contract: "
            f"{action_contract_mismatches}"
        )
    if policy_config["cameras"] != ["cam_main"]:
        raise ValueError("Official HRP reproduction requires cam_main only")
    if policy_config.get("precision") != "float32":
        raise ValueError("Official HRP reproduction requires float32 training")
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
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train_backbone = not args.freeze_backbone
    backbone.requires_grad_(train_backbone)
    backbone.train(train_backbone)
    train_transform = build_hrp_image_transform(
        int(metadata["image_size"]),
        training=True,
        augmentation=config.get("augmentation"),
    )
    eval_transform = build_hrp_image_transform(
        int(metadata["image_size"]), training=False
    )
    all_train_views = HRPImageDataset(
        args.image_buffer, "train", transform=train_transform
    )
    all_eval_views = HRPImageDataset(
        args.image_buffer, "train", transform=eval_transform
    )
    split_config = config["split"]
    if split_config["protocol"] != "hrp_fixed_transition_holdout":
        raise ValueError("Official HRP training requires its transition holdout")
    train_indices, heldout_indices = official_hrp_transition_split(
        len(all_train_views),
        held_out_transitions=int(split_config["held_out_transitions"]),
        shuffle_seed=int(split_config["shuffle_seed"]),
    )
    train_data = Subset(all_train_views, train_indices)
    heldout_data = Subset(all_eval_views, heldout_indices)
    identity_mean = torch.zeros(int(policy_config["action_dim"]))
    identity_std = torch.ones(int(policy_config["action_dim"]))
    state_mean, state_std, action_mean, action_std = (
        all_train_views.state_action_statistics(train_indices)
    )
    state_normalizer = ActionNormalizer(
        state_mean if bool(policy_config["normalize_state"]) else identity_mean,
        state_std if bool(policy_config["normalize_state"]) else identity_std,
    ).to(device)
    action_normalizer = ActionNormalizer(
        action_mean if bool(policy_config["normalize_actions"]) else identity_mean,
        action_std if bool(policy_config["normalize_actions"]) else identity_std,
    ).to(device)
    model = HRPSingleViewGaussianMixturePolicy(
        feature_dim=int(metadata["feature_dim"]),
        action_dim=int(policy_config["action_dim"]),
        hidden_dims=tuple(policy_config["hidden_dimensions"]),
        state_dim=int(policy_config["proprioception_dim"]),
        dropout=float(policy_config["dropout"]),
        num_modes=int(policy_config["num_modes"]),
        min_std=float(policy_config["min_std"]),
    ).to(device)
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(policy_config["num_workers"])
    )
    batch_size = int(policy_config["batch_size"])
    iterations = args.iterations or int(policy_config["training_steps"])
    train_sampler = RandomSampler(
        train_data,
        replacement=True,
        num_samples=iterations * batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=True,
    )
    heldout_loader = DataLoader(
        heldout_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(workers, 4),
    )
    optimized_parameters = (
        chain(backbone.parameters(), model.parameters())
        if train_backbone
        else model.parameters()
    )
    optimizer = torch.optim.Adam(
        optimized_parameters,
        lr=float(policy_config["learning_rate"]),
        weight_decay=float(policy_config["weight_decay"]),
    )
    eval_every = int(policy_config["eval_every"])
    heldout_samples = int(split_config["held_out_transitions"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    train_iterator = iter(train_loader)
    save_checkpoint(
        output_dir / "checkpoint_000000.pt",
        backbone=backbone,
        model=model,
        config=config,
        metadata=metadata,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        step=0,
        heldout_loss=float("nan"),
    )
    for step in range(1, iterations + 1):
        images, states, actions, tasks = next(train_iterator)
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        tasks = tasks.to(device, non_blocking=True)
        if train_backbone:
            features = backbone(images).float()
        else:
            with torch.no_grad():
                features = backbone(images).float()
        loss = model.negative_log_likelihood(
            action_normalizer.normalize(actions),
            features,
            None,
            tasks,
            state_normalizer.normalize(states),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, "train_gmm_nll": float(loss.detach())}))
        if step % eval_every == 0 or step == iterations:
            heldout_loss, heldout_l2, heldout_lsig, heldout_mae = evaluate(
                backbone,
                model,
                heldout_loader,
                state_normalizer,
                action_normalizer,
                device,
                heldout_samples,
                train_backbone=train_backbone,
            )
            row = {
                "step": step,
                "heldout_normalized_gmm_nll": heldout_loss,
                "heldout_normalized_action_l2": heldout_l2,
                "heldout_normalized_sign_disagreement": heldout_lsig,
                "heldout_joint_position_mae": heldout_mae,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True))
            save_checkpoint(
                output_dir / f"checkpoint_{step:06d}.pt",
                backbone=backbone,
                model=model,
                config=config,
                metadata=metadata,
                state_normalizer=state_normalizer,
                action_normalizer=action_normalizer,
                step=step,
                heldout_loss=heldout_loss,
            )
    shutil.copyfile(
        output_dir / f"checkpoint_{iterations:06d}.pt",
        output_dir / "checkpoint_last.pt",
    )
    metrics = {
        "history": history,
        "split_protocol": split_config,
        "train_transitions": len(train_indices),
        "heldout_transitions": len(heldout_indices),
        "training_iterations": iterations,
        "state_normalization": bool(policy_config["normalize_state"]),
        "action_normalization": bool(policy_config["normalize_actions"]),
        "backbone_frozen": not train_backbone,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
