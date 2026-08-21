#!/usr/bin/env python3
"""Train the single-view HRP MLP-GMM policy end to end from JPEG observations."""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import chain
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from tcc_real_robot.hrp_image_data import HRPImageDataset
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
    return parser.parse_args()


def image_transform(image_size: int, training: bool) -> transforms.Compose:
    operations: list[torch.nn.Module] = [
        transforms.ConvertImageDtype(torch.float32)
    ]
    if training:
        kernel = int(0.05 * image_size)
        kernel += 1 - kernel % 2
        operations.extend(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.9, 1.0), antialias=False
                ),
                transforms.GaussianBlur(kernel_size=kernel),
            ]
        )
    else:
        operations.append(
            transforms.Resize((image_size, image_size), antialias=False)
        )
    operations.append(
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    )
    return transforms.Compose(operations)


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
    validation_loss: float,
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
            "best_validation_normalized_loss": validation_loss,
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
) -> tuple[float, float]:
    backbone.eval()
    model.eval()
    loss_total = 0.0
    error_total = 0.0
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
        prediction = action_normalizer.denormalize(
            model(features, None, tasks, normalized_state)
        )
        batch_count = images.shape[0]
        loss_total += float(loss) * batch_count
        error_total += float(torch.abs(prediction - actions).mean()) * batch_count
        count += batch_count
    backbone.train()
    model.train()
    return loss_total / count, error_total / count


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    policy_config = config["policy"]
    if policy_config["cameras"] != ["cam_main"]:
        raise ValueError("Official HRP reproduction requires cam_main only")
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
    train_transform = image_transform(int(metadata["image_size"]), True)
    eval_transform = image_transform(int(metadata["image_size"]), False)
    train_data = HRPImageDataset(
        args.image_buffer, "train", transform=train_transform
    )
    validation_data = HRPImageDataset(
        args.image_buffer, "validation", transform=eval_transform
    )
    test_data = HRPImageDataset(
        args.image_buffer, "test", transform=eval_transform
    )
    state_mean, state_std, action_mean, action_std = (
        train_data.state_action_statistics()
    )
    state_normalizer = ActionNormalizer(state_mean, state_std).to(device)
    action_normalizer = ActionNormalizer(action_mean, action_std).to(device)
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
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    optimizer = torch.optim.Adam(
        chain(backbone.parameters(), model.parameters()),
        lr=float(policy_config["learning_rate"]),
        weight_decay=float(policy_config["weight_decay"]),
    )
    iterations = args.iterations or int(policy_config["training_steps"])
    eval_every = int(policy_config["eval_every"])
    eval_samples = int(policy_config["evaluation_samples"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    train_iterator = iter(train_loader)
    for step in range(1, iterations + 1):
        try:
            images, states, actions, tasks = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, states, actions, tasks = next(train_iterator)
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        tasks = tasks.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
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
            print(json.dumps({"step": step, "train_gmm_nll": float(loss)}))
        if step % eval_every == 0 or step == iterations:
            validation_loss, validation_mae = evaluate(
                backbone,
                model,
                validation_loader,
                state_normalizer,
                action_normalizer,
                device,
                eval_samples,
            )
            row = {
                "step": step,
                "validation_gmm_nll": validation_loss,
                "validation_velocity_mae": validation_mae,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True))
            if validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(
                    output_dir / "checkpoint_best.pt",
                    backbone=backbone,
                    model=model,
                    config=config,
                    metadata=metadata,
                    state_normalizer=state_normalizer,
                    action_normalizer=action_normalizer,
                    step=step,
                    validation_loss=validation_loss,
                )
    test_loss, test_mae = evaluate(
        backbone,
        model,
        test_loader,
        state_normalizer,
        action_normalizer,
        device,
        eval_samples,
    )
    save_checkpoint(
        output_dir / "checkpoint_last.pt",
        backbone=backbone,
        model=model,
        config=config,
        metadata=metadata,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        step=iterations,
        validation_loss=best_loss,
    )
    shutil.copyfile(
        output_dir / "checkpoint_best.pt",
        output_dir / f"checkpoint_{iterations:06d}.pt",
    )
    metrics = {
        "history": history,
        "best_validation_gmm_nll": best_loss,
        "test_gmm_nll": test_loss,
        "test_velocity_mae": test_mae,
        "training_iterations": iterations,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
