#!/usr/bin/env python3
"""Train the offline TCC-MLP-BC v0 policy from cached features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from tcc_real_robot.policy import ActionNormalizer, TCCMLPPolicy
from tcc_real_robot.policy_data import load_cached_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--cache-root", type=Path, default=Path("runs/feature_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tcc_mlp_bc_v0"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--episodes-per-task", type=int)
    return parser.parse_args()


def sample_batch(
    data: dict[str, torch.Tensor], batch_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    indices = torch.randint(data["action"].shape[0], (batch_size,))
    return {key: value[indices].to(device) for key, value in data.items()}


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    cache_manifest_path = args.cache_root / "manifest.json"
    if not cache_manifest_path.is_file():
        raise FileNotFoundError(cache_manifest_path)
    cache_manifest = json.loads(cache_manifest_path.read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    selected_episode_ids: dict[int, set[int]] | None = None
    if args.episodes_per_task is not None:
        if args.episodes_per_task < 1:
            raise ValueError("--episodes-per-task must be positive")
        records = cache_manifest.get("episode_records")
        if not isinstance(records, list):
            raise ValueError(
                "Feature cache manifest lacks ordered episode_records; "
                "rebuild or extend the cache before selecting a subset"
            )
        selected_episode_ids = {}
        for row in records:
            if row.get("split") != "train":
                continue
            task_index = int(row["task_index"])
            selected = selected_episode_ids.setdefault(task_index, set())
            if len(selected) < args.episodes_per_task:
                selected.add(int(row["episode_index"]))
        expected_tasks = int(config["policy"]["number_of_tasks"])
        if len(selected_episode_ids) != expected_tasks or any(
            len(selected) != args.episodes_per_task
            for selected in selected_episode_ids.values()
        ):
            raise ValueError(
                f"Cache does not contain {args.episodes_per_task} ordered "
                "training episodes for every task"
            )
        config["split"]["train_episodes_per_task"] = args.episodes_per_task
        config["split"]["validation_episodes_per_task"] = 0
        config["split"]["test_episodes_per_task"] = 0
        config["split"]["unused_episodes_per_task"] = (
            int(config["dataset"]["demonstrations_per_task"])
            - args.episodes_per_task
        )

    train = load_cached_split(
        args.cache_root, "train", episode_ids_by_task=selected_episode_ids
    )
    feature_dim = int(train["cam_main"].shape[1])
    if train["cam_wrist"].shape[1] != feature_dim:
        raise ValueError("Camera feature dimensions differ")

    policy_config = config["policy"]
    if policy_config["proprioception"] is not False:
        raise ValueError("v0 is intentionally configured without proprioception")
    model = TCCMLPPolicy(
        feature_dim=feature_dim,
        num_tasks=int(policy_config["number_of_tasks"]),
        action_dim=int(policy_config["action_dim"]),
        hidden_dims=tuple(policy_config["hidden_dimensions"]),
        proprio_dim=0,
        input_batch_norm=bool(policy_config["input_batch_norm"]),
    ).to(device)
    action_mean = train["action"].mean(dim=0)
    action_std = train["action"].std(dim=0)
    normalizer = ActionNormalizer(action_mean, action_std).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(policy_config["learning_rate"])
    )
    loss_function = nn.MSELoss()
    steps = args.steps or int(policy_config["training_steps"])
    log_every = args.log_every or int(policy_config["log_every"])
    batch_size = int(policy_config["batch_size"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for step in range(1, steps + 1):
        model.train()
        batch = sample_batch(train, batch_size, device)
        prediction = model(
            batch["cam_main"].float(),
            batch["cam_wrist"].float(),
            batch["task_index"],
        )
        loss = loss_function(prediction, normalizer.normalize(batch["action"]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            row = {"step": step, "train_normalized_mse": float(loss.detach())}
            history.append(row)
            print(json.dumps(row, sort_keys=True))
    torch.save(
        {
            "model": model.state_dict(),
            "action_mean": action_mean,
            "action_std": action_std,
            "feature_dim": feature_dim,
            "feature_cache_manifest": cache_manifest,
            "training_episode_ids": (
                {
                    task_index: sorted(episode_ids)
                    for task_index, episode_ids in selected_episode_ids.items()
                }
                if selected_episode_ids is not None
                else None
            ),
            "config": config,
            "step": steps,
            "history": history,
            "actuation_enabled": False,
        },
        output_dir / f"checkpoint_{steps:06d}.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
