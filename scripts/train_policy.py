#!/usr/bin/env python3
"""Train a frozen-feature TCC-MLP behavior-cloning policy."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from tcc_real_robot.policy import ActionNormalizer, TCCMLPPolicy
from tcc_real_robot.policy_data import (
    load_cached_future_delta_split,
    load_cached_split,
)
from tcc_real_robot.policy_runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--cache-root", type=Path, default=Path("runs/feature_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tcc_mlp_bc_v1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--episodes-per-task", type=int)
    return parser.parse_args()


def sample_batch(
    data: dict[str, torch.Tensor], batch_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    indices = torch.randint(data["action"].shape[0], (batch_size,))
    return {key: value[indices].to(device) for key, value in data.items()}


@torch.inference_mode()
def evaluate_policy(
    model: TCCMLPPolicy,
    normalizer: ActionNormalizer,
    state_normalizer: ActionNormalizer | None,
    data: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    """Evaluate normalized Huber loss and denormalized action-space MAE."""
    model.eval()
    loss_total = 0.0
    absolute_error = torch.zeros(7, device=device)
    examples = int(data["action"].shape[0])
    for start in range(0, examples, batch_size):
        stop = min(start + batch_size, examples)
        batch = {key: value[start:stop].to(device) for key, value in data.items()}
        proprioception = (
            state_normalizer.normalize(batch["state"].float())
            if state_normalizer is not None
            else None
        )
        prediction = model(
            batch["cam_main"].float(),
            batch["cam_wrist"].float(),
            batch["task_index"],
            proprioception,
            batch["progress"].float() if model.progress_dim else None,
        )
        normalized_target = normalizer.normalize(batch["action"])
        loss = nn.functional.smooth_l1_loss(prediction, normalized_target)
        count = stop - start
        loss_total += float(loss) * count
        absolute_error += torch.sum(
            torch.abs(normalizer.denormalize(prediction) - batch["action"]), dim=0
        )
    return loss_total / examples, (absolute_error / examples).cpu().tolist()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    cache_manifest_path = args.cache_root / "manifest.json"
    if not cache_manifest_path.is_file():
        raise FileNotFoundError(cache_manifest_path)
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if args.episodes_per_task is None:
        expected_split = {
            key: int(config["split"][key])
            for key in (
                "train_episodes_per_task",
                "validation_episodes_per_task",
                "test_episodes_per_task",
                "unused_episodes_per_task",
            )
        }
        manifest_split = cache_manifest.get("split")
        if manifest_split != expected_split:
            raise ValueError(
                "Feature-cache split does not match the experiment config: "
                f"{manifest_split} != {expected_split}. Use a clean cache root."
            )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    print(f"Training device: {device}")
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
        config["split"]["unused_episodes_per_task"] = (
            int(config["dataset"]["demonstrations_per_task"])
            - args.episodes_per_task
            - int(config["split"]["validation_episodes_per_task"])
            - int(config["split"]["test_episodes_per_task"])
        )
        if config["split"]["unused_episodes_per_task"] < 0:
            raise ValueError(
                "Requested train/validation/test episodes exceed the dataset"
            )

    policy_config = config["policy"]
    action_representation = policy_config.get("action_representation", "absolute")
    uses_proprioception = policy_config.get("proprioception") is True
    if action_representation == "future_delta":
        if not uses_proprioception:
            raise ValueError("future_delta training requires proprioception")
        lookahead_frames = int(policy_config["lookahead_frames"])
        train = load_cached_future_delta_split(
            args.cache_root,
            "train",
            lookahead_frames,
            episode_ids_by_task=selected_episode_ids,
        )
        validation = load_cached_future_delta_split(
            args.cache_root, "validation", lookahead_frames
        )
        test = load_cached_future_delta_split(
            args.cache_root, "test", lookahead_frames
        )
    elif action_representation == "absolute":
        train = load_cached_split(
            args.cache_root,
            "train",
            episode_ids_by_task=selected_episode_ids,
        )
        validation = load_cached_split(args.cache_root, "validation")
        test = load_cached_split(args.cache_root, "test")
    else:
        raise ValueError(
            f"Unsupported action_representation: {action_representation}"
        )
    feature_dim = int(train["cam_main"].shape[1])
    if train["cam_wrist"].shape[1] != feature_dim:
        raise ValueError("Camera feature dimensions differ")

    model = TCCMLPPolicy(
        feature_dim=feature_dim,
        num_tasks=int(policy_config["number_of_tasks"]),
        action_dim=int(policy_config["action_dim"]),
        hidden_dims=tuple(policy_config["hidden_dimensions"]),
        proprio_dim=(
            int(policy_config.get("proprioception_dim", 7))
            if uses_proprioception
            else 0
        ),
        progress_dim=(
            1
            if policy_config.get("progress_conditioning")
            == "normalized_episode_time"
            else 0
        ),
        input_batch_norm=bool(policy_config["input_batch_norm"]),
        input_layer_norm=bool(policy_config["input_layer_norm"]),
    ).to(device)
    action_mean = train["action"].mean(dim=0)
    action_std = train["action"].std(dim=0)
    normalizer = ActionNormalizer(action_mean, action_std).to(device)
    state_mean: torch.Tensor | None = None
    state_std: torch.Tensor | None = None
    state_normalizer: ActionNormalizer | None = None
    if uses_proprioception:
        state_mean = train["state"].mean(dim=0)
        state_std = train["state"].std(dim=0)
        state_normalizer = ActionNormalizer(state_mean, state_std).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(policy_config["learning_rate"])
    )
    if policy_config.get("loss") != "smooth_l1":
        raise ValueError("The MLP policy trainer requires smooth_l1 loss")
    loss_function = nn.SmoothL1Loss()
    steps = args.steps or int(policy_config["training_steps"])
    log_every = args.log_every or int(policy_config["log_every"])
    batch_size = int(policy_config["batch_size"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_validation_loss = float("inf")
    best_step = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    for step in range(1, steps + 1):
        model.train()
        batch = sample_batch(train, batch_size, device)
        proprioception = (
            state_normalizer.normalize(batch["state"].float())
            if state_normalizer is not None
            else None
        )
        prediction = model(
            batch["cam_main"].float(),
            batch["cam_wrist"].float(),
            batch["task_index"],
            proprioception,
            batch["progress"].float() if model.progress_dim else None,
        )
        loss = loss_function(prediction, normalizer.normalize(batch["action"]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            validation_loss, validation_mae = evaluate_policy(
                model,
                normalizer,
                state_normalizer,
                validation,
                batch_size * 4,
                device,
            )
            metric_name = (
                "validation_delta_mae"
                if action_representation == "future_delta"
                else "validation_action_mae"
            )
            row = {
                "step": step,
                "train_normalized_smooth_l1": float(loss.detach()),
                "validation_normalized_smooth_l1": validation_loss,
                metric_name: validation_mae,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                best_model_state = copy.deepcopy(model.state_dict())

    def checkpoint_payload(
        model_state: dict[str, torch.Tensor], checkpoint_step: int
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model_state,
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
            "step": checkpoint_step,
            "best_validation_step": best_step,
            "best_validation_normalized_smooth_l1": best_validation_loss,
            "history": history,
            "actuation_enabled": False,
        }
        if state_mean is not None and state_std is not None:
            payload["state_mean"] = state_mean
            payload["state_std"] = state_std
        return payload

    if best_model_state is None:
        raise RuntimeError("No best validation checkpoint was selected")
    final_model_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_state)
    test_loss, test_mae = evaluate_policy(
        model, normalizer, state_normalizer, test, batch_size * 4, device
    )
    model.load_state_dict(final_model_state)

    torch.save(
        checkpoint_payload(final_model_state, steps),
        output_dir / "checkpoint_last.pt",
    )
    torch.save(
        checkpoint_payload(best_model_state, best_step),
        output_dir / "checkpoint_best.pt",
    )
    # Keep the Hub-compatible filename while deploying validation-selected weights.
    torch.save(
        checkpoint_payload(best_model_state, best_step),
        output_dir / f"checkpoint_{steps:06d}.pt",
    )
    test_metric_name = (
        "test_delta_mae_at_best"
        if action_representation == "future_delta"
        else "test_action_mae_at_best"
    )
    metrics = {
        "history": history,
        "best_validation_step": best_step,
        "best_validation_normalized_smooth_l1": best_validation_loss,
        "test_normalized_smooth_l1_at_best": test_loss,
        test_metric_name: test_mae,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
