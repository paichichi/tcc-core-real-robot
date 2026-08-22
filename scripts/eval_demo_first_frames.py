#!/usr/bin/env python3
"""Evaluate a trained policy on recorded demo first frames without a robot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from tcc_real_robot.config import load_yaml
from tcc_real_robot.model_assets import resolve_model_assets
from tcc_real_robot.offline_eval import compare_first_frame
from tcc_real_robot.policy_data import build_episode_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained policy on recorded demo first frames and compare "
            "predictions with recorded actions. This never imports the robot driver."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robot.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--backbone", default="ours_rn50")
    parser.add_argument("--demonstrations", type=int, default=80)
    parser.add_argument("--task", default="carrot")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--tcc-source-root", type=Path, required=True)
    parser.add_argument("--hub-cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gmm-inference",
        choices=("checkpoint", "highest-probability-mode"),
        default="highest-probability-mode",
        help="Use deterministic highest-probability HRP mode by default.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def resolve_task(task: str, task_names: list[str]) -> tuple[int, str]:
    aliases = {
        "carrot": "pick_and_place_carrot_100",
        "pineapple": "pick_and_place_pineapple_100",
        "starfruit": "pick_and_place_starfruit_100",
        "strawberry": "pick_and_place_strawberry_100",
    }
    name = aliases.get(task.lower(), task)
    if name not in task_names:
        raise ValueError(f"Unknown task {task!r}; expected one of {task_names}")
    return task_names.index(name), name


def first_rgb_frame(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"Video contains no frames: {path}")


def resolve_dataset_root(config: dict, override: Path | None) -> Path:
    value = override if override is not None else config["dataset"].get("local_root")
    if value in (None, "TBD"):
        raise ValueError("Set --dataset-root or dataset.local_root")
    root = Path(str(value)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    config = load_yaml(args.config)
    robot_config = load_yaml(args.robot_config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    task_names = [str(value) for value in config["dataset"]["tasks"]]
    task_index, task_name = resolve_task(args.task, task_names)
    split = config["split"]
    if all(
        key in split
        for key in (
            "train_episodes_per_task",
            "validation_episodes_per_task",
            "test_episodes_per_task",
        )
    ):
        split_sizes = (
            int(split["train_episodes_per_task"]),
            int(split["validation_episodes_per_task"]),
            int(split["test_episodes_per_task"]),
        )
        shuffle_episodes = True
    elif split.get("protocol") == "hrp_fixed_transition_holdout":
        split_sizes = (
            int(config["dataset"]["demonstrations_per_task"]),
            0,
            0,
        )
        shuffle_episodes = False
    else:
        raise ValueError(f"Unsupported episode selection protocol: {split}")
    records = build_episode_records(
        dataset_root,
        task_names,
        int(config["seed"]),
        split_sizes,
        shuffle=shuffle_episodes,
    )
    selected = [
        record
        for record in records
        if record.task_index == task_index and record.split == "train"
    ][: args.episodes]
    if len(selected) != args.episodes:
        raise RuntimeError(
            f"Only {len(selected)} training episodes available; "
            f"requested {args.episodes}"
        )

    from tcc_real_robot.policy_runtime import (
        load_policy_bundle,
        predict_action,
        resolve_device,
        restore_policy_backbone,
    )
    from tcc_real_robot.tcc_backbone import load_frozen_tcc_backbone

    device = resolve_device(args.device)
    assets = resolve_model_assets(
        config,
        args.backbone,
        args.demonstrations,
        cache_dir=args.hub_cache_dir,
        local_files_only=args.offline,
    )
    backbone, backbone_metadata = load_frozen_tcc_backbone(
        assets.backbone_path, args.tcc_source_root, device
    )
    bundle = load_policy_bundle(
        assets.policy_path,
        expected_feature_dim=int(backbone_metadata["feature_dim"]),
        device=device,
    )
    fine_tuned_backbone_restored = restore_policy_backbone(backbone, bundle)
    trained_tasks = [str(value) for value in bundle.config["dataset"]["tasks"]]
    if trained_tasks != task_names:
        raise RuntimeError("Checkpoint and current config use different task ordering")

    rows = []
    for record in selected:
        table = pq.read_table(
            record.parquet_path,
            columns=["action", "observation.state"],
        )
        target = np.asarray(table["action"][0].as_py(), dtype=np.float64)
        state = np.asarray(table["observation.state"][0].as_py(), dtype=np.float64)
        cam_main = first_rgb_frame(record.video_path("cam_main"))
        cam_wrist = (
            first_rgb_frame(record.video_path("cam_wrist"))
            if "cam_wrist" in bundle.model.camera_names
            else cam_main
        )
        prediction = (
            predict_action(
                backbone,
                bundle,
                cam_main,
                cam_wrist,
                task_index,
                int(backbone_metadata["image_size"]),
                device,
                observation_state=state,
                episode_progress=0.0,
                gmm_inference_override=(
                    None
                    if args.gmm_inference == "checkpoint"
                    else args.gmm_inference
                ),
            )
            .numpy()
            .astype(np.float64)
        )
        comparison = compare_first_frame(state, target, prediction)
        rows.append(
            {
                "episode": record.episode_index,
                "state": state,
                "target": target,
                "prediction": prediction,
                "comparison": comparison,
            }
        )

    max_joint_delta = float(robot_config["safety"]["max_joint_delta_rad"])
    max_gripper_delta = float(robot_config["safety"]["max_gripper_delta_m"])
    target_arm_safe = all(
        row["comparison"].target_max_arm_delta_rad <= max_joint_delta for row in rows
    )
    target_gripper_safe = all(
        row["comparison"].target_gripper_delta_m <= max_gripper_delta for row in rows
    )
    prediction_arm_safe = all(
        row["comparison"].prediction_max_arm_delta_rad <= max_joint_delta
        for row in rows
    )
    prediction_gripper_safe = all(
        row["comparison"].prediction_gripper_delta_m <= max_gripper_delta
        for row in rows
    )
    checks = {
        "recorded_first_actions_are_safe": target_arm_safe and target_gripper_safe,
        "predicted_first_arm_actions_are_safe": prediction_arm_safe,
        "predicted_first_gripper_actions_are_safe": prediction_gripper_safe,
    }
    decision = "PASS" if all(checks.values()) else "BLOCKED"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output_dir / f"demo_first_frames_{args.backbone}_{task_name}_{stamp}.txt"
    )
    with output_path.open("w", encoding="utf-8") as report:
        report.write("Demo First-Frame Offline Policy Evaluation\n")
        report.write("==========================================\n")
        report.write("Mode: OFFLINE (no robot driver imported)\n")
        report.write(f"Task: {task_name} (index {task_index})\n")
        report.write(f"Backbone: {args.backbone}\n")
        report.write(f"Demonstrations: {args.demonstrations}\n")
        report.write(f"Training episodes checked: {len(rows)}\n")
        report.write(f"Hub revision: {assets.revision}\n")
        report.write(f"Policy SHA256: {assets.policy_sha256}\n")
        report.write(
            "Fine-tuned backbone restored: "
            f"{'YES' if fine_tuned_backbone_restored else 'NO'}\n"
        )
        report.write(f"GMM inference override: {args.gmm_inference}\n")
        report.write(f"Device: {device}\n\n")
        for row in rows:
            comparison = row["comparison"]
            report.write(f"Episode {row['episode']:06d}\n")
            for name in ("state", "target", "prediction"):
                values = ", ".join(f"{value:.7f}" for value in row[name])
                report.write(f"  {name}: [{values}]\n")
            report.write(
                "  target_max_arm_delta_rad: "
                f"{comparison.target_max_arm_delta_rad:.7f}\n"
            )
            report.write(
                "  prediction_max_arm_delta_rad: "
                f"{comparison.prediction_max_arm_delta_rad:.7f}\n"
            )
            report.write(
                "  prediction_gripper_delta_m: "
                f"{comparison.prediction_gripper_delta_m:.7f}\n"
            )
            report.write(
                f"  prediction_action_mae: {comparison.prediction_action_mae:.7f}\n\n"
            )

        arm_deltas = [row["comparison"].prediction_max_arm_delta_rad for row in rows]
        gripper_deltas = [row["comparison"].prediction_gripper_delta_m for row in rows]
        action_maes = [row["comparison"].prediction_action_mae for row in rows]
        report.write("Summary\n")
        report.write(
            f"Prediction arm delta median/max: {np.median(arm_deltas):.7f} / "
            f"{max(arm_deltas):.7f} rad\n"
        )
        report.write(
            f"Prediction gripper delta median/max: "
            f"{np.median(gripper_deltas):.7f} / {max(gripper_deltas):.7f} m\n"
        )
        report.write(
            f"Prediction action MAE median/max: {np.median(action_maes):.7f} / "
            f"{max(action_maes):.7f}\n"
        )
        report.write("Checks:\n")
        for name, passed in checks.items():
            report.write(f"- {name}: {'PASS' if passed else 'FAIL'}\n")
        report.write(f"Decision: {decision}\n")

    print(output_path.read_text(encoding="utf-8"), end="")
    print(f"report: {output_path}")
    if decision != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
