#!/usr/bin/env python3
"""Replay one audited dataset demonstration through the official arm driver."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tcc_real_robot.config import load_yaml
from tcc_real_robot.demo_replay import audit_demo_trajectory
from tcc_real_robot.policy_home import PolicyHomeSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robot.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--task", default="carrot")
    parser.add_argument("--episode", type=int, default=33)
    parser.add_argument("--controller-timeout", type=float, default=20.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--emergency-stop-ready", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def resolve_task(task: str, configured: list[str]) -> str:
    aliases = {
        "carrot": "pick_and_place_carrot_100",
        "pineapple": "pick_and_place_pineapple_100",
        "starfruit": "pick_and_place_starfruit_100",
        "strawberry": "pick_and_place_strawberry_100",
    }
    resolved = aliases.get(task.lower(), task)
    if resolved not in configured:
        raise ValueError(f"Unknown task {task!r}; expected one of {configured}")
    return resolved


def main() -> None:
    args = parse_args()
    if args.episode < 0:
        raise SystemExit("--episode must be non-negative")
    if args.controller_timeout <= 0:
        raise SystemExit("--controller-timeout must be positive")
    if args.execute and not args.emergency_stop_ready:
        raise SystemExit("--execute requires --emergency-stop-ready")

    import pyarrow.parquet as pq

    config = load_yaml(args.config)
    robot_config = load_yaml(args.robot_config)
    task_name = resolve_task(
        args.task, [str(value) for value in config["dataset"]["tasks"]]
    )
    dataset_root = Path(
        args.dataset_root or config["dataset"]["local_root"]
    ).expanduser()
    parquet_path = (
        dataset_root
        / task_name
        / "data"
        / "chunk-000"
        / f"episode_{args.episode:06d}.parquet"
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    actions = np.asarray(
        pq.read_table(parquet_path, columns=["action"])["action"].to_pylist(),
        dtype=np.float64,
    )
    fps = float(config["observations"]["fps"])
    replay = robot_config["demo_replay"]
    task_limits = robot_config["policy_evaluation"]["clipped_rollout"][
        "dataset_action_limits"
    ].get(task_name)
    if task_limits is None:
        raise ValueError(f"No dataset action envelope configured for {task_name}")
    audit = audit_demo_trajectory(
        actions,
        fps=fps,
        absolute_min=[float(value) for value in task_limits["min"]],
        absolute_max=[float(value) for value in task_limits["max"]],
        max_arm_velocity_rad_s=float(
            robot_config["safety"]["max_joint_velocity_rad_s"]
        ),
        max_gripper_velocity_m_s=float(
            robot_config["safety"]["max_gripper_velocity_m_s"]
        ),
    )
    expected_frames = int(config["evaluation"]["max_rollout_steps"])
    if audit.frames != expected_frames:
        raise RuntimeError(
            f"Demo contains {audit.frames} frames; expected {expected_frames}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output_dir / f"demo_replay_{task_name}_{args.episode:06d}_{stamp}.txt"
    )
    completed = 0
    observed_rates: list[float] = []
    final_error: list[float] | None = None
    failure = ""
    with output_path.open("w", encoding="utf-8", buffering=1) as report:
        report.write("TCC Dataset Demonstration Replay\n")
        report.write("================================\n")
        report.write(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}\n")
        report.write(f"Task: {task_name}\n")
        report.write(f"Episode: {args.episode}\n")
        report.write(f"Dataset file: {parquet_path}\n")
        report.write(f"Frames: {audit.frames}\n")
        report.write(f"Dataset rate: {fps:.3f} Hz\n")
        report.write(f"Maximum step: {list(audit.max_step)}\n")
        report.write(f"Maximum implied velocity: {list(audit.max_velocity)}\n")
        report.write("Trajectory audit: PASS\n\n")
        if args.execute:
            try:
                import trossen_arm

                session = PolicyHomeSession(
                    trossen_arm, robot_config, args.controller_timeout
                )
                try:
                    preparation = session.prepare()
                    report.write(f"Driver: {preparation.driver_version}\n")
                    report.write(f"Firmware: {preparation.firmware_version}\n")
                    if session.driver is None:
                        raise RuntimeError("Driver unavailable after home preparation")
                    limits = session.driver.get_joint_limits()
                    if len(limits) != 7:
                        raise RuntimeError("Expected seven controller joint limits")
                    targets = actions.copy()
                    for index, limit in enumerate(limits):
                        targets[:, index] = np.clip(
                            targets[:, index],
                            float(limit.position_min),
                            float(limit.position_max),
                        )
                    first = targets[0].tolist()
                    session.driver.set_arm_positions(
                        first[:6], float(replay["start_goal_time_s"]), True
                    )
                    session.driver.set_gripper_position(
                        first[6], float(replay["start_gripper_goal_time_s"]), True
                    )
                    start_observed = session.read_positions()
                    start_error = [
                        abs(actual - target)
                        for actual, target in zip(start_observed, first, strict=True)
                    ]
                    if max(start_error[:6]) > float(replay["max_tracking_error"][0]):
                        raise RuntimeError("Failed to align with demo start pose")
                    if start_error[6] > float(replay["max_tracking_error"][6]):
                        raise RuntimeError(
                            "Failed to align gripper with demo start pose"
                        )
                    report.write(f"Start target: {first}\n")
                    report.write(f"Start observed: {start_observed}\n\n")

                    period = 1.0 / fps
                    goal_time = float(replay["command_goal_time_s"])
                    started = time.monotonic()
                    previous_sample: tuple[float, list[float]] | None = None
                    for step, target_array in enumerate(targets):
                        delay = started + step * period - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                        sampled_at = time.monotonic()
                        observed = session.read_positions()
                        if previous_sample is not None:
                            previous_at, previous_observed = previous_sample
                            sample_period = sampled_at - previous_at
                            velocity = [
                                abs(current - previous) / sample_period
                                for current, previous in zip(
                                    observed, previous_observed, strict=True
                                )
                            ]
                            if max(velocity[:6]) > float(
                                robot_config["safety"]["max_joint_velocity_rad_s"]
                            ) or velocity[6] > float(
                                robot_config["safety"]["max_gripper_velocity_m_s"]
                            ):
                                raise RuntimeError(
                                    "Observed replay velocity exceeds limit"
                                )
                            observed_rates.append(1.0 / sample_period)
                        target = target_array.tolist()
                        session.driver.set_all_positions(target, goal_time, False)
                        error = str(session.driver.get_error_information())
                        if error.lower() != "no error":
                            raise RuntimeError(f"Controller reports an error: {error}")
                        previous_sample = (sampled_at, observed)
                        completed += 1
                        report.write(f"step={step:03d} target={target}\n")

                    time.sleep(goal_time)
                    median_rate = float(np.median(observed_rates))
                    if median_rate < float(
                        robot_config["policy_evaluation"]["minimum_observed_rate_hz"]
                    ):
                        raise RuntimeError(
                            f"Observed replay rate {median_rate:.3f} Hz is too low"
                        )
                    final_observed = session.read_positions()
                    final_error = [
                        abs(actual - target)
                        for actual, target in zip(
                            final_observed, targets[-1], strict=True
                        )
                    ]
                    tracking_limits = [
                        float(value) for value in replay["max_tracking_error"]
                    ]
                    if any(
                        error > limit
                        for error, limit in zip(
                            final_error, tracking_limits, strict=True
                        )
                    ):
                        raise RuntimeError("Final replay target tracking failed")
                    report.write(f"Final observed: {final_observed}\n")
                finally:
                    session.close()
            except Exception as exc:  # noqa: BLE001 - preserve hardware report
                failure = f"{type(exc).__name__}: {exc}"

        report.write("\nSummary\n")
        report.write(f"Completed steps: {completed}/{audit.frames}\n")
        if observed_rates:
            report.write(
                f"Observed command rate median: {float(np.median(observed_rates)):.3f} Hz\n"
            )
        if final_error is not None:
            report.write(f"Final tracking error: {final_error}\n")
        if failure:
            report.write(f"Failure: {failure}\n")
        decision = (
            "DRY_RUN_PASS"
            if not args.execute
            else "PASS"
            if completed == audit.frames and not failure
            else "FAIL"
        )
        report.write(f"Decision: {decision}\n")

    print(output_path.read_text(encoding="utf-8"), end="")
    print(f"report: {output_path}")
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
