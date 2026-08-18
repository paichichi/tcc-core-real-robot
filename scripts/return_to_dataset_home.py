#!/usr/bin/env python3
"""Slowly move the real arm to the configured dataset collection home."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml
from tcc_real_robot.policy_home import PolicyHomeSession


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slowly return the real arm to dataset_collection_home."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/robot.yaml"))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that this moves the real arm.",
    )
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to move the arm without --execute")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    config = load_yaml(args.config)
    assert_actuation_disabled(config)
    robot = config["robot"]
    goal_time = float(config["policy_evaluation"]["home_goal_time_s"])
    print(
        f"Moving slowly to {robot['home_name']} over {goal_time:.1f} seconds."
    )
    print("Keep the workspace clear and the physical E-stop ready.")

    try:
        import trossen_arm
    except ImportError as exc:
        raise SystemExit(
            "Trossen driver is not installed. Run: pip install -e '.[robot]'"
        ) from exc

    captured_at = datetime.now(timezone.utc)
    with PolicyHomeSession(trossen_arm, config, args.timeout) as session:
        result = session.prepare()

    report = "\n".join(
        [
            "TCC-Core Dataset Home Return",
            "============================",
            "Overall: PASS",
            f"Captured at: {captured_at.isoformat()}",
            f"Controller IP: {robot['controller_ip']}",
            f"Driver version: {result.driver_version}",
            f"Firmware version: {result.firmware_version}",
            f"Home: {robot['home_name']}",
            f"Goal time: {goal_time:.6f} s",
            f"Target: {list(result.target)}",
            f"Observed: {list(result.observed)}",
            f"Maximum arm tracking error: {result.max_arm_error_rad:.9f} rad",
            f"Gripper tracking error: {result.gripper_error_m:.9f} m",
            "Idle restored: True",
            "",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"robot_dataset_home_return_{stamp}.txt"
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
