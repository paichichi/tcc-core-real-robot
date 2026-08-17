#!/usr/bin/env python3
"""Return from dataset collection home to the recorded folded pose."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml
from tcc_real_robot.reporting import format_folded_pose_return_report
from tcc_real_robot.robot_inspection import run_folded_pose_return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slowly return the real arm from dataset home to its folded pose."
    )
    parser.add_argument("--config", default="configs/robot.yaml")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that this moves the real arm.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.execute:
        raise SystemExit("Refusing to move the arm without --execute")

    config = load_yaml(Path(args.config))
    assert_actuation_disabled(config)
    robot = config["robot"]
    goal_time = config["diagnostic_tests"]["folded_pose_return"]["goal_time_s"]
    print(
        f"Returning from {robot['home_name']} to {robot['folded_pose_name']} "
        f"over {goal_time:.1f} seconds."
    )
    print("The gripper stays idle. Keep the workspace clear and E-stop ready.")

    try:
        import trossen_arm
    except ImportError as exc:
        raise SystemExit(
            "Trossen driver is not installed. Run: pip install -e '.[robot]'"
        ) from exc

    report = run_folded_pose_return(trossen_arm, config, args.timeout)
    report["captured_at"] = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"robot_folded_pose_return_{stamp}.txt"
    report_text = format_folded_pose_return_report(report)
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"report: {output_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
