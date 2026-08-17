#!/usr/bin/env python3
"""Run one bounded Cartesian translation step and return to the origin."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml
from tcc_real_robot.reporting import format_cartesian_step_report
from tcc_real_robot.robot_inspection import run_cartesian_step_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move one Cartesian axis by a small amount, then return."
    )
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--config", default="configs/robot.yaml")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs/workspace_steps")
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
    home_name = config["robot"]["home_name"]
    print(f"This test first moves slowly to {home_name}.")
    print(
        f"It then moves Cartesian {args.axis} by {args.distance:+.3f} m "
        "and returns to the home Cartesian origin."
    )
    print("Clear the full arm workspace and keep the emergency stop ready.")

    try:
        import trossen_arm
    except ImportError as exc:
        raise SystemExit(
            "Trossen driver is not installed. Run: pip install -e '.[robot]'"
        ) from exc

    report = run_cartesian_step_test(
        trossen_arm,
        config,
        args.timeout,
        axis=args.axis,
        distance_m=args.distance,
    )
    report["captured_at"] = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    direction = "positive" if args.distance > 0 else "negative"
    output_path = output_dir / (
        f"cartesian_step_{args.axis}_{direction}_{stamp}.txt"
    )
    report_text = format_cartesian_step_report(report)
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"report: {output_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
