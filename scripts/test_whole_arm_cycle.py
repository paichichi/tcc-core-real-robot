#!/usr/bin/env python3
"""Run a bounded, explicitly gated whole-arm outbound-and-return test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml
from tcc_real_robot.reporting import format_whole_arm_cycle_report
from tcc_real_robot.robot_inspection import run_whole_arm_cycle_test

CONFIRMATION = "I CONFIRM WHOLE ARM WORKSPACE CLEAR"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move all six arm joints by 0.03 rad, then return them."
    )
    parser.add_argument("--config", default="configs/robot.yaml")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that this moves the whole real arm.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.execute:
        raise SystemExit("Refusing to move the arm without --execute")

    print("This test moves all six arm joints together by 0.03 rad, then returns.")
    print("The gripper stays idle. Clear the full workspace and ready the E-stop.")
    typed = input(f"Type exactly '{CONFIRMATION}' to continue: ")
    if typed != CONFIRMATION:
        raise SystemExit("Confirmation did not match; no connection was attempted")

    config = load_yaml(Path(args.config))
    assert_actuation_disabled(config)
    try:
        import trossen_arm
    except ImportError as exc:
        raise SystemExit(
            "Trossen driver is not installed. Run: pip install -e '.[robot]'"
        ) from exc

    report = run_whole_arm_cycle_test(trossen_arm, config, args.timeout)
    report["captured_at"] = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"robot_whole_arm_cycle_{stamp}.txt"
    report_text = format_whole_arm_cycle_report(report)
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"report: {output_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
