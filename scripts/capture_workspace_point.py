#!/usr/bin/env python3
"""Capture one read-only Cartesian workspace calibration point."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

from tcc_real_robot.config import load_yaml
from tcc_real_robot.reporting import format_workspace_point_report
from tcc_real_robot.robot_inspection import preflight_robot

LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read and save one safe workspace boundary point."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", default="configs/robot.yaml")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs/workspace_points")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not LABEL_PATTERN.fullmatch(args.label):
        parser.error("--label must contain only lowercase letters, digits, _ or -")

    config = load_yaml(Path(args.config))
    try:
        import trossen_arm
    except ImportError as exc:
        raise SystemExit(
            "Trossen driver is not installed. Run: pip install -e '.[robot]'"
        ) from exc

    report = preflight_robot(trossen_arm, config, args.timeout)
    report["label"] = args.label
    report["captured_at"] = datetime.now(timezone.utc).isoformat()
    report_text = format_workspace_point_report(report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"workspace_point_{args.label}_{stamp}.txt"
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"report: {output_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
