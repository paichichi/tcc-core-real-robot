#!/usr/bin/env python3
"""Create a read-only Trossen controller safety baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml
from tcc_real_robot.robot_inspection import preflight_robot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Trossen controller preflight."
    )
    parser.add_argument("--config", default="configs/robot.yaml")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    config = load_yaml(Path(args.config))
    assert_actuation_disabled(config)

    try:
        import trossen_arm
    except ImportError as exc:
        raise SystemExit(
            "Trossen driver is not installed. Run: pip install -e '.[robot]'"
        ) from exc

    report = preflight_robot(trossen_arm, config, args.timeout)
    report["captured_at"] = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"robot_preflight_{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"report: {output_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
