#!/usr/bin/env python3
"""Continuously read Trossen joint state without commanding motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tcc_real_robot.config import assert_actuation_disabled, load_yaml
from tcc_real_robot.robot_inspection import monitor_robot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Trossen arm state monitor."
    )
    parser.add_argument("--config", default="configs/robot.yaml")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=20.0)
    args = parser.parse_args()
    for name in ("timeout", "duration", "rate"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")

    config = load_yaml(Path(args.config))
    assert_actuation_disabled(config)

    try:
        import trossen_arm
    except ImportError as exc:
        raise SystemExit(
            "Trossen driver is not installed. Run: pip install -e '.[robot]'"
        ) from exc

    def print_sample(sample: dict[str, Any]) -> None:
        print(json.dumps(sample), flush=True)

    summary = monitor_robot(
        trossen_arm,
        config,
        timeout=args.timeout,
        duration=args.duration,
        rate_hz=args.rate,
        on_sample=print_sample,
    )
    print("summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
