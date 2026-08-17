#!/usr/bin/env python3
"""Connect to a Trossen follower and print state without commanding motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml
from tcc_real_robot.robot_inspection import inspect_robot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Trossen arm connection and state inspection."
    )
    parser.add_argument("--config", default="configs/robot.yaml")
    parser.add_argument("--timeout", type=float, default=20.0)
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

    state = inspect_robot(trossen_arm, config, args.timeout)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
