#!/usr/bin/env python3
"""Fetch one pinned backbone-policy pair from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcc_real_robot.config import load_yaml
from tcc_real_robot.model_assets import resolve_model_assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify a frozen backbone and matching policy."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--backbone", default="ours_rn50")
    parser.add_argument("--demonstrations", type=int, default=60)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use already cached files without accessing Hugging Face.",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    assets = resolve_model_assets(
        config,
        args.backbone,
        args.demonstrations,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    print(json.dumps(assets.to_json_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
