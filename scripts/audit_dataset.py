#!/usr/bin/env python3
"""Read-only audit of the configured Hugging Face dataset repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from tcc_real_robot.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Inspect repository metadata without downloading dataset files.",
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    dataset = config["dataset"]
    repo_id = dataset["repository"]
    revision = dataset["revision"] if dataset["revision"] != "TBD" else None
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)

    print(f"repository: {repo_id}")
    print(f"revision: {info.sha}")
    print(f"private: {info.private}")
    print(f"configured tasks: {len(dataset['tasks'])}")
    print(f"repository files: {len(info.siblings or [])}")
    if not args.metadata_only:
        print("No download was performed; dataset transfer is intentionally separate.")


if __name__ == "__main__":
    main()
