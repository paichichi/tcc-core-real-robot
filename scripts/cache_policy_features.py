#!/usr/bin/env python3
"""Cache frozen TCC features for both cameras. This script never actuates a robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from tcc_real_robot.model_assets import resolve_backbone_asset
from tcc_real_robot.policy_data import (
    build_episode_records,
    cache_shard_path,
)
from tcc_real_robot.policy_runtime import preprocess_rgb_frames
from tcc_real_robot.tcc_backbone import load_frozen_tcc_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hub-backbone")
    parser.add_argument("--hub-cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tcc-source-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("runs/feature_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--limit-episodes-per-split", type=int)
    parser.add_argument("--train-episodes-per-task", type=int)
    return parser.parse_args()


def resolve_path(value: object, override: Path | None, name: str) -> Path:
    chosen = override if override is not None else value
    if chosen in (None, "TBD"):
        raise ValueError(f"Set {name} in the config or command line")
    return Path(str(chosen)).expanduser().resolve()


@torch.inference_mode()
def encode_camera(
    backbone: torch.nn.Module,
    video_path: Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    outputs = []
    frame_batch: list[np.ndarray] = []

    def flush() -> None:
        if not frame_batch:
            return
        batch = preprocess_rgb_frames(frame_batch, image_size).to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs.append(backbone(batch).float().cpu())
        frame_batch.clear()

    frame_count = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame_batch.append(frame.to_ndarray(format="rgb24"))
            frame_count += 1
            if len(frame_batch) == batch_size:
                flush()
    flush()
    return torch.cat(outputs).half(), frame_count


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    dataset_root = resolve_path(
        config["dataset"].get("local_root"), args.dataset_root, "dataset.local_root"
    )
    if args.checkpoint is not None and args.hub_backbone is not None:
        raise ValueError("Use either --checkpoint or --hub-backbone, not both")
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
    elif config["backbone"].get("source") == "huggingface" or args.hub_backbone:
        hub_backbone = args.hub_backbone or str(config["backbone"]["hub_name"])
        checkpoint, _ = resolve_backbone_asset(
            config,
            hub_backbone,
            cache_dir=args.hub_cache_dir,
            local_files_only=args.offline,
        )
    else:
        checkpoint = resolve_path(
            config["backbone"].get("checkpoint"), None, "backbone.checkpoint"
        )
    source_root = resolve_path(
        config["backbone"].get("tcc_source_root"),
        args.tcc_source_root,
        "backbone.tcc_source_root",
    )
    cache_root = args.cache_root.expanduser().resolve()
    device = torch.device(args.device)
    backbone, metadata = load_frozen_tcc_backbone(checkpoint, source_root, device)

    split = config["split"]
    if args.train_episodes_per_task is not None:
        if not 1 <= args.train_episodes_per_task <= int(
            config["dataset"]["demonstrations_per_task"]
        ):
            raise ValueError("--train-episodes-per-task is outside the dataset")
        split["train_episodes_per_task"] = args.train_episodes_per_task
        split["unused_episodes_per_task"] = (
            int(config["dataset"]["demonstrations_per_task"])
            - args.train_episodes_per_task
            - int(split["validation_episodes_per_task"])
            - int(split["test_episodes_per_task"])
        )
        if split["unused_episodes_per_task"] < 0:
            raise ValueError(
                "Requested train/validation/test episodes exceed the dataset"
            )
    records = build_episode_records(
        dataset_root=dataset_root,
        task_names=list(config["dataset"]["tasks"]),
        seed=int(config["seed"]),
        split_sizes=(
            int(split["train_episodes_per_task"]),
            int(split["validation_episodes_per_task"]),
            int(split["test_episodes_per_task"]),
        ),
    )
    if args.limit_episodes is not None:
        records = records[: args.limit_episodes]
    if args.limit_episodes_per_split is not None:
        records = [
            record
            for split_name in ("train", "validation", "test")
            for record in [row for row in records if row.split == split_name][
                : args.limit_episodes_per_split
            ]
        ]

    for number, record in enumerate(records, start=1):
        output = cache_shard_path(cache_root, record)
        if output.exists() and not args.overwrite:
            print(f"[{number}/{len(records)}] cached {output}")
            continue
        table = pq.read_table(
            record.parquet_path,
            columns=["action", "observation.state"],
        )
        action = torch.from_numpy(np.asarray(table["action"].to_pylist())).float()
        state = torch.from_numpy(
            np.asarray(table["observation.state"].to_pylist())
        ).float()
        main_features, main_count = encode_camera(
            backbone,
            record.video_path("cam_main"),
            metadata["image_size"],
            args.batch_size,
            device,
        )
        wrist_features, wrist_count = encode_camera(
            backbone,
            record.video_path("cam_wrist"),
            metadata["image_size"],
            args.batch_size,
            device,
        )
        lengths = {main_count, wrist_count, action.shape[0], state.shape[0]}
        if len(lengths) != 1:
            raise RuntimeError(f"Unsynchronized episode {record}: lengths={lengths}")
        payload = {
            "cam_main": main_features,
            "cam_wrist": wrist_features,
            "action": action,
            "state": state,
            "task_index": torch.full((action.shape[0],), record.task_index),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        print(f"[{number}/{len(records)}] wrote {output}")

    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        **metadata,
        "dataset_revision": config["dataset"]["revision"],
        "seed": config["seed"],
        "episodes_cached": len(records),
        "episode_records": [
            {
                "task_index": record.task_index,
                "task_name": record.task_name,
                "episode_index": record.episode_index,
                "split": record.split,
            }
            for record in records
        ],
        "split": dict(split),
        "policy_actuation": False,
    }
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
