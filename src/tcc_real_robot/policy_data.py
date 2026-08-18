"""Episode splitting and feature-cache helpers for the policy baseline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class EpisodeRecord:
    task_index: int
    task_name: str
    episode_index: int
    task_root: Path
    split: str

    @property
    def parquet_path(self) -> Path:
        return self.task_root / "data" / "chunk-000" / (
            f"episode_{self.episode_index:06d}.parquet"
        )

    def video_path(self, camera: str) -> Path:
        return self.task_root / "videos" / "chunk-000" / (
            f"observation.images.{camera}"
        ) / f"episode_{self.episode_index:06d}.mp4"


def build_episode_records(
    dataset_root: str | Path,
    task_names: list[str],
    seed: int,
    split_sizes: tuple[int, int, int],
) -> list[EpisodeRecord]:
    """Make deterministic, episode-level train/validation/test splits."""
    root = Path(dataset_root)
    split_names = ("train", "validation", "test")
    records: list[EpisodeRecord] = []
    for task_index, task_name in enumerate(task_names):
        task_root = root / task_name
        episode_file = task_root / "meta" / "episodes.jsonl"
        rows = [json.loads(line) for line in episode_file.read_text().splitlines()]
        episode_ids = [int(row["episode_index"]) for row in rows]
        if sum(split_sizes) > len(episode_ids):
            raise ValueError(
                f"Split sizes {split_sizes} exceed {task_name}: "
                f"{len(episode_ids)} episodes"
            )
        random.Random(seed + task_index).shuffle(episode_ids)
        offset = 0
        for split_name, size in zip(split_names, split_sizes):
            for episode_index in episode_ids[offset : offset + size]:
                records.append(
                    EpisodeRecord(
                        task_index=task_index,
                        task_name=task_name,
                        episode_index=episode_index,
                        task_root=task_root,
                        split=split_name,
                    )
                )
            offset += size
    return records


def cache_shard_path(cache_root: str | Path, record: EpisodeRecord) -> Path:
    return Path(cache_root) / record.split / f"task_{record.task_index}" / (
        f"episode_{record.episode_index:06d}.pt"
    )


def load_cached_split(
    cache_root: str | Path,
    split: str,
    episode_ids_by_task: dict[int, set[int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Load and concatenate cached episode tensors for one split."""
    paths = sorted((Path(cache_root) / split).glob("task_*/episode_*.pt"))
    if episode_ids_by_task is not None:
        paths = [
            path
            for path in paths
            if int(path.parent.name.removeprefix("task_"))
            in episode_ids_by_task
            and int(path.stem.removeprefix("episode_"))
            in episode_ids_by_task[
                int(path.parent.name.removeprefix("task_"))
            ]
        ]
    if not paths:
        raise FileNotFoundError(f"No cached {split} episodes under {cache_root}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {"cam_main", "cam_wrist", "action", "state", "task_index"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f"Malformed feature-cache shard: {path}")
        rows.append(payload)
    return {
        "cam_main": torch.cat([row["cam_main"] for row in rows]),
        "cam_wrist": torch.cat([row["cam_wrist"] for row in rows]),
        "action": torch.cat([row["action"] for row in rows]).float(),
        "state": torch.cat([row["state"] for row in rows]).float(),
        "task_index": torch.cat([row["task_index"] for row in rows]).long(),
    }
