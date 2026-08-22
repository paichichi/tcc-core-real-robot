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
    *,
    shuffle: bool = True,
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
        if shuffle:
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


def _episode_progress(frames: int) -> torch.Tensor:
    """Return normalized [0, 1] frame progress for one complete episode."""
    if frames <= 0:
        raise ValueError("Cached episode must contain at least one frame")
    if frames == 1:
        return torch.zeros((1, 1), dtype=torch.float32)
    return torch.linspace(0.0, 1.0, frames).unsqueeze(1)


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
        lengths = {int(payload[key].shape[0]) for key in required}
        if len(lengths) != 1:
            raise ValueError(f"Cached episode tensors have different lengths: {path}")
        payload["progress"] = _episode_progress(lengths.pop())
        rows.append(payload)
    return {
        "cam_main": torch.cat([row["cam_main"] for row in rows]),
        "cam_wrist": torch.cat([row["cam_wrist"] for row in rows]),
        "action": torch.cat([row["action"] for row in rows]).float(),
        "state": torch.cat([row["state"] for row in rows]).float(),
        "task_index": torch.cat([row["task_index"] for row in rows]).long(),
        "progress": torch.cat([row["progress"] for row in rows]).float(),
    }


def load_cached_future_delta_split(
    cache_root: str | Path,
    split: str,
    lookahead_frames: int,
    episode_ids_by_task: dict[int, set[int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Load episode-local observations with a future-action delta target.

    Each label is ``action[t + lookahead] - state[t]``. Slicing happens before
    concatenation so a training example can never cross an episode boundary.
    """
    if lookahead_frames <= 0:
        raise ValueError("lookahead_frames must be positive")
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

    rows: list[dict[str, torch.Tensor]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {"cam_main", "cam_wrist", "action", "state", "task_index"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f"Malformed feature-cache shard: {path}")
        lengths = {int(payload[key].shape[0]) for key in required}
        if len(lengths) != 1:
            raise ValueError(f"Cached episode tensors have different lengths: {path}")
        frames = lengths.pop()
        if frames <= lookahead_frames:
            raise ValueError(
                f"Cached episode {path} is shorter than its lookahead"
            )
        current = slice(None, -lookahead_frames)
        future = slice(lookahead_frames, None)
        rows.append(
            {
                "cam_main": payload["cam_main"][current],
                "cam_wrist": payload["cam_wrist"][current],
                "state": payload["state"][current].float(),
                "action": (
                    payload["action"][future].float()
                    - payload["state"][current].float()
                ),
                "task_index": payload["task_index"][current].long(),
                "progress": _episode_progress(frames)[current],
            }
        )
    return {
        key: torch.cat([row[key] for row in rows])
        for key in (
            "cam_main",
            "cam_wrist",
            "state",
            "action",
            "task_index",
            "progress",
        )
    }


def load_cached_current_delta_split(
    cache_root: str | Path,
    split: str,
    episode_ids_by_task: dict[int, set[int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Load current observations with one-step joint-command delta labels.

    The dataset stores absolute joint/gripper commands. For a delta-action baseline,
    the closest measured target available here is
    ``action[t] - state[t]``. Conversion happens independently inside every
    episode and never crosses a trajectory boundary.
    """
    data = load_cached_split(
        cache_root,
        split,
        episode_ids_by_task=episode_ids_by_task,
    )
    data["action"] = data["action"] - data["state"]
    return data
