"""Trossen-native synchronized image and joint-action training data."""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg

JOINT_STATE_SEMANTICS = "trossen_joint_position_6_plus_gripper_position"
JOINT_ACTION_SEMANTICS = "trossen_joint_position_goal_6_plus_gripper_position"


def require_complete_trossen_buffer(
    database: str | Path,
    *,
    dataset_revision: str,
    tasks: Sequence[str],
    episodes_per_task: int,
    frames_per_episode: int,
) -> dict[str, object]:
    """Fail closed unless the dual-camera buffer is complete and joint-native."""
    path = Path(database).expanduser().resolve()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'manifest'"
        ).fetchone()
        if row is None:
            raise ValueError(f"Image buffer has no manifest: {path}")
        manifest = json.loads(str(row[0]))
        expected_episodes = len(tasks) * episodes_per_task
        expected_samples = expected_episodes * frames_per_episode
        expected = {
            "dataset_revision": dataset_revision,
            "tasks": list(tasks),
            "cameras": ["cam_main", "cam_wrist"],
            "state_semantics": JOINT_STATE_SEMANTICS,
            "action_semantics": JOINT_ACTION_SEMANTICS,
            "action_source": "original_lerobot_action_column",
            "driver_command": "set_all_positions",
            "episodes": expected_episodes,
            "episodes_per_task": episodes_per_task,
            "frames_per_episode": frames_per_episode,
            "samples": expected_samples,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Trossen image-buffer contract mismatch: {mismatches}")
        summary = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT task_index), COUNT(DISTINCT split) "
            "FROM samples"
        ).fetchone()
        if summary != (expected_samples, len(tasks), 1):
            raise ValueError(f"Incomplete Trossen image buffer: {summary}")
        malformed = connection.execute(
            "SELECT COUNT(*) FROM samples WHERE length(jpeg_main) = 0 "
            "OR length(jpeg_wrist) = 0 OR length(state) != ? OR length(action) != ?",
            (7 * np.dtype(np.float32).itemsize, 7 * np.dtype(np.float32).itemsize),
        ).fetchone()[0]
        if int(malformed):
            raise ValueError(f"Image buffer has {malformed} malformed rows")
        episodes = connection.execute(
            "SELECT task_index, episode_index, COUNT(*), "
            "COUNT(DISTINCT frame_index), MIN(frame_index), MAX(frame_index) "
            "FROM samples GROUP BY task_index, episode_index"
        ).fetchall()
        if len(episodes) != expected_episodes or any(
            tuple(int(value) for value in row[2:])
            != (frames_per_episode, frames_per_episode, 0, frames_per_episode - 1)
            for row in episodes
        ):
            raise ValueError("Image-buffer episodes are incomplete or non-contiguous")
    return manifest


def episode_split_indices(
    database: str | Path,
    *,
    train_episodes: int,
    validation_episodes: int,
    test_episodes: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return leak-free positional indices split by complete episodes."""
    sizes = (train_episodes, validation_episodes, test_episodes)
    if any(size <= 0 for size in sizes):
        raise ValueError("Every episode split must be positive")
    path = Path(database).expanduser().resolve()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = [
            (int(task), int(episode))
            for task, episode in connection.execute(
                "SELECT task_index, episode_index FROM samples "
                "WHERE split = 'train' ORDER BY id"
            )
        ]
    episodes_by_task: dict[int, list[int]] = {}
    for task, episode in rows:
        if episode not in episodes_by_task.setdefault(task, []):
            episodes_by_task[task].append(episode)
    memberships = [set(), set(), set()]
    for task, episodes in sorted(episodes_by_task.items()):
        if sum(sizes) != len(episodes):
            raise ValueError(f"Split {sizes} does not exhaust task {task}")
        random.Random(seed + task).shuffle(episodes)
        offset = 0
        for membership, size in zip(memberships, sizes):
            membership.update((task, episode) for episode in episodes[offset : offset + size])
            offset += size
    result = [[], [], []]
    for index, pair in enumerate(rows):
        matches = [slot for slot, membership in enumerate(memberships) if pair in membership]
        if len(matches) != 1:
            raise RuntimeError(f"Invalid episode split membership: {pair}")
        result[matches[0]].append(index)
    return result[0], result[1], result[2]


class TrossenMultiViewDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Synchronized RGB camera observations and original 7-D joint goals."""

    def __init__(
        self,
        database: str | Path,
        transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        include_state: bool = False,
    ) -> None:
        self.database = Path(database).expanduser().resolve()
        self.transform = transform
        self.include_state = include_state
        self._connection: sqlite3.Connection | None = None
        with self._connect() as connection:
            self.row_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM samples WHERE split = 'train' ORDER BY id"
                )
            ]
        if not self.row_ids:
            raise ValueError(f"No samples in {self.database}")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.database}?mode=ro", uri=True)

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._connect()
        return self._connection

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_connection"] = None
        return state

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self._get_connection().execute(
            "SELECT jpeg_main, jpeg_wrist, state, action FROM samples WHERE id = ?",
            (self.row_ids[index],),
        ).fetchone()
        if row is None:
            raise IndexError(index)
        images = []
        for jpeg in row[:2]:
            encoded = torch.from_numpy(np.frombuffer(jpeg, dtype=np.uint8).copy())
            images.append(self.transform(decode_jpeg(encoded, mode="RGB")))
        state = torch.from_numpy(np.frombuffer(row[2], dtype=np.float32).copy())
        action = torch.from_numpy(np.frombuffer(row[3], dtype=np.float32).copy())
        if state.shape != (7,) or not torch.isfinite(state).all():
            raise ValueError("Trossen state must be finite and seven-dimensional")
        if action.shape != (7,) or not torch.isfinite(action).all():
            raise ValueError("Trossen action must be finite and seven-dimensional")
        if self.include_state:
            return images[0], images[1], state, action
        return images[0], images[1], action

    def state_statistics(
        self, indices: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._vector_statistics(indices, "state")

    def action_statistics(
        self, indices: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._vector_statistics(indices, "action")

    def _vector_statistics(
        self, indices: Sequence[int], column: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if column not in {"state", "action"}:
            raise ValueError(f"Unsupported statistics column: {column}")
        if not indices:
            raise ValueError(f"{column.title()} statistics require training samples")
        selected_ids = {self.row_ids[index] for index in indices}
        vectors = []
        with self._connect() as connection:
            for row_id, vector_bytes in connection.execute(
                f"SELECT id, {column} FROM samples WHERE split = 'train'"
            ):
                if int(row_id) in selected_ids:
                    vectors.append(
                        np.frombuffer(vector_bytes, dtype=np.float32).copy()
                    )
        tensor = torch.from_numpy(np.stack(vectors))
        return tensor.mean(0), tensor.std(0, unbiased=False)
