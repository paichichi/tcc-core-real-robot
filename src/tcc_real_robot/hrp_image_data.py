"""SQLite-backed image observations for HRP-style end-to-end training."""

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


def read_image_buffer_manifest(database: str | Path) -> dict[str, object]:
    """Read the semantic contract embedded by ``cache_hrp_images.py``."""
    path = Path(database).expanduser().resolve()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'manifest'"
        ).fetchone()
    if row is None:
        raise ValueError(f"Image buffer has no semantic manifest: {path}")
    manifest = json.loads(str(row[0]))
    if not isinstance(manifest, dict):
        raise TypeError(f"Invalid image buffer manifest: {path}")
    return manifest


def validate_joint_position_manifest(
    manifest: dict[str, object],
) -> dict[str, object]:
    """Validate a decoded image-buffer semantic contract."""
    expected = {
        "state_semantics": JOINT_STATE_SEMANTICS,
        "action_semantics": JOINT_ACTION_SEMANTICS,
        "action_source": "original_lerobot_action_column",
        "driver_command": "set_all_positions",
        "action_leads_measured_state_frames": 2,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Image buffer is not the required joint-position contract: {mismatches}"
        )
    return manifest


def require_joint_position_manifest(database: str | Path) -> dict[str, object]:
    """Fail closed if a buffer is not the original Trossen joint-action data."""
    return validate_joint_position_manifest(read_image_buffer_manifest(database))


def require_complete_joint_position_buffer(
    database: str | Path,
    *,
    dataset_revision: str,
    tasks: Sequence[str],
    episodes_per_task: int,
    frames_per_episode: int,
    camera: str = "cam_main",
) -> dict[str, object]:
    """Prove that a formal image buffer contains the complete fixed dataset."""
    if not tasks or episodes_per_task <= 0 or frames_per_episode <= 0:
        raise ValueError("Complete-buffer expectations must be positive and non-empty")
    path = Path(database).expanduser().resolve()
    manifest = require_joint_position_manifest(path)
    expected_episodes = len(tasks) * episodes_per_task
    expected_samples = expected_episodes * frames_per_episode
    expected_manifest = {
        "dataset_revision": dataset_revision,
        "tasks": list(tasks),
        "camera": camera,
        "episodes": expected_episodes,
        "episodes_per_task": episodes_per_task,
        "frames_per_episode": frames_per_episode,
        "samples": expected_samples,
    }
    manifest_mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if manifest_mismatches:
        raise ValueError(
            "Image buffer manifest is not the complete formal dataset: "
            f"{manifest_mismatches}"
        )

    expected_pairs = {
        (task_index, episode_index)
        for task_index in range(len(tasks))
        for episode_index in range(episodes_per_task)
    }
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        summary = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT task_index),
                   COUNT(DISTINCT split)
            FROM samples
            """
        ).fetchone()
        if summary != (expected_samples, len(tasks), 1):
            raise ValueError(
                "Image buffer row/task/split counts are incomplete: "
                f"observed={summary}, expected={(expected_samples, len(tasks), 1)}"
            )
        split_rows = connection.execute("SELECT DISTINCT split FROM samples").fetchall()
        if split_rows != [("train",)]:
            raise ValueError(
                f"Formal HRP buffer must contain only train rows: {split_rows}"
            )

        episode_rows = connection.execute(
            """
            SELECT task_index, episode_index, COUNT(*),
                   COUNT(DISTINCT frame_index), MIN(frame_index), MAX(frame_index)
            FROM samples
            GROUP BY task_index, episode_index
            """
        ).fetchall()
        observed_pairs = {(int(row[0]), int(row[1])) for row in episode_rows}
        if observed_pairs != expected_pairs:
            missing = sorted(expected_pairs - observed_pairs)
            unexpected = sorted(observed_pairs - expected_pairs)
            raise ValueError(
                "Image buffer episode indices are incomplete: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        malformed_episodes = [
            row
            for row in episode_rows
            if tuple(int(value) for value in row[2:])
            != (frames_per_episode, frames_per_episode, 0, frames_per_episode - 1)
        ]
        if malformed_episodes:
            raise ValueError(
                "Image buffer episodes are not contiguous full trajectories: "
                f"{malformed_episodes[:10]}"
            )

        malformed_blobs = connection.execute(
            """
            SELECT COUNT(*) FROM samples
            WHERE length(jpeg) = 0 OR length(state) != ? OR length(action) != ?
            """,
            (7 * np.dtype(np.float32).itemsize, 7 * np.dtype(np.float32).itemsize),
        ).fetchone()[0]
        if int(malformed_blobs) != 0:
            raise ValueError(
                f"Image buffer contains {malformed_blobs} malformed image/vector blobs"
            )
        for row_id, state_bytes, action_bytes in connection.execute(
            "SELECT id, state, action FROM samples"
        ):
            state = np.frombuffer(state_bytes, dtype=np.float32)
            action = np.frombuffer(action_bytes, dtype=np.float32)
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(
                    f"Image buffer sample {row_id} contains non-finite state/action"
                )
    return manifest


def official_hrp_transition_split(
    number_of_transitions: int,
    held_out_transitions: int = 500,
    shuffle_seed: int = 3904767649,
) -> tuple[list[int], list[int]]:
    """Reproduce HRP's fixed shuffled transition-level train/test split."""
    if number_of_transitions <= held_out_transitions or held_out_transitions <= 0:
        raise ValueError("HRP split requires more transitions than its holdout")
    indices = list(range(number_of_transitions))
    random.Random(shuffle_seed).shuffle(indices)
    return indices[held_out_transitions:], indices[:held_out_transitions]


def episode_level_transition_split(
    database: str | Path,
    *,
    train_episodes_per_task: int,
    validation_episodes_per_task: int,
    test_episodes_per_task: int,
    shuffle_seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split transition positions by whole episodes with no frame leakage."""
    sizes = (
        train_episodes_per_task,
        validation_episodes_per_task,
        test_episodes_per_task,
    )
    if any(size < 0 for size in sizes) or min(sizes) <= 0:
        raise ValueError("Episode split sizes must all be positive")
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
    episode_splits: list[set[tuple[int, int]]] = [set(), set(), set()]
    for task, episodes in sorted(episodes_by_task.items()):
        if sum(sizes) != len(episodes):
            raise ValueError(
                f"Episode split sizes {sizes} do not exhaust task {task}: "
                f"{len(episodes)} episodes"
            )
        shuffled = episodes.copy()
        random.Random(shuffle_seed + task).shuffle(shuffled)
        offset = 0
        for selected, size in zip(episode_splits, sizes):
            selected.update((task, episode) for episode in shuffled[offset : offset + size])
            offset += size
    transition_splits = [[], [], []]
    for position, pair in enumerate(rows):
        matches = [index for index, selected in enumerate(episode_splits) if pair in selected]
        if len(matches) != 1:
            raise RuntimeError(f"Transition episode has invalid split membership: {pair}")
        transition_splits[matches[0]].append(position)
    return tuple(transition_splits)  # type: ignore[return-value]


def start_weighted_transition_weights(
    database: str | Path,
    indices: Sequence[int],
    *,
    start_frames: int,
    start_weight: float,
) -> torch.Tensor:
    """Return subset-local sampling weights that emphasize episode starts."""
    if start_frames <= 0 or start_weight < 1.0:
        raise ValueError("Start sampling requires positive frames and weight >= 1")
    path = Path(database).expanduser().resolve()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        frame_indices = [
            int(row[0])
            for row in connection.execute(
                "SELECT frame_index FROM samples WHERE split = 'train' ORDER BY id"
            )
        ]
    if any(index < 0 or index >= len(frame_indices) for index in indices):
        raise IndexError("Sampling-weight index is outside the image buffer")
    return torch.tensor(
        [start_weight if frame_indices[index] < start_frames else 1.0 for index in indices],
        dtype=torch.double,
    )


class HRPImageDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Random-access JPEG observations without freezing visual features."""

    def __init__(
        self,
        database: str | Path,
        split: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.database = Path(database).expanduser().resolve()
        if not self.database.is_file():
            raise FileNotFoundError(self.database)
        self.split = split
        self.transform = transform
        self._connection: sqlite3.Connection | None = None
        with self._connect() as connection:
            self.row_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM samples WHERE split = ? ORDER BY id", (split,)
                )
            ]
        if not self.row_ids:
            raise ValueError(f"No {split} samples in {self.database}")

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
        row = (
            self._get_connection()
            .execute(
                "SELECT jpeg, state, action, task_index FROM samples WHERE id = ?",
                (self.row_ids[index],),
            )
            .fetchone()
        )
        if row is None:
            raise IndexError(index)
        jpeg, state_bytes, action_bytes, task_index = row
        encoded = torch.from_numpy(np.frombuffer(jpeg, dtype=np.uint8).copy())
        image = decode_jpeg(encoded, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        state = torch.from_numpy(np.frombuffer(state_bytes, dtype=np.float32).copy())
        action = torch.from_numpy(np.frombuffer(action_bytes, dtype=np.float32).copy())
        if state.shape != (7,) or action.shape != (7,):
            raise ValueError("HRP image buffer requires 7-D state and action")
        return image, state, action, torch.tensor(int(task_index))

    def state_action_statistics(
        self,
        indices: Sequence[int] | None = None,
        *,
        action_representation: str = "absolute",
    ) -> tuple[torch.Tensor, ...]:
        """Compute statistics from only the selected dataset rows.

        ``indices`` use this dataset's positional indexing, so callers can pass
        the training side of a transition-level split without leaking held-out
        transitions into normalization statistics.
        """
        selected_row_ids: set[int] | None = None
        if indices is not None:
            if len(indices) == 0:
                raise ValueError("Statistics require at least one sample")
            if any(index < 0 or index >= len(self.row_ids) for index in indices):
                raise IndexError("Statistics index is outside the dataset")
            selected_row_ids = {self.row_ids[index] for index in indices}
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, state, action FROM samples WHERE split = ?", (self.split,)
            )
            for row_id, state_bytes, action_bytes in rows:
                if selected_row_ids is not None and int(row_id) not in selected_row_ids:
                    continue
                state = np.frombuffer(state_bytes, dtype=np.float32).copy()
                action = np.frombuffer(action_bytes, dtype=np.float32).copy()
                states.append(state)
                if action_representation == "absolute":
                    actions.append(action)
                elif action_representation == "current_delta":
                    actions.append(action - state)
                else:
                    raise ValueError(
                        f"Unsupported action representation: {action_representation}"
                    )
        if not states:
            raise ValueError("Statistics selection contains no samples")
        state_tensor = torch.from_numpy(np.stack(states))
        action_tensor = torch.from_numpy(np.stack(actions))
        return (
            state_tensor.mean(0),
            state_tensor.std(0, unbiased=False),
            action_tensor.mean(0),
            action_tensor.std(0, unbiased=False),
        )
