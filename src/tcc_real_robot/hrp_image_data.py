"""SQLite-backed image observations for HRP-style end-to-end training."""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg


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
        row = self._get_connection().execute(
            "SELECT jpeg, state, action, task_index FROM samples WHERE id = ?",
            (self.row_ids[index],),
        ).fetchone()
        if row is None:
            raise IndexError(index)
        jpeg, state_bytes, action_bytes, task_index = row
        encoded = torch.from_numpy(np.frombuffer(jpeg, dtype=np.uint8).copy())
        image = decode_jpeg(encoded, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        state = torch.from_numpy(
            np.frombuffer(state_bytes, dtype=np.float32).copy()
        )
        action = torch.from_numpy(
            np.frombuffer(action_bytes, dtype=np.float32).copy()
        )
        if state.shape != (7,) or action.shape != (7,):
            raise ValueError("HRP image buffer requires 7-D state and action")
        return image, state, action, torch.tensor(int(task_index))

    def state_action_statistics(self) -> tuple[torch.Tensor, ...]:
        """Compute training-only HRP state and velocity-action statistics."""
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT state, action FROM samples WHERE split = ?", (self.split,)
            )
            for state_bytes, action_bytes in rows:
                state = np.frombuffer(state_bytes, dtype=np.float32).copy()
                action = np.frombuffer(action_bytes, dtype=np.float32).copy()
                states.append(state)
                actions.append(action)
        state_tensor = torch.from_numpy(np.stack(states))
        action_tensor = torch.from_numpy(np.stack(actions))
        return (
            state_tensor.mean(0),
            state_tensor.std(0, unbiased=False),
            action_tensor.mean(0),
            action_tensor.std(0, unbiased=False),
        )
