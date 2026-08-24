import json
import sqlite3

import numpy as np
import pytest
import torch
from torchvision.io import encode_jpeg

from tcc_real_robot.trossen_image_data import (
    JOINT_ACTION_SEMANTICS,
    JOINT_STATE_SEMANTICS,
    TrossenSingleViewChunkDataset,
    episode_split_indices,
    require_complete_trossen_buffer,
)


def make_buffer(path) -> None:
    manifest = {
        "dataset_revision": "revision",
        "tasks": ["task"],
        "cameras": ["cam_main", "cam_wrist"],
        "state_semantics": JOINT_STATE_SEMANTICS,
        "action_semantics": JOINT_ACTION_SEMANTICS,
        "action_source": "original_lerobot_action_column",
        "driver_command": "set_all_positions",
        "action_leads_measured_state_frames": 2,
        "jpeg_quality": 95,
        "episodes": 2,
        "episodes_per_task": 2,
        "frames_per_episode": 1,
        "samples": 2,
    }
    vector = np.zeros(7, dtype=np.float32).tobytes()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE samples (id INTEGER PRIMARY KEY, split TEXT, "
            "task_index INTEGER, episode_index INTEGER, frame_index INTEGER, "
            "jpeg_main BLOB, jpeg_wrist BLOB, state BLOB, action BLOB)"
        )
        connection.execute(
            "INSERT INTO metadata VALUES ('manifest', ?)",
            (json.dumps(manifest),),
        )
        connection.executemany(
            "INSERT INTO samples VALUES (?, 'train', 0, ?, 0, ?, ?, ?, ?)",
            [(1, 0, b"jpeg", b"jpeg", vector, vector), (2, 1, b"jpeg", b"jpeg", vector, vector)],
        )


def validate(path, lead=2):
    return require_complete_trossen_buffer(
        path,
        dataset_revision="revision",
        tasks=["task"],
        episodes_per_task=2,
        frames_per_episode=1,
        action_leads_measured_state_frames=lead,
    )


def test_buffer_contract_requires_exact_action_lead(tmp_path) -> None:
    path = tmp_path / "buffer.sqlite3"
    make_buffer(path)

    assert validate(path)["action_leads_measured_state_frames"] == 2
    with pytest.raises(ValueError, match="contract mismatch"):
        validate(path, lead=1)


def test_buffer_contract_rejects_non_finite_vectors(tmp_path) -> None:
    path = tmp_path / "buffer.sqlite3"
    make_buffer(path)
    bad = np.full(7, np.nan, dtype=np.float32).tobytes()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE samples SET action = ? WHERE id = 1", (bad,))

    with pytest.raises(ValueError, match="non-finite"):
        validate(path)


def make_chunk_buffer(path) -> None:
    jpeg = bytes(encode_jpeg(torch.zeros((3, 8, 8), dtype=torch.uint8)).numpy())
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE samples (id INTEGER PRIMARY KEY, split TEXT, "
            "task_index INTEGER, episode_index INTEGER, frame_index INTEGER, "
            "jpeg_main BLOB, jpeg_wrist BLOB, state BLOB, action BLOB)"
        )
        rows = []
        row_id = 1
        for episode in range(2):
            for frame in range(3):
                state = np.full(7, episode + frame, dtype=np.float32).tobytes()
                action = np.full(7, 10 * episode + frame, dtype=np.float32).tobytes()
                rows.append(
                    (
                        row_id,
                        episode,
                        frame,
                        jpeg,
                        jpeg,
                        state,
                        action,
                    )
                )
                row_id += 1
        connection.executemany(
            "INSERT INTO samples VALUES (?, 'train', 0, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def test_single_view_chunk_dataset_repeats_final_absolute_target(tmp_path) -> None:
    path = tmp_path / "chunks.sqlite3"
    make_chunk_buffer(path)
    dataset = TrossenSingleViewChunkDataset(
        path,
        lambda image: image,
        action_chunk_size=4,
    )

    image, state, action, task = dataset[1]

    assert image.shape == (3, 8, 8)
    assert torch.equal(state, torch.ones(7))
    assert task.item() == 0
    assert torch.equal(
        action.reshape(4, 7)[:, 0],
        torch.tensor([1.0, 2.0, 2.0, 2.0]),
    )


def test_episode_split_accepts_act_train_validation_without_test(tmp_path) -> None:
    path = tmp_path / "split.sqlite3"
    make_chunk_buffer(path)

    train, validation, test = episode_split_indices(
        path,
        train_episodes=1,
        validation_episodes=1,
        test_episodes=0,
        seed=7,
    )

    assert len(train) == 3
    assert len(validation) == 3
    assert test == []
    assert set(train).isdisjoint(validation)
