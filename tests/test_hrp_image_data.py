import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from tcc_real_robot.hrp_image_data import (
    JOINT_ACTION_SEMANTICS,
    JOINT_STATE_SEMANTICS,
    HRPImageDataset,
    official_hrp_transition_split,
    require_complete_joint_position_buffer,
    require_joint_position_manifest,
)


def test_official_hrp_transition_split_is_fixed_and_exhaustive() -> None:
    train, heldout = official_hrp_transition_split(1000)
    repeated_train, repeated_heldout = official_hrp_transition_split(1000)

    assert len(train) == 500
    assert len(heldout) == 500
    assert set(train).isdisjoint(heldout)
    assert set(train) | set(heldout) == set(range(1000))
    assert (train, heldout) == (repeated_train, repeated_heldout)


def test_official_hrp_transition_split_rejects_invalid_holdout() -> None:
    with pytest.raises(ValueError):
        official_hrp_transition_split(500)


def test_hrp_image_dataset_decodes_jpeg_and_joint_action(tmp_path: Path) -> None:
    database = tmp_path / "buffer.sqlite3"
    image = torch.zeros((3, 16, 16), dtype=torch.uint8)
    jpeg = torchvision.io.encode_jpeg(image).numpy().tobytes()
    state = np.arange(7, dtype=np.float32)
    action = np.full(7, 0.25, dtype=np.float32)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY,
                split TEXT,
                jpeg BLOB,
                state BLOB,
                action BLOB,
                task_index INTEGER
            )
            """
        )
        connection.execute(
            "INSERT INTO samples(split, jpeg, state, action, task_index) "
            "VALUES (?, ?, ?, ?, ?)",
            ("train", jpeg, state.tobytes(), action.tobytes(), 2),
        )

    dataset = HRPImageDataset(database, "train")
    decoded, decoded_state, joint_action, task = dataset[0]
    state_mean, _, action_mean, _ = dataset.state_action_statistics()

    assert decoded.shape == (3, 16, 16)
    assert torch.equal(decoded_state, torch.from_numpy(state))
    assert torch.allclose(joint_action, torch.full((7,), 0.25))
    assert int(task) == 2
    assert torch.allclose(state_mean, torch.from_numpy(state))
    assert torch.allclose(action_mean, torch.full((7,), 0.25))


def test_hrp_statistics_can_exclude_heldout_rows(tmp_path: Path) -> None:
    database = tmp_path / "buffer.sqlite3"
    image = torch.zeros((3, 16, 16), dtype=torch.uint8)
    jpeg = torchvision.io.encode_jpeg(image).numpy().tobytes()
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY,
                split TEXT,
                jpeg BLOB,
                state BLOB,
                action BLOB,
                task_index INTEGER
            )
            """
        )
        for value in (1.0, 3.0, 100.0):
            vector = np.full(7, value, dtype=np.float32)
            connection.execute(
                "INSERT INTO samples(split, jpeg, state, action, task_index) "
                "VALUES (?, ?, ?, ?, ?)",
                ("train", jpeg, vector.tobytes(), vector.tobytes(), 0),
            )

    dataset = HRPImageDataset(database, "train")
    state_mean, state_std, action_mean, action_std = dataset.state_action_statistics(
        [0, 1]
    )

    assert torch.allclose(state_mean, torch.full((7,), 2.0))
    assert torch.allclose(action_mean, torch.full((7,), 2.0))
    assert torch.allclose(state_std, torch.ones(7))
    assert torch.allclose(action_std, torch.ones(7))


def test_joint_position_manifest_rejects_old_action_semantics(tmp_path: Path) -> None:
    database = tmp_path / "buffer.sqlite3"
    manifest = {
        "state_semantics": JOINT_STATE_SEMANTICS,
        "action_semantics": JOINT_ACTION_SEMANTICS,
        "action_source": "original_lerobot_action_column",
        "driver_command": "set_all_positions",
        "action_leads_measured_state_frames": 2,
    }
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('manifest', ?)",
            (json.dumps(manifest),),
        )
    assert require_joint_position_manifest(database) == manifest

    manifest["action_semantics"] = "cartesian_velocity"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'manifest'",
            (json.dumps(manifest),),
        )
    with pytest.raises(ValueError, match="joint-position contract"):
        require_joint_position_manifest(database)


def _write_complete_joint_buffer(database: Path) -> None:
    manifest = {
        "dataset_revision": "fixed-revision",
        "tasks": ["carrot"],
        "camera": "cam_main",
        "state_semantics": JOINT_STATE_SEMANTICS,
        "action_semantics": JOINT_ACTION_SEMANTICS,
        "action_source": "original_lerobot_action_column",
        "driver_command": "set_all_positions",
        "action_leads_measured_state_frames": 2,
        "episodes": 2,
        "episodes_per_task": 2,
        "frames_per_episode": 3,
        "samples": 6,
    }
    vector = np.arange(7, dtype=np.float32).tobytes()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            """
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY,
                split TEXT NOT NULL,
                task_index INTEGER NOT NULL,
                episode_index INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                jpeg BLOB NOT NULL,
                state BLOB NOT NULL,
                action BLOB NOT NULL,
                UNIQUE(task_index, episode_index, frame_index)
            )
            """
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('manifest', ?)",
            (json.dumps(manifest),),
        )
        connection.executemany(
            """
            INSERT INTO samples(
                split, task_index, episode_index, frame_index, jpeg, state, action
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("train", 0, episode, frame, b"jpeg", vector, vector)
                for episode in range(2)
                for frame in range(3)
            ],
        )


def test_complete_joint_position_buffer_proves_exact_shape(tmp_path: Path) -> None:
    database = tmp_path / "buffer.sqlite3"
    _write_complete_joint_buffer(database)

    manifest = require_complete_joint_position_buffer(
        database,
        dataset_revision="fixed-revision",
        tasks=["carrot"],
        episodes_per_task=2,
        frames_per_episode=3,
    )

    assert manifest["samples"] == 6


def test_complete_joint_position_buffer_rejects_missing_frame(tmp_path: Path) -> None:
    database = tmp_path / "buffer.sqlite3"
    _write_complete_joint_buffer(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM samples WHERE episode_index = 1 AND frame_index = 2"
        )

    with pytest.raises(ValueError, match="counts are incomplete"):
        require_complete_joint_position_buffer(
            database,
            dataset_revision="fixed-revision",
            tasks=["carrot"],
            episodes_per_task=2,
            frames_per_episode=3,
        )


def test_complete_joint_position_buffer_rejects_nonfinite_action(
    tmp_path: Path,
) -> None:
    database = tmp_path / "buffer.sqlite3"
    _write_complete_joint_buffer(database)
    invalid = np.full(7, np.nan, dtype=np.float32).tobytes()
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE samples SET action = ? WHERE id = 1", (invalid,))

    with pytest.raises(ValueError, match="non-finite"):
        require_complete_joint_position_buffer(
            database,
            dataset_revision="fixed-revision",
            tasks=["carrot"],
            episodes_per_task=2,
            frames_per_episode=3,
        )
