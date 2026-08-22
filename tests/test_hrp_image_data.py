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
    HRPMultiViewImageDataset,
    episode_level_transition_split,
    official_hrp_transition_split,
    require_complete_joint_position_buffer,
    require_joint_position_manifest,
    start_weighted_transition_weights,
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


def test_hrp_multiview_dataset_decodes_synchronized_cameras(tmp_path: Path) -> None:
    database = tmp_path / "dual.sqlite3"
    main = torch.zeros((3, 16, 16), dtype=torch.uint8)
    wrist = torch.full((3, 16, 16), 255, dtype=torch.uint8)
    main_jpeg = torchvision.io.encode_jpeg(main).numpy().tobytes()
    wrist_jpeg = torchvision.io.encode_jpeg(wrist).numpy().tobytes()
    vector = np.arange(7, dtype=np.float32)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE samples (id INTEGER PRIMARY KEY, split TEXT, "
            "jpeg_main BLOB, jpeg_wrist BLOB, state BLOB, action BLOB, "
            "task_index INTEGER)"
        )
        connection.execute(
            "INSERT INTO samples(split, jpeg_main, jpeg_wrist, state, action, "
            "task_index) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "train",
                main_jpeg,
                wrist_jpeg,
                vector.tobytes(),
                vector.tobytes(),
                0,
            ),
        )

    dataset = HRPMultiViewImageDataset(database, "train")
    decoded_main, decoded_wrist, state, action, task = dataset[0]

    assert decoded_main.shape == decoded_wrist.shape == (3, 16, 16)
    assert float(decoded_main.float().mean()) < 1.0
    assert float(decoded_wrist.float().mean()) > 254.0
    assert torch.equal(state, torch.from_numpy(vector))
    assert torch.equal(action, torch.from_numpy(vector))
    assert int(task) == 0


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

    _, _, delta_mean, delta_std = dataset.state_action_statistics(
        [0, 1], action_representation="current_delta"
    )
    assert torch.allclose(delta_mean, torch.zeros(7))
    assert torch.allclose(delta_std, torch.zeros(7))


def test_episode_level_transition_split_has_no_frame_leakage(tmp_path: Path) -> None:
    database = tmp_path / "episodes.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE samples (id INTEGER PRIMARY KEY, split TEXT, "
            "task_index INTEGER, episode_index INTEGER, frame_index INTEGER)"
        )
        connection.executemany(
            "INSERT INTO samples(split, task_index, episode_index, frame_index) "
            "VALUES ('train', ?, ?, ?)",
            [
                (task, episode, frame)
                for task in range(2)
                for episode in range(5)
                for frame in range(3)
            ],
        )

    train, validation, test = episode_level_transition_split(
        database,
        train_episodes_per_task=3,
        validation_episodes_per_task=1,
        test_episodes_per_task=1,
        shuffle_seed=7,
    )

    assert [len(part) for part in (train, validation, test)] == [18, 6, 6]
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert set(train) | set(validation) | set(test) == set(range(30))


def test_start_weighted_transition_weights_use_frame_index(tmp_path: Path) -> None:
    database = tmp_path / "weights.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE samples (id INTEGER PRIMARY KEY, split TEXT, "
            "frame_index INTEGER)"
        )
        connection.executemany(
            "INSERT INTO samples(split, frame_index) VALUES ('train', ?)",
            [(0,), (1,), (2,), (0,), (1,), (2,)],
        )

    weights = start_weighted_transition_weights(
        database, [0, 2, 3, 5], start_frames=1, start_weight=10.0
    )

    assert weights.tolist() == [10.0, 1.0, 10.0, 1.0]


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
