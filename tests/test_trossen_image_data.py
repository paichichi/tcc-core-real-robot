import json
import sqlite3

import numpy as np
import pytest

from tcc_real_robot.trossen_image_data import (
    JOINT_ACTION_SEMANTICS,
    JOINT_STATE_SEMANTICS,
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
