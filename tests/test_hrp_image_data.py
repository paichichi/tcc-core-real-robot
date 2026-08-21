import sqlite3
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from tcc_real_robot.hrp_image_data import HRPImageDataset


def test_hrp_image_dataset_decodes_jpeg_and_velocity_action(tmp_path: Path) -> None:
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
    decoded, decoded_state, velocity, task = dataset[0]
    state_mean, _, action_mean, _ = dataset.state_action_statistics()

    assert decoded.shape == (3, 16, 16)
    assert torch.equal(decoded_state, torch.from_numpy(state))
    assert torch.allclose(velocity, torch.full((7,), 0.25))
    assert int(task) == 2
    assert torch.allclose(state_mean, torch.from_numpy(state))
    assert torch.allclose(action_mean, torch.full((7,), 0.25))
