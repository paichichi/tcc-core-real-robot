from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from tcc_real_robot.policy_data import (
    build_episode_records,
    load_cached_current_delta_split,
    load_cached_future_delta_split,
    load_cached_split,
)


def test_episode_split_has_no_frame_level_leakage(tmp_path: Path) -> None:
    tasks = ["task_a", "task_b"]
    for task in tasks:
        metadata = tmp_path / task / "meta"
        metadata.mkdir(parents=True)
        rows = [f'{{"episode_index": {index}}}' for index in range(10)]
        (metadata / "episodes.jsonl").write_text("\n".join(rows) + "\n")

    records = build_episode_records(tmp_path, tasks, seed=1, split_sizes=(6, 2, 2))
    for task_index in range(2):
        task_records = [row for row in records if row.task_index == task_index]
        by_split = {
            split: {row.episode_index for row in task_records if row.split == split}
            for split in ("train", "validation", "test")
        }
        assert len(by_split["train"]) == 6
        assert by_split["train"].isdisjoint(by_split["validation"])
        assert by_split["train"].isdisjoint(by_split["test"])
        assert by_split["validation"].isdisjoint(by_split["test"])


def test_load_cached_split_filters_episode_ids_per_task(tmp_path: Path) -> None:
    for task_index in range(2):
        for episode_index in range(3):
            path = (
                tmp_path
                / "train"
                / f"task_{task_index}"
                / f"episode_{episode_index:06d}.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            value = torch.tensor([[10 * task_index + episode_index]])
            torch.save(
                {
                    "cam_main": value,
                    "cam_wrist": value,
                    "action": value.float(),
                    "state": value.float(),
                    "task_index": torch.tensor([task_index]),
                },
                path,
            )

    loaded = load_cached_split(
        tmp_path,
        "train",
        episode_ids_by_task={0: {1}, 1: {0, 2}},
    )

    assert loaded["action"].flatten().tolist() == [1.0, 10.0, 12.0]
    assert loaded["progress"].flatten().tolist() == [0.0, 0.0, 0.0]


def test_future_delta_labels_are_shifted_within_each_episode(tmp_path: Path) -> None:
    path = tmp_path / "train" / "task_0" / "episode_000000.pt"
    path.parent.mkdir(parents=True)
    state = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    action = state + 10.0
    torch.save(
        {
            "cam_main": torch.arange(8).reshape(4, 2),
            "cam_wrist": torch.arange(8).reshape(4, 2),
            "action": action,
            "state": state,
            "task_index": torch.zeros(4, dtype=torch.long),
        },
        path,
    )

    loaded = load_cached_future_delta_split(tmp_path, "train", 2)

    assert loaded["state"].shape == (2, 5)
    assert torch.equal(loaded["state"], state[:2])
    assert torch.equal(loaded["action"], action[2:] - state[:2])
    assert loaded["cam_main"].shape == (2, 2)
    assert loaded["progress"].flatten().tolist() == pytest.approx([0.0, 1.0 / 3.0])


def test_cached_progress_resets_at_each_episode_boundary(tmp_path: Path) -> None:
    for episode_index in range(2):
        path = tmp_path / "train" / "task_0" / f"episode_{episode_index:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = torch.zeros((3, 2))
        torch.save(
            {
                "cam_main": values,
                "cam_wrist": values,
                "action": torch.zeros((3, 7)),
                "state": torch.zeros((3, 7)),
                "task_index": torch.zeros(3, dtype=torch.long),
            },
            path,
        )

    loaded = load_cached_split(tmp_path, "train")

    assert loaded["progress"].flatten().tolist() == [0.0, 0.5, 1.0] * 2


def test_current_delta_labels_use_matching_state_and_action(tmp_path: Path) -> None:
    path = tmp_path / "train" / "task_0" / "episode_000000.pt"
    path.parent.mkdir(parents=True)
    state = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    action = state + torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.01])
    torch.save(
        {
            "cam_main": torch.zeros((3, 2)),
            "cam_wrist": torch.zeros((3, 2)),
            "action": action,
            "state": state,
            "task_index": torch.zeros(3, dtype=torch.long),
        },
        path,
    )

    loaded = load_cached_current_delta_split(tmp_path, "train")

    assert torch.allclose(loaded["action"], action - state)
    assert torch.equal(loaded["state"], state)
