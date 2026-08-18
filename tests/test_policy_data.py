from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from tcc_real_robot.policy_data import build_episode_records, load_cached_split


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
