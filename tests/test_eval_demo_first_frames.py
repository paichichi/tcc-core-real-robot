from pathlib import Path

import pytest

from tcc_real_robot.policy_data import EpisodeRecord


def load_evaluator():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "eval_demo_first_frames.py"
    spec = importlib.util.spec_from_file_location("eval_demo_first_frames_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_record(episode: int, split: str) -> EpisodeRecord:
    return EpisodeRecord(0, "task", episode, Path("/dataset/task"), split)


def test_first_frame_gate_defaults_can_select_held_out_test_episodes() -> None:
    module = load_evaluator()
    records = [
        make_record(0, "train"),
        make_record(1, "train"),
        make_record(2, "test"),
        make_record(3, "test"),
    ]

    selected = module.select_records(
        records, task_index=0, split="test", episodes=2
    )

    assert [record.episode_index for record in selected] == [2, 3]
    assert all(record.split == "test" for record in selected)


def test_first_frame_gate_fails_if_requested_split_is_too_small() -> None:
    module = load_evaluator()
    records = [make_record(0, "train"), make_record(1, "test")]

    with pytest.raises(RuntimeError, match="Only 1 test episodes"):
        module.select_records(records, task_index=0, split="test", episodes=2)


def test_first_frame_gate_can_select_validation_episodes() -> None:
    module = load_evaluator()
    records = [
        make_record(0, "train"),
        make_record(1, "validation"),
        make_record(2, "validation"),
    ]

    selected = module.select_records(
        records, task_index=0, split="validation", episodes=2
    )

    assert [record.episode_index for record in selected] == [1, 2]
