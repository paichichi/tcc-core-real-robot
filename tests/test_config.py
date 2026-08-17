from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml

ROOT = Path(__file__).parents[1]


def test_initial_revision_disables_actuation() -> None:
    config = load_yaml(ROOT / "configs" / "robot.yaml")
    assert_actuation_disabled(config)
    assert config["safety"]["dry_run_by_default"] is True
    assert config["safety"]["require_emergency_stop_ready"] is True


def test_dataset_scope_is_four_tasks() -> None:
    config = load_yaml(ROOT / "configs" / "experiment.yaml")
    assert config["dataset"]["demonstrations_per_task"] == 100
    assert len(config["dataset"]["tasks"]) == 4
