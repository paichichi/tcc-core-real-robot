from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml

ROOT = Path(__file__).parents[1]


def test_initial_revision_disables_actuation() -> None:
    config = load_yaml(ROOT / "configs" / "robot.yaml")
    assert_actuation_disabled(config)
    assert config["safety"]["dry_run_by_default"] is True
    assert config["safety"]["require_emergency_stop_ready"] is True
    assert config["robot"]["home_gripper_position_m"] == 0.0
    assert config["policy_evaluation"]["inference_warmup_steps"] >= 1
    assert config["policy_evaluation"]["minimum_observed_rate_hz"] == 18.0
    assert config["policy_evaluation"]["camera_capture_fps"] == 30.0
    clipped = config["policy_evaluation"]["clipped_rollout"]
    assert clipped["max_steps"] == 10
    assert clipped["max_joint_delta_rad"] <= 0.02
    assert clipped["max_gripper_delta_m"] <= 0.001
    assert clipped["max_cumulative_joint_delta_rad"] <= 0.06
    assert clipped["max_cumulative_gripper_delta_m"] <= 0.003


def test_dataset_scope_is_four_tasks() -> None:
    config = load_yaml(ROOT / "configs" / "experiment.yaml")
    assert config["dataset"]["demonstrations_per_task"] == 100
    assert len(config["dataset"]["tasks"]) == 4


def test_first_policy_is_single_step_offline_bc() -> None:
    config = load_yaml(ROOT / "configs" / "experiment.yaml")
    policy = config["policy"]
    assert policy["implementation"] == "tcc_mlp_bc_v0"
    assert policy["hidden_dimensions"] == [256, 256]
    assert policy["action_chunk_size"] == 1
    assert policy["proprioception"] is False
    assert config["evaluation"]["max_rollout_steps"] == 359
    assert config["split"]["train_episodes_per_task"] == 60
    assert config["split"]["validation_episodes_per_task"] == 0
    assert config["split"]["test_episodes_per_task"] == 0
    assert "horizon" not in policy


def test_model_hub_is_pinned_to_an_immutable_revision() -> None:
    config = load_yaml(ROOT / "configs" / "experiment.yaml")
    hub = config["model_hub"]
    assert hub["repository"] == "Chipaipai/tcc-core-real-robot-policies"
    assert len(hub["revision"]) == 40
    assert all(character in "0123456789abcdef" for character in hub["revision"])
    assert config["backbone"]["source"] == "huggingface"
    assert config["backbone"]["hub_name"] == "ours_rn50"
