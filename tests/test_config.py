from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml

ROOT = Path(__file__).parents[1]


def test_initial_revision_disables_actuation() -> None:
    config = load_yaml(ROOT / "configs" / "robot.yaml")
    assert_actuation_disabled(config)
    assert config["safety"]["dry_run_by_default"] is True
    assert config["safety"]["require_emergency_stop_ready"] is True
    assert config["robot"]["home_gripper_position_m"] == 0.0
    assert config["robot"]["motor_parameters"] == "wxai_v0_20250509"
    assert config["robot"]["expected_driver_version"] == "1.9.3"
    assert config["robot"]["expected_firmware_version"] == "1.9.2"
    assert config["cameras"]["cam_main"]["serial_number"] == "838212073584"
    assert config["cameras"]["cam_main"]["asic_serial_number"] == "843213020438"
    assert config["cameras"]["cam_main"]["usb_serial"] == "UNAVAILABLE"
    assert config["cameras"]["cam_wrist"]["serial_number"] == "409122274608"
    assert config["cameras"]["cam_wrist"]["asic_serial_number"] == "242623072067"
    assert config["cameras"]["stream"] == "color"
    assert config["cameras"]["format"] == "rgb8"
    assert config["policy_evaluation"]["inference_warmup_steps"] >= 1
    assert config["policy_evaluation"]["minimum_observed_rate_hz"] == 18.0
    assert config["policy_evaluation"]["camera_capture_fps"] == 30.0
    clipped = config["policy_evaluation"]["clipped_rollout"]
    assert clipped["max_steps"] == 359
    assert clipped["max_action_delta"] == [
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.001,
    ]
    assert clipped["max_tracking_error"] == [
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.001,
    ]
    assert clipped["control_fps"] == 20.0
    assert clipped["min_time_to_move_multiplier"] == 6.0
    assert clipped["command_blocking"] is False
    assert "max_cumulative_joint_delta_rad" not in clipped
    assert "max_cumulative_gripper_delta_m" not in clipped
    carrot_limits = clipped["dataset_action_limits"]["pick_and_place_carrot_100"]
    assert len(carrot_limits["min"]) == 7
    assert len(carrot_limits["max"]) == 7
    probe = config["workspace_probe"]
    assert probe["step_m"] <= 0.002
    assert probe["hard_travel_limits_m"]["z_negative"] <= 0.130


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
