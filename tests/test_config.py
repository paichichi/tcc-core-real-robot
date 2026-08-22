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
    hrp = config["hrp_driver_contract"]
    assert hrp["execution"] == "bounded_joint_position_rollout"
    assert hrp["action_representation"] == "absolute"
    assert hrp["action_space"] == "joint_position"
    assert hrp["driver_call"] == "set_all_positions"
    assert hrp["ik_required"] is False
    assert hrp["recommended_firmware"] == "1.9.3"
    assert hrp["learned_policy_release"] == (
        "BLOCKED_PENDING_SHADOW_AND_SUPERVISED_TEST"
    )
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
    assert config["policy_evaluation"]["action_ema_alpha"] == 1.0
    assert config["policy_evaluation"]["force_first_action_home"] is True
    clipped = config["policy_evaluation"]["clipped_rollout"]
    assert clipped["max_steps"] == 900
    assert clipped["max_action_delta"] == [
        0.04615854099392891,
        0.060654640197753906,
        0.08506894111633301,
        0.13237200677394867,
        0.0644693672657013,
        0.08545053005218506,
        0.004429406486451626,
    ]
    assert clipped["max_command_lead"] == [
        0.09231708198785782,
        0.12130928039550781,
        0.17013788223266602,
        0.26474401354789734,
        0.1289387345314026,
        0.17090106010437012,
        0.008858812972903252,
    ]
    assert clipped["max_command_lead"] == [
        value * clipped["min_time_to_move_multiplier"]
        for value in clipped["max_action_delta"]
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
    assert clipped["min_time_to_move_multiplier"] == 2.0
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


def test_future_delta_policy_configuration() -> None:
    config = load_yaml(ROOT / "configs" / "experiment.yaml")
    policy = config["policy"]
    assert policy["implementation"] == "tcc_mlp_bc_v1_future_delta"
    assert policy["hidden_dimensions"] == [256, 256]
    assert policy["action_chunk_size"] == 1
    assert policy["proprioception"] is True
    assert policy["proprioception_dim"] == 7
    assert policy["action_representation"] == "future_delta"
    assert policy["lookahead_frames"] == 10
    assert policy["execution_delta_gain"] == 0.6
    assert policy["input_batch_norm"] is False
    assert policy["input_layer_norm"] is True
    assert policy["loss"] == "smooth_l1"
    assert config["evaluation"]["max_rollout_steps"] == 359
    assert config["split"]["train_episodes_per_task"] == 80
    assert config["split"]["validation_episodes_per_task"] == 10
    assert config["split"]["test_episodes_per_task"] == 10
    assert "horizon" not in policy


def test_visual_absolute_60_policy_configuration() -> None:
    config = load_yaml(ROOT / "configs" / "experiment_visual_absolute_60.yaml")
    policy = config["policy"]
    assert policy["implementation"] == "tcc_mlp_bc_v2_visual_absolute"
    assert policy["proprioception"] is False
    assert policy["proprioception_dim"] == 0
    assert policy["action_representation"] == "absolute"
    assert "lookahead_frames" not in policy
    assert "execution_delta_gain" not in policy
    assert config["backbone"]["hub_name"] == "ours_vit"
    assert set(config["model_hub"]["supported_backbones"]) == {
        "ours_vit",
        "ours_rn50",
    }
    assert config["model_hub"]["supported_demonstrations"] == [60]
    assert config["split"] == {
        "train_episodes_per_task": 60,
        "validation_episodes_per_task": 20,
        "test_episodes_per_task": 20,
        "unused_episodes_per_task": 0,
    }
    assert policy["input_batch_norm"] is False
    assert policy["input_layer_norm"] is True
    assert policy["loss"] == "smooth_l1"
    assert config["evaluation"]["max_rollout_steps"] == 359
    assert "horizon" not in policy


def test_proprio_absolute_60_policy_configuration() -> None:
    config = load_yaml(ROOT / "configs" / "experiment_proprio_absolute_60.yaml")
    policy = config["policy"]
    assert policy["implementation"] == "tcc_mlp_bc_v3_proprio_absolute"
    assert policy["proprioception"] is True
    assert policy["proprioception_dim"] == 7
    assert policy["action_representation"] == "absolute"
    assert policy["hidden_dimensions"] == [256, 256]
    assert policy["input_batch_norm"] is False
    assert policy["input_layer_norm"] is True
    assert "lookahead_frames" not in policy
    assert "execution_delta_gain" not in policy
    assert config["split"] == {
        "train_episodes_per_task": 60,
        "validation_episodes_per_task": 20,
        "test_episodes_per_task": 20,
        "unused_episodes_per_task": 0,
    }


def test_r3m_reference_policy_configuration() -> None:
    config = load_yaml(ROOT / "configs" / "experiment_r3m_mlp_60.yaml")
    policy = config["policy"]
    assert policy["implementation"] == "tcc_mlp_bc_v4_r3m_reference"
    assert policy["proprioception"] is False
    assert "progress_conditioning" not in policy
    assert policy["action_representation"] == "absolute"
    assert policy["hidden_dimensions"] == [256, 256]
    assert policy["input_batch_norm"] is True
    assert policy["input_layer_norm"] is False
    assert policy["output_layer_scale"] == 0.01
    assert policy["normalize_actions"] is True
    assert policy["loss"] == "mse"
    assert policy["optimizer"] == "adam"
    assert policy["training_steps"] == 50_000
    assert policy["batch_size"] == 32
    assert policy["learning_rate"] == 0.001
    assert config["evaluation"]["max_rollout_steps"] == 359


def test_r3m_multiview_proprio_policy_configuration() -> None:
    config = load_yaml(ROOT / "configs" / "experiment_r3m_multiview_proprio_60.yaml")
    policy = config["policy"]
    assert policy["implementation"] == "tcc_mlp_bc_v5_r3m_multiview_proprio"
    assert policy["cameras"] == ["cam_main", "cam_wrist"]
    assert policy["camera_fusion"] == "project_then_concat"
    assert policy["camera_projection_dim"] == 128
    assert policy["proprioception"] is True
    assert policy["proprioception_dim"] == 7
    assert policy["action_representation"] == "absolute"
    assert policy["hidden_dimensions"] == [256, 256]
    assert policy["input_batch_norm"] is True
    assert policy["loss"] == "mse"
    assert config["observations"]["cameras"] == ["cam_main", "cam_wrist"]


def test_r3m_single_view_proprio_policy_configuration() -> None:
    config = load_yaml(ROOT / "configs" / "experiment_r3m_single_view_proprio_60.yaml")
    policy = config["policy"]
    assert policy["implementation"] == "tcc_mlp_bc_v5_r3m_single_view_proprio"
    assert policy["cameras"] == ["cam_main"]
    assert policy["camera_fusion"] == "raw_concat"
    assert policy["camera_projection_dim"] == 0
    assert policy["proprioception"] is True
    assert policy["proprioception_dim"] == 7
    assert policy["action_representation"] == "absolute"
    assert policy["hidden_dimensions"] == [256, 256]
    assert policy["input_batch_norm"] is True
    assert policy["loss"] == "mse"
    assert config["observations"]["cameras"] == ["cam_main", "cam_wrist"]
    assert set(config["model_hub"]["supported_backbones"]) == {
        "ours_rn50",
        "ours_vit",
        "hralign",
        "r3m_unadapted",
        "d4r_imagenet",
        "hrp_imagenet",
    }
    assert config["model_hub"]["revision"] == (
        "7b79ed9cefe5121ed510c74843f650310c564ada"
    )


def test_v6_gated_multiview_proprio_policy_configuration() -> None:
    config = load_yaml(
        ROOT / "configs" / "experiment_v6_gated_multiview_proprio_60.yaml"
    )
    policy = config["policy"]
    assert policy["implementation"] == "tcc_mlp_bc_v6_gated_multiview_proprio"
    assert policy["cameras"] == ["cam_main", "cam_wrist"]
    assert policy["camera_fusion"] == "gated_residual"
    assert policy["camera_projection_dim"] == 128
    assert policy["camera_gate_hidden_dim"] == 128
    assert policy["proprioception"] is True
    assert policy["proprioception_dim"] == 7
    assert policy["action_representation"] == "absolute"
    assert policy["hidden_dimensions"] == [256, 256]
    assert policy["input_batch_norm"] is True
    assert policy["loss"] == "mse"
    assert config["model_hub"]["supported_backbones"] == [
        "ours_rn50",
        "ours_vit",
        "hralign",
        "r3m_unadapted",
        "d4r_imagenet",
        "hrp_imagenet",
    ]
    assert config["model_hub"]["revision"] == (
        "94dc379f5ee00d7e410ac211396cd06d9e65953a"
    )


def test_v6_gated_multiview_no_proprio_policy_configuration() -> None:
    config = load_yaml(
        ROOT / "configs" / "experiment_v6_gated_multiview_no_proprio_60.yaml"
    )
    policy = config["policy"]
    assert policy["implementation"] == "tcc_mlp_bc_v6_gated_multiview_proprio"
    assert policy["cameras"] == ["cam_main", "cam_wrist"]
    assert policy["camera_fusion"] == "gated_residual"
    assert policy["camera_projection_dim"] == 128
    assert policy["camera_gate_hidden_dim"] == 128
    assert policy["proprioception"] is False
    assert policy["proprioception_dim"] == 0
    assert policy["action_representation"] == "absolute"
    assert policy["hidden_dimensions"] == [256, 256]
    assert policy["input_batch_norm"] is True
    assert policy["loss"] == "mse"
    assert config["model_hub"]["supported_backbones"] == [
        "ours_rn50",
        "ours_vit",
        "r3m_unadapted",
        "d4r_imagenet",
    ]
    assert config["model_hub"]["revision"] == (
        "e40f3fe4a459cc6cbbbf338d07cd651455978213"
    )
    assert config["model_hub"]["policy_checkpoint_template"].startswith(
        "policies_v6_no_proprio/"
    )
    assert config["model_hub"]["policy_metrics_template"].startswith(
        "policies_v6_no_proprio/"
    )


def test_model_hub_is_pinned_to_an_immutable_revision() -> None:
    config = load_yaml(ROOT / "configs" / "experiment.yaml")
    hub = config["model_hub"]
    assert hub["repository"] == "Chipaipai/tcc-core-real-robot-policies"
    assert len(hub["revision"]) == 40
    assert all(character in "0123456789abcdef" for character in hub["revision"])
    assert config["backbone"]["source"] == "huggingface"
    assert config["backbone"]["hub_name"] == "ours_rn50"


def test_v7_hrp_gmm_delta_policy_configuration() -> None:
    config = load_yaml(ROOT / "configs" / "experiment_v7_hrp_gmm_delta_60.yaml")
    policy = config["policy"]

    assert policy["reference"] == "hrp_data4robotics"
    assert policy["cameras"] == ["cam_main", "cam_wrist"]
    assert policy["camera_fusion"] == "raw_concat"
    assert policy["proprioception"] is True
    assert policy["progress_conditioning"] is None
    assert policy["action_representation"] == "current_delta"
    assert policy["action_distribution"] == "gaussian_mixture"
    assert policy["num_modes"] == 5
    assert policy["hidden_dimensions"] == [512, 512]
    assert policy["dropout"] == 0.2
    assert policy["loss"] == "gmm_nll"
    assert policy["batch_size"] == 150
    assert policy["learning_rate"] == 0.0001


def test_v8_matches_hrp_release_single_view_defaults() -> None:
    config = load_yaml(
        ROOT / "configs" / "experiment_v8_hrp_official_single_view_60.yaml"
    )
    policy = config["policy"]

    assert config["model_hub"]["supported_backbones"] == [
        "ours_rn50",
        "ours_vit",
        "r3m_unadapted",
        "d4r_imagenet",
    ]
    assert config["backbone"]["frozen"] is False
    assert config["backbone"]["fine_tuning"] == "full_end_to_end"
    assert config["dataset"]["tasks"] == ["pick_and_place_carrot_100"]
    assert config["dataset"]["demonstrations_per_task"] == 100
    assert config["dataset"]["action_leads_measured_state_frames"] == 2
    assert policy["architecture"] == "hrp_state_token_gmm"
    assert policy["architecture_reference"] == "data4robotics_hrp_release_default"
    assert policy["action_adapter"] == "trossen_joint_position_passthrough"
    assert policy["cameras"] == ["cam_main"]
    assert policy["task_conditioning"] is None
    assert policy["number_of_tasks"] == 1
    assert policy["action_representation"] == "absolute"
    assert policy["action_space"] == "joint_position"
    assert policy["state_representation"] == "measured_joint_position"
    assert policy["action_source"] == "original_lerobot_action"
    assert policy["action_leads_measured_state_frames"] == 2
    assert policy["action_frame"] == "joint"
    assert policy["gripper_action"] == "position"
    assert policy["action_distribution"] == "gaussian_mixture"
    assert policy["num_modes"] == 5
    assert policy["hidden_dimensions"] == [512, 512]
    assert policy["dropout"] == 0.2
    assert policy["training_steps"] == 40000
    assert policy["batch_size"] == 150
    assert policy["learning_rate"] == 0.0003
    assert policy["precision"] == "float32"
    assert policy["weight_decay"] == 0.0001
    assert policy["normalize_state"] is False
    assert policy["normalize_actions"] is True
    assert policy["inference"] == "official_zero_std_mixture_sample"
    assert config["split"] == {
        "protocol": "hrp_fixed_transition_holdout",
        "shuffle_seed": 3904767649,
        "held_out_transitions": 500,
    }
    assert config["model_hub"]["supported_demonstrations"] == [100]
    assert config["model_hub"]["revision"] == (
        "dfbc7a76194d4aad41c06441dd2d7e4abce397cc"
    )
    assert config["augmentation"] == {
        "name": "hrp_release_medium",
        "random_resized_crop_scale": [0.9, 1.0],
        "random_resized_crop_ratio": [0.75, 4 / 3],
        "color_jitter_probability": 0.0,
        "gaussian_blur": True,
        "gaussian_blur_probability": 1.0,
        "imagenet_normalization": True,
    }


def test_v9_is_trossen_native_joint_delta() -> None:
    config = load_yaml(ROOT / "configs" / "experiment_v9_trossen_joint_delta_100.yaml")
    policy = config["policy"]

    assert policy["architecture"] == "dual_encoder_mlp_gmm"
    assert policy["architecture_reference"] == "robomimic_multiview_late_fusion"
    assert policy["cameras"] == ["cam_main", "cam_wrist"]
    assert policy["shared_camera_backbone"] is False
    assert policy["camera_fusion"] == "project_then_concat"
    assert policy["camera_projection_dim"] == 128
    assert policy["action_representation"] == "current_delta"
    assert policy["action_space"] == "joint_position"
    assert policy["action_frame"] == "joint"
    assert policy["action_adapter"] == "current_state_plus_joint_delta_to_position"
    assert policy["state_representation"] == "measured_joint_position"
    assert policy["gripper_action"] == "position_delta"
    assert policy["normalize_state"] is True
    assert policy["normalize_actions"] is True
    assert policy["learning_rate"] == 0.0001
    assert policy["backbone_learning_rate"] == 0.00001
    assert policy["batch_size"] == 32
    assert policy["training_steps"] == 50000
    assert policy["deterministic_inference"] == "highest_probability_mode_mean"
    assert config["model_hub"]["revision"] == (
        "dfbc7a76194d4aad41c06441dd2d7e4abce397cc"
    )
    assert config["sampling"] == {
        "protocol": "trossen_start_weighted",
        "start_frames": 5,
        "start_weight": 10.0,
    }
    assert config["split"] == {
        "protocol": "trossen_episode_holdout",
        "shuffle_seed": 3904767649,
        "train_episodes_per_task": 80,
        "validation_episodes_per_task": 10,
        "test_episodes_per_task": 10,
    }
