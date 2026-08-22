from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def load_run_policy() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "run_policy.py"
    spec = importlib.util.spec_from_file_location("run_policy_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_policy_preset_has_short_fixed_defaults(monkeypatch) -> None:
    module = load_run_policy()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_policy.py", "--execute-policy", "--emergency-stop-ready"],
    )

    args = module.parse_args()

    assert args.task == "carrot"
    assert args.backbone == "ours_rn50"
    assert args.demonstrations is None
    assert args.camera_backend == "realsense-sdk"
    assert args.tcc_source_root == Path("/home/robotarm/TCC-core")
    assert args.offline is True
    assert args.execute_policy is True
    assert args.emergency_stop_ready is True


def test_first_executed_action_is_exact_dataset_home() -> None:
    torch = pytest.importorskip("torch")
    module = load_run_policy()
    raw_action = torch.tensor([0.4, 0.3, 0.2, 0.1, -0.1, -0.2, 0.03])
    home = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.0]

    action, applied = module.apply_first_action_home_anchor(
        raw_action, 0, home, True
    )

    assert applied is True
    assert action.tolist() == pytest.approx(home)


def test_home_anchor_preserves_later_live_policy_actions() -> None:
    torch = pytest.importorskip("torch")
    module = load_run_policy()
    raw_action = torch.tensor([0.4, 0.3, 0.2, 0.1, -0.1, -0.2, 0.03])

    action, applied = module.apply_first_action_home_anchor(
        raw_action, 1, [0.0] * 7, True
    )

    assert applied is False
    assert action is raw_action


def test_joint_position_driver_contract_matches_collection_api() -> None:
    module = load_run_policy()
    experiment = {
        "observations": {"fps": 20},
        "policy": {
            "action_space": "joint_position",
            "action_representation": "absolute",
            "action_adapter": "trossen_joint_position_passthrough",
        },
    }
    robot = {
        "hrp_driver_contract": {
            "action_representation": "absolute",
            "action_space": "joint_position",
            "driver_call": "set_all_positions",
            "ik_required": False,
        },
        "policy_evaluation": {"clipped_rollout": {"control_fps": 20}},
    }

    module.validate_joint_position_driver_contract(experiment, robot)
    robot["hrp_driver_contract"]["driver_call"] = "set_cartesian_positions"
    with pytest.raises(RuntimeError, match="Driver contract mismatch"):
        module.validate_joint_position_driver_contract(experiment, robot)


def test_joint_delta_policy_reconstructs_for_position_driver() -> None:
    module = load_run_policy()
    experiment = {
        "observations": {"fps": 20},
        "policy": {
            "action_space": "joint_position",
            "action_representation": "current_delta",
            "action_adapter": "current_state_plus_joint_delta_to_position",
        },
    }
    robot = {
        "hrp_driver_contract": {
            "action_representation": "absolute",
            "action_space": "joint_position",
            "driver_call": "set_all_positions",
            "ik_required": False,
        },
        "policy_evaluation": {"clipped_rollout": {"control_fps": 20}},
    }

    module.validate_joint_position_driver_contract(experiment, robot)
    experiment["policy"]["action_adapter"] = "trossen_joint_position_passthrough"
    with pytest.raises(RuntimeError, match="Driver contract mismatch"):
        module.validate_joint_position_driver_contract(experiment, robot)


def test_all_actuation_modes_obey_shadow_release_gate() -> None:
    module = load_run_policy()
    robot = {
        "action_contract": {"enabled": False},
        "safety": {"workspace_limits": "CALIBRATING"},
    }

    with pytest.raises(RuntimeError, match="Run shadow evaluation first"):
        module.assert_shadow_only(robot, True)

    module.assert_shadow_only(robot, False)


class FakeCapture:
    def __init__(self, grabs: list[bool]) -> None:
        self.grabs = iter(grabs)
        self.frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:
        return True

    def set(self, key: int, value: float) -> bool:
        self.properties[key] = value
        return True

    def get(self, key: int) -> float:
        return self.properties.get(key, 0.0)

    def grab(self) -> bool:
        return next(self.grabs)

    def retrieve(self) -> tuple[bool, np.ndarray]:
        return True, self.frame

    def release(self) -> None:
        return None


class FakeCV2:
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3
    CAP_PROP_BUFFERSIZE = 4
    CAP_PROP_FOURCC = 5
    CAP_V4L2 = 6
    COLOR_BGR2RGB = 7

    def __init__(self, captures: list[FakeCapture]) -> None:
        self.captures = iter(captures)

    def VideoCapture(self, *_: object) -> FakeCapture:
        return next(self.captures)

    @staticmethod
    def cvtColor(frame: np.ndarray, _: int) -> np.ndarray:
        return frame


def test_camera_pair_retries_a_transient_grab_failure() -> None:
    module = load_run_policy()
    cv2 = FakeCV2([FakeCapture([False, True]), FakeCapture([True, True])])
    cameras = module.SynchronizedCameras(
        cv2,
        "/dev/video10",
        "/dev/video2",
        640,
        480,
        20.0,
        startup_delay_s=0.0,
        read_attempts=2,
        retry_delay_s=0.0,
    )

    main, wrist = cameras.read_rgb_pair()

    assert main.shape == (2, 3, 3)
    assert wrist.shape == (2, 3, 3)


def test_camera_pair_error_identifies_the_failed_stream() -> None:
    module = load_run_policy()
    cv2 = FakeCV2([FakeCapture([True, True]), FakeCapture([False, False])])
    cameras = module.SynchronizedCameras(
        cv2,
        "/dev/video10",
        "/dev/video2",
        640,
        480,
        20.0,
        startup_delay_s=0.0,
        read_attempts=2,
        retry_delay_s=0.0,
    )

    with pytest.raises(RuntimeError, match="main_grab=PASS, wrist_grab=FAIL"):
        cameras.read_rgb_pair()


def test_camera_pair_rejects_a_solid_green_frame() -> None:
    module = load_run_policy()
    main = FakeCapture([True])
    wrist = FakeCapture([True])
    wrist.frame = np.full((4, 5, 3), (0, 255, 0), dtype=np.uint8)
    cv2 = FakeCV2([main, wrist])
    cameras = module.SynchronizedCameras(
        cv2,
        "/dev/video10",
        "/dev/video2",
        640,
        480,
        20.0,
        startup_delay_s=0.0,
        read_attempts=1,
        retry_delay_s=0.0,
        minimum_channel_std=2.0,
    )

    with pytest.raises(RuntimeError, match="wrist_frame=FLAT"):
        cameras.read_rgb_pair()
