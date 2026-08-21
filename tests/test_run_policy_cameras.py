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
    assert args.demonstrations == 80
    assert args.camera_backend == "realsense-sdk"
    assert args.tcc_source_root == Path("/home/robotarm/TCC-core")
    assert args.offline is True
    assert args.execute_policy is True
    assert args.emergency_stop_ready is True


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
