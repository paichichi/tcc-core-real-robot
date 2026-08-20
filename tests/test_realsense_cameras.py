from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tcc_real_robot.realsense_cameras import RealSenseColorCameras


class FakeConfig:
    def __init__(self) -> None:
        self.serial = ""
        self.stream_args: tuple[object, ...] = ()

    def enable_device(self, serial: str) -> None:
        self.serial = serial

    def enable_stream(self, *args: object) -> None:
        self.stream_args = args


class FakeDevice:
    def __init__(self, serial: str) -> None:
        self.serial = serial

    def get_info(self, field: str) -> str:
        return self.serial if field == "serial_number" else f"device-{self.serial}"


class FakeVideoProfile:
    def as_video_stream_profile(self) -> FakeVideoProfile:
        return self

    def format(self) -> str:
        return "rgb8"

    def width(self) -> int:
        return 640

    def height(self) -> int:
        return 480

    def fps(self) -> int:
        return 30


class FakePipelineProfile:
    def __init__(self, serial: str) -> None:
        self.device = FakeDevice(serial)

    def get_device(self) -> FakeDevice:
        return self.device

    def get_stream(self, stream: str) -> FakeVideoProfile:
        assert stream == "color"
        return FakeVideoProfile()


class FakeColorFrame:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def get_data(self) -> np.ndarray:
        return self.frame

    def __bool__(self) -> bool:
        return True


class FakeFrameSet:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def get_color_frame(self) -> FakeColorFrame:
        return FakeColorFrame(self.frame)


class FakePipeline:
    def __init__(self, owner: FakeRS) -> None:
        self.owner = owner
        self.serial = ""
        self.stopped = False

    def start(self, config: FakeConfig) -> FakePipelineProfile:
        self.serial = config.serial
        self.owner.started_configs.append(config)
        return FakePipelineProfile(self.serial)

    def wait_for_frames(self, timeout_ms: int) -> FakeFrameSet:
        assert timeout_ms == 3000
        return FakeFrameSet(self.owner.frames[self.serial])

    def stop(self) -> None:
        self.stopped = True


class FakeRS:
    stream = SimpleNamespace(color="color")
    format = SimpleNamespace(rgb8="rgb8")
    camera_info = SimpleNamespace(serial_number="serial_number", name="name")

    def __init__(self) -> None:
        frame = np.arange(480 * 640 * 3, dtype=np.uint32)
        frame = (frame % 251).astype(np.uint8).reshape(480, 640, 3)
        self.frames = {"main": frame, "wrist": np.flip(frame, axis=1).copy()}
        self.started_configs: list[FakeConfig] = []
        self.pipelines: list[FakePipeline] = []

    def pipeline(self) -> FakePipeline:
        pipeline = FakePipeline(self)
        self.pipelines.append(pipeline)
        return pipeline

    def config(self) -> FakeConfig:
        return FakeConfig()


def test_serial_pinned_cameras_request_only_rgb8_color() -> None:
    rs = FakeRS()
    with RealSenseColorCameras(rs, "main", "wrist", 640, 480, 30) as cameras:
        main, wrist = cameras.read_rgb_pair()

        assert main.shape == (480, 640, 3)
        assert wrist.shape == (480, 640, 3)
        assert cameras.main_properties["serial"] == "main"
        assert cameras.wrist_properties["serial"] == "wrist"
        assert cameras.last_pair_skew_ms is not None

    assert [config.stream_args for config in rs.started_configs] == [
        ("color", 640, 480, "rgb8", 30),
        ("color", 640, 480, "rgb8", 30),
    ]
    assert all(pipeline.stopped for pipeline in rs.pipelines)


def test_serial_pinned_cameras_require_distinct_devices() -> None:
    with pytest.raises(ValueError, match="distinct"):
        RealSenseColorCameras(FakeRS(), "same", "same", 640, 480, 30)
