"""Serial-pinned RGB acquisition for two Intel RealSense cameras."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Self

import numpy as np


class RealSenseColorCameras:
    """Read only explicit RGB8 color streams from two serial-pinned devices."""

    def __init__(
        self,
        rs: Any,
        main_serial: str,
        wrist_serial: str,
        width: int,
        height: int,
        fps: int,
        *,
        timeout_ms: int = 3000,
        read_attempts: int = 3,
        minimum_channel_std: float = 2.0,
        maximum_pair_skew_ms: float = 50.0,
    ) -> None:
        if not main_serial or not wrist_serial or main_serial == wrist_serial:
            raise ValueError("Two distinct RealSense serial numbers are required")
        if min(width, height, fps, timeout_ms, read_attempts) <= 0:
            raise ValueError("RealSense dimensions, rates, and retries must be positive")
        if minimum_channel_std < 0 or maximum_pair_skew_ms <= 0:
            raise ValueError("RealSense frame validation settings are invalid")
        self.rs = rs
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_ms = timeout_ms
        self.read_attempts = read_attempts
        self.minimum_channel_std = minimum_channel_std
        self.maximum_pair_skew_ms = maximum_pair_skew_ms
        self.last_pair_skew_ms: float | None = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.main_pipeline: Any | None = None
        self.wrist_pipeline: Any | None = None
        try:
            self.main_pipeline, main_profile = self._start(main_serial)
            self.wrist_pipeline, wrist_profile = self._start(wrist_serial)
            self.main_properties = self._properties(main_profile)
            self.wrist_properties = self._properties(wrist_profile)
        except Exception:
            self.close()
            raise

    def _start(self, serial: str) -> tuple[Any, Any]:
        pipeline = self.rs.pipeline()
        config = self.rs.config()
        config.enable_device(serial)
        config.enable_stream(
            self.rs.stream.color,
            self.width,
            self.height,
            self.rs.format.rgb8,
            self.fps,
        )
        profile = pipeline.start(config)
        actual_serial = str(
            profile.get_device().get_info(self.rs.camera_info.serial_number)
        )
        if actual_serial != serial:
            pipeline.stop()
            raise RuntimeError(
                f"RealSense serial mismatch: requested {serial}, got {actual_serial}"
            )
        return pipeline, profile

    def _properties(self, pipeline_profile: Any) -> dict[str, object]:
        device = pipeline_profile.get_device()
        stream = pipeline_profile.get_stream(
            self.rs.stream.color
        ).as_video_stream_profile()
        return {
            "serial": str(device.get_info(self.rs.camera_info.serial_number)),
            "name": str(device.get_info(self.rs.camera_info.name)),
            "stream": "color",
            "format": str(stream.format()),
            "width": int(stream.width()),
            "height": int(stream.height()),
            "fps": int(stream.fps()),
        }

    def _wait_rgb(self, pipeline: Any) -> tuple[np.ndarray, float]:
        frames = pipeline.wait_for_frames(self.timeout_ms)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense frameset did not contain a color frame")
        rgb = np.asanyarray(color.get_data()).copy()
        received_at = time.monotonic()
        return rgb, received_at

    def read_rgb_pair(self) -> tuple[np.ndarray, np.ndarray]:
        last_status = "no attempts made"
        if self.main_pipeline is None or self.wrist_pipeline is None:
            raise RuntimeError("RealSense pipelines are not running")
        for attempt in range(1, self.read_attempts + 1):
            main_future = self._executor.submit(self._wait_rgb, self.main_pipeline)
            wrist_future = self._executor.submit(self._wait_rgb, self.wrist_pipeline)
            main_rgb, main_at = main_future.result()
            wrist_rgb, wrist_at = wrist_future.result()
            pair_skew_ms = abs(wrist_at - main_at) * 1000.0
            expected_shape = (self.height, self.width, 3)
            main_valid = (
                main_rgb.shape == expected_shape
                and main_rgb.dtype == np.uint8
                and float(np.max(np.std(main_rgb, axis=(0, 1))))
                >= self.minimum_channel_std
            )
            wrist_valid = (
                wrist_rgb.shape == expected_shape
                and wrist_rgb.dtype == np.uint8
                and float(np.max(np.std(wrist_rgb, axis=(0, 1))))
                >= self.minimum_channel_std
            )
            skew_valid = pair_skew_ms <= self.maximum_pair_skew_ms
            if main_valid and wrist_valid and skew_valid:
                self.last_pair_skew_ms = pair_skew_ms
                return main_rgb, wrist_rgb
            last_status = (
                f"attempt={attempt}, main={'PASS' if main_valid else 'INVALID'}, "
                f"wrist={'PASS' if wrist_valid else 'INVALID'}, "
                f"pair_skew_ms={pair_skew_ms:.3f}, "
                f"skew={'PASS' if skew_valid else 'FAIL'}"
            )
        raise RuntimeError(
            "Failed to read a valid RealSense RGB pair after "
            f"{self.read_attempts} attempts ({last_status})"
        )

    def close(self) -> None:
        for name in ("main_pipeline", "wrist_pipeline"):
            pipeline = getattr(self, name, None)
            if pipeline is not None:
                try:
                    pipeline.stop()
                finally:
                    setattr(self, name, None)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
