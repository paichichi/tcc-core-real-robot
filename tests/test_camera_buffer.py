import time
from typing import ClassVar

import numpy as np

from tcc_real_robot.camera_buffer import LatestFramePairBuffer


class FakeCameraSource:
    main_properties: ClassVar[dict[str, str]] = {"name": "main"}
    wrist_properties: ClassVar[dict[str, str]] = {"name": "wrist"}

    def __init__(self) -> None:
        self.last_pair_skew_ms = None
        self.counter = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def read_rgb_pair(self) -> tuple[np.ndarray, np.ndarray]:
        time.sleep(0.005)
        self.counter += 1
        self.last_pair_skew_ms = 0.25
        frame = np.full((2, 3, 3), self.counter, dtype=np.uint8)
        return frame, frame.copy()


def test_latest_frame_buffer_delivers_fresh_pairs_and_closes_source() -> None:
    source = FakeCameraSource()
    with LatestFramePairBuffer(source, timeout_s=0.5) as buffer:
        first, _ = buffer.read_rgb_pair()
        second, _ = buffer.read_rgb_pair()
        assert int(second[0, 0, 0]) > int(first[0, 0, 0])
        assert buffer.last_pair_skew_ms == 0.25
        assert buffer.main_properties == {"name": "main"}
        assert buffer.captured_pairs >= 2
    assert source.closed is True


def test_latest_frame_buffer_skips_stale_pairs() -> None:
    source = FakeCameraSource()
    with LatestFramePairBuffer(source, timeout_s=0.5) as buffer:
        buffer.read_rgb_pair()
        time.sleep(0.03)
        buffer.read_rgb_pair()
        assert buffer.dropped_pairs > 0
