from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def load_diagnostic() -> Any:
    pytest.importorskip("cv2")
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "diagnose_policy_inputs.py"
    spec = importlib.util.spec_from_file_location("diagnose_policy_inputs_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jpeg_round_trip_preserves_rgb_contract() -> None:
    module = load_diagnostic()
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[:, :12, 0] = 255
    image[:, 12:, 1] = 128

    result = module.jpeg_round_trip(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert float(np.abs(result.astype(float) - image).mean()) < 12.0


def test_channel_statistics_and_arm_difference() -> None:
    module = load_diagnostic()
    image = np.full((4, 5, 3), [10, 20, 30], dtype=np.uint8)
    mean, std = module.channel_statistics(image)
    left = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.01])
    right = np.array([0.0, 1.2, 2.0, 2.5, 4.0, 5.0, 0.04])

    assert mean.tolist() == [10.0, 20.0, 30.0]
    assert std.tolist() == [0.0, 0.0, 0.0]
    assert module.maximum_arm_action_difference(left, right) == 0.5
