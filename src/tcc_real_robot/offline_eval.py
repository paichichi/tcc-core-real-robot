"""Metrics for comparing demo first-frame policy predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FirstFrameComparison:
    target_max_arm_delta_rad: float
    target_gripper_delta_m: float
    prediction_max_arm_delta_rad: float
    prediction_gripper_delta_m: float
    prediction_action_mae: float
    prediction_action_max_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compare_first_frame(
    state: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> FirstFrameComparison:
    """Compare a first-frame prediction with its state and recorded action."""
    arrays = {
        "state": np.asarray(state, dtype=np.float64),
        "target": np.asarray(target, dtype=np.float64),
        "prediction": np.asarray(prediction, dtype=np.float64),
    }
    for name, value in arrays.items():
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain seven finite values")
    state_value = arrays["state"]
    target_value = arrays["target"]
    prediction_value = arrays["prediction"]
    target_delta = np.abs(target_value - state_value)
    prediction_delta = np.abs(prediction_value - state_value)
    prediction_error = np.abs(prediction_value - target_value)
    return FirstFrameComparison(
        target_max_arm_delta_rad=float(target_delta[:6].max()),
        target_gripper_delta_m=float(target_delta[6]),
        prediction_max_arm_delta_rad=float(prediction_delta[:6].max()),
        prediction_gripper_delta_m=float(prediction_delta[6]),
        prediction_action_mae=float(prediction_error.mean()),
        prediction_action_max_error=float(prediction_error.max()),
    )
