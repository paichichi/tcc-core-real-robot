import numpy as np
import pytest

from tcc_real_robot.offline_eval import compare_first_frame


def test_compare_first_frame_reports_action_and_state_deltas() -> None:
    state = np.zeros(7)
    target = np.array([0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.001])
    prediction = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002])

    result = compare_first_frame(state, target, prediction)

    assert result.target_max_arm_delta_rad == pytest.approx(0.02)
    assert result.target_gripper_delta_m == pytest.approx(0.001)
    assert result.prediction_max_arm_delta_rad == pytest.approx(0.03)
    assert result.prediction_gripper_delta_m == pytest.approx(0.002)
    assert result.prediction_action_max_error == pytest.approx(0.02)


def test_compare_first_frame_rejects_invalid_vectors() -> None:
    with pytest.raises(ValueError, match="seven finite"):
        compare_first_frame(np.zeros(6), np.zeros(7), np.zeros(7))
