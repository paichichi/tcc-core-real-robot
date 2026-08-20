from __future__ import annotations

import numpy as np
import pytest

from tcc_real_robot.demo_replay import audit_demo_trajectory


def test_audit_accepts_bounded_absolute_trajectory() -> None:
    actions = np.zeros((3, 7), dtype=np.float64)
    actions[1, 0] = 0.01
    actions[2, 0] = 0.02
    result = audit_demo_trajectory(
        actions,
        fps=20.0,
        absolute_min=[-1.0] * 7,
        absolute_max=[1.0] * 7,
        max_arm_velocity_rad_s=1.5,
        max_gripper_velocity_m_s=0.06,
    )
    assert result.frames == 3
    assert result.max_step[0] == pytest.approx(0.01)
    assert result.max_velocity[0] == pytest.approx(0.2)


def test_audit_rejects_velocity_violation() -> None:
    actions = np.zeros((2, 7), dtype=np.float64)
    actions[1, 0] = 0.1
    with pytest.raises(ValueError, match="arm velocity"):
        audit_demo_trajectory(
            actions,
            fps=20.0,
            absolute_min=[-1.0] * 7,
            absolute_max=[1.0] * 7,
            max_arm_velocity_rad_s=1.5,
            max_gripper_velocity_m_s=0.06,
        )
