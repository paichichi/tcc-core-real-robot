from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tcc_real_robot.hrp_driver_adapter import TrossenHRPDriverAdapter


class FakeDriver:
    def __init__(self) -> None:
        self.arm_mode = None
        self.gripper_mode = None
        self.cartesian_commands = []
        self.gripper_commands = []

    def get_cartesian_positions(self):
        return [0.3, 0.0, 0.2, 0.0, 0.1, 0.0]

    def get_all_positions(self):
        return [0.0] * 6 + [0.02]

    def set_arm_modes(self, mode):
        self.arm_mode = mode

    def set_gripper_mode(self, mode):
        self.gripper_mode = mode

    def set_cartesian_velocities(self, values, interpolation, **kwargs):
        self.cartesian_commands.append((values, interpolation, kwargs))

    def set_gripper_velocity(self, value, **kwargs):
        self.gripper_commands.append((value, kwargs))

    def get_error_information(self):
        return "No error"


def test_adapter_is_a_thin_official_velocity_driver_boundary() -> None:
    api = SimpleNamespace(
        Mode=SimpleNamespace(velocity="velocity"),
        InterpolationSpace=SimpleNamespace(cartesian="cartesian"),
    )
    driver = FakeDriver()
    adapter = TrossenHRPDriverAdapter(
        api,
        driver,
        action_min=[-1.0] * 7,
        action_max=[1.0] * 7,
        control_fps=20.0,
    )
    adapter.start()
    step = adapter.execute([2.0, -2.0, 0.2, 0.0, 0.0, 0.0, 0.01])

    assert driver.arm_mode == "velocity"
    assert driver.gripper_mode == "velocity"
    assert np.allclose(step.commanded_velocity, [1.0, -1.0, 0.2, 0, 0, 0, 0.01])
    assert driver.cartesian_commands[-1] == (
        [1.0, -1.0, pytest.approx(0.2), 0.0, 0.0, 0.0],
        "cartesian",
        {"goal_time": 0.05, "blocking": False},
    )
    assert driver.gripper_commands[-1] == (
        pytest.approx(0.01),
        {"goal_time": 0.05, "blocking": False},
    )

    adapter.stop()
    assert driver.cartesian_commands[-1][0] == [0.0] * 6
    assert driver.gripper_commands[-1][0] == 0.0
