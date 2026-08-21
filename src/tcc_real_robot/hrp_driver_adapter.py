"""The only robot-specific boundary in the HRP reproduction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from tcc_real_robot.hrp_action_space import clip_hrp_action, hrp_state


@dataclass(frozen=True)
class HRPDriverStep:
    """One clipped HRP velocity command sent through the official driver API."""

    raw_velocity: tuple[float, ...]
    commanded_velocity: tuple[float, ...]
    observed_state: tuple[float, ...]


class TrossenHRPDriverAdapter:
    """Map official HRP actions onto Trossen Cartesian/gripper velocity calls."""

    def __init__(
        self,
        driver_api: Any,
        driver: Any,
        *,
        action_min: list[float],
        action_max: list[float],
        control_fps: float,
    ) -> None:
        if not isfinite(control_fps) or control_fps <= 0.0:
            raise ValueError("control_fps must be positive")
        self.driver_api = driver_api
        self.driver = driver
        self.action_min = np.asarray(action_min, dtype=np.float64)
        self.action_max = np.asarray(action_max, dtype=np.float64)
        # At 20 Hz this requests linear interpolation over exactly one policy step.
        self.goal_time = 1.0 / control_fps
        clip_hrp_action(np.zeros(7), self.action_min, self.action_max)
        self.active = False

    def read_state(self) -> np.ndarray:
        """Return HRP's 7-D Cartesian-pose/gripper observation."""
        pose = np.asarray(self.driver.get_cartesian_positions(), dtype=np.float64)
        positions = np.asarray(self.driver.get_all_positions(), dtype=np.float64)
        if positions.shape != (7,) or not np.isfinite(positions).all():
            raise RuntimeError("Controller returned invalid joint/gripper positions")
        return hrp_state(pose, float(positions[6]))

    def start(self) -> None:
        """Enter the velocity modes required by the official HRP action space."""
        self.driver.set_arm_modes(self.driver_api.Mode.velocity)
        self.driver.set_gripper_mode(self.driver_api.Mode.velocity)
        self.active = True

    def execute(self, raw_velocity: list[float]) -> HRPDriverStep:
        """Clip externally, then issue one non-blocking official driver command."""
        if not self.active:
            raise RuntimeError("HRP driver adapter has not entered velocity mode")
        raw = np.asarray(raw_velocity, dtype=np.float64)
        command = clip_hrp_action(raw, self.action_min, self.action_max)
        observed = self.read_state()
        self.driver.set_cartesian_velocities(
            command[:6].tolist(),
            self.driver_api.InterpolationSpace.cartesian,
            goal_time=self.goal_time,
            blocking=False,
        )
        self.driver.set_gripper_velocity(
            float(command[6]),
            goal_time=self.goal_time,
            blocking=False,
        )
        error = str(self.driver.get_error_information())
        if error.lower() != "no error":
            raise RuntimeError(f"Controller reports an error: {error}")
        return HRPDriverStep(
            raw_velocity=tuple(float(value) for value in raw),
            commanded_velocity=tuple(float(value) for value in command),
            observed_state=tuple(float(value) for value in observed),
        )

    def stop(self) -> None:
        """Command zero velocity before the owning session returns to idle."""
        if not self.active:
            return
        zeros = [0.0] * 6
        self.driver.set_cartesian_velocities(
            zeros,
            self.driver_api.InterpolationSpace.cartesian,
            goal_time=self.goal_time,
            blocking=False,
        )
        self.driver.set_gripper_velocity(
            0.0,
            goal_time=self.goal_time,
            blocking=False,
        )
        self.active = False
