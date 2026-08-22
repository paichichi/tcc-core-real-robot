"""Safety-gated dataset-home staging for shadow policy evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite
from typing import Any

from typing_extensions import Self

from tcc_real_robot.continuous_control import StatefulPositionLimiter
from tcc_real_robot.driver_config import apply_motor_parameters, validate_versions


@dataclass(frozen=True)
class HomePreparation:
    """Measured state after the controller reaches dataset collection home."""

    driver_version: str
    firmware_version: str
    target: tuple[float, ...]
    observed: tuple[float, ...]
    max_arm_error_rad: float
    gripper_error_m: float


@dataclass(frozen=True)
class BoundedPolicyStep:
    """Measured result of one clipped absolute-policy command."""

    raw_target: tuple[float, ...]
    start: tuple[float, ...]
    commanded: tuple[float, ...]
    observed: tuple[float, ...]
    max_commanded_arm_delta_rad: float
    commanded_gripper_delta_m: float
    max_arm_command_lead_rad: float
    gripper_command_lead_m: float
    max_arm_command_gap_rad: float
    gripper_command_gap_m: float
    sampled_at_monotonic: float


@dataclass(frozen=True)
class PolicyTargetVerification:
    """Final tracking result after the last non-blocking command settles."""

    target: tuple[float, ...]
    observed: tuple[float, ...]
    max_arm_tracking_error_rad: float
    gripper_tracking_error_m: float


class PolicyHomeSession:
    """Keep the arm at dataset home while a shadow evaluation runs."""

    def __init__(
        self,
        driver_api: Any,
        config: dict[str, Any],
        timeout: float,
    ) -> None:
        self.driver_api = driver_api
        self.config = config
        self.timeout = timeout
        self.driver: Any | None = None
        self.configured = False
        self.arm_mode_requested = False
        self.gripper_mode_requested = False
        self.joint_position_limits: tuple[tuple[float, float], ...] | None = None
        self.policy_limiter: StatefulPositionLimiter | None = None

    def prepare(self) -> HomePreparation:
        robot = self.config["robot"]
        settings = self.config["policy_evaluation"]
        arm_target = tuple(float(value) for value in robot["home_arm_positions_rad"])
        gripper_target = float(robot["home_gripper_position_m"])
        if len(arm_target) != 6 or not all(isfinite(value) for value in arm_target):
            raise ValueError("Dataset home must contain six finite arm targets")
        if not isfinite(gripper_target):
            raise ValueError("Dataset home gripper target must be finite")
        if self.timeout <= 0:
            raise ValueError("Controller timeout must be positive")

        model = getattr(self.driver_api.Model, robot["driver_model"])
        end_effector = getattr(
            self.driver_api.StandardEndEffector, robot["end_effector"]
        )
        self.driver = self.driver_api.TrossenArmDriver()
        try:
            self.driver.configure(
                model,
                end_effector,
                robot["controller_ip"],
                False,
                self.timeout,
            )
            self.configured = True
            driver_version, firmware_version = validate_versions(self.driver, robot)
            apply_motor_parameters(self.driver_api, self.driver, robot)
            error_before = str(self.driver.get_error_information())
            if error_before.lower() != "no error":
                raise RuntimeError(f"Controller reports an error: {error_before}")
            modes = [
                getattr(mode, "value", str(mode)) for mode in self.driver.get_modes()
            ]
            if len(modes) != 7 or not all(mode in (0, "idle") for mode in modes):
                raise RuntimeError(f"All joints must start idle; got {modes}")

            limits = self.driver.get_joint_limits()
            targets = (*arm_target, gripper_target)
            if len(limits) != 7:
                raise RuntimeError("Expected seven controller joint limits")
            for index, (target, limit) in enumerate(zip(targets, limits, strict=True)):
                if not float(limit.position_min) <= target <= float(limit.position_max):
                    raise RuntimeError(
                        f"Dataset home target {index}={target} is outside limits"
                    )
            self.joint_position_limits = tuple(
                (float(limit.position_min), float(limit.position_max))
                for limit in limits
            )

            self.driver.set_arm_modes(self.driver_api.Mode.position)
            self.arm_mode_requested = True
            self.driver.set_gripper_mode(self.driver_api.Mode.position)
            self.gripper_mode_requested = True
            self.driver.set_arm_positions(
                list(arm_target),
                float(settings["home_goal_time_s"]),
                True,
            )
            self.driver.set_gripper_position(
                gripper_target,
                float(settings["gripper_goal_time_s"]),
                True,
            )
            observed = self.read_positions()
            arm_error = max(
                abs(actual - target)
                for actual, target in zip(observed[:6], arm_target, strict=True)
            )
            gripper_error = abs(observed[6] - gripper_target)
            if arm_error > float(settings["max_home_tracking_error_rad"]):
                raise RuntimeError(f"Home arm error {arm_error:.6f} rad exceeds limit")
            if gripper_error > float(settings["max_home_gripper_error_m"]):
                raise RuntimeError(
                    f"Home gripper error {gripper_error:.6f} m exceeds limit"
                )
            error_after = str(self.driver.get_error_information())
            if error_after.lower() != "no error":
                raise RuntimeError(
                    f"Controller reports an error after home: {error_after}"
                )
            return HomePreparation(
                driver_version=driver_version,
                firmware_version=firmware_version,
                target=targets,
                observed=tuple(observed),
                max_arm_error_rad=arm_error,
                gripper_error_m=gripper_error,
            )
        except Exception:
            self.close()
            raise

    def read_positions(self) -> list[float]:
        if self.driver is None or not self.configured:
            raise RuntimeError("Home session is not connected")
        positions = [float(value) for value in self.driver.get_all_positions()]
        if len(positions) != 7 or not all(isfinite(value) for value in positions):
            raise RuntimeError("Controller returned invalid joint positions")
        return positions

    def read_cartesian_positions(self) -> list[float]:
        """Read the six-dimensional Cartesian tool pose."""
        if self.driver is None or not self.configured:
            raise RuntimeError("Home session is not connected")
        positions = [float(value) for value in self.driver.get_cartesian_positions()]
        if len(positions) != 6 or not all(isfinite(value) for value in positions):
            raise RuntimeError("Controller returned invalid Cartesian positions")
        return positions

    def move_cartesian(
        self,
        target: list[float],
        *,
        goal_time_s: float,
        trajectory_check_samples: int,
    ) -> list[float]:
        """Move to one checked Cartesian target and return the observed pose."""
        if self.driver is None or not self.configured:
            raise RuntimeError("Home session is not connected")
        if len(target) != 6 or not all(isfinite(value) for value in target):
            raise ValueError("Cartesian target must contain six finite values")
        if goal_time_s <= 0.2:
            raise ValueError("Cartesian goal time must exceed 0.2 seconds")
        if trajectory_check_samples <= 0:
            raise ValueError("Trajectory check samples must be positive")
        self.driver.set_cartesian_positions(
            target,
            self.driver_api.InterpolationSpace.cartesian,
            goal_time=goal_time_s,
            blocking=True,
            num_trajectory_check_samples=trajectory_check_samples,
        )
        error = str(self.driver.get_error_information())
        if error.lower() != "no error":
            raise RuntimeError(f"Controller reports an error: {error}")
        return self.read_cartesian_positions()

    def execute_bounded_policy_step(
        self,
        raw_target: list[float],
        reference: list[float],
        *,
        absolute_min: list[float] | None = None,
        absolute_max: list[float] | None = None,
    ) -> BoundedPolicyStep:
        """Execute one statefully limited, non-blocking absolute-policy command."""
        if self.driver is None or not self.configured:
            raise RuntimeError("Home session is not connected")
        if len(raw_target) != 7 or not all(isfinite(value) for value in raw_target):
            raise ValueError("Policy target must contain seven finite values")
        if len(reference) != 7 or not all(isfinite(value) for value in reference):
            raise ValueError("Rollout reference must contain seven finite values")
        use_absolute_limits = absolute_min is not None or absolute_max is not None
        if use_absolute_limits:
            if absolute_min is None or absolute_max is None:
                raise ValueError("Both absolute action bounds must be provided")
            if (
                len(absolute_min) != 7
                or len(absolute_max) != 7
                or not all(isfinite(value) for value in (*absolute_min, *absolute_max))
                or any(
                    low > high
                    for low, high in zip(absolute_min, absolute_max, strict=True)
                )
            ):
                raise ValueError(
                    "Absolute action bounds must be seven finite ordered pairs"
                )

        settings = self.config["policy_evaluation"]["clipped_rollout"]
        max_action_delta = [float(value) for value in settings["max_action_delta"]]
        max_command_lead = [float(value) for value in settings["max_command_lead"]]
        max_cumulative_arm_delta = float(
            settings.get("max_cumulative_joint_delta_rad", float("inf"))
        )
        max_cumulative_gripper_delta = float(
            settings.get("max_cumulative_gripper_delta_m", float("inf"))
        )
        control_fps = float(settings["control_fps"])
        min_time_to_move_multiplier = float(settings["min_time_to_move_multiplier"])
        command_blocking = settings["command_blocking"]
        goal_time = min_time_to_move_multiplier / control_fps
        control_period = 1.0 / control_fps
        max_tracking_error = [float(value) for value in settings["max_tracking_error"]]
        if (
            len(max_action_delta) != 7
            or any(value <= 0 for value in max_action_delta)
            or len(max_command_lead) != 7
            or any(value <= 0 for value in max_command_lead)
            or len(max_tracking_error) != 7
            or any(value <= 0 for value in max_tracking_error)
            or (
                not use_absolute_limits
                and max_cumulative_arm_delta < min(max_action_delta[:6])
            )
            or (
                not use_absolute_limits
                and max_cumulative_gripper_delta < max_action_delta[6]
            )
            or control_fps <= 0
            or min_time_to_move_multiplier <= 0
            or goal_time + 1e-12 < control_period
            or command_blocking is not False
        ):
            raise ValueError("Clipped single-step limits and goal time are invalid")

        start = self.read_positions()
        sampled_at = time.monotonic()
        if self.joint_position_limits is None:
            raise RuntimeError("Controller joint limits were not cached during prepare")
        if self.policy_limiter is None:
            if use_absolute_limits:
                configured_low = absolute_min
                configured_high = absolute_max
            else:
                configured_low = [
                    value
                    - (
                        max_cumulative_arm_delta
                        if index < 6
                        else max_cumulative_gripper_delta
                    )
                    for index, value in enumerate(reference)
                ]
                configured_high = [
                    value
                    + (
                        max_cumulative_arm_delta
                        if index < 6
                        else max_cumulative_gripper_delta
                    )
                    for index, value in enumerate(reference)
                ]
            lower_bounds = [
                max(configured, controller[0])
                for configured, controller in zip(
                    configured_low, self.joint_position_limits, strict=True
                )
            ]
            upper_bounds = [
                min(configured, controller[1])
                for configured, controller in zip(
                    configured_high, self.joint_position_limits, strict=True
                )
            ]
            self.policy_limiter = StatefulPositionLimiter(
                start,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                maximum_steps=max_action_delta,
                maximum_leads=max_command_lead,
            )
        limited = self.policy_limiter.limit(raw_target, start)
        commanded = list(limited.commanded)

        # Match Trossen's official LeRobot follower implementation: policy
        # commands are non-blocking so the outer loop can keep its control rate.
        self.driver.set_all_positions(commanded, goal_time, False)
        observed = self.read_positions()
        arm_command_gaps = [
            abs(actual - target)
            for actual, target in zip(observed[:6], commanded[:6], strict=True)
        ]
        arm_command_gap = max(arm_command_gaps)
        gripper_command_gap = abs(observed[6] - commanded[6])
        error_after = str(self.driver.get_error_information())
        if error_after.lower() != "no error":
            raise RuntimeError(f"Controller reports an error: {error_after}")

        return BoundedPolicyStep(
            raw_target=tuple(raw_target),
            start=tuple(start),
            commanded=tuple(commanded),
            observed=tuple(observed),
            max_commanded_arm_delta_rad=max(
                abs(value) for value in limited.command_step[:6]
            ),
            commanded_gripper_delta_m=abs(limited.command_step[6]),
            max_arm_command_lead_rad=max(
                abs(value) for value in limited.command_lead[:6]
            ),
            gripper_command_lead_m=abs(limited.command_lead[6]),
            max_arm_command_gap_rad=arm_command_gap,
            gripper_command_gap_m=gripper_command_gap,
            sampled_at_monotonic=sampled_at,
        )

    def settle_and_verify_policy_target(
        self, target: list[float]
    ) -> PolicyTargetVerification:
        """Let the final non-blocking command finish, then verify tracking."""
        if self.driver is None or not self.configured:
            raise RuntimeError("Home session is not connected")
        if len(target) != 7 or not all(isfinite(value) for value in target):
            raise ValueError("Final policy target must contain seven finite values")
        settings = self.config["policy_evaluation"]["clipped_rollout"]
        goal_time = float(settings["min_time_to_move_multiplier"]) / float(
            settings["control_fps"]
        )
        max_tracking_error = [float(value) for value in settings["max_tracking_error"]]
        if goal_time <= 0.0 or len(max_tracking_error) != 7:
            raise ValueError("Final policy verification settings are invalid")
        time.sleep(goal_time)
        observed = self.read_positions()
        errors = [
            abs(actual - desired)
            for actual, desired in zip(observed, target, strict=True)
        ]
        violations = [
            (index, error, max_tracking_error[index])
            for index, error in enumerate(errors)
            if error > max_tracking_error[index]
        ]
        if violations:
            index, error, limit = violations[0]
            unit = "rad" if index < 6 else "m"
            raise RuntimeError(
                f"Final joint {index} tracking error {error:.6f} {unit} "
                f"exceeds limit {limit:.6f} {unit}"
            )
        error_after = str(self.driver.get_error_information())
        if error_after.lower() != "no error":
            raise RuntimeError(f"Controller reports an error: {error_after}")
        return PolicyTargetVerification(
            target=tuple(target),
            observed=tuple(observed),
            max_arm_tracking_error_rad=max(errors[:6]),
            gripper_tracking_error_m=errors[6],
        )

    def close(self) -> None:
        if self.driver is None:
            return
        first_error: Exception | None = None
        try:
            if self.arm_mode_requested:
                try:
                    self.driver.set_arm_modes(self.driver_api.Mode.idle)
                    self.arm_mode_requested = False
                except Exception as exc:  # noqa: BLE001 - continue safe cleanup
                    first_error = exc
            if self.gripper_mode_requested:
                try:
                    self.driver.set_gripper_mode(self.driver_api.Mode.idle)
                    self.gripper_mode_requested = False
                except Exception as exc:  # noqa: BLE001 - continue safe cleanup
                    if first_error is None:
                        first_error = exc
        finally:
            if self.configured:
                try:
                    self.driver.cleanup(False)
                    self.configured = False
                except Exception as exc:  # noqa: BLE001 - report after all attempts
                    if first_error is None:
                        first_error = exc
        self.policy_limiter = None
        self.joint_position_limits = None
        if first_error is not None:
            raise first_error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
