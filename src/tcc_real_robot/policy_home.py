"""Safety-gated dataset-home staging for shadow policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Self

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
        """Execute exactly one policy step after clipping it around current state."""
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
                or any(low > high for low, high in zip(absolute_min, absolute_max, strict=True))
            ):
                raise ValueError("Absolute action bounds must be seven finite ordered pairs")

        settings = self.config["policy_evaluation"]["clipped_rollout"]
        max_action_delta = [float(value) for value in settings["max_action_delta"]]
        max_cumulative_arm_delta = float(
            settings.get("max_cumulative_joint_delta_rad", float("inf"))
        )
        max_cumulative_gripper_delta = float(
            settings.get("max_cumulative_gripper_delta_m", float("inf"))
        )
        control_fps = float(settings["control_fps"])
        min_time_to_move_multiplier = float(
            settings["min_time_to_move_multiplier"]
        )
        goal_time = min_time_to_move_multiplier / control_fps
        max_tracking_error = [
            float(value) for value in settings["max_tracking_error"]
        ]
        if (
            len(max_action_delta) != 7
            or any(value <= 0 for value in max_action_delta)
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
            or goal_time <= 0.2
        ):
            raise ValueError("Clipped single-step limits and goal time are invalid")

        start = self.read_positions()
        limits = self.driver.get_joint_limits()
        if len(limits) != 7:
            raise RuntimeError("Expected seven controller joint limits")
        commanded: list[float] = []
        for index, (current, target, limit) in enumerate(
            zip(start, raw_target, limits, strict=True)
        ):
            maximum_delta = max_action_delta[index]
            maximum_cumulative_delta = (
                max_cumulative_arm_delta
                if index < 6
                else max_cumulative_gripper_delta
            )
            delta = max(-maximum_delta, min(maximum_delta, target - current))
            bounded = current + delta
            if use_absolute_limits:
                bounded = max(absolute_min[index], bounded)
                bounded = min(absolute_max[index], bounded)
            else:
                bounded = max(reference[index] - maximum_cumulative_delta, bounded)
                bounded = min(reference[index] + maximum_cumulative_delta, bounded)
            bounded = max(float(limit.position_min), bounded)
            bounded = min(float(limit.position_max), bounded)
            commanded.append(bounded)

        self.driver.set_all_positions(commanded, goal_time, True)
        observed = self.read_positions()
        arm_tracking_errors = [
            abs(actual - target)
            for actual, target in zip(observed[:6], commanded[:6], strict=True)
        ]
        arm_tracking_error = max(arm_tracking_errors)
        gripper_tracking_error = abs(observed[6] - commanded[6])
        violations = [
            (index, error, max_tracking_error[index])
            for index, error in enumerate(arm_tracking_errors)
            if error > max_tracking_error[index]
        ]
        if violations:
            index, error, limit = violations[0]
            raise RuntimeError(
                f"Joint {index} tracking error {error:.6f} rad exceeds "
                f"dataset limit {limit:.6f} rad"
            )
        if gripper_tracking_error > max_tracking_error[6]:
            raise RuntimeError(
                f"Single-step gripper tracking error {gripper_tracking_error:.6f} m "
                f"exceeds dataset limit {max_tracking_error[6]:.6f} m"
            )
        error_after = str(self.driver.get_error_information())
        if error_after.lower() != "no error":
            raise RuntimeError(f"Controller reports an error: {error_after}")

        return BoundedPolicyStep(
            raw_target=tuple(raw_target),
            start=tuple(start),
            commanded=tuple(commanded),
            observed=tuple(observed),
            max_commanded_arm_delta_rad=max(
                abs(target - current)
                for target, current in zip(commanded[:6], start[:6], strict=True)
            ),
            commanded_gripper_delta_m=abs(commanded[6] - start[6]),
            max_arm_tracking_error_rad=arm_tracking_error,
            gripper_tracking_error_m=gripper_tracking_error,
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
        if first_error is not None:
            raise first_error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
