"""Safety-gated dataset-home staging for shadow policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Self


def _version_series(version: str) -> str:
    parts = version.removeprefix("v").split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise RuntimeError(f"Unrecognized Trossen version: {version!r}")
    return ".".join(parts[:2])


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
            driver_version = str(self.driver.get_driver_version())
            firmware_version = str(self.driver.get_controller_version())
            if _version_series(driver_version) != str(robot["expected_driver_series"]):
                raise RuntimeError(f"Unexpected driver version {driver_version}")
            if _version_series(firmware_version) != str(
                robot["expected_firmware_series"]
            ):
                raise RuntimeError(f"Unexpected firmware version {firmware_version}")
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

    def execute_bounded_policy_step(
        self,
        raw_target: list[float],
    ) -> BoundedPolicyStep:
        """Execute exactly one policy step after clipping it around current state."""
        if self.driver is None or not self.configured:
            raise RuntimeError("Home session is not connected")
        if len(raw_target) != 7 or not all(isfinite(value) for value in raw_target):
            raise ValueError("Policy target must contain seven finite values")

        settings = self.config["policy_evaluation"]["clipped_single_step"]
        max_arm_delta = float(settings["max_joint_delta_rad"])
        max_gripper_delta = float(settings["max_gripper_delta_m"])
        goal_time = float(settings["goal_time_s"])
        max_arm_tracking_error = float(settings["max_arm_tracking_error_rad"])
        max_gripper_tracking_error = float(settings["max_gripper_tracking_error_m"])
        if max_arm_delta <= 0 or max_gripper_delta <= 0 or goal_time <= 0.2:
            raise ValueError("Clipped single-step limits and goal time are invalid")

        start = self.read_positions()
        limits = self.driver.get_joint_limits()
        if len(limits) != 7:
            raise RuntimeError("Expected seven controller joint limits")
        commanded: list[float] = []
        for index, (current, target, limit) in enumerate(
            zip(start, raw_target, limits, strict=True)
        ):
            maximum_delta = max_arm_delta if index < 6 else max_gripper_delta
            delta = max(-maximum_delta, min(maximum_delta, target - current))
            bounded = current + delta
            bounded = max(float(limit.position_min), bounded)
            bounded = min(float(limit.position_max), bounded)
            commanded.append(bounded)

        self.driver.set_arm_positions(commanded[:6], goal_time, True)
        self.driver.set_gripper_position(commanded[6], goal_time, True)
        observed = self.read_positions()
        arm_tracking_error = max(
            abs(actual - target)
            for actual, target in zip(observed[:6], commanded[:6], strict=True)
        )
        gripper_tracking_error = abs(observed[6] - commanded[6])
        if arm_tracking_error > max_arm_tracking_error:
            raise RuntimeError(
                f"Single-step arm tracking error {arm_tracking_error:.6f} rad "
                "exceeds limit"
            )
        if gripper_tracking_error > max_gripper_tracking_error:
            raise RuntimeError(
                f"Single-step gripper tracking error {gripper_tracking_error:.6f} m "
                "exceeds limit"
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
