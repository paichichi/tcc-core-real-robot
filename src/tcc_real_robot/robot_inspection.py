"""Read-only inspection helpers for a Trossen follower arm."""

from __future__ import annotations

import time
from collections.abc import Callable
from math import isfinite
from typing import Any


def _version_series(version: str) -> str:
    """Return the major.minor portion of a version such as ``v1.9.2``."""
    parts = version.removeprefix("v").split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise RuntimeError(f"Unrecognized Trossen version: {version!r}")
    return ".".join(parts[:2])


def _validate_versions(driver: Any, robot: dict[str, Any]) -> tuple[str, str]:
    driver_version = str(driver.get_driver_version())
    firmware_version = str(driver.get_controller_version())
    expected_driver = str(robot["expected_driver_series"])
    expected_firmware = str(robot["expected_firmware_series"])

    if _version_series(driver_version) != expected_driver:
        raise RuntimeError(
            f"Driver {driver_version} does not match expected {expected_driver}.x"
        )
    if _version_series(firmware_version) != expected_firmware:
        raise RuntimeError(
            f"Firmware {firmware_version} does not match expected "
            f"{expected_firmware}.x"
        )
    return driver_version, firmware_version


def inspect_robot(driver_api: Any, config: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Connect, read state, and clean up without changing modes or commanding motion."""
    robot = config["robot"]
    model = getattr(driver_api.Model, robot["driver_model"])
    end_effector = getattr(driver_api.StandardEndEffector, robot["end_effector"])

    driver = driver_api.TrossenArmDriver()
    configured = False
    try:
        # Positional arguments mirror the vendor's configure_cleanup Python demo.
        # clear_error=False is intentional: inspection must not erase a fault.
        driver.configure(
            model,
            end_effector,
            robot["controller_ip"],
            False,
            timeout,
        )
        configured = True

        driver_version, firmware_version = _validate_versions(driver, robot)

        modes = [getattr(mode, "value", str(mode)) for mode in driver.get_modes()]
        return {
            "controller_ip": robot["controller_ip"],
            "driver_version": driver_version,
            "firmware_version": firmware_version,
            "num_joints": int(driver.get_num_joints()),
            "modes": modes,
            "arm_positions_rad": list(driver.get_arm_positions()),
            "gripper_position_m": float(driver.get_gripper_position()),
        }
    finally:
        if configured:
            driver.cleanup(False)


def monitor_robot(
    driver_api: Any,
    config: dict[str, Any],
    timeout: float,
    duration: float,
    rate_hz: float,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Read joint state at a fixed rate without changing modes or commanding motion."""
    robot = config["robot"]
    model = getattr(driver_api.Model, robot["driver_model"])
    end_effector = getattr(driver_api.StandardEndEffector, robot["end_effector"])
    driver = driver_api.TrossenArmDriver()
    configured = False
    sample_count = max(1, round(duration * rate_hz))
    period = 1.0 / rate_hz
    started = 0.0
    first_arm: list[float] | None = None
    last_arm: list[float] = []
    first_gripper: float | None = None
    last_gripper = 0.0

    try:
        driver.configure(
            model,
            end_effector,
            robot["controller_ip"],
            False,
            timeout,
        )
        configured = True
        driver_version, firmware_version = _validate_versions(driver, robot)
        started = time.monotonic()

        for index in range(sample_count):
            arm = list(driver.get_arm_positions())
            gripper = float(driver.get_gripper_position())
            if first_arm is None:
                first_arm = arm
                first_gripper = gripper
            last_arm = arm
            last_gripper = gripper
            sample = {
                "sample": index + 1,
                "elapsed_s": time.monotonic() - started,
                "arm_positions_rad": arm,
                "gripper_position_m": gripper,
            }
            if on_sample is not None:
                on_sample(sample)

            deadline = started + (index + 1) * period
            remaining = deadline - time.monotonic()
            if remaining > 0 and index + 1 < sample_count:
                time.sleep(remaining)

        elapsed = time.monotonic() - started
        assert first_arm is not None and first_gripper is not None
        return {
            "controller_ip": robot["controller_ip"],
            "driver_version": driver_version,
            "firmware_version": firmware_version,
            "samples": sample_count,
            "elapsed_s": elapsed,
            "observed_rate_hz": (
                (sample_count - 1) / elapsed if sample_count > 1 and elapsed > 0 else 0.0
            ),
            "max_arm_change_rad": max(
                abs(end - start) for start, end in zip(first_arm, last_arm, strict=True)
            ),
            "gripper_change_m": last_gripper - first_gripper,
        }
    finally:
        if configured:
            driver.cleanup(False)


def preflight_robot(
    driver_api: Any, config: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Collect and validate a read-only safety baseline from the controller."""
    robot = config["robot"]
    model = getattr(driver_api.Model, robot["driver_model"])
    end_effector = getattr(driver_api.StandardEndEffector, robot["end_effector"])
    driver = driver_api.TrossenArmDriver()
    configured = False

    try:
        driver.configure(
            model,
            end_effector,
            robot["controller_ip"],
            False,
            timeout,
        )
        configured = True
        driver_version, firmware_version = _validate_versions(driver, robot)

        modes = [getattr(mode, "value", str(mode)) for mode in driver.get_modes()]
        positions = [float(value) for value in driver.get_all_positions()]
        rotor_temperatures = [
            float(value) for value in driver.get_all_rotor_temperatures()
        ]
        driver_temperatures = [
            float(value) for value in driver.get_all_driver_temperatures()
        ]
        cartesian_positions = [
            float(value) for value in driver.get_cartesian_positions()
        ]
        limits = []
        for index, limit in enumerate(driver.get_joint_limits()):
            limits.append(
                {
                    "joint": index,
                    "position_min": float(limit.position_min),
                    "position_max": float(limit.position_max),
                    "position_tolerance": float(limit.position_tolerance),
                    "velocity_max": float(limit.velocity_max),
                    "velocity_tolerance": float(limit.velocity_tolerance),
                    "effort_max": float(limit.effort_max),
                    "effort_tolerance": float(limit.effort_tolerance),
                }
            )

        checks: dict[str, bool] = {
            "joint_count_is_7": len(positions) == 7 and len(limits) == 7,
            "all_modes_idle": len(modes) == 7
            and all(mode in (0, "idle") for mode in modes),
            "positions_finite": all(isfinite(value) for value in positions),
            "temperatures_finite": all(
                isfinite(value) for value in rotor_temperatures + driver_temperatures
            ),
            "cartesian_position_finite": all(
                isfinite(value) for value in cartesian_positions
            ),
            "limits_finite_and_ordered": all(
                isfinite(limit["position_min"])
                and isfinite(limit["position_max"])
                and limit["position_min"] <= limit["position_max"]
                for limit in limits
            ),
            "positions_within_limits_and_tolerance": len(positions) == len(limits)
            and all(
                limit["position_min"] - limit["position_tolerance"]
                <= position
                <= limit["position_max"] + limit["position_tolerance"]
                for position, limit in zip(positions, limits, strict=True)
            ),
        }

        return {
            "passed": all(checks.values()),
            "checks": checks,
            "controller_ip": robot["controller_ip"],
            "driver_version": driver_version,
            "firmware_version": firmware_version,
            "error_information": str(driver.get_error_information()),
            "modes": modes,
            "positions": positions,
            "joint_limits": limits,
            "rotor_temperatures": rotor_temperatures,
            "driver_temperatures": driver_temperatures,
            "cartesian_positions": cartesian_positions,
        }
    finally:
        if configured:
            driver.cleanup(False)


def run_position_hold_test(
    driver_api: Any,
    config: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Briefly enter position mode without sending a position target."""
    robot = config["robot"]
    settings = config["diagnostic_tests"]["position_hold"]
    duration = float(settings["duration_s"])
    rate_hz = float(settings["sample_rate_hz"])
    max_arm_drift = float(settings["max_arm_drift_rad"])
    max_gripper_drift = float(settings["max_gripper_drift_m"])
    if min(duration, rate_hz, max_arm_drift, max_gripper_drift) <= 0:
        raise ValueError("Position-hold diagnostic settings must be positive")

    model = getattr(driver_api.Model, robot["driver_model"])
    end_effector = getattr(driver_api.StandardEndEffector, robot["end_effector"])
    driver = driver_api.TrossenArmDriver()
    configured = False
    position_mode_requested = False

    try:
        driver.configure(
            model,
            end_effector,
            robot["controller_ip"],
            False,
            timeout,
        )
        configured = True
        driver_version, firmware_version = _validate_versions(driver, robot)
        error_information = str(driver.get_error_information())
        modes_before = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if error_information != "No error":
            raise RuntimeError(f"Controller reports an error: {error_information}")
        if len(modes_before) != 7 or not all(
            mode in (0, "idle") for mode in modes_before
        ):
            raise RuntimeError(f"All joints must start idle; got {modes_before}")

        initial = [float(value) for value in driver.get_all_positions()]
        if len(initial) != 7 or not all(isfinite(value) for value in initial):
            raise RuntimeError("Expected seven finite initial joint positions")

        driver.set_all_modes(driver_api.Mode.position)
        position_mode_requested = True
        modes_during = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        position_value = getattr(driver_api.Mode.position, "value", 1)
        if len(modes_during) != 7 or not all(
            mode == position_value for mode in modes_during
        ):
            raise RuntimeError(f"Position mode was not applied: {modes_during}")

        samples = max(1, round(duration * rate_hz))
        period = 1.0 / rate_hz
        started = time.monotonic()
        peak_arm_drift = 0.0
        peak_gripper_drift = 0.0
        final = initial
        for index in range(samples):
            final = [float(value) for value in driver.get_all_positions()]
            if len(final) != 7 or not all(isfinite(value) for value in final):
                raise RuntimeError("Received invalid joint positions during hold")
            peak_arm_drift = max(
                peak_arm_drift,
                max(
                    abs(current - start)
                    for start, current in zip(initial[:6], final[:6], strict=True)
                ),
            )
            peak_gripper_drift = max(
                peak_gripper_drift, abs(final[6] - initial[6])
            )
            if peak_arm_drift > max_arm_drift:
                raise RuntimeError(
                    f"Arm drift {peak_arm_drift:.6f} rad exceeded "
                    f"{max_arm_drift:.6f} rad"
                )
            if peak_gripper_drift > max_gripper_drift:
                raise RuntimeError(
                    f"Gripper drift {peak_gripper_drift:.6f} m exceeded "
                    f"{max_gripper_drift:.6f} m"
                )
            deadline = started + (index + 1) * period
            remaining = deadline - time.monotonic()
            if remaining > 0 and index + 1 < samples:
                time.sleep(remaining)

        driver.set_all_modes(driver_api.Mode.idle)
        position_mode_requested = False
        modes_after = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        passed = len(modes_after) == 7 and all(
            mode in (0, "idle") for mode in modes_after
        )
        return {
            "passed": passed,
            "controller_ip": robot["controller_ip"],
            "driver_version": driver_version,
            "firmware_version": firmware_version,
            "error_information": error_information,
            "duration_s": duration,
            "sample_rate_hz": rate_hz,
            "samples": samples,
            "modes_before": modes_before,
            "modes_during": modes_during,
            "modes_after": modes_after,
            "initial_positions": initial,
            "final_positions": final,
            "peak_arm_drift_rad": peak_arm_drift,
            "peak_gripper_drift_m": peak_gripper_drift,
            "max_allowed_arm_drift_rad": max_arm_drift,
            "max_allowed_gripper_drift_m": max_gripper_drift,
        }
    finally:
        if configured:
            try:
                if position_mode_requested:
                    driver.set_all_modes(driver_api.Mode.idle)
            finally:
                driver.cleanup(False)


def run_current_position_hold_test(
    driver_api: Any,
    config: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Read the current pose, command that same pose, and measure tracking error."""
    robot = config["robot"]
    settings = config["diagnostic_tests"]["current_position_hold"]
    goal_time = float(settings["goal_time_s"])
    duration = float(settings["duration_s"])
    rate_hz = float(settings["sample_rate_hz"])
    max_arm_error = float(settings["max_arm_error_rad"])
    max_gripper_error = float(settings["max_gripper_error_m"])
    if min(
        goal_time,
        duration,
        rate_hz,
        max_arm_error,
        max_gripper_error,
    ) <= 0:
        raise ValueError("Current-position hold settings must be positive")

    model = getattr(driver_api.Model, robot["driver_model"])
    end_effector = getattr(driver_api.StandardEndEffector, robot["end_effector"])
    driver = driver_api.TrossenArmDriver()
    configured = False
    position_mode_requested = False

    try:
        driver.configure(
            model,
            end_effector,
            robot["controller_ip"],
            False,
            timeout,
        )
        configured = True
        driver_version, firmware_version = _validate_versions(driver, robot)
        error_information = str(driver.get_error_information())
        modes_before = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if error_information != "No error":
            raise RuntimeError(f"Controller reports an error: {error_information}")
        if len(modes_before) != 7 or not all(
            mode in (0, "idle") for mode in modes_before
        ):
            raise RuntimeError(f"All joints must start idle; got {modes_before}")

        target = [float(value) for value in driver.get_all_positions()]
        if len(target) != 7 or not all(isfinite(value) for value in target):
            raise RuntimeError("Expected seven finite initial joint positions")

        limits = driver.get_joint_limits()
        if len(limits) != 7 or not all(
            float(limit.position_min) - float(limit.position_tolerance)
            <= position
            <= float(limit.position_max) + float(limit.position_tolerance)
            for position, limit in zip(target, limits, strict=True)
        ):
            raise RuntimeError("Initial positions are outside controller limits")

        driver.set_all_modes(driver_api.Mode.position)
        position_mode_requested = True
        driver.set_all_positions(target, goal_time, True)

        modes_during = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        position_value = getattr(driver_api.Mode.position, "value", 1)
        if len(modes_during) != 7 or not all(
            mode == position_value for mode in modes_during
        ):
            raise RuntimeError(f"Position mode was not applied: {modes_during}")

        samples = max(1, round(duration * rate_hz))
        period = 1.0 / rate_hz
        started = time.monotonic()
        peak_joint_errors = [0.0] * 7
        samples_observed = 0
        failure_reasons: list[str] = []
        final = target
        for index in range(samples):
            final = [float(value) for value in driver.get_all_positions()]
            if len(final) != 7 or not all(isfinite(value) for value in final):
                raise RuntimeError("Received invalid joint positions during hold")
            samples_observed += 1
            for joint, (goal, current) in enumerate(
                zip(target, final, strict=True)
            ):
                peak_joint_errors[joint] = max(
                    peak_joint_errors[joint], abs(current - goal)
                )
            peak_arm_error = max(peak_joint_errors[:6])
            peak_gripper_error = peak_joint_errors[6]
            if peak_arm_error > max_arm_error:
                failure_reasons.append(
                    f"Arm error {peak_arm_error:.6f} rad exceeded "
                    f"{max_arm_error:.6f} rad"
                )
            if peak_gripper_error > max_gripper_error:
                failure_reasons.append(
                    f"Gripper error {peak_gripper_error:.6f} m exceeded "
                    f"{max_gripper_error:.6f} m"
                )
            if failure_reasons:
                break
            deadline = started + (index + 1) * period
            remaining = deadline - time.monotonic()
            if remaining > 0 and index + 1 < samples:
                time.sleep(remaining)

        driver.set_all_modes(driver_api.Mode.idle)
        position_mode_requested = False
        modes_after = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        idle_restored = len(modes_after) == 7 and all(
            mode in (0, "idle") for mode in modes_after
        )
        if not idle_restored:
            failure_reasons.append(f"Idle mode was not restored: {modes_after}")
        error_information_after = str(driver.get_error_information())
        if error_information_after != "No error":
            failure_reasons.append(
                f"Controller reported an error after the test: "
                f"{error_information_after}"
            )
        return {
            "passed": not failure_reasons,
            "failure_reasons": failure_reasons,
            "controller_ip": robot["controller_ip"],
            "driver_version": driver_version,
            "firmware_version": firmware_version,
            "error_information": error_information,
            "error_information_after": error_information_after,
            "goal_time_s": goal_time,
            "duration_s": duration,
            "sample_rate_hz": rate_hz,
            "samples": samples,
            "samples_observed": samples_observed,
            "modes_before": modes_before,
            "modes_during": modes_during,
            "modes_after": modes_after,
            "target_positions": target,
            "final_positions": final,
            "peak_arm_error_rad": peak_arm_error,
            "peak_gripper_error_m": peak_gripper_error,
            "peak_joint_errors": peak_joint_errors,
            "max_allowed_arm_error_rad": max_arm_error,
            "max_allowed_gripper_error_m": max_gripper_error,
        }
    finally:
        if configured:
            try:
                if position_mode_requested:
                    driver.set_all_modes(driver_api.Mode.idle)
            finally:
                driver.cleanup(False)
