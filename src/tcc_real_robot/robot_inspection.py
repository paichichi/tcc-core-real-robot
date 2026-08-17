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


def run_gripper_cycle_test(
    driver_api: Any,
    config: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Open the gripper by a small amount, return it, and keep the arm idle."""
    robot = config["robot"]
    settings = config["diagnostic_tests"]["gripper_cycle"]
    open_delta = float(settings["open_delta_m"])
    goal_time = float(settings["goal_time_s"])
    hold_duration = float(settings["hold_duration_s"])
    rate_hz = float(settings["sample_rate_hz"])
    max_tracking_error = float(settings["max_tracking_error_m"])
    if min(
        open_delta,
        goal_time,
        hold_duration,
        rate_hz,
        max_tracking_error,
    ) <= 0:
        raise ValueError("Gripper-cycle settings must be positive")

    model = getattr(driver_api.Model, robot["driver_model"])
    end_effector = getattr(driver_api.StandardEndEffector, robot["end_effector"])
    driver = driver_api.TrossenArmDriver()
    configured = False
    gripper_position_mode_requested = False

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
        error_before = str(driver.get_error_information())
        modes_before = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if error_before != "No error":
            raise RuntimeError(f"Controller reports an error: {error_before}")
        if len(modes_before) != 7 or not all(
            mode in (0, "idle") for mode in modes_before
        ):
            raise RuntimeError(f"All joints must start idle; got {modes_before}")

        limits = driver.get_joint_limits()
        if len(limits) != 7:
            raise RuntimeError("Expected seven controller joint limits")
        gripper_limit = limits[6]
        position_min = float(gripper_limit.position_min)
        position_max = float(gripper_limit.position_max)
        tolerance = float(gripper_limit.position_tolerance)
        start = float(driver.get_gripper_position())
        if not all(
            isfinite(value)
            for value in (position_min, position_max, tolerance, start)
        ):
            raise RuntimeError("Received non-finite gripper state or limits")
        if not position_min - tolerance <= start <= position_max + tolerance:
            raise RuntimeError("Initial gripper position is outside controller limits")

        return_target = min(max(start, position_min), position_max)
        open_target = return_target + open_delta
        if open_target > position_max:
            raise RuntimeError(
                f"Requested gripper target {open_target:.6f} m exceeds "
                f"limit {position_max:.6f} m"
            )

        driver.set_gripper_mode(driver_api.Mode.position)
        gripper_position_mode_requested = True
        modes_during = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        position_value = getattr(driver_api.Mode.position, "value", 1)
        if len(modes_during) != 7 or not (
            all(mode in (0, "idle") for mode in modes_during[:6])
            and modes_during[6] == position_value
        ):
            raise RuntimeError(
                f"Expected arm idle and gripper position mode; got {modes_during}"
            )

        driver.set_gripper_position(open_target, goal_time, True)
        samples = max(1, round(hold_duration * rate_hz))
        period = 1.0 / rate_hz
        started = time.monotonic()
        peak_open_error = 0.0
        observed_open = start
        for index in range(samples):
            observed_open = float(driver.get_gripper_position())
            if not isfinite(observed_open):
                raise RuntimeError("Received a non-finite gripper position")
            peak_open_error = max(
                peak_open_error, abs(observed_open - open_target)
            )
            deadline = started + (index + 1) * period
            remaining = deadline - time.monotonic()
            if remaining > 0 and index + 1 < samples:
                time.sleep(remaining)

        failure_reasons: list[str] = []
        if peak_open_error > max_tracking_error:
            failure_reasons.append(
                f"Open tracking error {peak_open_error:.6f} m exceeded "
                f"{max_tracking_error:.6f} m"
            )

        error_after_open = str(driver.get_error_information())
        return_command_sent = error_after_open == "No error"
        if return_command_sent:
            driver.set_gripper_position(return_target, goal_time, True)
        else:
            failure_reasons.append(
                f"Controller error after opening: {error_after_open}"
            )

        observed_return = float(driver.get_gripper_position())
        if not isfinite(observed_return):
            raise RuntimeError("Received a non-finite return position")
        return_error = abs(observed_return - return_target)
        if return_command_sent and return_error > max_tracking_error:
            failure_reasons.append(
                f"Return tracking error {return_error:.6f} m exceeded "
                f"{max_tracking_error:.6f} m"
            )

        driver.set_gripper_mode(driver_api.Mode.idle)
        gripper_position_mode_requested = False
        modes_after = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if len(modes_after) != 7 or not all(
            mode in (0, "idle") for mode in modes_after
        ):
            failure_reasons.append(f"Idle mode was not restored: {modes_after}")
        error_after = str(driver.get_error_information())
        if error_after != "No error":
            failure_reasons.append(
                f"Controller reported an error after the test: {error_after}"
            )

        return {
            "passed": not failure_reasons,
            "failure_reasons": failure_reasons,
            "controller_ip": robot["controller_ip"],
            "driver_version": driver_version,
            "firmware_version": firmware_version,
            "error_before": error_before,
            "error_after_open": error_after_open,
            "error_after": error_after,
            "goal_time_s": goal_time,
            "hold_duration_s": hold_duration,
            "sample_rate_hz": rate_hz,
            "samples": samples,
            "modes_before": modes_before,
            "modes_during": modes_during,
            "modes_after": modes_after,
            "position_min_m": position_min,
            "position_max_m": position_max,
            "initial_position_m": start,
            "open_delta_m": open_delta,
            "open_target_m": open_target,
            "observed_open_m": observed_open,
            "peak_open_error_m": peak_open_error,
            "return_target_m": return_target,
            "return_command_sent": return_command_sent,
            "observed_return_m": observed_return,
            "return_error_m": return_error,
            "max_tracking_error_m": max_tracking_error,
        }
    finally:
        if configured:
            try:
                if gripper_position_mode_requested:
                    driver.set_gripper_mode(driver_api.Mode.idle)
            finally:
                driver.cleanup(False)


def run_whole_arm_cycle_test(
    driver_api: Any,
    config: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Move all six arm joints by bounded deltas and return; keep gripper idle."""
    robot = config["robot"]
    settings = config["diagnostic_tests"]["whole_arm_cycle"]
    deltas = [float(value) for value in settings["joint_deltas_rad"]]
    goal_time = float(settings["goal_time_s"])
    hold_duration = float(settings["hold_duration_s"])
    rate_hz = float(settings["sample_rate_hz"])
    max_tracking_error = float(settings["max_tracking_error_rad"])
    limit_margin = float(settings["joint_limit_margin_rad"])
    if len(deltas) != 6 or not all(isfinite(value) for value in deltas):
        raise ValueError("Whole-arm cycle requires six finite joint deltas")
    if max(abs(value) for value in deltas) > 0.03:
        raise ValueError("Whole-arm joint deltas may not exceed 0.03 rad")
    if min(goal_time, hold_duration, rate_hz, max_tracking_error) <= 0:
        raise ValueError("Whole-arm timing and error settings must be positive")
    if limit_margin < 0:
        raise ValueError("Whole-arm joint-limit margin may not be negative")

    model = getattr(driver_api.Model, robot["driver_model"])
    end_effector = getattr(driver_api.StandardEndEffector, robot["end_effector"])
    driver = driver_api.TrossenArmDriver()
    configured = False
    arm_position_mode_requested = False

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
        error_before = str(driver.get_error_information())
        modes_before = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if error_before != "No error":
            raise RuntimeError(f"Controller reports an error: {error_before}")
        if len(modes_before) != 7 or not all(
            mode in (0, "idle") for mode in modes_before
        ):
            raise RuntimeError(f"All joints must start idle; got {modes_before}")

        start = [float(value) for value in driver.get_arm_positions()]
        limits = driver.get_joint_limits()
        if len(start) != 6 or not all(isfinite(value) for value in start):
            raise RuntimeError("Expected six finite initial arm positions")
        if len(limits) != 7:
            raise RuntimeError("Expected seven controller joint limits")

        return_target: list[float] = []
        outbound_target: list[float] = []
        for joint, (position, delta, limit) in enumerate(
            zip(start, deltas, limits[:6], strict=True)
        ):
            position_min = float(limit.position_min)
            position_max = float(limit.position_max)
            tolerance = float(limit.position_tolerance)
            if not position_min - tolerance <= position <= position_max + tolerance:
                raise RuntimeError(f"Joint {joint} starts outside controller limits")
            safe_return = min(max(position, position_min), position_max)
            target = safe_return + delta
            if not (
                position_min + limit_margin
                <= target
                <= position_max - limit_margin
            ):
                raise RuntimeError(
                    f"Joint {joint} target {target:.6f} rad violates "
                    f"the {limit_margin:.6f} rad limit margin"
                )
            return_target.append(safe_return)
            outbound_target.append(target)

        driver.set_arm_modes(driver_api.Mode.position)
        arm_position_mode_requested = True
        modes_during = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        position_value = getattr(driver_api.Mode.position, "value", 1)
        if len(modes_during) != 7 or not (
            all(mode == position_value for mode in modes_during[:6])
            and modes_during[6] in (0, "idle")
        ):
            raise RuntimeError(
                f"Expected arm position mode and gripper idle; got {modes_during}"
            )

        driver.set_arm_positions(outbound_target, goal_time, True)
        samples = max(1, round(hold_duration * rate_hz))
        period = 1.0 / rate_hz
        started = time.monotonic()
        peak_outbound_errors = [0.0] * 6
        observed_outbound = start
        for index in range(samples):
            observed_outbound = [
                float(value) for value in driver.get_arm_positions()
            ]
            if len(observed_outbound) != 6 or not all(
                isfinite(value) for value in observed_outbound
            ):
                raise RuntimeError("Received invalid arm positions")
            for joint, (target, observed) in enumerate(
                zip(outbound_target, observed_outbound, strict=True)
            ):
                peak_outbound_errors[joint] = max(
                    peak_outbound_errors[joint], abs(observed - target)
                )
            deadline = started + (index + 1) * period
            remaining = deadline - time.monotonic()
            if remaining > 0 and index + 1 < samples:
                time.sleep(remaining)

        failure_reasons: list[str] = []
        peak_outbound_error = max(peak_outbound_errors)
        if peak_outbound_error > max_tracking_error:
            failure_reasons.append(
                f"Outbound tracking error {peak_outbound_error:.6f} rad exceeded "
                f"{max_tracking_error:.6f} rad"
            )
        error_after_outbound = str(driver.get_error_information())
        return_command_sent = error_after_outbound == "No error"
        if return_command_sent:
            driver.set_arm_positions(return_target, goal_time, True)
        else:
            failure_reasons.append(
                f"Controller error after outbound move: {error_after_outbound}"
            )

        observed_return = [float(value) for value in driver.get_arm_positions()]
        if len(observed_return) != 6 or not all(
            isfinite(value) for value in observed_return
        ):
            raise RuntimeError("Received invalid arm return positions")
        return_errors = [
            abs(observed - target)
            for target, observed in zip(
                return_target, observed_return, strict=True
            )
        ]
        peak_return_error = max(return_errors)
        if return_command_sent and peak_return_error > max_tracking_error:
            failure_reasons.append(
                f"Return tracking error {peak_return_error:.6f} rad exceeded "
                f"{max_tracking_error:.6f} rad"
            )

        driver.set_arm_modes(driver_api.Mode.idle)
        arm_position_mode_requested = False
        modes_after = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if len(modes_after) != 7 or not all(
            mode in (0, "idle") for mode in modes_after
        ):
            failure_reasons.append(f"Idle mode was not restored: {modes_after}")
        error_after = str(driver.get_error_information())
        if error_after != "No error":
            failure_reasons.append(
                f"Controller reported an error after the test: {error_after}"
            )

        return {
            "passed": not failure_reasons,
            "failure_reasons": failure_reasons,
            "controller_ip": robot["controller_ip"],
            "driver_version": driver_version,
            "firmware_version": firmware_version,
            "error_before": error_before,
            "error_after_outbound": error_after_outbound,
            "error_after": error_after,
            "goal_time_s": goal_time,
            "hold_duration_s": hold_duration,
            "sample_rate_hz": rate_hz,
            "samples": samples,
            "joint_deltas_rad": deltas,
            "limit_margin_rad": limit_margin,
            "max_tracking_error_rad": max_tracking_error,
            "modes_before": modes_before,
            "modes_during": modes_during,
            "modes_after": modes_after,
            "initial_positions_rad": start,
            "outbound_target_rad": outbound_target,
            "observed_outbound_rad": observed_outbound,
            "peak_outbound_errors_rad": peak_outbound_errors,
            "return_target_rad": return_target,
            "return_command_sent": return_command_sent,
            "observed_return_rad": observed_return,
            "return_errors_rad": return_errors,
        }
    finally:
        if configured:
            try:
                if arm_position_mode_requested:
                    driver.set_arm_modes(driver_api.Mode.idle)
            finally:
                driver.cleanup(False)


def run_folded_pose_return(
    driver_api: Any,
    config: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Return from the configured dataset home to the recorded folded pose."""
    robot = config["robot"]
    settings = config["diagnostic_tests"]["folded_pose_return"]
    expected_start = [float(value) for value in robot["home_arm_positions_rad"]]
    target = [float(value) for value in robot["folded_arm_positions_rad"]]
    goal_time = float(settings["goal_time_s"])
    start_tolerance = float(settings["expected_start_tolerance_rad"])
    max_tracking_error = float(settings["max_tracking_error_rad"])
    if len(expected_start) != 6 or not all(
        isfinite(value) for value in expected_start
    ):
        raise ValueError("Expected start pose must contain six finite positions")
    if len(target) != 6 or not all(isfinite(value) for value in target):
        raise ValueError("Folded target must contain six finite positions")
    if min(goal_time, start_tolerance, max_tracking_error) <= 0:
        raise ValueError("Folded-pose return settings must be positive")

    model = getattr(driver_api.Model, robot["driver_model"])
    end_effector = getattr(driver_api.StandardEndEffector, robot["end_effector"])
    driver = driver_api.TrossenArmDriver()
    configured = False
    arm_position_mode_requested = False

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
        error_before = str(driver.get_error_information())
        modes_before = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if error_before != "No error":
            raise RuntimeError(f"Controller reports an error: {error_before}")
        if len(modes_before) != 7 or not all(
            mode in (0, "idle") for mode in modes_before
        ):
            raise RuntimeError(f"All joints must start idle; got {modes_before}")

        initial = [float(value) for value in driver.get_arm_positions()]
        limits = driver.get_joint_limits()
        if len(initial) != 6 or not all(isfinite(value) for value in initial):
            raise RuntimeError("Expected six finite initial arm positions")
        if len(limits) != 7:
            raise RuntimeError("Expected seven controller joint limits")
        start_errors = [
            abs(observed - expected)
            for expected, observed in zip(expected_start, initial, strict=True)
        ]
        if max(start_errors) > start_tolerance:
            raise RuntimeError(
                f"Arm is not at {robot['home_name']}; maximum start error "
                f"{max(start_errors):.6f} rad exceeds {start_tolerance:.6f} rad"
            )
        for joint, (position, limit) in enumerate(
            zip(target, limits[:6], strict=True)
        ):
            position_min = float(limit.position_min)
            position_max = float(limit.position_max)
            if not position_min <= position <= position_max:
                raise RuntimeError(
                    f"Folded target for joint {joint} is outside controller limits"
                )

        driver.set_arm_modes(driver_api.Mode.position)
        arm_position_mode_requested = True
        modes_during = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        position_value = getattr(driver_api.Mode.position, "value", 1)
        if len(modes_during) != 7 or not (
            all(mode == position_value for mode in modes_during[:6])
            and modes_during[6] in (0, "idle")
        ):
            raise RuntimeError(
                f"Expected arm position mode and gripper idle; got {modes_during}"
            )

        driver.set_arm_positions(target, goal_time, True)
        observed = [float(value) for value in driver.get_arm_positions()]
        if len(observed) != 6 or not all(isfinite(value) for value in observed):
            raise RuntimeError("Received invalid folded-pose joint positions")
        tracking_errors = [
            abs(actual - desired)
            for desired, actual in zip(target, observed, strict=True)
        ]
        failure_reasons: list[str] = []
        if max(tracking_errors) > max_tracking_error:
            failure_reasons.append(
                f"Folded-pose tracking error {max(tracking_errors):.6f} rad "
                f"exceeded {max_tracking_error:.6f} rad"
            )
        error_after_move = str(driver.get_error_information())
        if error_after_move != "No error":
            failure_reasons.append(
                f"Controller error after folded-pose move: {error_after_move}"
            )

        driver.set_arm_modes(driver_api.Mode.idle)
        arm_position_mode_requested = False
        modes_after = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if len(modes_after) != 7 or not all(
            mode in (0, "idle") for mode in modes_after
        ):
            failure_reasons.append(f"Idle mode was not restored: {modes_after}")
        error_after = str(driver.get_error_information())
        if error_after != "No error":
            failure_reasons.append(
                f"Controller reported an error after the test: {error_after}"
            )

        return {
            "passed": not failure_reasons,
            "failure_reasons": failure_reasons,
            "controller_ip": robot["controller_ip"],
            "driver_version": driver_version,
            "firmware_version": firmware_version,
            "error_before": error_before,
            "error_after_move": error_after_move,
            "error_after": error_after,
            "expected_start_name": robot["home_name"],
            "expected_start_rad": expected_start,
            "expected_start_tolerance_rad": start_tolerance,
            "initial_positions_rad": initial,
            "start_errors_rad": start_errors,
            "target_name": robot["folded_pose_name"],
            "target_source": robot["folded_pose_source"],
            "target_positions_rad": target,
            "goal_time_s": goal_time,
            "max_tracking_error_rad": max_tracking_error,
            "observed_positions_rad": observed,
            "tracking_errors_rad": tracking_errors,
            "modes_before": modes_before,
            "modes_during": modes_during,
            "modes_after": modes_after,
        }
    finally:
        if configured:
            try:
                if arm_position_mode_requested:
                    driver.set_arm_modes(driver_api.Mode.idle)
            finally:
                driver.cleanup(False)


def run_cartesian_step_test(
    driver_api: Any,
    config: dict[str, Any],
    timeout: float,
    axis: str,
    distance_m: float,
) -> dict[str, Any]:
    """Move one Cartesian translation axis by a bounded step, then return."""
    if axis not in {"x", "y", "z"}:
        raise ValueError("Cartesian step axis must be x, y, or z")
    if not isfinite(distance_m) or distance_m == 0:
        raise ValueError("Cartesian step distance must be finite and non-zero")

    robot = config["robot"]
    settings = config["diagnostic_tests"]["cartesian_step"]
    home_target = [float(value) for value in robot["home_arm_positions_rad"]]
    home_goal_time = float(settings["home_goal_time_s"])
    max_home_tracking_error = float(settings["max_home_tracking_error_rad"])
    goal_time = float(settings["goal_time_s"])
    hold_duration = float(settings["hold_duration_s"])
    max_step = float(settings["max_translation_step_m"])
    max_downward_step = float(settings["max_downward_step_m"])
    max_tracking_error = float(settings["max_tracking_error_m"])
    trajectory_check_samples = int(settings["trajectory_check_samples"])
    if min(
        home_goal_time,
        max_home_tracking_error,
        goal_time,
        hold_duration,
        max_step,
        max_downward_step,
        max_tracking_error,
        trajectory_check_samples,
    ) <= 0:
        raise ValueError("Cartesian-step settings must be positive")
    if len(home_target) != 6 or not all(isfinite(value) for value in home_target):
        raise ValueError("Configured home must contain six finite arm positions")
    permitted_step = max_downward_step if axis == "z" and distance_m < 0 else max_step
    if abs(distance_m) > permitted_step:
        raise ValueError(
            f"Requested {distance_m:.6f} m exceeds the {permitted_step:.6f} m "
            f"limit for this direction"
        )

    model = getattr(driver_api.Model, robot["driver_model"])
    end_effector = getattr(driver_api.StandardEndEffector, robot["end_effector"])
    driver = driver_api.TrossenArmDriver()
    configured = False
    arm_position_mode_requested = False

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
        error_before = str(driver.get_error_information())
        modes_before = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if error_before != "No error":
            raise RuntimeError(f"Controller reports an error: {error_before}")
        if len(modes_before) != 7 or not all(
            mode in (0, "idle") for mode in modes_before
        ):
            raise RuntimeError(f"All joints must start idle; got {modes_before}")

        initial_cartesian = [
            float(value) for value in driver.get_cartesian_positions()
        ]
        initial_joints = [float(value) for value in driver.get_all_positions()]
        if len(initial_cartesian) != 6 or not all(
            isfinite(value) for value in initial_cartesian
        ):
            raise RuntimeError("Expected six finite initial Cartesian positions")
        if len(initial_joints) != 7 or not all(
            isfinite(value) for value in initial_joints
        ):
            raise RuntimeError("Expected seven finite initial joint positions")

        limits = driver.get_joint_limits()
        if len(limits) != 7 or not all(
            float(limit.position_min) <= target <= float(limit.position_max)
            for target, limit in zip(home_target, limits[:6], strict=True)
        ):
            raise RuntimeError("Configured home is outside controller joint limits")

        driver.set_arm_modes(driver_api.Mode.position)
        arm_position_mode_requested = True
        driver.set_arm_positions(home_target, home_goal_time, True)
        observed_home = [float(value) for value in driver.get_arm_positions()]
        if len(observed_home) != 6 or not all(
            isfinite(value) for value in observed_home
        ):
            raise RuntimeError("Received invalid joint positions at home")
        home_errors = [
            abs(observed - target)
            for target, observed in zip(home_target, observed_home, strict=True)
        ]
        peak_home_error = max(home_errors)
        if peak_home_error > max_home_tracking_error:
            raise RuntimeError(
                f"Home tracking error {peak_home_error:.6f} rad exceeded "
                f"{max_home_tracking_error:.6f} rad"
            )
        error_after_home = str(driver.get_error_information())
        if error_after_home != "No error":
            raise RuntimeError(
                f"Controller reported an error after moving home: "
                f"{error_after_home}"
            )

        origin = [float(value) for value in driver.get_cartesian_positions()]
        if len(origin) != 6 or not all(isfinite(value) for value in origin):
            raise RuntimeError("Expected six finite Cartesian home positions")
        target = origin.copy()
        target[{"x": 0, "y": 1, "z": 2}[axis]] += distance_m
        modes_during = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        position_value = getattr(driver_api.Mode.position, "value", 1)
        if len(modes_during) != 7 or not (
            all(mode == position_value for mode in modes_during[:6])
            and modes_during[6] in (0, "idle")
        ):
            raise RuntimeError(
                f"Expected arm position mode and gripper idle; got {modes_during}"
            )

        interpolation_space = driver_api.InterpolationSpace.cartesian
        driver.set_cartesian_positions(
            target,
            interpolation_space,
            goal_time=goal_time,
            blocking=True,
            num_trajectory_check_samples=trajectory_check_samples,
        )
        time.sleep(hold_duration)
        observed_target = [
            float(value) for value in driver.get_cartesian_positions()
        ]
        if len(observed_target) != 6 or not all(
            isfinite(value) for value in observed_target
        ):
            raise RuntimeError("Received invalid Cartesian target state")
        target_errors = [
            abs(observed - goal)
            for goal, observed in zip(target, observed_target, strict=True)
        ]
        translation_target_error = max(target_errors[:3])
        failure_reasons: list[str] = []
        if translation_target_error > max_tracking_error:
            failure_reasons.append(
                f"Target translation error {translation_target_error:.6f} m "
                f"exceeded {max_tracking_error:.6f} m"
            )

        error_after_target = str(driver.get_error_information())
        return_command_sent = error_after_target == "No error"
        if return_command_sent:
            driver.set_cartesian_positions(
                origin,
                interpolation_space,
                goal_time=goal_time,
                blocking=True,
                num_trajectory_check_samples=trajectory_check_samples,
            )
        else:
            failure_reasons.append(
                f"Controller error after target: {error_after_target}"
            )

        observed_return = [
            float(value) for value in driver.get_cartesian_positions()
        ]
        if len(observed_return) != 6 or not all(
            isfinite(value) for value in observed_return
        ):
            raise RuntimeError("Received invalid Cartesian return state")
        return_errors = [
            abs(observed - goal)
            for goal, observed in zip(origin, observed_return, strict=True)
        ]
        translation_return_error = max(return_errors[:3])
        if return_command_sent and translation_return_error > max_tracking_error:
            failure_reasons.append(
                f"Return translation error {translation_return_error:.6f} m "
                f"exceeded {max_tracking_error:.6f} m"
            )

        driver.set_arm_modes(driver_api.Mode.idle)
        arm_position_mode_requested = False
        modes_after = [
            getattr(mode, "value", str(mode)) for mode in driver.get_modes()
        ]
        if len(modes_after) != 7 or not all(
            mode in (0, "idle") for mode in modes_after
        ):
            failure_reasons.append(f"Idle mode was not restored: {modes_after}")
        error_after = str(driver.get_error_information())
        if error_after != "No error":
            failure_reasons.append(
                f"Controller reported an error after the test: {error_after}"
            )

        return {
            "passed": not failure_reasons,
            "failure_reasons": failure_reasons,
            "controller_ip": robot["controller_ip"],
            "driver_version": driver_version,
            "firmware_version": firmware_version,
            "error_before": error_before,
            "error_after_target": error_after_target,
            "error_after": error_after,
            "axis": axis,
            "distance_m": distance_m,
            "home_name": robot["home_name"],
            "home_source": robot["home_source"],
            "home_goal_time_s": home_goal_time,
            "max_home_tracking_error_rad": max_home_tracking_error,
            "home_target_rad": home_target,
            "observed_home_rad": observed_home,
            "home_errors_rad": home_errors,
            "error_after_home": error_after_home,
            "goal_time_s": goal_time,
            "hold_duration_s": hold_duration,
            "max_tracking_error_m": max_tracking_error,
            "trajectory_check_samples": trajectory_check_samples,
            "modes_before": modes_before,
            "modes_during": modes_during,
            "modes_after": modes_after,
            "initial_cartesian": initial_cartesian,
            "origin_cartesian": origin,
            "target_cartesian": target,
            "observed_target_cartesian": observed_target,
            "target_errors": target_errors,
            "return_command_sent": return_command_sent,
            "observed_return_cartesian": observed_return,
            "return_errors": return_errors,
            "initial_joint_positions": initial_joints,
            "final_joint_positions": [
                float(value) for value in driver.get_all_positions()
            ],
        }
    finally:
        if configured:
            try:
                if arm_position_mode_requested:
                    driver.set_arm_modes(driver_api.Mode.idle)
            finally:
                driver.cleanup(False)
