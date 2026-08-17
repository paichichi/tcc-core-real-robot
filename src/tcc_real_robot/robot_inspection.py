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
