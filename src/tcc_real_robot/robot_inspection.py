"""Read-only inspection helpers for a Trossen follower arm."""

from __future__ import annotations

import time
from collections.abc import Callable
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
            "observed_rate_hz": sample_count / elapsed if elapsed > 0 else 0.0,
            "max_arm_change_rad": max(
                abs(end - start) for start, end in zip(first_arm, last_arm, strict=True)
            ),
            "gripper_change_m": last_gripper - first_gripper,
        }
    finally:
        if configured:
            driver.cleanup(False)
