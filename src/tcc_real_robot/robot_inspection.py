"""Read-only inspection helpers for a Trossen follower arm."""

from __future__ import annotations

from typing import Any


def _version_series(version: str) -> str:
    """Return the major.minor portion of a version such as ``v1.9.2``."""
    parts = version.removeprefix("v").split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise RuntimeError(f"Unrecognized Trossen version: {version!r}")
    return ".".join(parts[:2])


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
