"""Shared Trossen driver configuration helpers."""

from __future__ import annotations

from typing import Any


def _normalize_version(version: str) -> str:
    normalized = version.removeprefix("v")
    parts = normalized.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise RuntimeError(f"Unrecognized Trossen version: {version!r}")
    return normalized


def validate_versions(driver: Any, robot: dict[str, Any]) -> tuple[str, str]:
    """Require the exact driver/firmware pair validated on the physical arm."""
    driver_version = str(driver.get_driver_version())
    firmware_version = str(driver.get_controller_version())
    expected_driver = str(robot["expected_driver_version"])
    expected_firmware = str(robot["expected_firmware_version"])
    if _normalize_version(driver_version) != _normalize_version(expected_driver):
        raise RuntimeError(
            f"Driver {driver_version} does not match validated {expected_driver}"
        )
    if _normalize_version(firmware_version) != _normalize_version(expected_firmware):
        raise RuntimeError(
            f"Firmware {firmware_version} does not match validated {expected_firmware}"
        )
    return driver_version, firmware_version


def apply_motor_parameters(
    driver_api: Any,
    driver: Any,
    robot: dict[str, Any],
) -> str:
    """Apply the explicitly pinned official motor-parameter preset."""
    preset_name = str(robot["motor_parameters"])
    try:
        preset = getattr(driver_api.StandardMotorParameters, preset_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Installed Trossen driver does not provide motor preset {preset_name!r}"
        ) from exc
    driver.set_motor_parameters(preset)
    return preset_name
