"""Human-readable text reports for real-robot diagnostics."""

from __future__ import annotations

from typing import Any


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def format_preflight_report(report: dict[str, Any]) -> str:
    """Format a preflight result for operator review."""
    lines = [
        "TCC-Core Robot Preflight Report",
        "=" * 32,
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        f"Captured at: {report['captured_at']}",
        f"Controller IP: {report['controller_ip']}",
        f"Driver version: {report['driver_version']}",
        f"Firmware version: {report['firmware_version']}",
        f"Controller error: {report['error_information']}",
        "",
        "Checks",
        "------",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {name}")

    lines.extend(["", "Joint state and limits", "----------------------"])
    lines.append(
        "joint | position | min | max | tolerance | velocity_max | effort_max"
    )
    for position, limit in zip(
        report["positions"], report["joint_limits"], strict=True
    ):
        lines.append(
            f"{limit['joint']:>5} | {position:>10.6f} | "
            f"{limit['position_min']:>9.6f} | {limit['position_max']:>9.6f} | "
            f"{limit['position_tolerance']:>9.6f} | "
            f"{limit['velocity_max']:>12.6f} | {limit['effort_max']:>10.6f}"
        )

    lines.extend(["", f"Modes: {report['modes']}", "", "Temperatures (C)"])
    lines.append("----------------")
    lines.append(f"Rotor:  {report['rotor_temperatures']}")
    lines.append(f"Driver: {report['driver_temperatures']}")
    lines.extend(["", "Cartesian position", "------------------"])
    lines.append(str(report["cartesian_positions"]))
    return "\n".join(lines) + "\n"


def format_position_hold_report(report: dict[str, Any]) -> str:
    """Format a position-mode hold diagnostic for operator review."""
    lines = [
        "TCC-Core Position-Mode Hold Report",
        "=" * 34,
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        f"Captured at: {report['captured_at']}",
        f"Controller IP: {report['controller_ip']}",
        f"Driver version: {report['driver_version']}",
        f"Firmware version: {report['firmware_version']}",
        f"Controller error: {report['error_information']}",
        "",
        "Test configuration",
        "------------------",
        f"Duration: {_format_value(report['duration_s'])} s",
        f"Sample rate: {_format_value(report['sample_rate_hz'])} Hz",
        f"Samples: {report['samples']}",
        "No position, velocity, or effort target was sent.",
        "",
        "Mode transition",
        "---------------",
        f"Before: {report['modes_before']}",
        f"During: {report['modes_during']}",
        f"After:  {report['modes_after']}",
        "",
        "Observed position change",
        "------------------------",
        f"Initial: {report['initial_positions']}",
        f"Final:   {report['final_positions']}",
        f"Peak arm drift: {_format_value(report['peak_arm_drift_rad'])} rad",
        f"Allowed arm drift: {_format_value(report['max_allowed_arm_drift_rad'])} rad",
        f"Peak gripper drift: {_format_value(report['peak_gripper_drift_m'])} m",
        f"Allowed gripper drift: {_format_value(report['max_allowed_gripper_drift_m'])} m",
    ]
    return "\n".join(lines) + "\n"


def format_current_position_hold_report(report: dict[str, Any]) -> str:
    """Format a commanded current-position hold diagnostic."""
    lines = [
        "TCC-Core Current-Position Hold Report",
        "=" * 37,
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        f"Captured at: {report['captured_at']}",
        f"Controller IP: {report['controller_ip']}",
        f"Driver version: {report['driver_version']}",
        f"Firmware version: {report['firmware_version']}",
        f"Controller error before: {report['error_information']}",
        f"Controller error after:  {report['error_information_after']}",
        "",
        "Test configuration",
        "------------------",
        f"Goal time: {_format_value(report['goal_time_s'])} s",
        f"Observation duration: {_format_value(report['duration_s'])} s",
        f"Sample rate: {_format_value(report['sample_rate_hz'])} Hz",
        f"Planned samples: {report['samples']}",
        f"Observed samples: {report['samples_observed']}",
        "The measured current position was sent back unchanged as the target.",
        "The test returns to idle immediately if an error limit is exceeded.",
        "",
        "Mode transition",
        "---------------",
        f"Before: {report['modes_before']}",
        f"During: {report['modes_during']}",
        f"After:  {report['modes_after']}",
        "",
        "Command and tracking error",
        "--------------------------",
        f"Target: {report['target_positions']}",
        f"Final:  {report['final_positions']}",
        f"Peak arm error: {_format_value(report['peak_arm_error_rad'])} rad",
        f"Allowed arm error: {_format_value(report['max_allowed_arm_error_rad'])} rad",
        f"Peak gripper error: {_format_value(report['peak_gripper_error_m'])} m",
        f"Allowed gripper error: {_format_value(report['max_allowed_gripper_error_m'])} m",
    ]
    lines.extend(["", "Peak error by joint", "-------------------"])
    for joint, error in enumerate(report["peak_joint_errors"]):
        unit = "rad" if joint < 6 else "m"
        lines.append(f"Joint {joint}: {_format_value(error)} {unit}")
    lines.extend(["", "Failure reasons", "---------------"])
    if report["failure_reasons"]:
        lines.extend(f"- {reason}" for reason in report["failure_reasons"])
    else:
        lines.append("None")
    return "\n".join(lines) + "\n"
