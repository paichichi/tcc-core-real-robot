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


def format_gripper_cycle_report(report: dict[str, Any]) -> str:
    """Format a small commanded gripper open-and-return diagnostic."""
    lines = [
        "TCC-Core Gripper Cycle Report",
        "=" * 29,
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        f"Captured at: {report['captured_at']}",
        f"Controller IP: {report['controller_ip']}",
        f"Driver version: {report['driver_version']}",
        f"Firmware version: {report['firmware_version']}",
        f"Controller error before:     {report['error_before']}",
        f"Controller error after open: {report['error_after_open']}",
        f"Controller error after:      {report['error_after']}",
        "",
        "Test configuration",
        "------------------",
        f"Open delta: {_format_value(report['open_delta_m'])} m",
        f"Goal time: {_format_value(report['goal_time_s'])} s",
        f"Hold duration: {_format_value(report['hold_duration_s'])} s",
        f"Sample rate: {_format_value(report['sample_rate_hz'])} Hz",
        f"Samples: {report['samples']}",
        f"Maximum tracking error: {_format_value(report['max_tracking_error_m'])} m",
        "Only the gripper entered position mode; all six arm joints stayed idle.",
        "",
        "Mode transition",
        "---------------",
        f"Before: {report['modes_before']}",
        f"During: {report['modes_during']}",
        f"After:  {report['modes_after']}",
        "",
        "Command and tracking",
        "--------------------",
        (
            f"Controller range: [{_format_value(report['position_min_m'])}, "
            f"{_format_value(report['position_max_m'])}] m"
        ),
        f"Initial: {_format_value(report['initial_position_m'])} m",
        f"Open target: {_format_value(report['open_target_m'])} m",
        f"Observed open: {_format_value(report['observed_open_m'])} m",
        f"Peak open error: {_format_value(report['peak_open_error_m'])} m",
        f"Return target: {_format_value(report['return_target_m'])} m",
        f"Return command sent: {report['return_command_sent']}",
        f"Observed return: {_format_value(report['observed_return_m'])} m",
        f"Return error: {_format_value(report['return_error_m'])} m",
        "",
        "Failure reasons",
        "---------------",
    ]
    if report["failure_reasons"]:
        lines.extend(f"- {reason}" for reason in report["failure_reasons"])
    else:
        lines.append("None")
    return "\n".join(lines) + "\n"


def format_whole_arm_cycle_report(report: dict[str, Any]) -> str:
    """Format a bounded six-joint outbound-and-return diagnostic."""
    lines = [
        "TCC-Core Whole-Arm Cycle Report",
        "=" * 31,
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        f"Captured at: {report['captured_at']}",
        f"Controller IP: {report['controller_ip']}",
        f"Driver version: {report['driver_version']}",
        f"Firmware version: {report['firmware_version']}",
        f"Controller error before: {report['error_before']}",
        f"Controller error after outbound: {report['error_after_outbound']}",
        f"Controller error after: {report['error_after']}",
        "",
        "Test configuration",
        "------------------",
        f"Joint deltas: {report['joint_deltas_rad']} rad",
        f"Goal time per move: {_format_value(report['goal_time_s'])} s",
        f"Hold duration: {_format_value(report['hold_duration_s'])} s",
        f"Samples: {report['samples']}",
        f"Joint-limit margin: {_format_value(report['limit_margin_rad'])} rad",
        f"Maximum tracking error: {_format_value(report['max_tracking_error_rad'])} rad",
        "All six arm joints moved together; the gripper stayed idle.",
        "",
        "Mode transition",
        "---------------",
        f"Before: {report['modes_before']}",
        f"During: {report['modes_during']}",
        f"After:  {report['modes_after']}",
        "",
        "Positions (rad)",
        "---------------",
        f"Initial: {report['initial_positions_rad']}",
        f"Outbound target: {report['outbound_target_rad']}",
        f"Observed outbound: {report['observed_outbound_rad']}",
        f"Outbound peak errors: {report['peak_outbound_errors_rad']}",
        f"Return target: {report['return_target_rad']}",
        f"Return command sent: {report['return_command_sent']}",
        f"Observed return: {report['observed_return_rad']}",
        f"Return errors: {report['return_errors_rad']}",
        "",
        "Failure reasons",
        "---------------",
    ]
    if report["failure_reasons"]:
        lines.extend(f"- {reason}" for reason in report["failure_reasons"])
    else:
        lines.append("None")
    return "\n".join(lines) + "\n"


def format_workspace_point_report(report: dict[str, Any]) -> str:
    """Format one read-only workspace calibration point."""
    cartesian = report["cartesian_positions"]
    lines = [
        "TCC-Core Workspace Calibration Point",
        "=" * 36,
        f"Label: {report['label']}",
        f"Captured at: {report['captured_at']}",
        f"Preflight passed: {report['passed']}",
        f"Controller IP: {report['controller_ip']}",
        f"Driver version: {report['driver_version']}",
        f"Firmware version: {report['firmware_version']}",
        f"Controller error: {report['error_information']}",
        f"Modes: {report['modes']}",
        "",
        "Cartesian position",
        "------------------",
        f"x_m: {_format_value(cartesian[0])}",
        f"y_m: {_format_value(cartesian[1])}",
        f"z_m: {_format_value(cartesian[2])}",
        f"angle_axis_x_rad: {_format_value(cartesian[3])}",
        f"angle_axis_y_rad: {_format_value(cartesian[4])}",
        f"angle_axis_z_rad: {_format_value(cartesian[5])}",
        "",
        "Joint positions (arm rad, gripper m)",
        "------------------------------------",
        str(report["positions"]),
        "",
        "Temperatures (C)",
        "----------------",
        f"Rotor: {report['rotor_temperatures']}",
        f"Driver: {report['driver_temperatures']}",
        "",
        "This command only read controller state; it sent no motion command.",
    ]
    return "\n".join(lines) + "\n"


def format_cartesian_step_report(report: dict[str, Any]) -> str:
    """Format one bounded Cartesian step-and-return diagnostic."""
    lines = [
        "TCC-Core Cartesian Step Report",
        "=" * 30,
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        f"Captured at: {report['captured_at']}",
        f"Controller IP: {report['controller_ip']}",
        f"Driver version: {report['driver_version']}",
        f"Firmware version: {report['firmware_version']}",
        f"Controller error before: {report['error_before']}",
        f"Controller error after home: {report['error_after_home']}",
        f"Controller error after target: {report['error_after_target']}",
        f"Controller error after: {report['error_after']}",
        "",
        "Test configuration",
        "------------------",
        f"Home: {report['home_name']}",
        f"Home source: {report['home_source']}",
        f"Home move time: {_format_value(report['home_goal_time_s'])} s",
        f"Home target: {report['home_target_rad']} rad",
        f"Observed home: {report['observed_home_rad']} rad",
        f"Home absolute errors: {report['home_errors_rad']} rad",
        f"Axis: {report['axis']}",
        f"Distance: {_format_value(report['distance_m'])} m",
        f"Goal time per move: {_format_value(report['goal_time_s'])} s",
        f"Hold duration: {_format_value(report['hold_duration_s'])} s",
        f"Trajectory check samples: {report['trajectory_check_samples']}",
        f"Maximum tracking error: {_format_value(report['max_tracking_error_m'])} m",
        "Orientation was held constant and the gripper stayed idle.",
        "",
        "Mode transition",
        "---------------",
        f"Before: {report['modes_before']}",
        f"During: {report['modes_during']}",
        f"After:  {report['modes_after']}",
        "",
        "Cartesian positions [x, y, z, angle-axis]",
        "------------------------------------------",
        f"Before home: {report['initial_cartesian']}",
        f"Origin: {report['origin_cartesian']}",
        f"Target: {report['target_cartesian']}",
        f"Observed target: {report['observed_target_cartesian']}",
        f"Target absolute errors: {report['target_errors']}",
        f"Return command sent: {report['return_command_sent']}",
        f"Observed return: {report['observed_return_cartesian']}",
        f"Return absolute errors: {report['return_errors']}",
        "",
        "Joint positions (arm rad, gripper m)",
        "------------------------------------",
        f"Initial: {report['initial_joint_positions']}",
        f"Final: {report['final_joint_positions']}",
        "",
        "Failure reasons",
        "---------------",
    ]
    if report["failure_reasons"]:
        lines.extend(f"- {reason}" for reason in report["failure_reasons"])
    else:
        lines.append("None")
    return "\n".join(lines) + "\n"
