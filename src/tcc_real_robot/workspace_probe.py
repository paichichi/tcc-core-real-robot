"""Pure helpers for operator-stepped Cartesian workspace calibration."""

from __future__ import annotations

from math import isfinite


def next_probe_target(
    origin: list[float],
    current_target: list[float],
    *,
    axis: str,
    direction: str,
    step_m: float,
    hard_travel_limit_m: float,
) -> tuple[list[float], float, bool]:
    """Return the next bounded Cartesian target and cumulative travel."""
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be x, y, or z")
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be positive or negative")
    if len(origin) != 6 or len(current_target) != 6:
        raise ValueError("Cartesian poses must contain six values")
    if not all(isfinite(value) for value in (*origin, *current_target)):
        raise ValueError("Cartesian poses must be finite")
    if not isfinite(step_m) or step_m <= 0:
        raise ValueError("step_m must be positive and finite")
    if not isfinite(hard_travel_limit_m) or hard_travel_limit_m < step_m:
        raise ValueError("hard travel limit must be at least one step")

    index = {"x": 0, "y": 1, "z": 2}[axis]
    sign = 1.0 if direction == "positive" else -1.0
    travelled = abs(current_target[index] - origin[index])
    next_travel = min(travelled + step_m, hard_travel_limit_m)
    target = current_target.copy()
    target[index] = origin[index] + sign * next_travel
    reached_limit = next_travel >= hard_travel_limit_m - 1e-12
    return target, next_travel, reached_limit


def format_workspace_probe_report(report: dict[str, object]) -> str:
    """Format a human-readable interactive workspace probe report."""
    lines = [
        "TCC-Core Interactive Workspace Probe",
        "====================================",
        f"Overall: {'PASS' if report['passed'] else 'STOPPED/FAILED'}",
        f"Captured at: {report['captured_at']}",
        f"Axis: {report['axis']}",
        f"Direction: {report['direction']}",
        f"Step: {float(report['step_m']):.6f} m",
        f"Hard travel limit: {float(report['hard_travel_limit_m']):.6f} m",
        f"Stop reason: {report['stop_reason']}",
        f"Returned home: {report['returned_home']}",
        f"Idle restored: {report['idle_restored']}",
        f"Origin Cartesian: {report['origin_cartesian']}",
        f"Last safe Cartesian: {report['last_safe_cartesian']}",
        f"Cumulative travel: {float(report['cumulative_travel_m']):.6f} m",
        "",
        "Accepted points",
        "---------------",
    ]
    points = report["points"]
    assert isinstance(points, list)
    for point in points:
        assert isinstance(point, dict)
        lines.append(
            f"step={int(point['step']):03d} "
            f"travel_m={float(point['travel_m']):.6f} "
            f"tracking_error_m={float(point['tracking_error_m']):.6f} "
            f"cartesian={point['cartesian']}"
        )
    failure = str(report.get("failure", ""))
    lines.extend(("", "Failure", "-------", failure or "None"))
    return "\n".join(lines) + "\n"
