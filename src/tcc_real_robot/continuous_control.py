"""Stateful command shaping for continuous Trossen position control."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RelativeClippedCommand:
    """Absolute command after official measured-position relative clipping."""

    desired: tuple[float, ...]
    commanded: tuple[float, ...]
    relative_target: tuple[float, ...]


def clip_goal_to_measured_position(
    raw_target: list[float],
    observed: list[float],
    *,
    lower_bounds: list[float],
    upper_bounds: list[float],
    maximum_relative_targets: list[float],
) -> RelativeClippedCommand:
    """Match LeRobot's official ``ensure_safe_goal_position`` semantics.

    Each absolute policy target is first kept inside the certified absolute
    envelope, then clipped independently to ``measured +/- cap``. The function
    is intentionally stateless: a non-blocking command that the robot has not
    yet followed does not become the reference for the next command.
    """
    vectors = (
        raw_target,
        observed,
        lower_bounds,
        upper_bounds,
        maximum_relative_targets,
    )
    if any(len(vector) != 7 for vector in vectors):
        raise ValueError("Position clipping vectors must contain seven values")
    if not all(isfinite(value) for vector in vectors for value in vector):
        raise ValueError("Position clipping vectors must be finite")
    if any(
        low > high
        for low, high in zip(lower_bounds, upper_bounds, strict=True)
    ):
        raise ValueError("Position clipping bounds must be ordered")
    if any(value <= 0 for value in maximum_relative_targets):
        raise ValueError("Maximum relative targets must be positive")

    desired: list[float] = []
    commanded: list[float] = []
    for raw, measured, low, high, cap in zip(
        raw_target,
        observed,
        lower_bounds,
        upper_bounds,
        maximum_relative_targets,
        strict=True,
    ):
        bounded_desired = max(low, min(high, raw))
        safe_difference = max(-cap, min(cap, bounded_desired - measured))
        safe_goal = max(low, min(high, measured + safe_difference))
        desired.append(bounded_desired)
        commanded.append(safe_goal)

    return RelativeClippedCommand(
        desired=tuple(desired),
        commanded=tuple(commanded),
        relative_target=tuple(
            target - measured
            for target, measured in zip(commanded, observed, strict=True)
        ),
    )


class ExponentialActionFilter:
    """Low-pass noisy absolute policy targets without changing their units."""

    def __init__(self, initial_action: list[float], alpha: float) -> None:
        if len(initial_action) != 7 or not all(isfinite(v) for v in initial_action):
            raise ValueError("Initial action must contain seven finite values")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.alpha = alpha
        self.previous = tuple(initial_action)

    def update(self, raw_action: list[float]) -> tuple[float, ...]:
        if len(raw_action) != 7 or not all(isfinite(v) for v in raw_action):
            raise ValueError("Raw action must contain seven finite values")
        filtered = tuple(
            previous + self.alpha * (raw - previous)
            for raw, previous in zip(raw_action, self.previous, strict=True)
        )
        self.previous = filtered
        return filtered
