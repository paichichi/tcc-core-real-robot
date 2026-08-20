"""Validation helpers for replaying recorded absolute joint trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DemoReplayAudit:
    frames: int
    max_step: tuple[float, ...]
    max_velocity: tuple[float, ...]


def audit_demo_trajectory(
    actions: np.ndarray,
    *,
    fps: float,
    absolute_min: list[float],
    absolute_max: list[float],
    max_arm_velocity_rad_s: float,
    max_gripper_velocity_m_s: float,
) -> DemoReplayAudit:
    """Reject malformed or out-of-envelope replay trajectories."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or values.shape[0] < 2:
        raise ValueError("Demo actions must have shape [frames, 7]")
    if not np.isfinite(values).all():
        raise ValueError("Demo actions must all be finite")
    if fps <= 0:
        raise ValueError("Demo FPS must be positive")
    low = np.asarray(absolute_min, dtype=np.float64)
    high = np.asarray(absolute_max, dtype=np.float64)
    if low.shape != (7,) or high.shape != (7,) or np.any(low > high):
        raise ValueError("Absolute action bounds must contain seven ordered pairs")
    if np.any(values < low) or np.any(values > high):
        raise ValueError("Demo action lies outside the configured dataset envelope")

    max_step = np.max(np.abs(np.diff(values, axis=0)), axis=0)
    max_velocity = max_step * fps
    if float(max_velocity[:6].max()) > max_arm_velocity_rad_s:
        raise ValueError("Demo arm velocity exceeds the configured limit")
    if float(max_velocity[6]) > max_gripper_velocity_m_s:
        raise ValueError("Demo gripper velocity exceeds the configured limit")
    return DemoReplayAudit(
        frames=int(values.shape[0]),
        max_step=tuple(float(value) for value in max_step),
        max_velocity=tuple(float(value) for value in max_velocity),
    )
