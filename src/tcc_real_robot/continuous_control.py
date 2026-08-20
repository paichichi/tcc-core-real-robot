"""Stateful command shaping for continuous Trossen position control."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class LimitedCommand:
    """One absolute command after envelope, slew, and lead limiting."""

    desired: tuple[float, ...]
    previous: tuple[float, ...]
    commanded: tuple[float, ...]
    command_step: tuple[float, ...]
    command_lead: tuple[float, ...]


class StatefulPositionLimiter:
    """Keep consecutive non-blocking position commands continuous.

    Slew is measured from the previous command, not from the lagging measured
    position. A separate lead bound prevents the command stream from running
    too far ahead of the physical robot.
    """

    def __init__(
        self,
        initial_command: list[float],
        *,
        lower_bounds: list[float],
        upper_bounds: list[float],
        maximum_steps: list[float],
        maximum_leads: list[float],
    ) -> None:
        vectors = (
            initial_command,
            lower_bounds,
            upper_bounds,
            maximum_steps,
            maximum_leads,
        )
        if any(len(vector) != 7 for vector in vectors):
            raise ValueError("Position limiter vectors must contain seven values")
        if not all(isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Position limiter vectors must be finite")
        if any(
            low > high
            for low, high in zip(lower_bounds, upper_bounds, strict=True)
        ):
            raise ValueError("Position limiter bounds must be ordered")
        if any(value <= 0 for value in (*maximum_steps, *maximum_leads)):
            raise ValueError("Position limiter step and lead limits must be positive")

        self.lower_bounds = tuple(lower_bounds)
        self.upper_bounds = tuple(upper_bounds)
        self.maximum_steps = tuple(maximum_steps)
        self.maximum_leads = tuple(maximum_leads)
        self.previous_command = tuple(
            max(low, min(high, value))
            for value, low, high in zip(
                initial_command, lower_bounds, upper_bounds, strict=True
            )
        )

    def limit(self, raw_target: list[float], observed: list[float]) -> LimitedCommand:
        """Return and remember the next continuous absolute-position command."""
        if len(raw_target) != 7 or len(observed) != 7:
            raise ValueError("Policy target and observation must contain seven values")
        if not all(isfinite(value) for value in (*raw_target, *observed)):
            raise ValueError("Policy target and observation must be finite")

        desired: list[float] = []
        commanded: list[float] = []
        for raw, measured, previous, low, high, step, lead in zip(
            raw_target,
            observed,
            self.previous_command,
            self.lower_bounds,
            self.upper_bounds,
            self.maximum_steps,
            self.maximum_leads,
            strict=True,
        ):
            bounded_desired = max(low, min(high, raw))
            next_value = previous + max(
                -step, min(step, bounded_desired - previous)
            )
            next_value = max(measured - lead, min(measured + lead, next_value))
            next_value = max(low, min(high, next_value))
            desired.append(bounded_desired)
            commanded.append(next_value)

        previous_command = self.previous_command
        self.previous_command = tuple(commanded)
        return LimitedCommand(
            desired=tuple(desired),
            previous=previous_command,
            commanded=tuple(commanded),
            command_step=tuple(
                target - previous
                for target, previous in zip(
                    commanded, previous_command, strict=True
                )
            ),
            command_lead=tuple(
                target - measured
                for target, measured in zip(commanded, observed, strict=True)
            ),
        )
