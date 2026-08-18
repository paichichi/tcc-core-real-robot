import pytest

from tcc_real_robot.workspace_probe import next_probe_target


def test_next_probe_target_steps_positive_without_changing_orientation() -> None:
    origin = [0.25, 0.0, 0.16, 0.1, 0.2, 0.3]

    target, travel, reached = next_probe_target(
        origin,
        origin.copy(),
        axis="z",
        direction="positive",
        step_m=0.002,
        hard_travel_limit_m=0.1,
    )

    assert target == pytest.approx([0.25, 0.0, 0.162, 0.1, 0.2, 0.3])
    assert travel == pytest.approx(0.002)
    assert reached is False


def test_next_probe_target_stops_exactly_at_negative_hard_limit() -> None:
    origin = [0.25, 0.0, 0.16, 0.1, 0.2, 0.3]
    current = [0.25, 0.0, 0.141, 0.1, 0.2, 0.3]

    target, travel, reached = next_probe_target(
        origin,
        current,
        axis="z",
        direction="negative",
        step_m=0.002,
        hard_travel_limit_m=0.02,
    )

    assert target[2] == pytest.approx(0.14)
    assert travel == pytest.approx(0.02)
    assert reached is True
