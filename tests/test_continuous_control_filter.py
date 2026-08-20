import pytest

from tcc_real_robot.continuous_control import ExponentialActionFilter


def test_action_ema_starts_from_observed_home() -> None:
    action_filter = ExponentialActionFilter([0.0] * 7, alpha=0.25)

    first = action_filter.update([1.0] * 7)
    second = action_filter.update([1.0] * 7)

    assert first == pytest.approx([0.25] * 7)
    assert second == pytest.approx([0.4375] * 7)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1])
def test_action_ema_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        ExponentialActionFilter([0.0] * 7, alpha=alpha)
