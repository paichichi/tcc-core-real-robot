import pytest

torch = pytest.importorskip("torch")

from tcc_real_robot.policy import ActionNormalizer, TCCMLPPolicy


def test_two_camera_policy_predicts_one_action() -> None:
    policy = TCCMLPPolicy(feature_dim=8, num_tasks=4)
    output = policy(
        torch.randn(5, 8),
        torch.randn(5, 8),
        torch.tensor([0, 1, 2, 3, 0]),
    )
    assert output.shape == (5, 7)


def test_no_state_policy_rejects_proprioception() -> None:
    policy = TCCMLPPolicy(feature_dim=8, num_tasks=4, proprio_dim=0)
    with pytest.raises(ValueError, match="without proprioception"):
        policy(
            torch.randn(2, 8),
            torch.randn(2, 8),
            torch.tensor([0, 1]),
            torch.randn(2, 7),
        )


def test_action_normalizer_round_trip() -> None:
    normalizer = ActionNormalizer(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
    action = torch.tensor([[3.0, 6.0]])
    assert torch.allclose(normalizer.denormalize(normalizer.normalize(action)), action)
