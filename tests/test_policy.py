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


def test_r3m_multiview_policy_projects_both_camera_features() -> None:
    policy = TCCMLPPolicy(
        feature_dim=8,
        num_tasks=4,
        proprio_dim=7,
        camera_names=("cam_main", "cam_wrist"),
        camera_fusion="project_then_concat",
        camera_projection_dim=4,
    ).eval()
    main = torch.randn(5, 8)
    task = torch.tensor([0, 1, 2, 3, 0])
    state = torch.randn(5, 7)

    wrist = torch.randn(5, 8)
    first = policy(main, wrist, task, state)
    second = policy(main, wrist + 1.0, task, state)

    assert first.shape == (5, 7)
    assert policy.mlp[0].num_features == 4 + 4 + 4 + 7
    assert not torch.allclose(first, second)


def test_v6_gated_policy_fuses_wrist_into_compact_visual_input() -> None:
    policy = TCCMLPPolicy(
        feature_dim=8,
        num_tasks=4,
        proprio_dim=7,
        camera_names=("cam_main", "cam_wrist"),
        camera_fusion="gated_residual",
        camera_projection_dim=4,
        camera_gate_hidden_dim=5,
    ).eval()
    main = torch.randn(5, 8)
    wrist = torch.randn(5, 8)
    task = torch.tensor([0, 1, 2, 3, 0])
    state = torch.randn(5, 7)

    first = policy(main, wrist, task, state)
    second = policy(main, wrist + 1.0, task, state)

    assert first.shape == (5, 7)
    assert policy.camera_gate is not None
    assert policy.mlp[0].num_features == 4 + 4 + 7
    assert not torch.allclose(first, second)


def test_v6_gated_policy_requires_two_cameras() -> None:
    with pytest.raises(ValueError, match="gated_residual requires both cameras"):
        TCCMLPPolicy(
            feature_dim=8,
            num_tasks=4,
            camera_names=("cam_main",),
            camera_fusion="gated_residual",
            camera_projection_dim=4,
            camera_gate_hidden_dim=4,
        )


def test_r3m_single_view_policy_ignores_wrist_feature() -> None:
    policy = TCCMLPPolicy(
        feature_dim=8,
        num_tasks=4,
        proprio_dim=7,
        camera_names=("cam_main",),
    ).eval()
    main = torch.randn(5, 8)
    task = torch.tensor([0, 1, 2, 3, 0])
    state = torch.randn(5, 7)

    first = policy(main, torch.randn(5, 8), task, state)
    second = policy(main, torch.randn(5, 8), task, state)

    assert first.shape == (5, 7)
    assert policy.mlp[0].num_features == 8 + 4 + 7
    assert torch.allclose(first, second)


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


def test_future_delta_policy_accepts_normalized_state_and_layer_norm() -> None:
    policy = TCCMLPPolicy(
        feature_dim=8,
        num_tasks=4,
        proprio_dim=7,
        input_batch_norm=False,
        input_layer_norm=True,
    )

    output = policy(
        torch.randn(3, 8),
        torch.randn(3, 8),
        torch.tensor([0, 1, 2]),
        torch.randn(3, 7),
    )

    assert output.shape == (3, 7)


def test_progress_conditioned_policy_accepts_normalized_episode_time() -> None:
    policy = TCCMLPPolicy(
        feature_dim=8,
        num_tasks=4,
        progress_dim=1,
        input_batch_norm=False,
        input_layer_norm=True,
    )

    output = policy(
        torch.randn(3, 8),
        torch.randn(3, 8),
        torch.tensor([0, 1, 2]),
        progress=torch.tensor([[0.0], [0.5], [1.0]]),
    )

    assert output.shape == (3, 7)


def test_r3m_output_layer_uses_small_initialization() -> None:
    torch.manual_seed(1)
    baseline = TCCMLPPolicy(
        feature_dim=8, num_tasks=4, output_layer_scale=1.0
    )
    torch.manual_seed(1)
    r3m_style = TCCMLPPolicy(
        feature_dim=8, num_tasks=4, output_layer_scale=0.01
    )

    baseline_output = baseline.mlp[-1]
    r3m_output = r3m_style.mlp[-1]
    assert torch.allclose(r3m_output.weight, baseline_output.weight * 0.01)
    assert torch.allclose(r3m_output.bias, baseline_output.bias * 0.01)
