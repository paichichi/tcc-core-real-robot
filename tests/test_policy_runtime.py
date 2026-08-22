from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tcc_real_robot.policy import (
    R3MRobomimicPolicy,
    TCCMLPGaussianMixturePolicy,
    TCCMLPPolicy,
)
from tcc_real_robot.policy_runtime import (
    load_policy_bundle,
    predict_action,
    preprocess_rgb_frames,
    restore_policy_backbone,
    validate_policy_contract,
)
from tcc_real_robot.tcc_backbone import IndependentCameraBackbones


class MeanBackbone(torch.nn.Module):
    output_dim = 3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))


class MeanBackboneWithScale(MeanBackbone):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return super().forward(images) * self.scale


def checkpoint_payload() -> dict:
    model = TCCMLPPolicy(feature_dim=3, num_tasks=4)
    return {
        "model": model.state_dict(),
        "action_mean": torch.arange(7, dtype=torch.float32),
        "action_std": torch.ones(7),
        "feature_dim": 3,
        "config": {
            "dataset": {"tasks": ["a", "b", "c", "d"]},
            "policy": {
                "number_of_tasks": 4,
                "action_dim": 7,
                "action_chunk_size": 1,
                "hidden_dimensions": [256, 256],
                "input_batch_norm": True,
                "proprioception": False,
            },
        },
        "step": 50_000,
    }


def r3m_robomimic_checkpoint_payload() -> dict:
    model = R3MRobomimicPolicy(feature_dim=3)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    return {
        "model": model.state_dict(),
        "action_mean": torch.full((7,), 0.25),
        "action_std": torch.ones(7),
        "feature_dim": 3,
        "config": {
            "dataset": {"tasks": ["carrot"]},
            "policy": {
                "architecture": (
                    "r3m_deterministic_mlp_dual_independent_encoder"
                ),
                "number_of_tasks": 1,
                "action_dim": 7,
                "action_chunk_size": 1,
                "action_representation": "absolute",
                "action_distribution": "deterministic",
                "hidden_dimensions": [256, 256],
                "output_layer_scale": 0.01,
                "input_batch_norm": True,
                "input_layer_norm": False,
                "proprioception": False,
                "cameras": ["cam_main", "cam_wrist"],
                "camera_fusion": "raw_concat",
                "camera_projection_dim": 0,
                "camera_gate_hidden_dim": 0,
            },
        },
        "step": 5_000,
    }


def future_delta_checkpoint_payload() -> dict:
    model = TCCMLPPolicy(
        feature_dim=3,
        num_tasks=4,
        proprio_dim=7,
        input_batch_norm=False,
        input_layer_norm=True,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    return {
        "model": model.state_dict(),
        "action_mean": torch.full((7,), 0.1),
        "action_std": torch.ones(7),
        "state_mean": torch.zeros(7),
        "state_std": torch.ones(7),
        "feature_dim": 3,
        "config": {
            "dataset": {"tasks": ["a", "b", "c", "d"]},
            "policy": {
                "number_of_tasks": 4,
                "action_dim": 7,
                "action_chunk_size": 1,
                "action_representation": "future_delta",
                "lookahead_frames": 10,
                "execution_delta_gain": 0.1,
                "hidden_dimensions": [256, 256],
                "input_batch_norm": False,
                "input_layer_norm": True,
                "proprioception": True,
                "proprioception_dim": 7,
            },
        },
        "step": 10_000,
    }


def progress_checkpoint_payload() -> dict:
    model = TCCMLPPolicy(
        feature_dim=3,
        num_tasks=4,
        progress_dim=1,
        input_batch_norm=False,
        input_layer_norm=True,
    )
    return {
        "model": model.state_dict(),
        "action_mean": torch.zeros(7),
        "action_std": torch.ones(7),
        "feature_dim": 3,
        "config": {
            "dataset": {"tasks": ["a", "b", "c", "d"]},
            "policy": {
                "number_of_tasks": 4,
                "action_dim": 7,
                "action_chunk_size": 1,
                "action_representation": "absolute",
                "hidden_dimensions": [256, 256],
                "input_batch_norm": False,
                "input_layer_norm": True,
                "proprioception": False,
                "proprioception_dim": 0,
                "progress_conditioning": "normalized_episode_time",
                "progress_dim": 1,
            },
        },
        "step": 50_000,
    }


def gmm_checkpoint_payload() -> dict:
    model = TCCMLPGaussianMixturePolicy(
        feature_dim=3,
        num_tasks=4,
        proprio_dim=7,
        hidden_dims=(512, 512),
        input_batch_norm=True,
        dropout=0.2,
        num_modes=5,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    return {
        "model": model.state_dict(),
        "action_mean": torch.full((7,), 0.01),
        "action_std": torch.ones(7),
        "state_mean": torch.zeros(7),
        "state_std": torch.ones(7),
        "feature_dim": 3,
        "config": {
            "dataset": {"tasks": ["a", "b", "c", "d"]},
            "policy": {
                "number_of_tasks": 4,
                "action_dim": 7,
                "action_chunk_size": 1,
                "action_representation": "current_delta",
                "execution_delta_gain": 1.0,
                "action_distribution": "gaussian_mixture",
                "num_modes": 5,
                "min_std": 1e-4,
                "hidden_dimensions": [512, 512],
                "dropout": 0.2,
                "input_batch_norm": True,
                "input_layer_norm": False,
                "proprioception": True,
                "proprioception_dim": 7,
                "cameras": ["cam_main", "cam_wrist"],
                "camera_fusion": "raw_concat",
                "camera_projection_dim": 0,
                "camera_gate_hidden_dim": 0,
            },
        },
        "step": 50_000,
    }


def test_preprocess_rgb_frames_matches_training_shape() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    output = preprocess_rgb_frames([frame, frame], 32)
    assert output.shape == (2, 3, 32, 32)
    assert output.dtype == torch.float32


def test_restore_and_predict_denormalized_action(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(checkpoint_payload(), checkpoint)
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    action = predict_action(
        MeanBackbone().eval(),
        bundle,
        frame,
        frame,
        task_index=2,
        image_size=32,
        device=torch.device("cpu"),
    )
    assert action.shape == (7,)
    assert torch.isfinite(action).all()


def test_restore_and_predict_v6_gated_multiview_action(tmp_path: Path) -> None:
    model = TCCMLPPolicy(
        feature_dim=3,
        num_tasks=4,
        proprio_dim=7,
        camera_fusion="gated_residual",
        camera_projection_dim=4,
        camera_gate_hidden_dim=5,
    )
    checkpoint = tmp_path / "v6.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "action_mean": torch.zeros(7),
            "action_std": torch.ones(7),
            "state_mean": torch.zeros(7),
            "state_std": torch.ones(7),
            "feature_dim": 3,
            "config": {
                "dataset": {"tasks": ["a", "b", "c", "d"]},
                "policy": {
                    "number_of_tasks": 4,
                    "action_dim": 7,
                    "action_chunk_size": 1,
                    "action_representation": "absolute",
                    "hidden_dimensions": [256, 256],
                    "input_batch_norm": True,
                    "input_layer_norm": False,
                    "proprioception": True,
                    "proprioception_dim": 7,
                    "cameras": ["cam_main", "cam_wrist"],
                    "camera_fusion": "gated_residual",
                    "camera_projection_dim": 4,
                    "camera_gate_hidden_dim": 5,
                },
            },
            "step": 50_000,
        },
        checkpoint,
    )
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    action = predict_action(
        MeanBackbone().eval(),
        bundle,
        frame,
        frame,
        task_index=2,
        image_size=32,
        device=torch.device("cpu"),
        observation_state=np.zeros(7, dtype=np.float32),
    )

    assert action.shape == (7,)
    assert torch.isfinite(action).all()


def test_policy_feature_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(checkpoint_payload(), checkpoint)
    with pytest.raises(ValueError, match="feature mismatch"):
        load_policy_bundle(
            checkpoint, expected_feature_dim=4, device=torch.device("cpu")
        )


def test_runtime_rejects_checkpoint_with_different_proprioception(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(checkpoint_payload(), checkpoint)
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    runtime = {
        "policy": {
            "proprioception": True,
            "proprioception_dim": 7,
            "action_representation": "absolute",
        }
    }

    with pytest.raises(RuntimeError, match="contract mismatch"):
        validate_policy_contract(runtime, bundle)


def test_future_delta_policy_reconstructs_absolute_target(tmp_path: Path) -> None:
    checkpoint = tmp_path / "future_delta.pt"
    torch.save(future_delta_checkpoint_payload(), checkpoint)
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)

    action = predict_action(
        MeanBackbone().eval(),
        bundle,
        frame,
        frame,
        task_index=0,
        image_size=32,
        device=torch.device("cpu"),
        observation_state=state,
    )

    assert torch.allclose(action, torch.from_numpy(state) + 0.01)


def test_future_delta_policy_accepts_runtime_gain_override(tmp_path: Path) -> None:
    checkpoint = tmp_path / "future_delta.pt"
    torch.save(future_delta_checkpoint_payload(), checkpoint)
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)

    action = predict_action(
        MeanBackbone().eval(),
        bundle,
        frame,
        frame,
        task_index=0,
        image_size=32,
        device=torch.device("cpu"),
        observation_state=state,
        execution_delta_gain_override=0.6,
    )

    assert torch.allclose(action, torch.from_numpy(state) + 0.06)


def test_future_delta_policy_requires_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "future_delta.pt"
    torch.save(future_delta_checkpoint_payload(), checkpoint)
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="requires observation_state"):
        predict_action(
            MeanBackbone().eval(),
            bundle,
            frame,
            frame,
            task_index=0,
            image_size=32,
            device=torch.device("cpu"),
        )


def test_progress_policy_requires_episode_progress(tmp_path: Path) -> None:
    checkpoint = tmp_path / "progress.pt"
    torch.save(progress_checkpoint_payload(), checkpoint)
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="requires episode_progress"):
        predict_action(
            MeanBackbone().eval(),
            bundle,
            frame,
            frame,
            task_index=0,
            image_size=32,
            device=torch.device("cpu"),
        )


def test_r3m_robomimic_checkpoint_predicts_absolute_joint_goal(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "r3m_robomimic.pt"
    torch.save(r3m_robomimic_checkpoint_payload(), checkpoint)
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    backbone = IndependentCameraBackbones(
        MeanBackbone().eval(), MeanBackbone().eval()
    ).eval()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    action = predict_action(
        backbone,
        bundle,
        frame,
        frame,
        task_index=0,
        image_size=32,
        device=torch.device("cpu"),
    )

    assert torch.allclose(action, torch.full((7,), 0.25))


def test_progress_policy_predicts_with_normalized_episode_time(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "progress.pt"
    torch.save(progress_checkpoint_payload(), checkpoint)
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    action = predict_action(
        MeanBackbone().eval(),
        bundle,
        frame,
        frame,
        task_index=0,
        image_size=32,
        device=torch.device("cpu"),
        episode_progress=0.5,
    )

    assert action.shape == (7,)
    assert torch.isfinite(action).all()


def test_gmm_current_delta_reconstructs_absolute_target(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gmm.pt"
    torch.save(gmm_checkpoint_payload(), checkpoint)
    bundle = load_policy_bundle(
        checkpoint, expected_feature_dim=3, device=torch.device("cpu")
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)

    action = predict_action(
        MeanBackbone().eval(),
        bundle,
        frame,
        frame,
        task_index=0,
        image_size=32,
        device=torch.device("cpu"),
        observation_state=state,
    )

    assert torch.allclose(action, torch.from_numpy(state) + 0.01)


def test_end_to_end_checkpoint_restores_fine_tuned_backbone() -> None:
    backbone = torch.nn.Linear(3, 3)
    restored = torch.nn.Linear(3, 3)
    with torch.no_grad():
        backbone.weight.fill_(2.0)
        backbone.bias.fill_(1.0)
    policy_bundle = SimpleNamespace(backbone_state=backbone.state_dict())

    changed = restore_policy_backbone(restored, policy_bundle)

    assert changed is True
    assert torch.equal(restored.weight, backbone.weight)
    assert torch.equal(restored.bias, backbone.bias)


def test_independent_camera_backbones_restore_both_parameter_sets() -> None:
    original = IndependentCameraBackbones(
        MeanBackboneWithScale(2.0), MeanBackboneWithScale(3.0)
    )
    restored = IndependentCameraBackbones(
        MeanBackboneWithScale(0.0), MeanBackboneWithScale(0.0)
    )
    policy_bundle = SimpleNamespace(backbone_state=original.state_dict())

    changed = restore_policy_backbone(restored, policy_bundle)

    assert changed is True
    assert torch.equal(restored.cam_main.scale, torch.tensor(2.0))
    assert torch.equal(restored.cam_wrist.scale, torch.tensor(3.0))


def test_checkpoint_without_embedded_backbone_reports_not_restored() -> None:
    restored = torch.nn.Linear(3, 3)
    policy_bundle = SimpleNamespace(backbone_state=None)

    changed = restore_policy_backbone(restored, policy_bundle)

    assert changed is False
