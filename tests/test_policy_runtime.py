from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tcc_real_robot.policy import (
    HRPSingleViewGaussianMixturePolicy,
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


class MeanBackbone(torch.nn.Module):
    output_dim = 3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))


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


def hrp_gmm_checkpoint_payload() -> dict:
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


def hrp_single_view_checkpoint_payload() -> dict:
    model = HRPSingleViewGaussianMixturePolicy(
        feature_dim=3,
        action_dim=7,
        state_dim=7,
        hidden_dims=(4, 4),
        num_modes=5,
        dropout=0.0,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        model.mixture_means.bias.reshape(5, 7)[2].fill_(0.25)
        model.mixture_logits.bias.copy_(torch.tensor([0.0, 1.0, 5.0, 2.0, 0.0]))
    return {
        "model": model.state_dict(),
        "action_mean": torch.zeros(7),
        "action_std": torch.ones(7),
        "state_mean": torch.zeros(7),
        "state_std": torch.ones(7),
        "feature_dim": 3,
        "config": {
            "dataset": {"tasks": ["carrot"]},
            "policy": {
                "architecture": "hrp_state_token_gmm",
                "number_of_tasks": 1,
                "action_dim": 7,
                "action_chunk_size": 1,
                "action_representation": "absolute",
                "action_distribution": "gaussian_mixture",
                "num_modes": 5,
                "min_std": 1e-4,
                "hidden_dimensions": [4, 4],
                "dropout": 0.0,
                "precision": "float32",
                "deterministic_inference": "highest_probability_mode_mean",
                "proprioception": True,
                "proprioception_dim": 7,
                "cameras": ["cam_main"],
            },
        },
        "step": 40_000,
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


def test_hrp_gmm_current_delta_reconstructs_absolute_target(tmp_path: Path) -> None:
    checkpoint = tmp_path / "hrp_gmm.pt"
    torch.save(hrp_gmm_checkpoint_payload(), checkpoint)
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


def test_hrp_single_view_can_use_highest_probability_mode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "hrp_single_view.pt"
    torch.save(hrp_single_view_checkpoint_payload(), checkpoint)
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
        observation_state=np.zeros(7, dtype=np.float32),
        gmm_inference_override="highest-probability-mode",
    )

    assert torch.allclose(action, torch.full((7,), 0.25))


def test_hrp_checkpoint_can_require_deterministic_inference(tmp_path: Path) -> None:
    checkpoint = tmp_path / "hrp_deterministic.pt"
    torch.save(hrp_single_view_checkpoint_payload(), checkpoint)
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
        observation_state=np.zeros(7, dtype=np.float32),
    )

    assert torch.allclose(action, torch.full((7,), 0.25))


def test_hrp_single_view_delta_is_reconstructed_as_absolute_target(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "hrp_delta.pt"
    payload = hrp_single_view_checkpoint_payload()
    payload["config"]["policy"]["action_representation"] = "current_delta"
    torch.save(payload, checkpoint)
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

    assert torch.allclose(action, torch.from_numpy(state) + 0.25)


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


def test_checkpoint_without_embedded_backbone_reports_not_restored() -> None:
    restored = torch.nn.Linear(3, 3)
    policy_bundle = SimpleNamespace(backbone_state=None)

    changed = restore_policy_backbone(restored, policy_bundle)

    assert changed is False
