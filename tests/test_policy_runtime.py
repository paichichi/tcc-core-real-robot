from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tcc_real_robot.policy import TCCMLPPolicy
from tcc_real_robot.policy_runtime import (
    load_policy_bundle,
    predict_action,
    preprocess_rgb_frames,
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


def test_policy_feature_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(checkpoint_payload(), checkpoint)
    with pytest.raises(ValueError, match="feature mismatch"):
        load_policy_bundle(
            checkpoint, expected_feature_dim=4, device=torch.device("cpu")
        )
