import pytest

torch = pytest.importorskip("torch")

from tcc_real_robot.r3m_vision import (
    build_r3m_train_transform,
    build_r3m_transform,
)


def test_training_augmentation_is_seeded_and_preserves_tensor_contract() -> None:
    image = torch.arange(3 * 48 * 64, dtype=torch.uint8).reshape(3, 48, 64)
    transform = build_r3m_train_transform(32, blur_probability=1.0)

    torch.manual_seed(7)
    first = transform(image)
    torch.manual_seed(7)
    repeated = transform(image)
    torch.manual_seed(8)
    different = transform(image)

    assert first.shape == (3, 32, 32)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_evaluation_transform_remains_deterministic() -> None:
    image = torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8)
    transform = build_r3m_transform(32)

    assert torch.equal(transform(image), transform(image))
