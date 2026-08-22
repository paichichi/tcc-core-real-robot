from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tcc_real_robot import tcc_backbone


class FakeViT(torch.nn.Module):
    output_dim = 768

    def __init__(self) -> None:
        super().__init__()
        self.pooling = "patch_mean"
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.new_zeros((images.shape[0], self.output_dim))


def test_raw_hrp_d4r_mae_vit_uses_official_cls_pooling(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "tcc"
    (source_root / "xirl").mkdir(parents=True)
    (source_root / "xirl" / "models.py").write_text("# fake\n")
    checkpoint = tmp_path / "HRP_IN.pth"
    torch.save(
        {
            "model": {
                "cls_token": torch.zeros((1, 1, 768)),
                "blocks.0.norm1.weight": torch.ones(768),
            }
        },
        checkpoint,
    )
    fake = FakeViT()
    module = SimpleNamespace(build_backbone=lambda **_: fake)
    monkeypatch.setattr(tcc_backbone.importlib, "import_module", lambda _: module)

    loaded, metadata = tcc_backbone.load_frozen_tcc_backbone(
        checkpoint, source_root, torch.device("cpu")
    )

    assert loaded is fake
    assert loaded.pooling == "cls"
    assert metadata["source_format"] == "mae-vit-release"
    assert metadata["pooling"] == "cls"
