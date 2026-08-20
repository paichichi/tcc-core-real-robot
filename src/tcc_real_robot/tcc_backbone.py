"""Load the trained backbone from a TCC-Core checkpoint without camera-slot fusion."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    if isinstance(args, dict):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    raise TypeError("TCC checkpoint args must be a mapping or argparse namespace")


class _SpatialAveragePool(nn.Module):
    """Expose an HRAlign spatial backbone through the pooled feature contract."""

    def __init__(self, spatial_backbone: nn.Module) -> None:
        super().__init__()
        self.spatial_backbone = spatial_backbone
        self.output_dim = int(spatial_backbone.output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.spatial_backbone(images)
        return torch.flatten(F.adaptive_avg_pool2d(features, (1, 1)), 1)


def _freeze(backbone: nn.Module, device: torch.device) -> nn.Module:
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone = backbone.eval().to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        backbone = backbone.to(memory_format=torch.channels_last)
    return backbone


def load_frozen_tcc_backbone(
    checkpoint_path: str | Path,
    tcc_source_root: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore only ``model.backbone`` from a full TCC training checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    source_root = Path(tcc_source_root).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not (source_root / "xirl" / "models.py").is_file():
        raise FileNotFoundError(source_root / "xirl" / "models.py")

    source_string = str(source_root)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    models = importlib.import_module("xirl.models")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary checkpoint")

    model_state = checkpoint.get("model")
    if isinstance(model_state, dict) and any(
        key.startswith("backbone.") for key in model_state
    ):
        args = _checkpoint_args(checkpoint)
        backbone_name = str(args.get("backbone", "vit_b16"))
        backbone = models.build_backbone(
            backbone=backbone_name,
            pretrain_path="",
            train_norm_affine=False,
            train_adapters=False,
        )
        prefix = "backbone."
        state = {
            key.removeprefix(prefix): value
            for key, value in model_state.items()
            if key.startswith(prefix)
        }
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "TCC backbone checkpoint mismatch: "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}"
            )
        source_format = "tcc-training"
        image_size = int(args.get("image_size", 224))
    elif (
        isinstance(model_state, dict)
        and "cls_token" in model_state
        and any(key.startswith("blocks.") for key in model_state)
    ):
        # D4R/HRP ImageNet releases are raw MAE/timm-style ViT checkpoints.
        # ``ViTB16Backbone.load_pretrained`` performs the audited key mapping
        # into torchvision's ViT-B/16 implementation.
        backbone_name = "vit_b16"
        backbone = models.build_backbone(
            backbone=backbone_name,
            pretrain_path=str(checkpoint_path),
            train_norm_affine=False,
            train_adapters=False,
        )
        source_format = "mae-vit-release"
        image_size = 224
    elif checkpoint.get("format") == "hralign-reproduction-v1":
        recorded_root = checkpoint["config"].get("_meta", {}).get("project_root")
        reproduction_root = (
            Path(recorded_root)
            if recorded_root
            else source_root / "task_progression_annotation" / "hralign_reproduction"
        )
        if not (reproduction_root / "hralign" / "models.py").is_file():
            raise FileNotFoundError(reproduction_root / "hralign" / "models.py")
        reproduction_string = str(reproduction_root)
        if reproduction_string not in sys.path:
            sys.path.insert(0, reproduction_string)
        hralign_models = importlib.import_module("hralign.models")
        model_config = checkpoint["config"]["model"]
        core = hralign_models.HRAlignR3ML(
            pretrain_path=model_config["pretrain"],
            adapted_bn_mode=model_config["adapted_bn_mode"],
            normalize_language_query=model_config["normalize_language_query"],
            normalize_visual_tokens_for_attention=model_config[
                "normalize_visual_tokens_for_attention"
            ],
            normalize_pooled_features=model_config["normalize_pooled_features"],
        )
        core.load_trainable_state(checkpoint["trainable_state"])
        backbone = _SpatialAveragePool(core.adapted)
        backbone_name = "hralign_reproduced_r3m_l"
        source_format = "hralign-reproduction-v1"
        image_size = int(checkpoint["config"]["sampling"]["crop_size"])
    elif isinstance(checkpoint.get("r3m"), dict):
        backbone_name = "r3m_resnet50"
        backbone = models.build_backbone(
            backbone=backbone_name,
            pretrain_path=str(checkpoint_path),
            train_norm_affine=False,
            train_adapters=False,
        )
        source_format = "r3m-release"
        image_size = 224
    else:
        raise TypeError(
            "Unsupported TCC/R3M/HRAlign/D4R/HRP checkpoint format"
        )

    backbone = _freeze(backbone, device)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "backbone": backbone_name,
        "feature_dim": int(backbone.output_dim),
        "image_size": image_size,
        "source_format": source_format,
    }
    return backbone, metadata
