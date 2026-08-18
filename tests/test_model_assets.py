import hashlib
import json
from pathlib import Path

import pytest

from tcc_real_robot.model_assets import resolve_model_assets


def test_resolve_model_assets_pins_revision_and_verifies_backbone(
    tmp_path: Path,
) -> None:
    backbone = b"frozen-backbone"
    policy = b"policy-head"
    files = {
        "backbones/manifest.json": json.dumps(
            {
                "backbones": [
                    {
                        "name": "ours_rn50",
                        "repo_path": "backbones/ours_rn50/checkpoint.pt",
                        "size": len(backbone),
                        "sha256": hashlib.sha256(backbone).hexdigest(),
                    }
                ]
            }
        ).encode(),
        "backbones/ours_rn50/checkpoint.pt": backbone,
        "policies/ours_rn50/60/checkpoint.pt": policy,
        "policies/ours_rn50/60/metrics.json": b"[]",
    }
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        filename = str(kwargs["filename"])
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[filename])
        return str(path)

    config = {
        "model_hub": {
            "repository": "owner/repo",
            "revision": "a" * 40,
            "backbone_manifest": "backbones/manifest.json",
            "policy_checkpoint_template": (
                "policies/{backbone}/{demonstrations}/checkpoint.pt"
            ),
            "policy_metrics_template": (
                "policies/{backbone}/{demonstrations}/metrics.json"
            ),
            "supported_backbones": ["ours_rn50"],
            "supported_demonstrations": [60],
        }
    }
    assets = resolve_model_assets(
        config,
        "ours_rn50",
        60,
        cache_dir=tmp_path / "cache",
        download_file=fake_download,
    )

    assert assets.backbone_path.read_bytes() == backbone
    assert assets.policy_path.read_bytes() == policy
    assert assets.backbone_sha256 == hashlib.sha256(backbone).hexdigest()
    assert assets.policy_sha256 == hashlib.sha256(policy).hexdigest()
    assert all(call["revision"] == "a" * 40 for call in calls)


def test_resolve_model_assets_rejects_backbone_hash_mismatch(
    tmp_path: Path,
) -> None:
    files = {
        "manifest.json": json.dumps(
            {
                "backbones": [
                    {
                        "name": "ours_vit",
                        "repo_path": "backbone.pt",
                        "size": 3,
                        "sha256": "0" * 64,
                    }
                ]
            }
        ).encode(),
        "backbone.pt": b"bad",
    }

    def fake_download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        path = tmp_path / filename
        path.write_bytes(files[filename])
        return str(path)

    config = {
        "model_hub": {
            "repository": "owner/repo",
            "revision": "b" * 40,
            "backbone_manifest": "manifest.json",
            "policy_checkpoint_template": "unused",
            "policy_metrics_template": "unused",
            "supported_backbones": ["ours_vit"],
            "supported_demonstrations": [60],
        }
    }
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        resolve_model_assets(
            config,
            "ours_vit",
            60,
            download_file=fake_download,
        )
