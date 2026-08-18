"""Resolve pinned frozen-backbone and policy assets from Hugging Face."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

DownloadFile = Callable[..., str]


@dataclass(frozen=True)
class ResolvedModelAssets:
    repository: str
    revision: str
    backbone: str
    demonstrations: int
    backbone_path: Path
    backbone_sha256: str
    policy_path: Path
    policy_sha256: str
    metrics_path: Path

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("backbone_path", "policy_path", "metrics_path"):
            result[key] = str(result[key])
        return result


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a local file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    download_file: DownloadFile,
    repository: str,
    revision: str,
    filename: str,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> Path:
    return Path(
        download_file(
            repo_id=repository,
            filename=filename,
            revision=revision,
            repo_type="model",
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
    )


def resolve_backbone_asset(
    config: dict[str, Any],
    backbone: str,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    download_file: DownloadFile = hf_hub_download,
) -> tuple[Path, str]:
    """Download and SHA256-verify one pinned frozen-backbone checkpoint."""
    hub = config["model_hub"]
    supported = set(hub["supported_backbones"])
    if backbone not in supported:
        raise ValueError(
            f"Unsupported backbone {backbone!r}; expected one of {sorted(supported)}"
        )
    repository = str(hub["repository"])
    revision = str(hub["revision"])
    manifest_path = _download(
        download_file,
        repository,
        revision,
        str(hub["backbone_manifest"]),
        cache_dir,
        local_files_only,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {
        str(row["name"]): row for row in manifest.get("backbones", [])
    }
    if backbone not in entries:
        raise ValueError(f"Backbone {backbone!r} is missing from the Hub manifest")
    entry = entries[backbone]
    checkpoint = _download(
        download_file,
        repository,
        revision,
        str(entry["repo_path"]),
        cache_dir,
        local_files_only,
    )
    expected_size = int(entry["size"])
    if checkpoint.stat().st_size != expected_size:
        raise RuntimeError(
            f"Backbone size mismatch for {backbone}: "
            f"{checkpoint.stat().st_size} != {expected_size}"
        )
    expected_sha256 = str(entry["sha256"])
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Backbone SHA256 mismatch for {backbone}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return checkpoint, actual_sha256


def resolve_model_assets(
    config: dict[str, Any],
    backbone: str,
    demonstrations: int,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    download_file: DownloadFile = hf_hub_download,
) -> ResolvedModelAssets:
    """Resolve one verified backbone and its matching trained policy head."""
    hub = config["model_hub"]
    supported_demonstrations = {
        int(value) for value in hub["supported_demonstrations"]
    }
    if demonstrations not in supported_demonstrations:
        raise ValueError(
            f"Unsupported demonstration count {demonstrations}; expected one of "
            f"{sorted(supported_demonstrations)}"
        )
    backbone_path, backbone_sha256 = resolve_backbone_asset(
        config,
        backbone,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        download_file=download_file,
    )
    repository = str(hub["repository"])
    revision = str(hub["revision"])
    template_values = {
        "backbone": backbone,
        "demonstrations": demonstrations,
    }
    policy_path = _download(
        download_file,
        repository,
        revision,
        str(hub["policy_checkpoint_template"]).format(**template_values),
        cache_dir,
        local_files_only,
    )
    metrics_path = _download(
        download_file,
        repository,
        revision,
        str(hub["policy_metrics_template"]).format(**template_values),
        cache_dir,
        local_files_only,
    )
    return ResolvedModelAssets(
        repository=repository,
        revision=revision,
        backbone=backbone,
        demonstrations=demonstrations,
        backbone_path=backbone_path,
        backbone_sha256=backbone_sha256,
        policy_path=policy_path,
        policy_sha256=sha256_file(policy_path),
        metrics_path=metrics_path,
    )
