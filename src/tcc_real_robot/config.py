"""Configuration loading with fail-closed action validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return data


def assert_actuation_disabled(config: dict[str, Any]) -> None:
    """Fail unless the initial repository's action contract remains disabled."""
    enabled = config.get("action_contract", {}).get("enabled")
    if enabled is not False:
        raise RuntimeError("Actuation must remain disabled in the initial revision")
