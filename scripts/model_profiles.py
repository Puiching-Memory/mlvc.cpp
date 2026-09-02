#!/usr/bin/env python3
"""Shared MLVC/MLVC-S converter profile registration and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs" / "model_profiles.json"


def load_profiles() -> dict[str, dict[str, Any]]:
    document = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported profile schema: {PROFILE_PATH}")
    return document["profiles"]


def configure_model_factory(model_factory: Any) -> None:
    configs: dict[str, dict[str, Any]] = {}
    for profile in load_profiles().values():
        model_version = profile["model_version"]
        configs[model_version] = {
            "module": "._dmc61sb_model",
            "class": "TraceableMLVC",
            "weights_path": None,
            "weights_version": profile["weights_version"],
            "iframe_period": profile["iframe_period"],
            "reset_period": profile["reset_period"],
            "split_type": profile["split_type"],
            "params": profile["params"],
        }
    model_factory._config_cache = configs


def profile_for_model(model_version: str) -> dict[str, Any]:
    for profile in load_profiles().values():
        if profile["model_version"] == model_version:
            return profile
    raise ValueError(f"unknown model profile: {model_version}")


def verify_checkpoint(model_version: str, checkpoint: Path) -> None:
    profile = profile_for_model(model_version)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != profile["weights_sha256"]:
        raise ValueError(
            f"checkpoint SHA-256 mismatch for {model_version}: "
            f"expected {profile['weights_sha256']}, got {digest}"
        )
