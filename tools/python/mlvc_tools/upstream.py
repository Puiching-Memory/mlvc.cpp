"""Isolate upstream Microsoft MLVC imports from unused exporter dependencies."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEO_ROOT = REPO_ROOT / "third_party" / "mlvc" / "video"


def ensure_upstream_conversion() -> Path:
    """Register the upstream video source path and isolated conversion package."""
    video_path = str(VIDEO_ROOT)
    if video_path not in sys.path:
        sys.path.insert(0, video_path)
    # Upstream's conversion.__init__ imports every exporter and model bundler.
    # Register its namespace directly so tooling imports only used modules.
    if "conversion" not in sys.modules:
        package = ModuleType("conversion")
        package.__path__ = [str(VIDEO_ROOT / "conversion")]
        package.__spec__ = ModuleSpec("conversion", loader=None, is_package=True)
        sys.modules["conversion"] = package
    return VIDEO_ROOT
