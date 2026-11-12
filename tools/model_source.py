"""Construct official MLVC split models directly from registered checkpoints."""

import hashlib
import json
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/mlvc/video"))

# Upstream's convenience __init__ imports every exporter and model bundler.
# Load its source package namespace directly so checkpoint tooling imports only
# the model/codec modules it actually uses, without optional export dependencies.
if "conversion" not in sys.modules:
    package = ModuleType("conversion")
    package.__path__ = [str(ROOT / "third_party/mlvc/video/conversion")]
    package.__spec__ = ModuleSpec("conversion", loader=None, is_package=True)
    sys.modules["conversion"] = package


def load_profiles():
    return json.loads((ROOT / "models/profiles/profiles.json").read_text())["profiles"]


def load_model(profile_name, checkpoint, width, height):
    from conversion._full_model import _model_factory
    from conversion._split_model import split_full_model
    from conversion.types import ModelPrecision

    profiles = load_profiles()
    profile = profiles[profile_name]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != profile["weights_sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch for {profile_name}: {digest}")
    _model_factory._config_cache = {
        p["model_version"]: dict(
            module="._dmc61sb_model", **{"class": "TraceableMLVC"},
            weights_version=p["weights_version"], iframe_period=p["iframe_period"],
            reset_period=p["reset_period"], split_type=p["split_type"], params=p["params"])
        for p in profiles.values()
    }
    full = _model_factory.full_model_factory(
        profile["model_version"], weights_path=checkpoint.resolve())
    # Entropy tables use FP32 parameters, before neural graphs are cast to FP16.
    gaussian, estimator = full.gaussian_coder_pmf, full.bit_estimator_pmf
    full.eval()
    split = split_full_model(full, split_type=profile["split_type"],
                             model_width=width, model_height=height)
    # SplitModel constructs its Python entropy coders from the full model here.
    # Keep that construction in FP32 too, so Python and packaged C++ use the
    # very same tables. Casting first would silently regenerate different PMFs.
    full.half()
    for wrapper in split.model_parts.values():
        wrapper.torch_model.half()
        wrapper._metadata = replace(wrapper.metadata, precision=ModelPrecision.FP16)
    return full, split, gaussian, estimator


def example_inputs(part, params, width, height):
    import torch

    reference = torch.zeros(1, params.feature_channels,
        height // params.downsample_feature, width // params.downsample_feature,
        dtype=torch.float16)
    q_index = torch.tensor([21], dtype=torch.int32)
    if part == "MLVCEncoder":
        return (torch.full((1, 3, height, width), 0.5, dtype=torch.float16),
                reference, q_index)
    yh = (height + params.downsample_latent - 1) // params.downsample_latent
    yw = (width + params.downsample_latent - 1) // params.downsample_latent
    zh = (yh + params.downsample_hyperprior - 1) // params.downsample_hyperprior
    zw = (yw + params.downsample_hyperprior - 1) // params.downsample_hyperprior
    z = torch.zeros(1, params.hyperprior_channels, zh, zw, dtype=torch.float16)
    y = torch.zeros(1, params.latent_channels // 2, yh, yw, dtype=torch.float16)
    return z, y, y.clone(), reference, q_index
