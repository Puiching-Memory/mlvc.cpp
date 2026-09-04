#!/usr/bin/env python3
"""Export MLVC models or verify a registered checkpoint.

Usage:
  model_convert.py onnx <upstream convert.py arguments>
  model_convert.py torchscript <upstream convert.py arguments>
  model_convert.py verify --model-version NAME --checkpoint FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "third_party" / "mlvc" / "video"
PROFILE_PATH = ROOT / "models" / "profiles" / "profiles.json"
sys.path.insert(0, str(VIDEO_ROOT))


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


def save_benchmark_snapshot(
    model_name: str, model_data: Any, output_path: Path, np: Any
) -> None:
    values = {
        **{
            f"{model_name}_input_{name}": value
            for name, value in model_data.inputs._asdict().items()
        },
        **{
            f"{model_name}_output_{name}": value
            for name, value in model_data.outputs._asdict().items()
        },
    }
    destination = output_path / f"benchmark-{model_name}.npz"
    np.savez(destination, **values)
    print(f"Saved benchmark snapshot to {destination}")


def run_upstream_converter(model_factory: Any) -> None:
    configure_model_factory(model_factory)
    converter = VIDEO_ROOT / "convert.py"
    sys.argv = [str(converter), *sys.argv[2:]]
    runpy.run_path(str(converter), run_name="__main__")


def run_onnx_converter() -> None:
    import numpy as np

    from conversion._exporter._onnx_exporter import OnnxExporter
    from conversion._full_model import _model_factory
    from conversion.types import ModelPrecision

    original_export = OnnxExporter._export
    optimization_passes = [
        "onnxscript_optimizations",
        "use_space_to_depth",
        "depth_to_space_crd_to_dcr",
    ]

    def export_onnx(
        exporter: Any,
        model_name: str,
        model: Any,
        example_model_data: list[Any],
        output_path: Path,
        fake_quantized: bool = False,
    ) -> None:
        if exporter._precision != ModelPrecision.FP16:
            raise ValueError("mlvc.cpp supports FP16 ONNX exports only")
        exporter._optimization_passes = optimization_passes
        original_export(
            exporter,
            model_name,
            model,
            example_model_data,
            output_path,
            fake_quantized,
        )
        save_benchmark_snapshot(model_name, example_model_data[-1], output_path, np)

    OnnxExporter._export = export_onnx
    run_upstream_converter(_model_factory)


def run_torchscript_converter() -> None:
    import numpy as np
    import torch

    from conversion._exporter._torch_exporter import TorchExporter
    from conversion._full_model import _model_factory
    from conversion.types import ModelPrecision
    from conversion.utils import to_torch_namedtuple

    def move_unregistered_tensors(model: Any, device: Any) -> None:
        for module in model.modules():
            for name, value in list(vars(module).items()):
                if isinstance(value, torch.Tensor):
                    setattr(
                        module,
                        name,
                        value.to(
                            device=device,
                            dtype=(
                                torch.float16
                                if value.is_floating_point()
                                else value.dtype
                            ),
                        ),
                    )

    @torch.inference_mode()
    def export_torchscript(
        exporter: Any,
        model_name: str,
        model: Any,
        example_model_data: list[Any],
        output_path: Path,
        fake_quantized: bool = False,
    ) -> None:
        del fake_quantized
        if exporter._precision != ModelPrecision.FP16:
            raise ValueError("mlvc.cpp supports FP16 TorchScript exports only")
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda")
        if device.type != "cuda":
            device = torch.device("cuda")

        model = model.eval().half().to(device=device)
        move_unregistered_tensors(model, device)
        named_inputs = to_torch_namedtuple(example_model_data[-1].inputs)
        inputs = tuple(
            value.to(
                device=device,
                dtype=torch.float16 if value.is_floating_point() else value.dtype,
            )
            for value in named_inputs
        )
        traced = torch.jit.trace(model, inputs, strict=False, check_trace=False)
        traced = torch.jit.freeze(traced.eval())
        destination = output_path / f"{model_name}.ts"
        traced.save(str(destination))
        print(f"Saved FP16 TorchScript model to {destination}")
        save_benchmark_snapshot(model_name, example_model_data[-1], output_path, np)

    TorchExporter._export = export_torchscript
    run_upstream_converter(_model_factory)


def run_verification(arguments: list[str]) -> None:
    from conversion._full_model import _model_factory

    parser = argparse.ArgumentParser(prog="model_convert.py verify")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(arguments)

    checkpoint = args.checkpoint.resolve()
    verify_checkpoint(args.model_version, checkpoint)
    configure_model_factory(_model_factory)
    model = _model_factory.full_model_factory(
        args.model_version,
        weights_path=checkpoint,
    )
    params = model.model_params
    print(
        f"verified {params.model_version}: feature={params.feature_channels}, "
        f"y={params.latent_channels}, z={params.hyperprior_channels}, "
        f"hyper-downsample={params.downsample_hyperprior}"
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__.strip())
        return
    command = sys.argv[1]
    if command == "onnx":
        run_onnx_converter()
    elif command == "torchscript":
        run_torchscript_converter()
    elif command == "verify":
        run_verification(sys.argv[2:])
    else:
        raise SystemExit(
            f"unknown command {command!r}; expected onnx, torchscript, or verify"
        )


if __name__ == "__main__":
    main()
