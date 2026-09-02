#!/usr/bin/env python3
"""Run the upstream converter and capture FP16 ONNX benchmark tensors."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MLVC_ROOT = ROOT / "third_party" / "mlvc"
VIDEO_ROOT = MLVC_ROOT / "video"
sys.path.insert(0, str(VIDEO_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from conversion._exporter._onnx_exporter import OnnxExporter  # noqa: E402
from conversion._full_model import _model_factory  # noqa: E402
from conversion.types import ModelData, ModelPrecision  # noqa: E402
from model_profiles import configure_model_factory  # noqa: E402


ORIGINAL_EXPORT = OnnxExporter._export
NVIDIA_OPTIMIZATION_PASSES = [
    "onnxscript_optimizations",
    "use_space_to_depth",
    "depth_to_space_crd_to_dcr",
]


def save_benchmark_snapshot(
    model_name: str, model_data: ModelData, output_path: Path
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


def export_onnx(
    exporter: OnnxExporter,
    model_name: str,
    model,
    example_model_data: list[ModelData],
    output_path: Path,
    fake_quantized: bool = False,
) -> None:
    if exporter._precision != ModelPrecision.FP16:
        raise ValueError("mlvc.cpp supports FP16 ONNX exports only")
    exporter._optimization_passes = NVIDIA_OPTIMIZATION_PASSES
    ORIGINAL_EXPORT(
        exporter,
        model_name,
        model,
        example_model_data,
        output_path,
        fake_quantized,
    )
    save_benchmark_snapshot(model_name, example_model_data[-1], output_path)


OnnxExporter._export = export_onnx
configure_model_factory(_model_factory)
sys.argv[0] = str(VIDEO_ROOT / "convert.py")
runpy.run_path(sys.argv[0], run_name="__main__")
