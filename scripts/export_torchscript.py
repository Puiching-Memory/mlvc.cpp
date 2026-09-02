#!/usr/bin/env python3
"""Run the upstream converter with an FP16 TorchScript exporter."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MLVC_ROOT = ROOT / "third_party" / "mlvc"
VIDEO_ROOT = MLVC_ROOT / "video"
sys.path.insert(0, str(VIDEO_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from conversion._full_model import _model_factory  # noqa: E402
from conversion._exporter._torch_exporter import TorchExporter  # noqa: E402
from conversion.types import ModelData, ModelPrecision  # noqa: E402
from conversion.utils import to_torch_namedtuple  # noqa: E402
from model_profiles import configure_model_factory  # noqa: E402


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


def move_unregistered_tensors(model: torch.nn.Module, device: torch.device) -> None:
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if isinstance(value, torch.Tensor):
                setattr(
                    module,
                    name,
                    value.to(
                        device=device,
                        dtype=torch.float16 if value.is_floating_point() else value.dtype,
                    ),
                )


@torch.inference_mode()
def export_torchscript(
    exporter: TorchExporter,
    model_name: str,
    model: torch.nn.Module,
    example_model_data: list[ModelData],
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
    save_benchmark_snapshot(model_name, example_model_data[-1], output_path)


TorchExporter._export = export_torchscript
configure_model_factory(_model_factory)
sys.argv[0] = str(VIDEO_ROOT / "convert.py")
runpy.run_path(sys.argv[0], run_name="__main__")
