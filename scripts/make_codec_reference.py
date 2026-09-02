#!/usr/bin/env python3
"""Generate an official-Python MLVC codec conformance fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "third_party" / "mlvc" / "video"
sys.path.insert(0, str(VIDEO_ROOT))

import numpy as np  # noqa: E402

from conversion._frame_loop import FrameLoop  # noqa: E402
from conversion._split_model import load_split_model  # noqa: E402
from conversion.types import OnnxExecutionProvider, RuntimeParams  # noqa: E402
from conversion.utils import read_video_frames  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--q-index", type=int, default=21)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execution-provider",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    return parser.parse_args()


def save_tensor_set(
    root: Path,
    stage: str,
    frame_index: int,
    direction: str,
    values: Any,
) -> list[dict[str, Any]]:
    directory = root / stage / f"frame-{frame_index:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, value) in enumerate(values._asdict().items()):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            array = np.ascontiguousarray(array, dtype=np.float16)
            dtype = "fp16"
        elif np.issubdtype(array.dtype, np.integer):
            array = np.ascontiguousarray(array, dtype=np.int32)
            dtype = "int32"
        else:
            raise TypeError(f"unsupported reference tensor dtype: {array.dtype}")
        filename = f"{direction}-{index:02d}-{name}.{dtype}.raw"
        path = directory / filename
        path.write_bytes(array.tobytes(order="C"))
        records.append(
            {
                "direction": direction,
                "index": index,
                "name": name,
                "dtype": dtype,
                "shape": list(array.shape),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "path": path.relative_to(root).as_posix(),
            }
        )
    (directory / f"{direction}s.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    return records


def save_repeated_input(
    source: Path, destination: Path, width: int, height: int, frame_count: int
) -> None:
    def to_bytes(array: np.ndarray) -> bytes:
        values = np.asarray(array)
        if np.issubdtype(values.dtype, np.floating):
            values = np.rint(np.clip(values, 0.0, 1.0) * 255.0)
        return np.ascontiguousarray(values, dtype=np.uint8).tobytes()

    frames = read_video_frames(source, width, height, frame_count)
    with destination.open("wb") as stream:
        for y, uv in frames:
            stream.write(to_bytes(y))
            stream.write(to_bytes(uv[0]))
            stream.write(to_bytes(uv[1]))


def frame_payloads(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    position = 0
    records = []
    while position < len(data):
        if position + 8 > len(data):
            raise ValueError("truncated reference bitstream header")
        q_index, size = struct.unpack_from("<iI", data, position)
        position += 8
        payload = data[position : position + size]
        if len(payload) != size:
            raise ValueError("truncated reference bitstream payload")
        records.append(
            {
                "q_index": q_index,
                "payload_bytes": size,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        position += size
    return records


def main() -> None:
    args = parse_args()
    if args.frames < 2:
        raise ValueError("codec conformance fixtures require at least I and P frames")
    model_dir = args.model_dir.resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    provider = (
        OnnxExecutionProvider.CUDA
        if args.execution_provider == "cuda"
        else OnnxExecutionProvider.CPU
    )
    split_model = load_split_model(
        model_dir,
        RuntimeParams(onnx_execution_provider=provider),
    )
    frame_loop_output = output_dir / "python-output"
    results = FrameLoop(
        split_model,
        input_path,
        args.width,
        args.height,
        args.frames,
        output_data_dir=frame_loop_output,
    ).run(
        q_index=args.q_index,
        output_model_data=True,
        progress_bar=False,
        save_yuv=True,
    )

    generated = frame_loop_output / f"q_index={args.q_index}"
    shutil.move(generated / "output.mlvc", output_dir / "reference.mlvc")
    shutil.move(generated / "output.yuv", output_dir / "reference.yuv")
    shutil.rmtree(frame_loop_output)
    save_repeated_input(
        input_path,
        output_dir / "input.yuv",
        args.width,
        args.height,
        args.frames,
    )

    tensor_records = []
    for frame_index, frame in enumerate(results.frames):
        for model_part, model_data in frame.all_model_data.items():
            if model_part.value == "MLVCEncoder":
                stage = "encoder"
            elif model_part.value == "MLVCDecoder":
                stage = "decoder"
            else:
                raise ValueError(
                    f"conformance generator requires the e1d1 split, got {model_part.value}"
                )
            tensor_records.extend(
                save_tensor_set(
                    output_dir, stage, frame_index, "input", model_data.inputs
                )
            )
            tensor_records.extend(
                save_tensor_set(
                    output_dir, stage, frame_index, "output", model_data.outputs
                )
            )

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    bundle_path = model_dir / "model_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8")) if bundle_path.exists() else None
    manifest = {
        "schema_version": 1,
        "reference_implementation": "microsoft/mlvc-python-split-model",
        "execution_provider": args.execution_provider,
        "profile": metadata["name"],
        "model_version": metadata["params"]["full_model_params"]["model_version"],
        "model_metadata_sha256": sha256(model_dir / "metadata.json"),
        "entropy_model_sha256": bundle.get("entropy_model_sha256") if bundle else None,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "q_index": args.q_index,
        "input": {"path": "input.yuv", "sha256": sha256(output_dir / "input.yuv")},
        "bitstream": {
            "path": "reference.mlvc",
            "sha256": sha256(output_dir / "reference.mlvc"),
            "frames": frame_payloads(output_dir / "reference.mlvc"),
        },
        "reconstruction": {
            "path": "reference.yuv",
            "sha256": sha256(output_dir / "reference.yuv"),
        },
        "contract": {
            "exact_encoder_frames": [0],
            "decoder_max_yuv_sample_error": 1,
            "decoder_min_psnr_db": 60.0,
            "tensor_max_abs_error": 0.25,
            "tensor_max_rmse": 0.007,
        },
        "tensors": tensor_records,
    }
    (output_dir / "reference.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"generated {manifest['profile']} codec reference: "
        f"{args.frames} frames, {len(tensor_records)} tensors, {output_dir}"
    )


if __name__ == "__main__":
    main()
