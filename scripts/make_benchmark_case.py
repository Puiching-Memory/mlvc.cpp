#!/usr/bin/env python3
"""Convert one upstream MLVC model_data_*.npz snapshot to an FP16 case."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "numpy is required; run this script in the upstream MLVC Python environment"
    ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an mlvc_backend_bench case from upstream model debug data"
    )
    parser.add_argument("--input", required=True, type=Path, help="model_data_*.npz")
    parser.add_argument(
        "--model",
        required=True,
        help="model part name, for example MLVCEncoder or MLVCDecoder",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value) or "unnamed"


def normalize_array(value: np.ndarray, key: str) -> tuple[np.ndarray, str]:
    if np.issubdtype(value.dtype, np.floating):
        return np.ascontiguousarray(value, dtype=np.dtype("<f2")), "fp16"
    if np.issubdtype(value.dtype, np.integer):
        info = np.iinfo(np.int32)
        if value.size and (value.min() < info.min or value.max() > info.max):
            raise ValueError(f"{key}: integer value does not fit int32")
        return np.ascontiguousarray(value, dtype=np.dtype("<i4")), "int32"
    raise ValueError(
        f"{key}: unsupported dtype {value.dtype}; only floating-point and integer tensors are supported"
    )


def convert_group(
    archive: np.lib.npyio.NpzFile,
    keys: list[str],
    prefix: str,
    output_dir: Path,
    label: str,
) -> list[dict[str, object]]:
    tensors: list[dict[str, object]] = []
    for index, key in enumerate(keys):
        name = key[len(prefix) :]
        array, dtype = normalize_array(archive[key], key)
        file_name = f"{label}-{index:02d}-{safe_name(name)}.{dtype}.raw"
        array.tofile(output_dir / file_name)
        tensors.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": dtype,
                "file": file_name,
            }
        )
    return tensors


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_prefix = f"{args.model}_input_"
    output_prefix = f"{args.model}_output_"
    with np.load(args.input, allow_pickle=False) as archive:
        input_keys = [key for key in archive.files if key.startswith(input_prefix)]
        output_keys = [key for key in archive.files if key.startswith(output_prefix)]
        if not input_keys:
            available = sorted(
                {key.split("_input_", 1)[0] for key in archive.files if "_input_" in key}
            )
            raise ValueError(
                f"no inputs found for {args.model}; available model parts: {available}"
            )
        if not output_keys:
            raise ValueError(f"no outputs found for {args.model}")

        document = {
            "schema_version": 1,
            "precision": "fp16",
            "model": args.model,
            "source": str(args.input),
            "inputs": convert_group(
                archive, input_keys, input_prefix, args.output_dir, "input"
            ),
            "expected_outputs": convert_group(
                archive, output_keys, output_prefix, args.output_dir, "expected"
            ),
        }

    digest = hashlib.sha256()
    hash_document = dict(document)
    hash_document.pop("source")
    digest.update(
        json.dumps(hash_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for tensor in document["inputs"] + document["expected_outputs"]:
        digest.update((args.output_dir / str(tensor["file"])).read_bytes())
    document["case_id"] = digest.hexdigest()

    case_path = args.output_dir / "case.json"
    case_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(case_path)


if __name__ == "__main__":
    main()
