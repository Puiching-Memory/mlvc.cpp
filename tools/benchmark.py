#!/usr/bin/env python3
"""Prepare and compare FP16 backend benchmark data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def require_numpy() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise SystemExit(
            "numpy is required for make-case; run this command in the "
            "upstream MLVC Python environment"
        ) from error
    return numpy


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value) or "unnamed"


def normalize_array(value: Any, key: str, np: Any) -> tuple[Any, str]:
    if np.issubdtype(value.dtype, np.floating):
        return np.ascontiguousarray(value, dtype=np.dtype("<f2")), "fp16"
    if np.issubdtype(value.dtype, np.integer):
        info = np.iinfo(np.int32)
        if value.size and (value.min() < info.min or value.max() > info.max):
            raise ValueError(f"{key}: integer value does not fit int32")
        return np.ascontiguousarray(value, dtype=np.dtype("<i4")), "int32"
    raise ValueError(
        f"{key}: unsupported dtype {value.dtype}; only floating-point and "
        "integer tensors are supported"
    )


def convert_group(
    archive: Any,
    keys: list[str],
    prefix: str,
    output_dir: Path,
    label: str,
    np: Any,
) -> list[dict[str, object]]:
    tensors: list[dict[str, object]] = []
    for index, key in enumerate(keys):
        name = key[len(prefix) :]
        array, dtype = normalize_array(archive[key], key, np)
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


def make_case(args: argparse.Namespace) -> None:
    np = require_numpy()
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
                archive, input_keys, input_prefix, args.output_dir, "input", np
            ),
            "expected_outputs": convert_group(
                archive, output_keys, output_prefix, args.output_dir, "expected", np
            ),
        }

    digest = hashlib.sha256()
    hash_document = dict(document)
    hash_document.pop("source")
    digest.update(
        json.dumps(hash_document, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for tensor in document["inputs"] + document["expected_outputs"]:
        digest.update((args.output_dir / str(tensor["file"])).read_bytes())
    document["case_id"] = digest.hexdigest()

    case_path = args.output_dir / "case.json"
    case_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(case_path)


def load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    if result.get("precision") != "fp16":
        raise ValueError(f"{path}: result is not FP16")
    result["_path"] = str(path)
    return result


def worst_metric(result: dict[str, Any], name: str) -> float | None:
    reference = result.get("reference")
    if not reference:
        return None
    values = [
        output["comparison"].get(name)
        for output in reference["outputs"]
        if output["dtype"] == "fp16"
    ]
    values = [float(value) for value in values if value is not None]
    return max(values) if values else None


def format_number(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def compare(args: argparse.Namespace) -> None:
    results = [load_result(path) for path in args.results]
    comparable_fields = (
        "case_id",
        "model",
        "precision",
        "timing_scope",
        "implementation",
        "device_id",
        "warmup_iterations",
        "iterations",
    )
    baseline = results[0]
    if not baseline.get("case_id"):
        raise ValueError("results must contain a non-empty case_id")
    for result in results[1:]:
        for field in comparable_fields:
            if result.get(field) != baseline.get(field):
                raise ValueError(
                    f"incomparable results: {field} differs between "
                    f"{baseline['_path']} and {result['_path']}"
                )

    rows: list[dict[str, Any]] = []
    fastest_mean = min(float(result["latency_ms"]["mean"]) for result in results)
    for result in sorted(results, key=lambda item: float(item["latency_ms"]["mean"])):
        mean = float(result["latency_ms"]["mean"])
        rows.append(
            {
                "backend": result["backend"],
                "model_load_ms": float(result["model_load_ms"]),
                "mean_ms": mean,
                "p50_ms": float(result["latency_ms"]["p50"]),
                "p95_ms": float(result["latency_ms"]["p95"]),
                "p99_ms": float(result["latency_ms"]["p99"]),
                "throughput": float(result["throughput_inferences_per_second"]),
                "relative_to_fastest": mean / fastest_mean,
                "reference_pass": (
                    result["reference"]["pass"] if result.get("reference") else None
                ),
                "worst_max_abs_error": worst_metric(result, "max_abs_error"),
                "worst_rmse": worst_metric(result, "rmse"),
            }
        )

    summary = {
        "schema_version": 1,
        "case_id": baseline.get("case_id"),
        "model": baseline["model"],
        "precision": "fp16",
        "timing_scope": baseline["timing_scope"],
        "implementation": baseline["implementation"],
        "results": rows,
    }
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        "| backend | load ms | mean ms | p50 ms | p95 ms | p99 ms | infer/s | "
        "vs fastest | ref pass | worst abs | worst RMSE |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|")
    for row in rows:
        reference_pass = (
            "n/a" if row["reference_pass"] is None else str(row["reference_pass"]).lower()
        )
        print(
            f"| {row['backend']} | {row['model_load_ms']:.2f} | {row['mean_ms']:.4f} | "
            f"{row['p50_ms']:.4f} | {row['p95_ms']:.4f} | {row['p99_ms']:.4f} | "
            f"{row['throughput']:.2f} | {row['relative_to_fastest']:.3f}x | "
            f"{reference_pass} | {format_number(row['worst_max_abs_error'], 6)} | "
            f"{format_number(row['worst_rmse'], 6)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    case_parser = commands.add_parser(
        "make-case", help="convert one upstream NPZ snapshot into a benchmark case"
    )
    case_parser.add_argument("--input", required=True, type=Path, help="model_data_*.npz")
    case_parser.add_argument("--model", required=True)
    case_parser.add_argument("--output-dir", required=True, type=Path)
    case_parser.set_defaults(action=make_case)

    compare_parser = commands.add_parser(
        "compare", help="compare compatible backend result JSON files"
    )
    compare_parser.add_argument("results", nargs="+", type=Path)
    compare_parser.add_argument("--json", dest="json_path", type=Path)
    compare_parser.set_defaults(action=compare)

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
