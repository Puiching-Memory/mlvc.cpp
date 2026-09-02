#!/usr/bin/env python3
"""Summarize comparable mlvc_backend_bench result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latency and FP16 reference error across backend results"
    )
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    results = [load_result(path) for path in args.results]
    if not results:
        raise ValueError("at least one result is required")

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


if __name__ == "__main__":
    main()
