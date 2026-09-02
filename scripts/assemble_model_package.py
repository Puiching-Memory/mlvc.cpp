#!/usr/bin/env python3
"""Assemble one backend-independent MLVC deployment model package.

The ONNX export is the canonical source of metadata and entropy tables.  Other
backend artifacts are copied into the same directory, so every encoder and
decoder consumes byte-identical PMF data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs" / "model_profiles.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def model_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    params = metadata["params"]
    return {
        "name": metadata["name"],
        "full_model_params": params["full_model_params"],
        "split_model_params": params["split_model_params"],
    }


def copy_artifact(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"missing model artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument("--torchscript-dir", type=Path)
    parser.add_argument("--aot-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing non-empty output directory",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing package instead of assembling it",
    )
    return parser.parse_args()


def verify_package(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "model_bundle.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported model bundle schema: {manifest_path}")
    for name, record in manifest["artifacts"].items():
        path = output_dir / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"bundle artifact {name!r} is missing: {path}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"bundle artifact size mismatch: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"bundle artifact SHA-256 mismatch: {path}")
    return manifest


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.verify_only:
        manifest = verify_package(output_dir)
        print(
            f"verified canonical package {manifest['profile']}: "
            f"{len(manifest['artifacts'])} artifacts"
        )
        return
    if not args.profile or not args.onnx_dir:
        raise ValueError("--profile and --onnx-dir are required when assembling")
    source_dirs = [args.onnx_dir.resolve()]
    source_dirs.extend(
        path.resolve()
        for path in (args.torchscript_dir, args.aot_dir)
        if path is not None
    )
    if any(source == output_dir or source.is_relative_to(output_dir)
           for source in source_dirs):
        raise ValueError("output directory must not contain a source directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; "
                "use --verify-only or --overwrite"
            )
        shutil.rmtree(output_dir)

    profiles = load_json(PROFILE_PATH)["profiles"]
    if args.profile not in profiles:
        raise ValueError(f"unknown model profile: {args.profile}")
    profile = profiles[args.profile]
    onnx_dir = source_dirs[0]
    canonical_metadata = load_json(onnx_dir / "metadata.json")
    identity = model_identity(canonical_metadata)
    if identity["name"] != args.profile:
        raise ValueError(
            f"profile/metadata mismatch: {args.profile!r} != {identity['name']!r}"
        )
    if identity["full_model_params"]["model_version"] != profile["model_version"]:
        raise ValueError("metadata model_version does not match the registered profile")

    if args.torchscript_dir:
        torch_identity = model_identity(
            load_json(args.torchscript_dir.resolve() / "metadata.json")
        )
        if torch_identity != identity:
            raise ValueError("ONNX and TorchScript exports describe different models")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    canonical_files = [
        "metadata.json",
        "gaussian_pmf.json",
        "bit_estimator_pmf.json",
        "MLVCEncoder.onnx",
        "MLVCDecoder.onnx",
    ]
    for name in canonical_files:
        artifacts[name] = copy_artifact(onnx_dir / name, output_dir / name)

    if args.torchscript_dir:
        torch_dir = args.torchscript_dir.resolve()
        for name in ("MLVCEncoder.ts", "MLVCDecoder.ts"):
            artifacts[name] = copy_artifact(torch_dir / name, output_dir / name)

    if args.aot_dir:
        aot_dir = args.aot_dir.resolve()
        for path in sorted(aot_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = Path("aot") / path.relative_to(aot_dir)
            key = relative.as_posix()
            record = copy_artifact(path, output_dir / relative)
            record["path"] = key
            artifacts[key] = record

    entropy_hash = hashlib.sha256()
    for name in ("gaussian_pmf.json", "bit_estimator_pmf.json"):
        entropy_hash.update(bytes.fromhex(artifacts[name]["sha256"]))
    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "model_version": profile["model_version"],
        "model_size": {
            "width": identity["split_model_params"]["model_width"],
            "height": identity["split_model_params"]["model_height"],
        },
        "canonical_entropy_source": "onnx",
        "entropy_model_sha256": entropy_hash.hexdigest(),
        "compatibility": {
            "container": "mlvc-frame-le-v1",
            "entropy": "canonical-pmf-v1",
            "encoder_bit_exact_across_backends": False,
            "decoder_max_yuv_sample_error": 1,
            "decoder_min_psnr_db": 60.0,
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "model_bundle.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verified = verify_package(output_dir)
    print(
        f"assembled canonical package {verified['profile']} at {output_dir} "
        f"({len(verified['artifacts'])} artifacts)"
    )


if __name__ == "__main__":
    main()
