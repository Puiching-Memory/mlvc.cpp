#!/usr/bin/env python3
"""Copy backend-specific model artifacts into a release package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from assemble_model_package import PROFILE_PATH, verify_package


COMMON_ARTIFACTS = (
    "metadata.json",
    "gaussian_pmf.json",
    "bit_estimator_pmf.json",
)
BACKEND_ARTIFACTS = {
    "tensorrt": (
        "MLVCEncoder.onnx",
        "MLVCDecoder.onnx",
    ),
    "driver-cubin": (
        "aot/manifest.json",
        "aot/MLVCEncoder/graph.json",
        "aot/MLVCEncoder/weights.bin",
        "aot/MLVCDecoder/graph.json",
        "aot/MLVCDecoder/weights.bin",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def find_source_bundle(model_root: Path, profile: str) -> Path:
    canonical_root = model_root / profile / "canonical"
    manifests = sorted(canonical_root.glob("*/model_bundle.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"expected exactly one canonical bundle for {profile} below "
            f"{canonical_root}, found {len(manifests)}"
        )
    return manifests[0].parent


def package_bundle(
    backend: str,
    source_dir: Path,
    destination_dir: Path,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    dimensions = source_manifest["model_size"]
    expected_directory = f"{dimensions['width']}x{dimensions['height']}"
    if source_dir.name != expected_directory:
        raise ValueError(
            f"model bundle directory {source_dir.name!r} does not match "
            f"manifest dimensions {expected_directory!r}"
        )

    required = COMMON_ARTIFACTS + BACKEND_ARTIFACTS[backend]
    selected: dict[str, Any] = {}
    for name in required:
        try:
            record = source_manifest["artifacts"][name]
        except KeyError as error:
            raise FileNotFoundError(
                f"{backend} requires artifact {name!r} in {source_dir}"
            ) from error
        if record.get("path") != name:
            raise ValueError(
                f"artifact {name!r} must use its canonical relative path"
            )
        source = source_dir / name
        target = destination_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        selected[name] = record

    packaged_manifest = dict(source_manifest)
    packaged_manifest["packaged_backend"] = backend
    packaged_manifest["artifacts"] = selected
    (destination_dir / "model_bundle.json").write_text(
        json.dumps(packaged_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verify_package(destination_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(BACKEND_ARTIFACTS), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = args.model_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"model output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    profiles = load_json(PROFILE_PATH)["profiles"]
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        for profile in profiles:
            source_dir = find_source_bundle(model_root, profile)
            source_manifest = verify_package(source_dir)
            if source_manifest.get("profile") != profile:
                raise ValueError(
                    f"bundle profile mismatch: {source_dir} describes "
                    f"{source_manifest.get('profile')!r}, expected {profile!r}"
                )
            if (
                source_manifest.get("model_version")
                != profiles[profile]["model_version"]
            ):
                raise ValueError(
                    f"bundle model version mismatch for {profile}: "
                    f"{source_manifest.get('model_version')!r}"
                )
            size = source_manifest["model_size"]
            dimensions = f"{size['width']}x{size['height']}"
            destination_dir = temporary / profile / dimensions
            manifest = package_bundle(
                args.backend, source_dir, destination_dir, source_manifest
            )
            print(
                f"packaged {args.backend} model {profile}/{dimensions} "
                f"({len(manifest['artifacts'])} artifacts)"
            )
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
