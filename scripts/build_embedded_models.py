#!/usr/bin/env python3
"""Build a deterministic, linkable driver-cubin model data image."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from assemble_model_package import PROFILE_PATH, verify_package
from package_model_bundles import (
    BACKEND_ARTIFACTS,
    COMMON_ARTIFACTS,
    find_source_bundle,
    load_json,
)


ALIGNMENT = 16


def align(data: bytearray) -> None:
    data.extend(b"\0" * (-len(data) % ALIGNMENT))


def filtered_manifest(source: dict[str, Any]) -> bytes:
    required = COMMON_ARTIFACTS + BACKEND_ARTIFACTS["driver-cubin"]
    artifacts: dict[str, Any] = {}
    for name in required:
        try:
            record = source["artifacts"][name]
        except KeyError as error:
            raise FileNotFoundError(
                f"driver-cubin requires artifact {name!r}"
            ) from error
        if record.get("path") != name:
            raise ValueError(
                f"artifact {name!r} must use its canonical relative path"
            )
        artifacts[name] = record
    result = dict(source)
    result["packaged_backend"] = "driver-cubin"
    result["storage"] = "embedded"
    result["artifacts"] = artifacts
    return (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()


def replace_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = args.model_root.resolve()
    profiles = load_json(PROFILE_PATH)["profiles"]
    payload = bytearray()
    index: list[tuple[str, str, int, int]] = []

    for profile, registration in profiles.items():
        source_dir = find_source_bundle(model_root, profile)
        source_manifest = verify_package(source_dir)
        if source_manifest.get("profile") != profile:
            raise ValueError(
                f"bundle profile mismatch: {source_dir} describes "
                f"{source_manifest.get('profile')!r}, expected {profile!r}"
            )
        if source_manifest.get("model_version") != registration["model_version"]:
            raise ValueError(f"bundle model version mismatch for {profile}")
        size = source_manifest["model_size"]
        expected_directory = f"{size['width']}x{size['height']}"
        if source_dir.name != expected_directory:
            raise ValueError(
                f"model bundle directory {source_dir.name!r} does not match "
                f"manifest dimensions {expected_directory!r}"
            )

        assets: list[tuple[str, bytes]] = [
            ("model_bundle.json", filtered_manifest(source_manifest))
        ]
        required = COMMON_ARTIFACTS + BACKEND_ARTIFACTS["driver-cubin"]
        assets.extend((name, (source_dir / name).read_bytes()) for name in required)
        for name, content in assets:
            align(payload)
            offset = len(payload)
            payload.extend(content)
            index.append((profile, name, offset, len(content)))

    index_source = "".join(
        f"    {{{json.dumps(profile)}, {json.dumps(name)}, {offset}, {size}}},\n"
        for profile, name, offset, size in index
    ).encode()
    replace_file(args.output_data.resolve(), bytes(payload))
    replace_file(args.output_index.resolve(), index_source)
    print(
        f"embedded {len(profiles)} driver-cubin profiles: "
        f"{len(index)} assets, {len(payload)} bytes"
    )


if __name__ == "__main__":
    main()
