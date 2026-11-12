#!/usr/bin/env python3
"""Assemble, verify, and embed Driver+cubin AOT model bundles.

Model metadata, entropy tables, graph schedules, and weights are generated
from the same registered checkpoint and official network implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "models" / "profiles" / "profiles.json"
COMMON_ARTIFACTS = (
    "metadata.json",
    "gaussian_pmf.json",
    "bit_estimator_pmf.json",
)
AOT_ARTIFACTS = (
    "aot/manifest.json",
    "aot/MLVCEncoder/graph.json",
    "aot/MLVCEncoder/weights.bin",
    "aot/MLVCDecoder/graph.json",
    "aot/MLVCDecoder/weights.bin",
)
EMBED_ALIGNMENT = 16


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


def add_assemble_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing non-empty output directory",
    )


def verify_package(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "model_bundle.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported model bundle schema: {manifest_path}")
    required = set(COMMON_ARTIFACTS + AOT_ARTIFACTS)
    if set(manifest["artifacts"]) != required:
        raise ValueError("bundle must contain exactly the Driver+cubin AOT artifacts")
    for name, record in manifest["artifacts"].items():
        if record["path"] != name:
            raise ValueError(f"non-canonical artifact path: {name}")
        path = output_dir / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"bundle artifact {name!r} is missing: {path}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"bundle artifact size mismatch: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"bundle artifact SHA-256 mismatch: {path}")
    profiles = load_json(PROFILE_PATH)["profiles"]
    profile = profiles[manifest["profile"]]
    for part in ("MLVCEncoder", "MLVCDecoder"):
        graph = load_json(output_dir / "aot" / part / "graph.json")
        if graph.get("schema_version") != 2 or graph.get("source") != "checkpoint":
            raise ValueError("AOT graphs must use checkpoint graph schema 2")
        if graph.get("source_sha256") != profile["weights_sha256"]:
            raise ValueError("AOT graph checkpoint identity mismatch")
    return manifest


def assemble(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve()
    if source_dir == output_dir or source_dir.is_relative_to(output_dir):
        raise ValueError("output directory must not contain a source directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; "
                "use the verify subcommand or --overwrite"
            )
        shutil.rmtree(output_dir)

    profiles = load_json(PROFILE_PATH)["profiles"]
    if args.profile not in profiles:
        raise ValueError(f"unknown model profile: {args.profile}")
    profile = profiles[args.profile]
    canonical_metadata = load_json(source_dir / "metadata.json")
    identity = model_identity(canonical_metadata)
    if identity["name"] != args.profile:
        raise ValueError(
            f"profile/metadata mismatch: {args.profile!r} != {identity['name']!r}"
        )
    if identity["full_model_params"]["model_version"] != profile["model_version"]:
        raise ValueError("metadata model_version does not match the registered profile")

    aot = load_json(source_dir / "aot/manifest.json")
    if aot.get("source_sha256") != profile["weights_sha256"]:
        raise ValueError("AOT checkpoint identity does not match registered profile")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in COMMON_ARTIFACTS + AOT_ARTIFACTS:
        record = copy_artifact(source_dir / name, output_dir / name)
        record["path"] = name
        artifacts[name] = record

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
        "canonical_entropy_source": "checkpoint-fp32",
        "entropy_model_sha256": entropy_hash.hexdigest(),
        "compatibility": {
            "container": "mlvc-frame-le-v1",
            "entropy": "canonical-pmf-v1",
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


def verify(args: argparse.Namespace) -> None:
    manifest = verify_package(args.model_dir.resolve())
    print(
        f"verified model package {manifest['profile']}: "
        f"{len(manifest['artifacts'])} artifacts"
    )


def find_source_bundle(model_root: Path, profile: str) -> Path:
    profile_root = model_root / profile
    manifests = sorted(profile_root.glob("*/model_bundle.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"expected exactly one canonical bundle for {profile} below "
            f"{profile_root}, found {len(manifests)}"
        )
    return manifests[0].parent


def align(data: bytearray) -> None:
    data.extend(b"\0" * (-len(data) % EMBED_ALIGNMENT))


def embedded_manifest(source: dict[str, Any]) -> bytes:
    required = COMMON_ARTIFACTS + AOT_ARTIFACTS
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


def depfile_escape(path: Path) -> str:
    return (
        str(path.resolve())
        .replace("\\", "\\\\")
        .replace(" ", "\\ ")
        .replace("#", "\\#")
        .replace("$", "$$")
    )


def embed(args: argparse.Namespace) -> None:
    model_root = args.model_root.resolve()
    profiles = load_json(PROFILE_PATH)["profiles"]
    payload = bytearray()
    index: list[tuple[str, str, int, int]] = []
    dependencies = {PROFILE_PATH.resolve()}

    for profile, registration in profiles.items():
        source_dir = find_source_bundle(model_root, profile)
        dependencies.add((source_dir / "model_bundle.json").resolve())
        source_manifest = verify_package(source_dir)
        dependencies.update(
            (source_dir / record["path"]).resolve()
            for record in source_manifest["artifacts"].values()
        )
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
            ("model_bundle.json", embedded_manifest(source_manifest))
        ]
        required = COMMON_ARTIFACTS + AOT_ARTIFACTS
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
    output_data = args.output_data.resolve()
    output_index = args.output_index.resolve()
    replace_file(output_data, bytes(payload))
    replace_file(output_index, index_source)
    if args.depfile is not None:
        targets = f"{depfile_escape(output_data)} {depfile_escape(output_index)}"
        prerequisites = " ".join(
            depfile_escape(path) for path in sorted(dependencies)
        )
        replace_file(
            args.depfile.resolve(), f"{targets}: {prerequisites}\n".encode()
        )
    print(
        f"embedded {len(profiles)} driver-cubin profiles: "
        f"{len(index)} assets, {len(payload)} bytes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    assemble_parser = commands.add_parser(
        "assemble", help="assemble one canonical AOT bundle"
    )
    add_assemble_arguments(assemble_parser)
    assemble_parser.set_defaults(action=assemble)

    verify_parser = commands.add_parser(
        "verify", help="verify hashes and sizes in an existing bundle"
    )
    verify_parser.add_argument("--model-dir", type=Path, required=True)
    verify_parser.set_defaults(action=verify)

    embed_parser = commands.add_parser(
        "embed", help="build the deterministic driver-cubin model image"
    )
    embed_parser.add_argument("--model-root", type=Path, required=True)
    embed_parser.add_argument("--output-data", type=Path, required=True)
    embed_parser.add_argument("--output-index", type=Path, required=True)
    embed_parser.add_argument("--depfile", type=Path)
    embed_parser.set_defaults(action=embed)

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
