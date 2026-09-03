#!/usr/bin/env python3
"""Run one mlvc_demo backend against an official Python codec fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--engine-cache-dir", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="write all metrics and continue after contract violations",
    )
    return parser.parse_args()


def run_codec(
    args: argparse.Namespace,
    command: str,
    input_path: Path,
    output_path: Path,
    debug_dir: Path | None = None,
) -> None:
    manifest = json.loads(
        (args.reference_dir / "reference.json").read_text(encoding="utf-8")
    )
    cmd = [
        str(args.binary.resolve()),
        command,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--width",
        str(manifest["width"]),
        "--height",
        str(manifest["height"]),
        "--frames",
        str(manifest["frames"]),
        "--model-dir",
        str(args.model_dir.resolve()),
    ]
    if command == "encode":
        cmd.extend(["--q-index", str(manifest["q_index"])])
    if args.engine_cache_dir:
        cmd.extend(["--engine-cache-dir", str(args.engine_cache_dir.resolve())])
    if debug_dir:
        cmd.extend(["--debug-dir", str(debug_dir)])
    subprocess.run(cmd, check=True)


def read_frames(path: Path) -> list[tuple[int, bytes]]:
    data = path.read_bytes()
    position = 0
    frames = []
    while position < len(data):
        if position + 8 > len(data):
            raise AssertionError(f"truncated bitstream header: {path}")
        q_index, size = struct.unpack_from("<iI", data, position)
        position += 8
        payload = data[position : position + size]
        if len(payload) != size:
            raise AssertionError(f"truncated bitstream payload: {path}")
        frames.append((q_index, payload))
        position += size
    return frames


def compare_yuv(
    reference: bytes,
    actual: bytes,
    frame_bytes: int,
    max_error: int,
    min_psnr: float,
    enforce_contract: bool = True,
) -> list[dict[str, Any]]:
    if len(reference) != len(actual) or len(reference) % frame_bytes != 0:
        raise AssertionError("reconstructed YUV size mismatch")
    results = []
    for frame in range(len(reference) // frame_bytes):
        start = frame * frame_bytes
        expected = reference[start : start + frame_bytes]
        observed = actual[start : start + frame_bytes]
        squared_error = 0
        maximum = 0
        differing = 0
        for lhs, rhs in zip(expected, observed):
            delta = abs(lhs - rhs)
            maximum = max(maximum, delta)
            differing += delta != 0
            squared_error += delta * delta
        rmse = math.sqrt(squared_error / frame_bytes)
        psnr = 999.9 if rmse == 0 else 20.0 * math.log10(255.0 / rmse)
        within_contract = maximum <= max_error and psnr >= min_psnr
        if not within_contract and enforce_contract:
            raise AssertionError(
                f"frame {frame} reconstruction outside contract: "
                f"max={maximum}, psnr={psnr:.3f} dB"
            )
        results.append(
            {
                "frame": frame,
                "differing_samples": differing,
                "max_sample_error": maximum,
                "rmse": rmse,
                "psnr_db": psnr,
                "within_contract": within_contract,
            }
        )
    return results


def numeric_values(path: Path, dtype: str) -> list[float | int]:
    data = path.read_bytes()
    if dtype == "fp16":
        if len(data) % 2:
            raise AssertionError(f"invalid fp16 tensor: {path}")
        return [value[0] for value in struct.iter_unpack("<e", data)]
    if dtype == "int32":
        if len(data) % 4:
            raise AssertionError(f"invalid int32 tensor: {path}")
        return [value[0] for value in struct.iter_unpack("<i", data)]
    raise AssertionError(f"unsupported fixture dtype: {dtype}")


def compare_tensors(
    reference_dir: Path,
    debug_root: Path,
    records: list[dict[str, Any]],
    contract: dict[str, Any],
    enforce_contract: bool = True,
) -> list[dict[str, Any]]:
    results = []
    exact_encoder_frames = set(contract["exact_encoder_frames"])
    for record in records:
        relative = Path(record["path"])
        parts = relative.parts
        stage = parts[0]
        frame = int(parts[1].split("-")[1])
        direction = record["direction"]
        name = record["name"]
        reference_path = reference_dir / relative
        actual_path = debug_root / relative
        if not actual_path.exists():
            raise AssertionError(f"missing backend debug tensor: {actual_path}")
        expected = numeric_values(reference_path, record["dtype"])
        actual = numeric_values(actual_path, record["dtype"])
        if len(expected) != len(actual):
            raise AssertionError(f"tensor element count mismatch: {relative}")

        strict = record["dtype"] == "int32" or name == "x"
        if (
            stage == "encoder"
            and direction == "output"
            and name in {"z_raw", "y_raw_0", "y_raw_1"}
        ):
            strict = frame in exact_encoder_frames
        max_abs = 0.0
        squared = 0.0
        mismatches = 0
        for lhs, rhs in zip(expected, actual):
            if isinstance(lhs, float) and (not math.isfinite(lhs) or not math.isfinite(rhs)):
                if math.isnan(lhs) and math.isnan(rhs):
                    delta = 0.0
                elif lhs == rhs:
                    delta = 0.0
                else:
                    delta = math.inf
            else:
                delta = abs(float(lhs) - float(rhs))
            max_abs = max(max_abs, delta)
            squared += delta * delta
            mismatches += delta != 0
        rmse = math.sqrt(squared / len(expected)) if expected else 0.0

        enforce_tolerance = not (
            stage == "encoder"
            and direction == "output"
            and name in {"z_raw", "y_raw_0", "y_raw_1"}
            and frame not in exact_encoder_frames
        )
        within_contract = True
        if strict and mismatches:
            within_contract = False
        if strict and mismatches and enforce_contract:
            raise AssertionError(
                f"tensor must be numerically exact: {relative} ({mismatches} mismatches)"
            )
        if enforce_tolerance and not strict and (
            max_abs > contract["tensor_max_abs_error"]
            or rmse > contract["tensor_max_rmse"]
        ):
            within_contract = False
        if enforce_tolerance and not strict and not within_contract and enforce_contract:
            raise AssertionError(
                f"tensor outside contract: {relative}, max={max_abs}, rmse={rmse}"
            )
        results.append(
            {
                "path": relative.as_posix(),
                "mismatches": mismatches,
                "max_abs_error": max_abs,
                "rmse": rmse,
                "enforced": enforce_tolerance,
                "within_contract": within_contract,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    args.reference_dir = args.reference_dir.resolve()
    manifest = json.loads(
        (args.reference_dir / "reference.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        raise AssertionError("unsupported codec reference schema")
    bundle = json.loads(
        (args.model_dir / "model_bundle.json").read_text(encoding="utf-8")
    )
    if (
        bundle.get("profile") != manifest["profile"]
        or bundle.get("model_version") != manifest["model_version"]
        or bundle.get("entropy_model_sha256") != manifest["entropy_model_sha256"]
    ):
        raise AssertionError("model bundle does not match codec reference")
    for name, record in bundle["artifacts"].items():
        artifact = args.model_dir / record["path"]
        if not artifact.is_file() or artifact.stat().st_size != record["bytes"]:
            raise AssertionError(f"model bundle artifact missing or truncated: {name}")
        if sha256(artifact) != record["sha256"]:
            raise AssertionError(f"model bundle artifact hash mismatch: {name}")
    if sha256(args.model_dir / "metadata.json") != manifest["model_metadata_sha256"]:
        raise AssertionError("model metadata does not match codec reference")
    if sha256(args.reference_dir / manifest["input"]["path"]) != manifest["input"]["sha256"]:
        raise AssertionError("reference input hash mismatch")
    if sha256(args.reference_dir / manifest["bitstream"]["path"]) != manifest["bitstream"]["sha256"]:
        raise AssertionError("reference bitstream hash mismatch")
    if sha256(args.reference_dir / manifest["reconstruction"]["path"]) != manifest["reconstruction"]["sha256"]:
        raise AssertionError("reference reconstruction hash mismatch")
    for record in manifest["tensors"]:
        if sha256(args.reference_dir / record["path"]) != record["sha256"]:
            raise AssertionError(f"reference tensor hash mismatch: {record['path']}")

    with tempfile.TemporaryDirectory(prefix="mlvc-conformance-") as temporary:
        work = Path(temporary)
        encoded = work / "encoded.mlvc"
        encode_debug = work / "encode-debug"
        run_codec(
            args,
            "encode",
            args.reference_dir / manifest["input"]["path"],
            encoded,
            encode_debug,
        )
        expected_frames = read_frames(args.reference_dir / manifest["bitstream"]["path"])
        actual_frames = read_frames(encoded)
        if len(expected_frames) != manifest["frames"] or len(actual_frames) != manifest["frames"]:
            raise AssertionError("reference or backend encoder produced the wrong frame count")
        exact_frames = set(manifest["contract"]["exact_encoder_frames"])
        bitstream_comparison = []
        for index, ((expected_q, expected), (actual_q, actual)) in enumerate(
            zip(expected_frames, actual_frames)
        ):
            q_match = expected_q == actual_q
            payload_match = expected == actual
            if not q_match and not args.diagnostic:
                raise AssertionError(f"frame {index} q-index mismatch")
            if index in exact_frames and not payload_match and not args.diagnostic:
                raise AssertionError(f"frame {index} payload is not bit-exact")
            bitstream_comparison.append(
                {
                    "frame": index,
                    "q_index_match": q_match,
                    "expected_q_index": expected_q,
                    "actual_q_index": actual_q,
                    "expected_payload_bytes": len(expected),
                    "actual_payload_bytes": len(actual),
                    "payload_bit_exact": payload_match,
                    "payload_sha256": hashlib.sha256(actual).hexdigest(),
                    "expected_payload_sha256": hashlib.sha256(expected).hexdigest(),
                    "required_bit_exact": index in exact_frames,
                }
            )

        decoded = work / "decoded-reference.yuv"
        decode_debug = work / "decode-debug"
        run_codec(
            args,
            "decode",
            args.reference_dir / manifest["bitstream"]["path"],
            decoded,
            decode_debug,
        )
        roundtrip = work / "roundtrip.yuv"
        run_codec(args, "decode", encoded, roundtrip)

        frame_bytes = manifest["width"] * manifest["height"] * 3 // 2
        reference_decode_reconstruction = compare_yuv(
            (args.reference_dir / manifest["reconstruction"]["path"]).read_bytes(),
            decoded.read_bytes(),
            frame_bytes,
            manifest["contract"]["decoder_max_yuv_sample_error"],
            manifest["contract"]["decoder_min_psnr_db"],
            enforce_contract=not args.diagnostic,
        )
        roundtrip_reconstruction = compare_yuv(
            (args.reference_dir / manifest["reconstruction"]["path"]).read_bytes(),
            roundtrip.read_bytes(),
            frame_bytes,
            manifest["contract"]["decoder_max_yuv_sample_error"],
            manifest["contract"]["decoder_min_psnr_db"],
            enforce_contract=not args.diagnostic,
        )
        tensor_results = compare_tensors(
            args.reference_dir,
            encode_debug,
            [record for record in manifest["tensors"] if record["path"].startswith("encoder/")],
            manifest["contract"],
            enforce_contract=not args.diagnostic,
        )
        tensor_results.extend(
            compare_tensors(
                args.reference_dir,
                decode_debug,
                [record for record in manifest["tensors"] if record["path"].startswith("decoder/")],
                manifest["contract"],
                enforce_contract=not args.diagnostic,
            )
        )
        result = {
            "schema_version": 1,
            "profile": manifest["profile"],
            "backend": subprocess.check_output(
                [str(args.binary.resolve()), "--backend-name"], text=True
            ).strip(),
            "encoded_frame_bytes": [len(payload) for _, payload in actual_frames],
            "bit_exact_frames": sorted(exact_frames),
            "bitstream_comparison": bitstream_comparison,
            "reference_decode_reconstruction": reference_decode_reconstruction,
            "roundtrip_reconstruction": roundtrip_reconstruction,
            "tensors": tensor_results,
        }
        result["contract_passed"] = all(
            item["within_contract"]
            for item in result["reference_decode_reconstruction"]
            + result["roundtrip_reconstruction"]
            + result["tensors"]
        ) and all(
            item["q_index_match"]
            and (not item["required_bit_exact"] or item["payload_bit_exact"])
            for item in bitstream_comparison
        )
        if args.result:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.diagnostic:
            print(
                f"codec conformance diagnostic: backend={result['backend']} "
                f"profile={result['profile']} frames={manifest['frames']} "
                f"contract_passed={result['contract_passed']}"
            )
        else:
            print(
                f"codec conformance passed: backend={result['backend']} "
                f"profile={result['profile']} frames={manifest['frames']}"
            )


if __name__ == "__main__":
    main()
