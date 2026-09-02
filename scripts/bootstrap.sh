#!/usr/bin/env bash
# Initialize shallow source submodules and acquire one or all GPU SDKs.
# Usage: ./scripts/bootstrap.sh [--backend sources|all|onnxruntime|libtorch|tensorrt]
#                               [--no-proxy]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND=sources
NO_PROXY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="${2:?--backend needs a value}"; shift 2 ;;
        --no-proxy) NO_PROXY=1; shift ;;
        -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$BACKEND" in
    sources|all|onnxruntime|libtorch|tensorrt) ;;
    *) echo "error: invalid backend: $BACKEND" >&2; exit 2 ;;
esac

command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }
git -C "$ROOT" submodule sync --recursive
git -C "$ROOT" submodule update --init --recursive --depth 1
[[ "$BACKEND" == sources ]] && exit 0

MISSING_HOST_TOOL=0
for tool in cmake c++ curl sha256sum tar unzip; do
    command -v "$tool" >/dev/null || MISSING_HOST_TOOL=1
done
if [[ "$MISSING_HOST_TOOL" -eq 1 ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    [[ "$ID:$VERSION_ID" == ubuntu:26.04 ]] || {
        echo "error: automatic host dependency installation requires Ubuntu 26.04" >&2
        exit 1
    }
    PRIVILEGE=()
    if [[ "$EUID" -ne 0 ]]; then
        command -v sudo >/dev/null || {
            echo "error: root or sudo is required to install host dependencies" >&2
            exit 1
        }
        PRIVILEGE=(sudo)
    fi
    "${PRIVILEGE[@]}" apt-get update
    "${PRIVILEGE[@]}" apt-get install -y \
        build-essential ca-certificates cmake coreutils curl tar unzip
fi

"$ROOT/scripts/fetch_cuda.sh"

FETCH_OPTIONS=()
[[ "$NO_PROXY" -eq 1 ]] && FETCH_OPTIONS+=(--no-proxy)

if [[ "$BACKEND" == all || "$BACKEND" == onnxruntime ]]; then
    "$ROOT/scripts/fetch_onnxruntime.sh" "${FETCH_OPTIONS[@]}"
fi
if [[ "$BACKEND" == all || "$BACKEND" == libtorch ]]; then
    "$ROOT/scripts/fetch_libtorch.sh" "${FETCH_OPTIONS[@]}"
fi
if [[ "$BACKEND" == all || "$BACKEND" == tensorrt ]]; then
    "$ROOT/scripts/fetch_tensorrt.sh"
fi
