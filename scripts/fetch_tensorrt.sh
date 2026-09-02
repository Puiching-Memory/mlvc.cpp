#!/usr/bin/env bash
# Install the pinned TensorRT 11.2 C++ SDK from NVIDIA's Ubuntu 26.04 repo.
# Usage: ./scripts/fetch_tensorrt.sh [--print-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/dependencies.env
source "$ROOT/scripts/dependencies.env"

find_header() {
    local multiarch
    multiarch="$(gcc -print-multiarch)"
    [[ -f "/usr/include/$multiarch/NvInfer.h" ]] && {
        printf '%s\n' "/usr/include/$multiarch"
        return 0
    }
    [[ -f /usr/include/NvInfer.h ]] && printf '%s\n' /usr/include
}

check_installation() {
    local include_dir linker_cache major minor
    include_dir="$(find_header)" || return 1
    linker_cache="$(ldconfig -p)"
    grep -q 'libnvinfer\.so\.11' <<< "$linker_cache" || return 1
    grep -q 'libnvonnxparser\.so\.11' <<< "$linker_cache" || return 1
    major="$(sed -n 's/^#define TRT_MAJOR_ENTERPRISE \([0-9][0-9]*\).*/\1/p' \
        "$include_dir/NvInferVersion.h")"
    minor="$(sed -n 's/^#define TRT_MINOR_ENTERPRISE \([0-9][0-9]*\).*/\1/p' \
        "$include_dir/NvInferVersion.h")"
    [[ "$major.$minor" == 11.2 ]]
}

if [[ "${1:-}" == --print-dir ]]; then
    [[ $# -eq 1 ]] || { echo "error: --print-dir takes no value" >&2; exit 2; }
    check_installation || {
        echo "error: TensorRT 11.2 is not installed; run $0" >&2
        exit 1
    }
    printf '%s\n' system
    exit 0
fi
[[ $# -eq 0 ]] || { echo "unknown argument: $1" >&2; exit 2; }

"$ROOT/scripts/fetch_cuda.sh"

[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || {
    echo "error: only Linux x86_64 is supported" >&2
    exit 1
}
# shellcheck source=/dev/null
source /etc/os-release
[[ "$ID:$VERSION_ID" == ubuntu:26.04 ]] || {
    echo "error: the pinned SDK requires Ubuntu 26.04" >&2
    exit 1
}

PRIVILEGE=()
if [[ "$EUID" -ne 0 ]]; then
    command -v sudo >/dev/null || { echo "error: root or sudo is required" >&2; exit 1; }
    PRIVILEGE=(sudo)
fi

KEYRING="$(mktemp --suffix=.deb)"
trap 'rm -f -- "$KEYRING"' EXIT
curl --fail --location --retry 3 \
    --output "$KEYRING" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
"${PRIVILEGE[@]}" dpkg -i "$KEYRING"
"${PRIVILEGE[@]}" apt-get update
"${PRIVILEGE[@]}" apt-get install -y \
    "libnvinfer-dev=$MLVC_TENSORRT_DEB_VERSION" \
    "libnvonnxparsers-dev=$MLVC_TENSORRT_DEB_VERSION"
check_installation || { echo "error: TensorRT installation validation failed" >&2; exit 1; }
echo "Installed TensorRT $MLVC_TENSORRT_VERSION."
