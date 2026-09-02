#!/usr/bin/env bash
# Ensure the single supported CUDA 13.3 toolkit is available.
set -euo pipefail

CUDA_ROOT=/usr/local/cuda-13.3
if [[ -x "$CUDA_ROOT/bin/nvcc" && -f "$CUDA_ROOT/lib64/libcudart.so.13" ]]; then
    echo "Already present: $CUDA_ROOT"
    exit 0
fi

[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || {
    echo "error: only Linux x86_64 is supported" >&2
    exit 1
}
# shellcheck source=/dev/null
source /etc/os-release
[[ "$ID:$VERSION_ID" == ubuntu:26.04 ]] || {
    echo "error: the pinned CUDA toolkit requires Ubuntu 26.04" >&2
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
"${PRIVILEGE[@]}" apt-get install -y cuda-toolkit-13-3
[[ -x "$CUDA_ROOT/bin/nvcc" ]] || { echo "error: CUDA installation failed" >&2; exit 1; }
