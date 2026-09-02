#!/usr/bin/env bash
# Fetch the pinned Linux x86_64 ONNX Runtime CUDA 13 SDK.
# Usage: ./scripts/fetch_onnxruntime.sh [--no-proxy|--print-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/dependencies.env
source "$ROOT/scripts/dependencies.env"

NO_PROXY=0
PRINT_DIR=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-proxy) NO_PROXY=1 ;;
        --print-dir) PRINT_DIR=1 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || {
    echo "error: only Linux x86_64 is supported" >&2
    exit 1
}
for tool in curl sha256sum tar unzip; do
    command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 1; }
done

ARCHIVE_PACKAGE="onnxruntime-linux-x64-gpu_cuda13-${MLVC_ONNXRUNTIME_VERSION}"
SDK_PACKAGE="onnxruntime-linux-x64-gpu-${MLVC_ONNXRUNTIME_VERSION}"
DEST="$ROOT/third_party/onnxruntime"
SDK="$DEST/$SDK_PACKAGE"

sdk_is_complete() {
    [[ -f "$SDK/lib/libonnxruntime_providers_cuda.so" &&
       -f "$SDK/lib/libcudnn.so.9" &&
       -f "$SDK/LICENSE.nvidia-cudnn-cu13.txt" ]]
}

if [[ "$PRINT_DIR" -eq 1 ]]; then
    sdk_is_complete || {
        echo "error: ONNX Runtime SDK is missing; run $0" >&2
        exit 1
    }
    printf '%s\n' "$SDK"
    exit 0
fi

if sdk_is_complete; then
    echo "Already present: $SDK"
    exit 0
fi

mkdir -p "$DEST"
CURL_OPTIONS=(--fail --location --retry 3)
[[ "$NO_PROXY" -eq 1 ]] && CURL_OPTIONS+=(--noproxy '*')

if [[ ! -f "$SDK/lib/libonnxruntime_providers_cuda.so" ]]; then
    ARCHIVE="$DEST/$ARCHIVE_PACKAGE.tgz"
    URL="https://github.com/microsoft/onnxruntime/releases/download/v${MLVC_ONNXRUNTIME_VERSION}/$ARCHIVE_PACKAGE.tgz"
    echo "Downloading $URL ..."
    curl "${CURL_OPTIONS[@]}" --output "$ARCHIVE" "$URL"
    printf '%s  %s\n' "$MLVC_ONNXRUNTIME_SHA256" "$ARCHIVE" | sha256sum --check
    tar -xzf "$ARCHIVE" -C "$DEST"
    rm -f -- "$ARCHIVE"
fi

if [[ ! -f "$SDK/lib/libcudnn.so.9" ]]; then
    WHEEL="nvidia_cudnn_cu13-${MLVC_CUDNN_WHEEL_VERSION}-py3-none-manylinux_2_27_x86_64.whl"
    URL="https://pypi.nvidia.com/nvidia-cudnn-cu13/$WHEEL"
    if ! [[ -f "$DEST/$WHEEL" ]] ||
       ! printf '%s  %s\n' "$MLVC_CUDNN_WHEEL_SHA256" "$DEST/$WHEEL" |
            sha256sum --check --status; then
        curl "${CURL_OPTIONS[@]}" --output "$DEST/$WHEEL" "$URL"
    else
        echo "Using cached: $DEST/$WHEEL"
    fi
    printf '%s  %s\n' "$MLVC_CUDNN_WHEEL_SHA256" "$DEST/$WHEEL" | sha256sum --check
    unzip -q -o -j "$DEST/$WHEEL" 'nvidia/*/lib/*.so*' -d "$SDK/lib"
    LICENSE_ENTRY="$(unzip -Z1 "$DEST/$WHEEL" |
        grep -Ei '(^|/)(LICENSE|COPYING)([^/]*)$' | head -n 1 || true)"
    [[ -z "$LICENSE_ENTRY" ]] ||
        unzip -p "$DEST/$WHEEL" "$LICENSE_ENTRY" > "$SDK/LICENSE.nvidia-cudnn-cu13.txt"
    rm -f -- "$DEST/$WHEEL"
    ln -sfn libcudnn.so.9 "$SDK/lib/libcudnn.so"
fi
sdk_is_complete || { echo "error: ONNX Runtime dependency closure is incomplete" >&2; exit 1; }
echo "Installed: $SDK"
