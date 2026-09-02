#!/usr/bin/env bash
# Fetch the pinned Linux x86_64 libtorch 2.13 cu130 SDK.
# Usage: ./scripts/fetch_libtorch.sh [--no-proxy|--print-dir]
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
for tool in cp curl sha256sum unzip; do
    command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 1; }
done

DEST="$ROOT/third_party/libtorch/${MLVC_LIBTORCH_VERSION}-${MLVC_LIBTORCH_CUDA_VARIANT}"
SDK="$DEST/libtorch"

sdk_is_complete() {
    local library license
    for library in libtorch_cuda.so libcudnn.so.9 libcusparseLt.so.0 \
                   libnccl.so.2 libnvshmem_host.so.3 libcupti.so.13; do
        [[ -f "$SDK/lib/$library" ]] || return 1
    done
    for license in LICENSE.pytorch.txt \
                   "LICENSE.cuda-toolkit-$MLVC_CUDA_VERSION.txt" \
                   LICENSE.nvidia-cudnn-cu13.txt \
                   LICENSE.nvidia-cusparselt-cu13.txt \
                   LICENSE.nvidia-nccl-cu13.txt \
                   LICENSE.nvidia-nvshmem-cu13.txt; do
        [[ -f "$SDK/$license" ]] || return 1
    done
}

if [[ "$PRINT_DIR" -eq 1 ]]; then
    sdk_is_complete || {
        echo "error: libtorch SDK is missing; run $0" >&2
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

if [[ ! -f "$SDK/lib/libtorch_cuda.so" ]]; then
    ARCHIVE="$DEST/libtorch-${MLVC_LIBTORCH_VERSION}-${MLVC_LIBTORCH_CUDA_VARIANT}.zip"
    URL="https://download.pytorch.org/libtorch/${MLVC_LIBTORCH_CUDA_VARIANT}/libtorch-shared-with-deps-${MLVC_LIBTORCH_VERSION}%2B${MLVC_LIBTORCH_CUDA_VARIANT}.zip"
    echo "Downloading $URL ..."
    curl "${CURL_OPTIONS[@]}" --output "$ARCHIVE" "$URL"
    printf '%s  %s\n' "$MLVC_LIBTORCH_SHA256" "$ARCHIVE" | sha256sum --check
    unzip -q "$ARCHIVE" -d "$DEST"
    rm -f -- "$ARCHIVE"
fi

install_wheel() {
    local name="$1"
    local filename="$2"
    local digest="$3"
    local wheel="$DEST/$filename"
    local url="https://pypi.nvidia.com/$name/$filename"
    if ! [[ -f "$wheel" ]] ||
       ! printf '%s  %s\n' "$digest" "$wheel" | sha256sum --check --status; then
        echo "Downloading $url ..."
        curl "${CURL_OPTIONS[@]}" --output "$wheel" "$url"
    else
        echo "Using cached: $wheel"
    fi
    printf '%s  %s\n' "$digest" "$wheel" | sha256sum --check
    unzip -q -o -j "$wheel" 'nvidia/*/lib/*.so*' -d "$SDK/lib"
    unzip -q -o -j "$wheel" 'nvidia/*/include/*' -d "$SDK/include" || true
    local license_entry
    license_entry="$(unzip -Z1 "$wheel" |
        grep -Ei '(^|/)(LICENSE|COPYING)([^/]*)$' | head -n 1 || true)"
    [[ -z "$license_entry" ]] ||
        unzip -p "$wheel" "$license_entry" > "$SDK/LICENSE.$name.txt"
    rm -f -- "$wheel"
}

[[ -f "$SDK/lib/libcudnn.so.9" &&
   -f "$SDK/LICENSE.nvidia-cudnn-cu13.txt" ]] || install_wheel \
    nvidia-cudnn-cu13 \
    "nvidia_cudnn_cu13-${MLVC_CUDNN_WHEEL_VERSION}-py3-none-manylinux_2_27_x86_64.whl" \
    "$MLVC_CUDNN_WHEEL_SHA256"
[[ -f "$SDK/lib/libcusparseLt.so.0" &&
   -f "$SDK/LICENSE.nvidia-cusparselt-cu13.txt" ]] || install_wheel \
    nvidia-cusparselt-cu13 \
    "nvidia_cusparselt_cu13-${MLVC_CUSPARSELT_WHEEL_VERSION}-py3-none-manylinux2014_x86_64.whl" \
    "$MLVC_CUSPARSELT_WHEEL_SHA256"
[[ -f "$SDK/lib/libnccl.so.2" &&
   -f "$SDK/LICENSE.nvidia-nccl-cu13.txt" ]] || install_wheel \
    nvidia-nccl-cu13 \
    "nvidia_nccl_cu13-${MLVC_NCCL_WHEEL_VERSION}-py3-none-manylinux_2_18_x86_64.whl" \
    "$MLVC_NCCL_WHEEL_SHA256"
[[ -f "$SDK/lib/libnvshmem_host.so.3" &&
   -f "$SDK/LICENSE.nvidia-nvshmem-cu13.txt" ]] || install_wheel \
    nvidia-nvshmem-cu13 \
    "nvidia_nvshmem_cu13-${MLVC_NVSHMEM_WHEEL_VERSION}-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" \
    "$MLVC_NVSHMEM_WHEEL_SHA256"

CUPTI_LIBRARY="/usr/local/cuda-${MLVC_CUDA_VERSION}/extras/CUPTI/lib64"
cp -a "$CUPTI_LIBRARY"/libcupti.so* "$SDK/lib/"
cp -a "/usr/local/cuda-${MLVC_CUDA_VERSION}/EULA.txt" \
    "$SDK/LICENSE.cuda-toolkit-$MLVC_CUDA_VERSION.txt"

PYTORCH_LICENSE="$SDK/LICENSE.pytorch.txt"
if ! [[ -f "$PYTORCH_LICENSE" ]] ||
   ! printf '%s  %s\n' "$MLVC_PYTORCH_LICENSE_SHA256" "$PYTORCH_LICENSE" |
        sha256sum --check --status; then
    curl "${CURL_OPTIONS[@]}" --output "$PYTORCH_LICENSE" \
        "https://raw.githubusercontent.com/pytorch/pytorch/v${MLVC_LIBTORCH_VERSION}/LICENSE"
fi
printf '%s  %s\n' "$MLVC_PYTORCH_LICENSE_SHA256" "$PYTORCH_LICENSE" |
    sha256sum --check
ln -sfn libcudnn.so.9 "$SDK/lib/libcudnn.so"
ln -sfn libcusparseLt.so.0 "$SDK/lib/libcusparseLt.so"
sdk_is_complete || { echo "error: libtorch dependency closure is incomplete" >&2; exit 1; }
echo "Installed: $SDK"
