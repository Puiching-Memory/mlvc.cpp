#!/usr/bin/env bash
# Initialize source submodules and acquire the pinned GPU SDKs.
# Usage: ./tools/bootstrap.sh [--backend sources|all|onnxruntime|libtorch|tensorrt]
#                             [--no-proxy]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tools/dependencies.env
source "$ROOT/tools/dependencies.env"

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

require_ubuntu_2604() {
    [[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || {
        echo "error: only Linux x86_64 is supported" >&2
        return 1
    }
    # shellcheck source=/dev/null
    source /etc/os-release
    [[ "$ID:$VERSION_ID" == ubuntu:26.04 ]] || {
        echo "error: the pinned NVIDIA stack requires Ubuntu 26.04" >&2
        return 1
    }
}

PRIVILEGE=()
configure_privilege() {
    if [[ "$EUID" -ne 0 ]]; then
        command -v sudo >/dev/null || {
            echo "error: root or sudo is required to install system packages" >&2
            return 1
        }
        PRIVILEGE=(sudo)
    fi
}

install_host_tools() {
    local missing=0 tool
    for tool in cmake c++ curl sha256sum tar unzip; do
        command -v "$tool" >/dev/null || missing=1
    done
    [[ "$missing" -eq 0 ]] && return
    require_ubuntu_2604
    configure_privilege
    "${PRIVILEGE[@]}" apt-get update
    "${PRIVILEGE[@]}" apt-get install -y \
        build-essential ca-certificates cmake coreutils curl tar unzip
}

CURL_OPTIONS=(--fail --location --retry 3)
[[ "$NO_PROXY" -eq 1 ]] && CURL_OPTIONS+=(--noproxy '*')

fetch_cuda() {
    local cuda_root="/usr/local/cuda-$MLVC_CUDA_VERSION"
    if [[ -x "$cuda_root/bin/nvcc" && -f "$cuda_root/lib64/libcudart.so.13" ]]; then
        echo "Already present: $cuda_root"
        return
    fi
    require_ubuntu_2604
    configure_privilege
    local keyring
    keyring="$(mktemp --suffix=.deb)"
    curl "${CURL_OPTIONS[@]}" --output "$keyring" \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
    "${PRIVILEGE[@]}" dpkg -i "$keyring"
    rm -f -- "$keyring"
    "${PRIVILEGE[@]}" apt-get update
    "${PRIVILEGE[@]}" apt-get install -y cuda-toolkit-13-3
    [[ -x "$cuda_root/bin/nvcc" ]] || {
        echo "error: CUDA installation failed" >&2
        return 1
    }
}

fetch_onnxruntime() {
    local archive_package="onnxruntime-linux-x64-gpu_cuda13-${MLVC_ONNXRUNTIME_VERSION}"
    local sdk_package="onnxruntime-linux-x64-gpu-${MLVC_ONNXRUNTIME_VERSION}"
    local destination="$ROOT/third_party/onnxruntime"
    local sdk="$destination/$sdk_package"
    if [[ -f "$sdk/lib/libonnxruntime_providers_cuda.so" &&
          -f "$sdk/lib/libcudnn.so.9" &&
          -f "$sdk/LICENSE.nvidia-cudnn-cu13.txt" ]]; then
        echo "Already present: $sdk"
        return
    fi

    mkdir -p "$destination"
    if [[ ! -f "$sdk/lib/libonnxruntime_providers_cuda.so" ]]; then
        local archive="$destination/$archive_package.tgz"
        local url="https://github.com/microsoft/onnxruntime/releases/download/v${MLVC_ONNXRUNTIME_VERSION}/$archive_package.tgz"
        echo "Downloading $url ..."
        curl "${CURL_OPTIONS[@]}" --output "$archive" "$url"
        printf '%s  %s\n' "$MLVC_ONNXRUNTIME_SHA256" "$archive" | sha256sum --check
        tar -xzf "$archive" -C "$destination"
        rm -f -- "$archive"
    fi

    if [[ ! -f "$sdk/lib/libcudnn.so.9" ]]; then
        local wheel="nvidia_cudnn_cu13-${MLVC_CUDNN_WHEEL_VERSION}-py3-none-manylinux_2_27_x86_64.whl"
        local wheel_path="$destination/$wheel"
        local url="https://pypi.nvidia.com/nvidia-cudnn-cu13/$wheel"
        if ! [[ -f "$wheel_path" ]] ||
           ! printf '%s  %s\n' "$MLVC_CUDNN_WHEEL_SHA256" "$wheel_path" |
                sha256sum --check --status; then
            curl "${CURL_OPTIONS[@]}" --output "$wheel_path" "$url"
        else
            echo "Using cached: $wheel_path"
        fi
        printf '%s  %s\n' "$MLVC_CUDNN_WHEEL_SHA256" "$wheel_path" | sha256sum --check
        unzip -q -o -j "$wheel_path" 'nvidia/*/lib/*.so*' -d "$sdk/lib"
        local license_entry
        license_entry="$(unzip -Z1 "$wheel_path" |
            grep -Ei '(^|/)(LICENSE|COPYING)([^/]*)$' | head -n 1 || true)"
        [[ -z "$license_entry" ]] ||
            unzip -p "$wheel_path" "$license_entry" > "$sdk/LICENSE.nvidia-cudnn-cu13.txt"
        rm -f -- "$wheel_path"
        ln -sfn libcudnn.so.9 "$sdk/lib/libcudnn.so"
    fi
    [[ -f "$sdk/lib/libonnxruntime_providers_cuda.so" &&
       -f "$sdk/lib/libcudnn.so.9" &&
       -f "$sdk/LICENSE.nvidia-cudnn-cu13.txt" ]] || {
        echo "error: ONNX Runtime dependency closure is incomplete" >&2
        return 1
    }
    echo "Installed: $sdk"
}

install_libtorch_wheel() {
    local destination="$1" sdk="$2" name="$3" filename="$4" digest="$5"
    local wheel="$destination/$filename"
    local url="https://pypi.nvidia.com/$name/$filename"
    if ! [[ -f "$wheel" ]] ||
       ! printf '%s  %s\n' "$digest" "$wheel" | sha256sum --check --status; then
        echo "Downloading $url ..."
        curl "${CURL_OPTIONS[@]}" --output "$wheel" "$url"
    else
        echo "Using cached: $wheel"
    fi
    printf '%s  %s\n' "$digest" "$wheel" | sha256sum --check
    unzip -q -o -j "$wheel" 'nvidia/*/lib/*.so*' -d "$sdk/lib"
    unzip -q -o -j "$wheel" 'nvidia/*/include/*' -d "$sdk/include" || true
    local license_entry
    license_entry="$(unzip -Z1 "$wheel" |
        grep -Ei '(^|/)(LICENSE|COPYING)([^/]*)$' | head -n 1 || true)"
    [[ -z "$license_entry" ]] ||
        unzip -p "$wheel" "$license_entry" > "$sdk/LICENSE.$name.txt"
    rm -f -- "$wheel"
}

libtorch_is_complete() {
    local sdk="$1" library license
    for library in libtorch_cuda.so libcudnn.so.9 libcusparseLt.so.0 \
                   libnccl.so.2 libnvshmem_host.so.3 libcupti.so.13; do
        [[ -f "$sdk/lib/$library" ]] || return 1
    done
    for license in LICENSE.pytorch.txt \
                   "LICENSE.cuda-toolkit-$MLVC_CUDA_VERSION.txt" \
                   LICENSE.nvidia-cudnn-cu13.txt \
                   LICENSE.nvidia-cusparselt-cu13.txt \
                   LICENSE.nvidia-nccl-cu13.txt \
                   LICENSE.nvidia-nvshmem-cu13.txt; do
        [[ -f "$sdk/$license" ]] || return 1
    done
}

fetch_libtorch() {
    local destination="$ROOT/third_party/libtorch/${MLVC_LIBTORCH_VERSION}-${MLVC_LIBTORCH_CUDA_VARIANT}"
    local sdk="$destination/libtorch"
    if libtorch_is_complete "$sdk"; then
        echo "Already present: $sdk"
        return
    fi

    mkdir -p "$destination"
    if [[ ! -f "$sdk/lib/libtorch_cuda.so" ]]; then
        local archive="$destination/libtorch-${MLVC_LIBTORCH_VERSION}-${MLVC_LIBTORCH_CUDA_VARIANT}.zip"
        local url="https://download.pytorch.org/libtorch/${MLVC_LIBTORCH_CUDA_VARIANT}/libtorch-shared-with-deps-${MLVC_LIBTORCH_VERSION}%2B${MLVC_LIBTORCH_CUDA_VARIANT}.zip"
        echo "Downloading $url ..."
        curl "${CURL_OPTIONS[@]}" --output "$archive" "$url"
        printf '%s  %s\n' "$MLVC_LIBTORCH_SHA256" "$archive" | sha256sum --check
        unzip -q "$archive" -d "$destination"
        rm -f -- "$archive"
    fi

    [[ -f "$sdk/lib/libcudnn.so.9" &&
       -f "$sdk/LICENSE.nvidia-cudnn-cu13.txt" ]] || install_libtorch_wheel \
        "$destination" "$sdk" nvidia-cudnn-cu13 \
        "nvidia_cudnn_cu13-${MLVC_CUDNN_WHEEL_VERSION}-py3-none-manylinux_2_27_x86_64.whl" \
        "$MLVC_CUDNN_WHEEL_SHA256"
    [[ -f "$sdk/lib/libcusparseLt.so.0" &&
       -f "$sdk/LICENSE.nvidia-cusparselt-cu13.txt" ]] || install_libtorch_wheel \
        "$destination" "$sdk" nvidia-cusparselt-cu13 \
        "nvidia_cusparselt_cu13-${MLVC_CUSPARSELT_WHEEL_VERSION}-py3-none-manylinux2014_x86_64.whl" \
        "$MLVC_CUSPARSELT_WHEEL_SHA256"
    [[ -f "$sdk/lib/libnccl.so.2" &&
       -f "$sdk/LICENSE.nvidia-nccl-cu13.txt" ]] || install_libtorch_wheel \
        "$destination" "$sdk" nvidia-nccl-cu13 \
        "nvidia_nccl_cu13-${MLVC_NCCL_WHEEL_VERSION}-py3-none-manylinux_2_18_x86_64.whl" \
        "$MLVC_NCCL_WHEEL_SHA256"
    [[ -f "$sdk/lib/libnvshmem_host.so.3" &&
       -f "$sdk/LICENSE.nvidia-nvshmem-cu13.txt" ]] || install_libtorch_wheel \
        "$destination" "$sdk" nvidia-nvshmem-cu13 \
        "nvidia_nvshmem_cu13-${MLVC_NVSHMEM_WHEEL_VERSION}-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" \
        "$MLVC_NVSHMEM_WHEEL_SHA256"

    cp -a "/usr/local/cuda-${MLVC_CUDA_VERSION}/extras/CUPTI/lib64"/libcupti.so* "$sdk/lib/"
    cp -a "/usr/local/cuda-${MLVC_CUDA_VERSION}/EULA.txt" \
        "$sdk/LICENSE.cuda-toolkit-$MLVC_CUDA_VERSION.txt"
    local pytorch_license="$sdk/LICENSE.pytorch.txt"
    if ! [[ -f "$pytorch_license" ]] ||
       ! printf '%s  %s\n' "$MLVC_PYTORCH_LICENSE_SHA256" "$pytorch_license" |
            sha256sum --check --status; then
        curl "${CURL_OPTIONS[@]}" --output "$pytorch_license" \
            "https://raw.githubusercontent.com/pytorch/pytorch/v${MLVC_LIBTORCH_VERSION}/LICENSE"
    fi
    printf '%s  %s\n' "$MLVC_PYTORCH_LICENSE_SHA256" "$pytorch_license" |
        sha256sum --check
    ln -sfn libcudnn.so.9 "$sdk/lib/libcudnn.so"
    ln -sfn libcusparseLt.so.0 "$sdk/lib/libcusparseLt.so"
    libtorch_is_complete "$sdk" || {
        echo "error: libtorch dependency closure is incomplete" >&2
        return 1
    }
    echo "Installed: $sdk"
}

find_tensorrt_header() {
    local multiarch
    multiarch="$(gcc -print-multiarch)"
    [[ -f "/usr/include/$multiarch/NvInfer.h" ]] && {
        printf '%s\n' "/usr/include/$multiarch"
        return
    }
    [[ -f /usr/include/NvInfer.h ]] && printf '%s\n' /usr/include
}

tensorrt_is_complete() {
    local include_dir linker_cache major minor
    include_dir="$(find_tensorrt_header)" || return 1
    linker_cache="$(ldconfig -p)"
    grep -q 'libnvinfer\.so\.11' <<< "$linker_cache" || return 1
    grep -q 'libnvonnxparser\.so\.11' <<< "$linker_cache" || return 1
    major="$(sed -n 's/^#define TRT_MAJOR_ENTERPRISE \([0-9][0-9]*\).*/\1/p' \
        "$include_dir/NvInferVersion.h")"
    minor="$(sed -n 's/^#define TRT_MINOR_ENTERPRISE \([0-9][0-9]*\).*/\1/p' \
        "$include_dir/NvInferVersion.h")"
    [[ "$major.$minor" == 11.2 ]]
}

fetch_tensorrt() {
    if tensorrt_is_complete; then
        echo "Already present: TensorRT $MLVC_TENSORRT_VERSION"
        return
    fi
    require_ubuntu_2604
    configure_privilege
    local keyring
    keyring="$(mktemp --suffix=.deb)"
    curl "${CURL_OPTIONS[@]}" --output "$keyring" \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
    "${PRIVILEGE[@]}" dpkg -i "$keyring"
    rm -f -- "$keyring"
    "${PRIVILEGE[@]}" apt-get update
    "${PRIVILEGE[@]}" apt-get install -y \
        "libnvinfer-dev=$MLVC_TENSORRT_DEB_VERSION" \
        "libnvonnxparsers-dev=$MLVC_TENSORRT_DEB_VERSION"
    tensorrt_is_complete || {
        echo "error: TensorRT installation validation failed" >&2
        return 1
    }
    echo "Installed TensorRT $MLVC_TENSORRT_VERSION."
}

command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }
git -C "$ROOT" submodule sync --recursive
git -C "$ROOT" submodule update --init --recursive --depth 1
[[ "$BACKEND" == sources ]] && exit 0

install_host_tools
fetch_cuda
if [[ "$BACKEND" == all || "$BACKEND" == onnxruntime ]]; then
    fetch_onnxruntime
fi
if [[ "$BACKEND" == all || "$BACKEND" == libtorch ]]; then
    fetch_libtorch
fi
if [[ "$BACKEND" == all || "$BACKEND" == tensorrt ]]; then
    fetch_tensorrt
fi
