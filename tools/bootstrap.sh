#!/usr/bin/env bash
# Initialize source submodules and optionally install the CUDA build toolkit.
# Usage: ./tools/bootstrap.sh [--with-cuda]
#                             [--no-proxy]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tools/dependencies.env
source "$ROOT/tools/dependencies.env"

WITH_CUDA=0
NO_PROXY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-cuda) WITH_CUDA=1; shift ;;
        --no-proxy) NO_PROXY=1; shift ;;
        -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

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
    for tool in cmake c++ curl sha256sum tar; do
        command -v "$tool" >/dev/null || missing=1
    done
    [[ "$missing" -eq 0 ]] && return
    require_ubuntu_2604
    configure_privilege
    "${PRIVILEGE[@]}" apt-get update
    "${PRIVILEGE[@]}" apt-get install -y \
        build-essential ca-certificates cmake coreutils curl tar
}

CURL_OPTIONS=(--fail --location --retry 3)
[[ "$NO_PROXY" -eq 1 ]] && CURL_OPTIONS+=(--noproxy '*')

fetch_cuda() {
    local cuda_root="/usr/local/cuda-$MLVC_CUDA_VERSION"
    if [[ -x "$cuda_root/bin/nvcc" && -f "$cuda_root/lib64/libcudart.so.13" ]]; then
        echo "Already present: $cuda_root"
    else
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
        echo "Installed: $cuda_root"
    fi
    # Driverless hosts only have the unversioned stub (SONAME libcuda.so.1), so
    # -Wl,-rpath-link cannot locate the codec's DT_NEEDED libcuda.so.1 by name.
    # Alias the stub in both deb-layout locations to make CI links succeed.
    local stub_dir
    for stub_dir in "$cuda_root/lib64/stubs" \
                    "$cuda_root/targets/x86_64-linux/lib/stubs"; do
        if [[ -f "$stub_dir/libcuda.so" && ! -e "$stub_dir/libcuda.so.1" ]]; then
            configure_privilege
            "${PRIVILEGE[@]}" ln -sf libcuda.so "$stub_dir/libcuda.so.1"
        fi
    done
    # Surface nvcc to later CI steps.
    [[ -z "${GITHUB_PATH:-}" ]] || echo "$cuda_root/bin" >> "$GITHUB_PATH"
}

command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }
git -C "$ROOT" submodule sync --recursive
git -C "$ROOT" submodule update --init --recursive --depth 1
[[ "$WITH_CUDA" -eq 0 ]] && exit 0

install_host_tools
fetch_cuda
