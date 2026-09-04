#!/usr/bin/env bash
# Rebuild the precompiled driver-only fatbin. The CUDA Toolkit is required only
# on this build host; the resulting runtime dynamically uses the NVIDIA driver.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NVCC="${NVCC:-/usr/local/cuda-13.3/bin/nvcc}"
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT="${2:?--output needs a value}"; shift 2 ;;
        --help|-h)
            printf 'usage: %s --output FILE\n' "$0"
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$OUTPUT" ]]; then
    echo "error: --output is required" >&2
    exit 2
fi

[[ -x "$NVCC" ]] || {
    echo "error: nvcc not found: $NVCC" >&2
    exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
temporary="$(mktemp "$OUTPUT.tmp.XXXXXX")"
trap 'rm -f "$temporary"' EXIT

"$NVCC" --fatbin -O3 --std=c++20 --expt-relaxed-constexpr \
    -I"$ROOT/backends/driver_cubin/include" \
    -I"$ROOT/backends/driver_cubin/kernels" \
    -I/usr/include \
    -Xcompiler=-Wno-template-body \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_89,code=compute_89 \
    "$ROOT/backends/driver_cubin/kernels/module.cu" \
    -o "$temporary"

mv "$temporary" "$OUTPUT"
chmod 0644 "$OUTPUT"
trap - EXIT

echo "fatbin: $OUTPUT"
sha256sum "$OUTPUT"
CUOBJDUMP="$(dirname "$NVCC")/cuobjdump"
if [[ -x "$CUOBJDUMP" ]]; then
    "$CUOBJDUMP" --list-elf "$OUTPUT"
    "$CUOBJDUMP" --list-ptx "$OUTPUT"
fi
