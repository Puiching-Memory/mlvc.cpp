#!/usr/bin/env bash
# Rebuild the precompiled driver-only fatbin. The CUDA Toolkit is required only
# on this build host; the resulting runtime dynamically uses the NVIDIA driver.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVCC="${NVCC:-/usr/local/cuda-13.3/bin/nvcc}"
OUTPUT="$ROOT/assets/cubin/mlvc_driver_kernels.fatbin"

[[ -x "$NVCC" ]] || {
    echo "error: nvcc not found: $NVCC" >&2
    exit 1
}

temporary="$(mktemp "$OUTPUT.tmp.XXXXXX")"
trap 'rm -f "$temporary"' EXIT

"$NVCC" --fatbin -O3 --std=c++20 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_89,code=compute_89 \
    "$ROOT/kernels/driver_kernels.cu" \
    -o "$temporary"

mv "$temporary" "$OUTPUT"
chmod 0644 "$OUTPUT"
trap - EXIT

echo "fatbin: $OUTPUT"
sha256sum "$OUTPUT"
"${NVCC%/bin/nvcc}/bin/cuobjdump" --list-elf "$OUTPUT"
"${NVCC%/bin/nvcc}/bin/cuobjdump" --list-ptx "$OUTPUT"
