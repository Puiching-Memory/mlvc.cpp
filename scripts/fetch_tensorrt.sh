#!/usr/bin/env bash
# Extract a TensorRT GA tarball into third_party/tensorrt/.
# Usage: ./scripts/fetch_tensorrt.sh /path/to/TensorRT-10.x.x.x.Linux.x86_64-gnu.cuda-12.x.tar.gz
#
# NVIDIA distributes the GA tarballs behind a login, so this script cannot
# download them directly. Get one from:
#   https://developer.nvidia.com/tensorrt  (Download section)
# or reuse a tarball you already have.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/TensorRT-*.tar.gz" >&2
    echo "Download: https://developer.nvidia.com/tensorrt (requires NVIDIA login)" >&2
    exit 1
fi
TARBALL="$1"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/tensorrt"
mkdir -p "$DEST"

echo "Extracting $TARBALL ..."
tar xzf "$TARBALL" -C "$DEST"
DIR="$(find "$DEST" -maxdepth 1 -type d -name 'TensorRT-*' | head -1)"
if [[ -z "$DIR" ]]; then
    echo "No TensorRT-* directory found after extraction" >&2
    exit 1
fi
echo "Extracted to $DIR"
echo
echo "Configure with:"
echo "  cmake -S . -B build -DMLVC_WITH_TENSORRT=ON -DTENSORRT_ROOT=\"$DIR\""
