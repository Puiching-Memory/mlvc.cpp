#!/usr/bin/env bash
# Fetch an ONNX Runtime package into third_party/onnxruntime/.
# Usage: ./scripts/fetch_onnxruntime.sh [--gpu] [--version 1.19.2]
set -euo pipefail

VERSION="1.19.2"
VARIANT="x64"          # cpu edition
EXTRA=""
for arg in "$@"; do
    case "$arg" in
        --gpu) VARIANT="x64-gpu" ;;
        --version) ;;
        *) if [[ "$VERSION" != *"$arg"* ]]; then EXTRA="$EXTRA $arg"; fi ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"
PKG="onnxruntime-linux-${VARIANT}-${VERSION}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PKG}.tgz"
TARBALL="$DEST/onnxruntime.tar.gz"

mkdir -p "$DEST"
if [[ -d "$DEST/$PKG" ]]; then
    echo "Already present: $DEST/$PKG"
    exit 0
fi

echo "Downloading $URL ..."
curl -L --fail -o "$TARBALL" "$URL"
tar xzf "$TARBALL" -C "$DEST"
rm -f "$TARBALL"
echo "Extracted to $DEST/$PKG"
echo
echo "Configure with:"
echo "  cmake -S . -B build -DONNXRUNTIME_ROOT=\"$DEST/$PKG\""
