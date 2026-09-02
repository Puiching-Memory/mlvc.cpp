#!/usr/bin/env bash
# Fetch an ONNX Runtime package into third_party/onnxruntime/.
#
# Usage: ./scripts/fetch_onnxruntime.sh [--gpu] [--version X.Y.Z]
#          [--os linux|windows|macos] [--arch x64|arm64]
#          [--no-proxy] [--no-verify] [--print-dir]
#
# Reproducibility: the version is pinned here and every download is verified
# against the pinned SHA256 table below. Platforms with an empty entry are
# not pinned yet; the script prints the computed sum so it can be added.
set -euo pipefail

VERSION="1.19.2"
GPU=0
NO_PROXY=0
VERIFY=1
PRINT_DIR=0
OS=""
ARCH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU=1; shift ;;
        --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
        --os) OS="${2:?--os needs a value}"; shift 2 ;;
        --arch) ARCH="${2:?--arch needs a value}"; shift 2 ;;
        --no-proxy) NO_PROXY=1; shift ;;
        --no-verify) VERIFY=0; shift ;;
        --print-dir) PRINT_DIR=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# --- platform detection -----------------------------------------------------
if [[ -z "$OS" ]]; then
    case "$(uname -s)" in
        Linux*)                OS=linux ;;
        Darwin*)               OS=macos ;;
        MINGW*|MSYS*|CYGWIN*)  OS=windows ;;
        *) echo "Unsupported OS: $(uname -s) (pass --os)" >&2; exit 1 ;;
    esac
fi
if [[ -z "$ARCH" ]]; then
    case "$(uname -m)" in
        x86_64|amd64|AMD64) ARCH=x64 ;;
        aarch64|arm64)      ARCH=arm64 ;;
        *) echo "Unsupported arch: $(uname -m) (pass --arch)" >&2; exit 1 ;;
    esac
fi

# Asset naming used by the official GitHub release.
case "$OS-$ARCH" in
    linux-x64)      PLAT="linux-x64" ;;
    linux-arm64)    PLAT="linux-aarch64" ;;
    macos-x64)      PLAT="osx-universal2" ;;
    macos-arm64)    PLAT="osx-arm64" ;;
    windows-x64)    PLAT="win-x64" ;;
    windows-arm64)  PLAT="win-arm64" ;;
    *) echo "No official ONNX Runtime asset for $OS-$ARCH" >&2; exit 1 ;;
esac
SUFFIX=""
if [[ "$GPU" == "1" ]]; then
    case "$PLAT" in
        linux-x64|win-x64) SUFFIX="-gpu" ;;
        *) echo "No official GPU asset for $OS-$ARCH" >&2; exit 1 ;;
    esac
fi

# --- pinned SHA256 of the release assets ------------------------------------
# Key: "<plat>[-gpu]". Empty = not pinned for this platform; the computed sum
# is printed after download so it can be pasted here.
pinned_sha256() {
    case "$1" in
        linux-x64) echo "eb00c64e0041f719913c4080e0fed7d9963dc3aa9b54664df6036d8308dbcd33" ;;
        *) echo "" ;;
    esac
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"
PKG="onnxruntime-${PLAT}${SUFFIX}-${VERSION}"
DIR="$DEST/$PKG"

if [[ "$PRINT_DIR" == "1" ]]; then
    [[ -d "$DIR" ]] || { echo "Not fetched yet: run $0 first" >&2; exit 1; }
    echo "$DIR"
    exit 0
fi

if [[ -d "$DIR" ]]; then
    echo "Already present: $DIR"
    exit 0
fi

EXT="tgz"
[[ "$OS" == "windows" ]] && EXT="zip"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PKG}.${EXT}"
ARCHIVE="$DEST/${PKG}.${EXT}"

mkdir -p "$DEST"
echo "Downloading $URL ..."
CURL_OPTS=(-L --fail)
[[ "$NO_PROXY" == "1" ]] && CURL_OPTS+=(--noproxy '*')
curl "${CURL_OPTS[@]}" -o "$ARCHIVE" "$URL"

if [[ "$VERIFY" == "1" ]]; then
    if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
    GOT="$($SHA "$ARCHIVE" | cut -d' ' -f1)"
    WANT="$(pinned_sha256 "${PLAT}${SUFFIX}")"
    if [[ -z "$WANT" ]]; then
        echo "WARNING: no pinned sha256 for ${PLAT}${SUFFIX} v${VERSION}; computed: $GOT" >&2
    elif [[ "$GOT" != "$WANT" ]]; then
        rm -f "$ARCHIVE"
        echo "SHA256 mismatch for $PKG: got $GOT, want $WANT" >&2
        exit 1
    fi
fi

echo "Extracting to $DIR ..."
if [[ "$EXT" == "zip" ]]; then
    if command -v unzip >/dev/null; then unzip -q "$ARCHIVE" -d "$DEST"
    elif command -v python3 >/dev/null; then python3 -m zipfile -e "$ARCHIVE" "$DEST"
    else python -m zipfile -e "$ARCHIVE" "$DEST"; fi
else
    tar xzf "$ARCHIVE" -C "$DEST"
fi
rm -f "$ARCHIVE"

echo "Configure with:"
echo "  cmake -S . -B build -DONNXRUNTIME_ROOT=\"$DIR\""
