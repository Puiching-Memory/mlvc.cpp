#!/usr/bin/env bash
# Fetch a libtorch package into third_party/libtorch/.
#
# Usage: ./scripts/fetch_libtorch.sh [--cuda cu126|cu130|...] [--version X.Y.Z]
#          [--os linux|windows] [--no-proxy] [--no-verify] [--print-dir]
# (CPU edition by default; libtorch >= 2.13 requires a C++20 toolchain.)
#
# Reproducibility: the version is pinned here and every download is verified
# against the pinned SHA256 table below. Variants with an empty entry are not
# pinned yet; the script prints the computed sum so it can be added.
# Note: download.pytorch.org has no stable macOS libtorch naming across
# releases, so macOS is not supported by this script.
set -euo pipefail

VERSION="2.13.0"
VARIANT="cpu"
NO_PROXY=0
VERIFY=1
PRINT_DIR=0
OS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) VARIANT="${2:?--cuda needs a tag, e.g. cu126}"; shift 2 ;;
        --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
        --os) OS="${2:?--os needs a value}"; shift 2 ;;
        --no-proxy) NO_PROXY=1; shift ;;
        --no-verify) VERIFY=0; shift ;;
        --print-dir) PRINT_DIR=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OS" ]]; then
    case "$(uname -s)" in
        Linux*)                OS=linux ;;
        MINGW*|MSYS*|CYGWIN*)  OS=windows ;;
        Darwin*) echo "macOS libtorch naming is not stable; download manually from https://download.pytorch.org/libtorch/cpu/" >&2; exit 1 ;;
        *) echo "Unsupported OS: $(uname -s) (pass --os)" >&2; exit 1 ;;
    esac
fi

# Asset naming changed in newer releases (the cxx11-abi infix was dropped);
# probe the known patterns and use the first that exists.
if [[ "$OS" == "windows" ]]; then
    CANDIDATES=("libtorch-win-shared-with-deps-${VERSION}%2B${VARIANT}.zip")
else
    CANDIDATES=(
        "libtorch-shared-with-deps-${VERSION}%2B${VARIANT}.zip"
        "libtorch-cxx11-abi-shared-with-deps-${VERSION}%2B${VARIANT}.zip"
    )
fi

# --- pinned SHA256 of the packages ------------------------------------------
# Key: "<os>-<variant>". Empty = not pinned for this variant; the computed sum
# is printed after download so it can be pasted here.
pinned_sha256() {
    case "$1" in
        linux-cpu) echo "edbf4cbed78433d803e90a65f1752e57783d164bce66c95c0872b2ab8f5c159e" ;;
        *) echo "" ;;
    esac
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/libtorch"
DIR="$DEST/libtorch"

if [[ "$PRINT_DIR" == "1" ]]; then
    [[ -d "$DIR" ]] || { echo "Not fetched yet: run $0 first" >&2; exit 1; }
    echo "$DIR"
    exit 0
fi

if [[ -d "$DIR" ]]; then
    echo "Already present: $DIR"
    exit 0
fi

CURL_OPTS=(-L --fail)
[[ "$NO_PROXY" == "1" ]] && CURL_OPTS+=(--noproxy '*')

URL=""
for name in "${CANDIDATES[@]}"; do
    candidate="https://download.pytorch.org/libtorch/${VARIANT}/${name}"
    if curl "${CURL_OPTS[@]}" -sI -o /dev/null "$candidate"; then
        URL="$candidate"
        break
    fi
done
[[ -n "$URL" ]] || { echo "No libtorch ${VERSION}+${VARIANT} package found for $OS" >&2; exit 1; }

mkdir -p "$DEST"
ARCHIVE="$DEST/libtorch.zip"
echo "Downloading $URL ..."
curl "${CURL_OPTS[@]}" -o "$ARCHIVE" "$URL"

if [[ "$VERIFY" == "1" ]]; then
    if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
    GOT="$($SHA "$ARCHIVE" | cut -d' ' -f1)"
    WANT="$(pinned_sha256 "${OS}-${VARIANT}")"
    if [[ -z "$WANT" ]]; then
        echo "WARNING: no pinned sha256 for ${OS}-${VARIANT} v${VERSION}; computed: $GOT" >&2
    elif [[ "$GOT" != "$WANT" ]]; then
        rm -f "$ARCHIVE"
        echo "SHA256 mismatch: got $GOT, want $WANT" >&2
        exit 1
    fi
fi

echo "Extracting to $DIR ..."
if command -v unzip >/dev/null; then unzip -q "$ARCHIVE" -d "$DEST"
elif command -v python3 >/dev/null; then python3 -m zipfile -e "$ARCHIVE" "$DEST"
else python -m zipfile -e "$ARCHIVE" "$DEST"; fi
rm -f "$ARCHIVE"

echo "Configure with:"
echo "  cmake -S . -B build -DMLVC_WITH_LIBTORCH=ON -DLIBTORCH_ROOT=\"$DIR\""
