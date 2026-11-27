#!/usr/bin/env bash
# Download official Microsoft MLVC checkpoints and compile canonical AOT bundles.
# Usage: ./tools/scripts/prepare_models.sh [--profile mlvc-psnr-v1|mlvc-s-psnr-v1|all]
#          [--width 640] [--height 368] [--checkpoint-dir DIR]
#          [--output-root DIR] [--overwrite]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PROFILE="all"
WIDTH=640
HEIGHT=368
CHECKPOINT_DIR="$ROOT/models/checkpoints"
OUTPUT_ROOT="$ROOT/models/canonical"
VENV_DIR="$ROOT/build/model-venv"
OVERWRITE=0

usage() {
    sed -n '2,5p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="${2:?--profile needs a value}"; shift 2 ;;
        --width) WIDTH="${2:?--width needs a value}"; shift 2 ;;
        --height) HEIGHT="${2:?--height needs a value}"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="${2:?--checkpoint-dir needs a value}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?--output-root needs a value}"; shift 2 ;;
        --venv) VENV_DIR="${2:?--venv needs a value}"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$PROFILE" in
    all) PROFILES=(mlvc-psnr-v1 mlvc-s-psnr-v1) ;;
    mlvc-psnr-v1|mlvc-s-psnr-v1) PROFILES=("$PROFILE") ;;
    *) echo "error: unsupported profile '$PROFILE' (expected mlvc-psnr-v1, mlvc-s-psnr-v1, or all)" >&2; exit 2 ;;
esac

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"
    fi
}

expected_sha256() {
    case "$1" in
        mlvc-psnr-v1) echo "834adaab680837c4106a9bb62e5778b3f13729b4fe6f751797e0747214f5cca1" ;;
        mlvc-s-psnr-v1) echo "1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896" ;;
    esac
}

checkpoint_url() {
    echo "https://mlvideopub.blob.core.windows.net/mlvc/models/${1}.ckpt"
}

ensure_venv() {
    local py="$VENV_DIR/bin/python"
    if [[ -x "$py" ]] && "$py" -c "import torch, google.protobuf, msrtc_rans, mlvc_tools" >/dev/null 2>&1; then
        echo "[models] Reusing Python environment: $VENV_DIR"
        return
    fi

    echo "[models] Setting up offline AOT compiler Python environment in $VENV_DIR ..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 "$VENV_DIR"
        uv pip install --python "$py" -r "$ROOT/tools/python/requirements-models.txt"
        uv pip install --python "$py" -e "$ROOT/tools/python"
        uv pip install --python "$py" "$ROOT/third_party/mlvc/packages/msrtc_rans"
    else
        command -v python3 >/dev/null 2>&1 || {
            echo "error: uv or python3 (3.12+) is required to compile checkpoints" >&2
            exit 1
        }
        python3 -m venv "$VENV_DIR"
        "$py" -m pip install --upgrade pip
        "$py" -m pip install -r "$ROOT/tools/python/requirements-models.txt"
        "$py" -m pip install -e "$ROOT/tools/python"
        "$py" -m pip install "$ROOT/third_party/mlvc/packages/msrtc_rans"
    fi
}

download_checkpoint() {
    local prof="$1"
    local dst="$CHECKPOINT_DIR/${prof}.ckpt"
    local want
    want="$(expected_sha256 "$prof")"
    mkdir -p "$CHECKPOINT_DIR"

    if [[ -f "$dst" ]]; then
        local got
        got="$(sha256_file "$dst")"
        if [[ "$got" == "$want" ]]; then
            echo "[models] Verified cached checkpoint: $dst"
            return
        fi
        echo "[models] Cached checkpoint checksum mismatch; re-downloading $prof ..." >&2
        rm -f "$dst"
    fi

    local url
    url="$(checkpoint_url "$prof")"
    echo "[models] Downloading $url ..."
    command -v curl >/dev/null 2>&1 || { echo "error: curl is required to download checkpoints" >&2; exit 1; }
    local tmp="${dst}.part"
    curl --fail --location --retry 3 --progress-bar --output "$tmp" "$url"
    local got
    got="$(sha256_file "$tmp")"
    if [[ "$got" != "$want" ]]; then
        rm -f "$tmp"
        echo "error: SHA-256 mismatch for $prof (got $got, expected $want)" >&2
        exit 1
    fi
    mv "$tmp" "$dst"
    echo "[models] Verified SHA-256: $dst"
}

[[ -f "$ROOT/third_party/mlvc/packages/msrtc_rans/CMakeLists.txt" ]] || {
    echo "error: submodules are not initialized; run ./tools/scripts/bootstrap.sh first" >&2
    exit 1
}

ensure_venv
PY="$VENV_DIR/bin/python"

for prof in "${PROFILES[@]}"; do
    download_checkpoint "$prof"
    gen_dir="$ROOT/models/generated/checkpoint/$prof/${WIDTH}x${HEIGHT}"
    out_dir="$OUTPUT_ROOT/$prof/${WIDTH}x${HEIGHT}"

    if [[ -f "$out_dir/model_bundle.pb" && "$OVERWRITE" -eq 0 ]]; then
        echo "[models] Canonical bundle already exists (use --overwrite to rebuild): $out_dir"
        continue
    fi

    echo "[models] Compiling AOT graph for $prof (${WIDTH}x${HEIGHT}) ..."
    "$PY" -m mlvc_tools.aot \
        --profile "$prof" \
        --checkpoint "$CHECKPOINT_DIR/${prof}.ckpt" \
        --model-width "$WIDTH" \
        --model-height "$HEIGHT" \
        --output-dir "$gen_dir"

    echo "[models] Assembling canonical bundle: $out_dir"
    assemble_args=(
        -m mlvc_tools.package assemble
        --profile "$prof"
        --source-dir "$gen_dir"
        --output-dir "$out_dir"
    )
    [[ "$OVERWRITE" -eq 1 ]] && assemble_args+=(--overwrite)
    "$PY" "${assemble_args[@]}"
done

echo "[models] Done. Canonical bundles ready under $OUTPUT_ROOT"
