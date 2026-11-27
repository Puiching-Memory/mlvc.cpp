#!/usr/bin/env bash
# End-to-end MP4 neural video coding demo & side-by-side H.264 comparison.
# Usage: ./tools/scripts/demo_mp4.sh [INPUT.mp4 | --synthetic]
#          [--output-dir DIR] [--profile mlvc-psnr-v1|mlvc-s-psnr-v1]
#          [--model-dir DIR] [--binary PATH] [--backend NAME] [--threads N]
#          [--q-index 21] [--frames 120] [--fps 30] [--width 640] [--height 360]
#          [--no-compare] [--play]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INPUT=""
SYNTHETIC=0
OUTPUT_DIR="$ROOT/demo_output"
PROFILE="mlvc-s-psnr-v1"
MODEL_DIR=""
BINARY=""
BACKEND=""
THREADS=""
Q_INDEX=21
FRAMES=120
FPS=30
WIDTH=640
HEIGHT=360
COMPARE_H264=1
PLAY=0

usage() {
    sed -n '2,7p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --synthetic) SYNTHETIC=1; shift ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir needs a value}"; shift 2 ;;
        --profile) PROFILE="${2:?--profile needs a value}"; shift 2 ;;
        --model-dir) MODEL_DIR="${2:?--model-dir needs a value}"; shift 2 ;;
        --binary) BINARY="${2:?--binary needs a value}"; shift 2 ;;
        --backend) BACKEND="${2:?--backend needs a value}"; shift 2 ;;
        --threads) THREADS="${2:?--threads needs a value}"; shift 2 ;;
        --q-index) Q_INDEX="${2:?--q-index needs a value}"; shift 2 ;;
        --frames) FRAMES="${2:?--frames needs a value}"; shift 2 ;;
        --fps) FPS="${2:?--fps needs a value}"; shift 2 ;;
        --width) WIDTH="${2:?--width needs a value}"; shift 2 ;;
        --height) HEIGHT="${2:?--height needs a value}"; shift 2 ;;
        --no-compare) COMPARE_H264=0; shift ;;
        --play) PLAY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            if [[ -z "$INPUT" ]]; then
                INPUT="$1"; shift
            else
                echo "error: unexpected positional argument: $1" >&2; exit 2
            fi
            ;;
    esac
done

if [[ -z "$INPUT" && "$SYNTHETIC" -eq 0 ]]; then
    echo "No input video provided; generating a ${FRAMES}-frame synthetic test clip (--synthetic)."
    SYNTHETIC=1
fi

command -v ffmpeg >/dev/null 2>&1 || {
    echo "error: ffmpeg is required for MP4 conversion and PSNR/SSIM comparison" >&2
    exit 1
}

find_binary() {
    if [[ -n "$BINARY" ]]; then
        [[ -x "$BINARY" ]] || { echo "error: mlvc_demo not executable: $BINARY" >&2; exit 1; }
        echo "$BINARY"
        return
    fi
    local cand
    for cand in \
        "$ROOT/build/driver-cubin-release/tools/mlvc_demo" \
        "$ROOT/build/all-release/tools/mlvc_demo" \
        "$ROOT/build/cpu-release/tools/mlvc_demo" \
        "$ROOT/build/cpu/tools/mlvc_demo" \
        "$(command -v mlvc_demo 2>/dev/null || true)"; do
        if [[ -n "$cand" && -x "$cand" ]]; then
            echo "$cand"
            return
        fi
    done
    echo "error: could not locate mlvc_demo; build the project or pass --binary PATH" >&2
    exit 1
}

BIN="$(find_binary)"
MODEL_ARGS=()
if [[ -n "$MODEL_DIR" ]]; then
    MODEL_ARGS=(--model-dir "$MODEL_DIR")
elif "$BIN" --list-model-profiles 2>/dev/null | grep -qx "$PROFILE"; then
    MODEL_ARGS=(--model-profile "$PROFILE")
else
    # Derive padded model height (multiple of 16) for canonical directory lookup.
    PADDED_H=$(( ((HEIGHT + 15) / 16) * 16 ))
    DEFAULT_MODEL_DIR="$ROOT/models/canonical/$PROFILE/${WIDTH}x${PADDED_H}"
    if [[ -f "$DEFAULT_MODEL_DIR/model_bundle.pb" ]]; then
        MODEL_ARGS=(--model-dir "$DEFAULT_MODEL_DIR")
    else
        echo "error: model '$PROFILE' not embedded and not found at $DEFAULT_MODEL_DIR" >&2
        echo "       Run ./tools/scripts/prepare_models.sh --profile $PROFILE or pass --model-dir DIR" >&2
        exit 1
    fi
fi

EXTRA_ARGS=()
[[ -n "$BACKEND" ]] && EXTRA_ARGS+=(--backend "$BACKEND")
[[ -n "$THREADS" ]] && EXTRA_ARGS+=(--threads "$THREADS")

mkdir -p "$OUTPUT_DIR"
SRC_YUV="$OUTPUT_DIR/source.yuv"
MLVC_BS="$OUTPUT_DIR/encoded.mlvc"
REC_YUV="$OUTPUT_DIR/reconstructed_mlvc.yuv"
REC_MP4="$OUTPUT_DIR/reconstructed_mlvc.mp4"

echo "[demo] Binary : $BIN"
echo "[demo] Model  : ${MODEL_ARGS[*]}"
echo "[demo] Config : ${WIDTH}x${HEIGHT} @ ${FPS}fps, frames=${FRAMES}, q_index=${Q_INDEX}"

if [[ "$SYNTHETIC" -eq 1 ]]; then
    DUR=$(awk "BEGIN { printf \"%.3f\", $FRAMES / $FPS }")
    echo "[demo] Generating ${FRAMES}-frame synthetic clip via ffmpeg testsrc2 ..."
    ffmpeg -hide_banner -loglevel error -y \
        -f lavfi -i "testsrc2=size=${WIDTH}x${HEIGHT}:rate=${FPS}:duration=${DUR}" \
        -frames:v "$FRAMES" -pix_fmt yuv420p -f rawvideo "$SRC_YUV"
else
    [[ -f "$INPUT" ]] || { echo "error: input file not found: $INPUT" >&2; exit 1; }
    echo "[demo] Decoding input video to ${WIDTH}x${HEIGHT} YUV420p (${FRAMES} frames) ..."
    ffmpeg -hide_banner -loglevel error -y \
        -i "$INPUT" -vf "scale=${WIDTH}:${HEIGHT}:flags=bicubic" \
        -frames:v "$FRAMES" -pix_fmt yuv420p -f rawvideo "$SRC_YUV"
fi

FRAME_BYTES=$(( WIDTH * HEIGHT * 3 / 2 ))
ACTUAL_BYTES=$(wc -c < "$SRC_YUV")
ACTUAL_FRAMES=$(( ACTUAL_BYTES / FRAME_BYTES ))
if [[ "$ACTUAL_FRAMES" -le 0 ]]; then
    echo "error: failed to produce any YUV420 frames" >&2
    exit 1
fi

echo "[demo] Encoding ${ACTUAL_FRAMES} frames with mlvc.cpp ..."
ENC_LOG="$("$BIN" encode \
    --input "$SRC_YUV" --output "$MLVC_BS" \
    --width "$WIDTH" --height "$HEIGHT" \
    --frames "$ACTUAL_FRAMES" --q-index "$Q_INDEX" \
    "${MODEL_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1)"
echo "       $ENC_LOG"

echo "[demo] Decoding ${ACTUAL_FRAMES} frames with mlvc.cpp ..."
DEC_LOG="$("$BIN" decode \
    --input "$MLVC_BS" --output "$REC_YUV" \
    --frames "$ACTUAL_FRAMES" \
    "${MODEL_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1)"
echo "       $DEC_LOG"

MLVC_BYTES=$(wc -c < "$MLVC_BS")
KBPS=$(awk "BEGIN { printf \"%.1f\", ($MLVC_BYTES * 8.0 * $FPS) / ($ACTUAL_FRAMES * 1000.0) }")

ffmpeg -hide_banner -loglevel error -y \
    -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i "$REC_YUV" \
    -c:v libx264 -crf 14 -pix_fmt yuv420p "$REC_MP4" 2>/dev/null || \
ffmpeg -hide_banner -loglevel error -y \
    -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i "$REC_YUV" \
    -pix_fmt yuv420p "$REC_MP4"

measure_quality() {
    local ref="$1" dist="$2"
    local stats
    stats="$(ffmpeg -hide_banner -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -i "$dist" \
        -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -i "$ref" \
        -lavfi "[0:v][1:v]psnr;[0:v][1:v]ssim" -f null - 2>&1)"
    local psnr ssim
    psnr="$(echo "$stats" | sed -n 's/.*average:\([0-9.]*\).*/\1/p' | tail -n1)"
    ssim="$(echo "$stats" | sed -n 's/.*All:\([0-9.]*\).*/\1/p' | tail -n1)"
    echo "${psnr:-n/a} ${ssim:-n/a}"
}

read -r MLVC_PSNR MLVC_SSIM <<<"$(measure_quality "$SRC_YUV" "$REC_YUV")"

echo ""
echo "==================== Results (${ACTUAL_FRAMES}f @ ${WIDTH}x${HEIGHT}, ${FPS}fps) ===================="
printf "%-14s %10s %12s %10s %10s\n" "Codec" "Size (KB)" "Bitrate" "PSNR (dB)" "SSIM"
printf "%-14s %10.1f %9s kbps %10s %10s\n" \
    "mlvc.cpp" "$(awk "BEGIN{printf \"%.1f\", $MLVC_BYTES/1024.0}")" "$KBPS" "$MLVC_PSNR" "$MLVC_SSIM"

if [[ "$COMPARE_H264" -eq 1 ]] && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264; then
    H264_MP4="$OUTPUT_DIR/anchor_x264.mp4"
    H264_YUV="$OUTPUT_DIR/anchor_x264.yuv"
    SBS_MP4="$OUTPUT_DIR/compare_mlvc_vs_x264.mp4"

    # Match the exact MLVC bitrate using 2-pass libx264 (medium preset, GOP=64).
    PASSLOG="$OUTPUT_DIR/x264_2pass"
    ffmpeg -hide_banner -loglevel error -y \
        -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i "$SRC_YUV" \
        -c:v libx264 -preset medium -g 64 -b:v "${KBPS}k" -pass 1 -passlogfile "$PASSLOG" \
        -an -f null /dev/null
    ffmpeg -hide_banner -loglevel error -y \
        -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i "$SRC_YUV" \
        -c:v libx264 -preset medium -g 64 -b:v "${KBPS}k" -pass 2 -passlogfile "$PASSLOG" \
        -an "$H264_MP4"
    rm -f "${PASSLOG}"*

    ffmpeg -hide_banner -loglevel error -y -i "$H264_MP4" -pix_fmt yuv420p -f rawvideo "$H264_YUV"
    H264_BYTES=$(wc -c < "$H264_MP4")
    H264_KBPS=$(awk "BEGIN { printf \"%.1f\", ($H264_BYTES * 8.0 * $FPS) / ($ACTUAL_FRAMES * 1000.0) }")
    read -r H264_PSNR H264_SSIM <<<"$(measure_quality "$SRC_YUV" "$H264_YUV")"

    printf "%-14s %10.1f %9s kbps %10s %10s\n" \
        "x264 (2-pass)" "$(awk "BEGIN{printf \"%.1f\", $H264_BYTES/1024.0}")" "$H264_KBPS" "$H264_PSNR" "$H264_SSIM"

    # Build a side-by-side comparison MP4 (MLVC left, H.264 right).
    ffmpeg -hide_banner -loglevel error -y \
        -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i "$REC_YUV" \
        -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i "$H264_YUV" \
        -filter_complex "[0:v][1:v]hstack=inputs=2" \
        -c:v libx264 -crf 14 -pix_fmt yuv420p "$SBS_MP4"
    echo "================================================================================"
    echo "[demo] Bitstream  : $MLVC_BS"
    echo "[demo] MLVC MP4   : $REC_MP4"
    echo "[demo] Side-by-Side (left=mlvc.cpp, right=x264): $SBS_MP4"
else
    echo "================================================================================"
    echo "[demo] Bitstream  : $MLVC_BS"
    echo "[demo] MLVC MP4   : $REC_MP4"
fi

if [[ "$PLAY" -eq 1 ]] && command -v ffplay >/dev/null 2>&1; then
    TARGET_PLAY="$REC_MP4"
    [[ -f "$OUTPUT_DIR/compare_mlvc_vs_x264.mp4" ]] && TARGET_PLAY="$OUTPUT_DIR/compare_mlvc_vs_x264.mp4"
    ffplay -autoexit -window_title "mlvc.cpp demo" "$TARGET_PLAY"
fi
