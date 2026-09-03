#!/usr/bin/env bash
# Build and package the embedded-cubin runtime. CUDA Toolkit is a build-only
# input; the produced package requires only the NVIDIA driver at runtime.
#
# Usage: ./scripts/package_driver_cubin.sh [--build-root DIR]
#          [--output-dir DIR] [--model-root DIR] [--jobs N] [--no-tar]
set -euo pipefail

BUILD_ROOT="build-driver-release"
OUTPUT_DIR="packages"
MODEL_ROOT="model-assets/models"
JOBS=""
NO_TAR=0

usage() {
    sed -n '2,6p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-root) BUILD_ROOT="${2:?--build-root needs a value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir needs a value}"; shift 2 ;;
        --model-root) MODEL_ROOT="${2:?--model-root needs a value}"; shift 2 ;;
        --jobs) JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        --no-tar) NO_TAR=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$BUILD_ROOT" = /* ]] || BUILD_ROOT="$ROOT/$BUILD_ROOT"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
[[ "$MODEL_ROOT" = /* ]] || MODEL_ROOT="$ROOT/$MODEL_ROOT"
if [[ -z "$JOBS" ]]; then
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: --jobs must be a positive integer" >&2
    exit 2
}

for tool in cmake ldd python3 readelf sha256sum tar nvidia-smi; do
    command -v "$tool" >/dev/null || {
        echo "error: $tool is required" >&2
        exit 1
    }
done
nvidia-smi -L >/dev/null || {
    echo "error: an NVIDIA driver and GPU are required" >&2
    exit 1
}

VERSION="$(sed -n 's/^project(mlvc_cpp VERSION \([0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt")"
PRODUCT="mlvc_cpp-${VERSION}-driver-cubin-nvidia-linux-x86_64"
mkdir -p "$BUILD_ROOT" "$OUTPUT_DIR"

cmake -S "$ROOT" -B "$BUILD_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLVC_DRIVER_CUBIN_ONLY=ON \
    -DMLVC_EMBEDDED_MODEL_ROOT="$MODEL_ROOT" \
    -DMLVC_ENABLE_IPO=ON \
    -DCMAKE_INSTALL_BINDIR=bin \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_INCLUDEDIR=include \
    -DBUILD_TESTING=ON
cmake --build "$BUILD_ROOT" --parallel "$JOBS"
ctest --test-dir "$BUILD_ROOT" --output-on-failure

stage_root="$(mktemp -d "$OUTPUT_DIR/.stage-driver-cubin.XXXXXX")"
trap 'cmake -E remove_directory "$stage_root"' EXIT
prefix="$stage_root/$PRODUCT"
cmake --install "$BUILD_ROOT" --prefix "$prefix"

demo="$prefix/bin/mlvc_demo"
benchmark="$prefix/bin/mlvc_backend_bench"
probe="$prefix/bin/mlvc_driver_probe"
codec_library="$(find "$prefix/lib" -maxdepth 1 -type f -name 'libmlvc_codec.so.*' -print -quit)"
[[ -n "$codec_library" ]] || {
    echo "error: packaged codec library is missing: libmlvc_codec.so.*" >&2
    exit 1
}
for binary in "$demo" "$benchmark" "$probe"; do
    [[ -x "$binary" ]] || {
        echo "error: packaged driver-cubin executable is missing: $binary" >&2
        exit 1
    }
    linkage="$(ldd "$binary")"
    [[ "$linkage" != *"not found"* ]] || {
        echo "$linkage" >&2
        exit 1
    }
    [[ "$linkage" == *"libcuda.so.1"* ]] || {
        echo "error: $binary does not require the NVIDIA driver" >&2
        exit 1
    }
    if grep -Eqi 'libcudart|libcudnn|libcublas|libnvinfer|libonnxruntime|libtorch' \
            <<<"$linkage"; then
        echo "error: a forbidden CUDA Toolkit or inference runtime was linked" >&2
        echo "$linkage" >&2
        exit 1
    fi
done
linkage="$(ldd "$codec_library")"
[[ "$linkage" != *"not found"* ]] || {
    echo "$linkage" >&2
    exit 1
}
[[ "$linkage" == *"libcuda.so.1"* ]] || {
    echo "error: $codec_library does not require the NVIDIA driver" >&2
    exit 1
}
if grep -Eqi 'libcudart|libcudnn|libcublas|libnvinfer|libonnxruntime|libtorch' \
        <<<"$linkage"; then
    echo "error: a forbidden CUDA Toolkit or inference runtime was linked" >&2
    echo "$linkage" >&2
    exit 1
fi
[[ "$("$demo" --backend-name)" == "driver-cubin" ]] || {
    echo "error: packaged codec is not the driver-cubin variant" >&2
    exit 1
}
[[ "$("$benchmark" --backend-name)" == "driver-cubin" ]] || {
    echo "error: packaged benchmark is not the driver-cubin variant" >&2
    exit 1
}
bundled_models="$("$demo" --list-model-profiles | paste -sd ',' -)"
[[ -n "$bundled_models" ]] || {
    echo "error: packaged codec contains no embedded models" >&2
    exit 1
}
[[ ! -e "$prefix/share/mlvc/models" ]] || {
    echo "error: driver-cubin models must be embedded, not installed separately" >&2
    exit 1
}
"$probe" --iterations 100 >/dev/null

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
fatbin_sha="$(sha256sum "$ROOT/assets/cubin/mlvc_driver_kernels.fatbin" | cut -d' ' -f1)"
embedded_models="$BUILD_ROOT/generated/embedded-models/mlvc_driver_models.bin"
[[ -f "$embedded_models" ]] || {
    echo "error: embedded model image is missing: $embedded_models" >&2
    exit 1
}
embedded_models_sha="$(sha256sum "$embedded_models" | cut -d' ' -f1)"
embedded_models_bytes="$(stat -c '%s' "$embedded_models")"
{
    printf 'name=%s\n' "$PRODUCT"
    printf 'version=%s\n' "$VERSION"
    printf 'backend=driver-cubin\n'
    printf 'codec_pipeline=complete\n'
    printf 'aot_graphs=MLVCEncoder,MLVCDecoder\n'
    printf 'codec_library=%s\n' "$(basename "$codec_library")"
    printf 'codec_api=c-abi,cxx\n'
    printf 'floating_point=fp16-only\n'
    printf 'runtime_gpu_dependency=libcuda.so.1\n'
    printf 'cuda_toolkit_runtime_dependency=false\n'
    printf 'build_driver=%s\n' "$driver_version"
    printf 'fatbin_sha256=%s\n' "$fatbin_sha"
    printf 'fatbin_targets=sm_75,sm_80,sm_86,sm_89,compute_89\n'
    printf 'model_storage=embedded:lib/libmlvc_codec.so\n'
    printf 'bundled_models=%s\n' "$bundled_models"
    printf 'embedded_models_bytes=%s\n' "$embedded_models_bytes"
    printf 'embedded_models_sha256=%s\n' "$embedded_models_sha"
} > "$prefix/BUILD-MANIFEST.txt"
(
    cd "$prefix"
    find bin include lib share -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$prefix/SHA256SUMS"

if [[ "$NO_TAR" -eq 1 ]]; then
    [[ ! -e "$OUTPUT_DIR/$PRODUCT" ]] || {
        echo "error: output already exists: $OUTPUT_DIR/$PRODUCT" >&2
        exit 1
    }
    mv "$prefix" "$OUTPUT_DIR/$PRODUCT"
    rmdir "$stage_root"
    trap - EXIT
    echo "package tree: $OUTPUT_DIR/$PRODUCT"
else
    tarball="$OUTPUT_DIR/${PRODUCT}.tar.gz"
    tar -C "$stage_root" -czf "$tarball" "$PRODUCT"
    cmake -E remove_directory "$stage_root"
    trap - EXIT
    echo "package: $tarball"
fi
