#!/usr/bin/env bash
# Build isolated NVIDIA GPU releases for ONNX Runtime, libtorch, and TensorRT.
#
# Usage: ./scripts/package.sh [--backend all|onnxruntime|libtorch|tensorrt]
#          [--build-root DIR] [--output-dir DIR] [--jobs N] [--no-tar]
set -euo pipefail

BACKEND="all"
BUILD_ROOT="build-release"
OUTPUT_DIR="packages"
BUILD_TYPE="Release"
JOBS=""
NO_TAR=0

usage() {
    sed -n '2,6p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="${2:?--backend needs a value}"; shift 2 ;;
        --build-root) BUILD_ROOT="${2:?--build-root needs a value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir needs a value}"; shift 2 ;;
        --jobs) JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        --no-tar) NO_TAR=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$BACKEND" in
    all) BACKENDS=(onnxruntime libtorch tensorrt) ;;
    onnxruntime|libtorch|tensorrt) BACKENDS=("$BACKEND") ;;
    *) echo "error: --backend must be all, onnxruntime, libtorch, or tensorrt" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/dependencies.env
source "$ROOT/scripts/dependencies.env"
[[ "$BUILD_ROOT" = /* ]] || BUILD_ROOT="$ROOT/$BUILD_ROOT"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$ROOT/$OUTPUT_DIR"

if [[ -z "$JOBS" ]]; then
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "error: --jobs must be a positive integer" >&2; exit 2; }

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: NVIDIA GPU release packaging currently supports Linux only" >&2
    exit 1
fi
command -v cmake >/dev/null || { echo "error: cmake is required" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "error: CUDA 13.3 toolkit is required" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "error: nvidia-smi is required" >&2; exit 1; }
for tool in ldd sha256sum tar; do
    command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 1; }
done
nvidia-smi -L >/dev/null || { echo "error: no usable NVIDIA GPU was detected" >&2; exit 1; }
[[ -f "$ROOT/third_party/mlvc/packages/msrtc_rans/CMakeLists.txt" &&
   -f "$ROOT/third_party/nlohmann_json/CMakeLists.txt" ]] || {
    echo "error: source submodules are missing; run scripts/bootstrap.sh" >&2
    exit 1
}

has_library() {
    local root="$1"
    local name="$2"
    [[ -n "$root" ]] || return 1
    compgen -G "$root/lib/$name" >/dev/null ||
        compgen -G "$root/lib64/$name" >/dev/null
}

ONNXRUNTIME_SDK="$("$ROOT/scripts/fetch_onnxruntime.sh" --print-dir 2>/dev/null || true)"
LIBTORCH_SDK="$("$ROOT/scripts/fetch_libtorch.sh" --print-dir 2>/dev/null || true)"
TENSORRT_SDK=system

preflight() {
    case "$1" in
        onnxruntime)
            [[ -f "$ONNXRUNTIME_SDK/include/onnxruntime_cxx_api.h" ]] || {
                echo "error: ONNX Runtime GPU SDK not found; run scripts/fetch_onnxruntime.sh" >&2
                return 1
            }
            has_library "$ONNXRUNTIME_SDK" 'libonnxruntime_providers_cuda.so*' || {
                echo "error: $ONNXRUNTIME_SDK is not an ONNX Runtime GPU package" >&2
                return 1
            }
            ;;
        libtorch)
            [[ -f "$LIBTORCH_SDK/share/cmake/Torch/TorchConfig.cmake" ]] || {
                echo "error: CUDA libtorch SDK not found; run scripts/fetch_libtorch.sh" >&2
                return 1
            }
            has_library "$LIBTORCH_SDK" 'libtorch_cuda.so*' || {
                echo "error: $LIBTORCH_SDK is a CPU-only libtorch package" >&2
                return 1
            }
            ;;
        tensorrt)
            "$ROOT/scripts/fetch_tensorrt.sh" --print-dir >/dev/null
            ;;
    esac
}

# Validate every requested SDK before producing any partial release.
for backend in "${BACKENDS[@]}"; do
    preflight "$backend"
done

VERSION="$(sed -n 's/^project(mlvc_cpp VERSION \([0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt")"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
DRIVER_CUDA_CAPABILITY="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
CUDA_TOOLKIT_VERSION="$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | head -n 1)"
[[ "$CUDA_TOOLKIT_VERSION" == "$MLVC_CUDA_VERSION" ]] || {
    echo "error: CUDA $MLVC_CUDA_VERSION is required, found $CUDA_TOOLKIT_VERSION" >&2
    exit 1
}
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | LC_ALL=C sort -u | paste -sd ',' -)"

mkdir -p "$BUILD_ROOT" "$OUTPUT_DIR"

audit_linkage() {
    local target="$1"
    local library_path="${2:-}"
    local report
    if ! report="$(LD_LIBRARY_PATH="$library_path" ldd "$target" 2>&1)"; then
        echo "error: ldd failed for $target" >&2
        echo "$report" >&2
        return 1
    fi
    if [[ "$report" == *"not found"* ]]; then
        echo "error: unresolved dependency in $target" >&2
        echo "$report" >&2
        return 1
    fi
}

package_backend() {
    local backend="$1"
    local build_dir="$BUILD_ROOT/$backend"
    local product="mlvc_cpp-${VERSION}-${backend}-nvidia-${OS}-${ARCH}"
    local stage_root
    local prefix
    local sdk_root
    local sdk_label
    local -a configure_args

    case "$backend" in
        onnxruntime)
            sdk_root="$ONNXRUNTIME_SDK"
            sdk_label="onnxruntime-$MLVC_ONNXRUNTIME_VERSION-cuda13"
            ;;
        libtorch)
            sdk_root="$LIBTORCH_SDK"
            sdk_label="libtorch-$MLVC_LIBTORCH_VERSION-$MLVC_LIBTORCH_CUDA_VARIANT"
            ;;
        tensorrt)
            sdk_root="$TENSORRT_SDK"
            sdk_label="tensorrt-$MLVC_TENSORRT_VERSION"
            ;;
    esac

    configure_args=(
        -S "$ROOT"
        -B "$build_dir"
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DMLVC_BACKEND="$backend"
        -DMLVC_ENABLE_IPO=ON
    )
    case "$backend" in
        onnxruntime) configure_args+=("-DONNXRUNTIME_ROOT=$sdk_root") ;;
        libtorch) configure_args+=("-DLIBTORCH_ROOT=$sdk_root") ;;
        tensorrt) ;;
    esac

    echo "==> configuring $backend"
    cmake "${configure_args[@]}"
    echo "==> building $backend"
    cmake --build "$build_dir" --config "$BUILD_TYPE" --parallel "$JOBS"

    stage_root="$(mktemp -d "$OUTPUT_DIR/.stage-${backend}.XXXXXX")"
    STAGE_TO_CLEAN="$stage_root"
    prefix="$stage_root/$product"
    cmake --install "$build_dir" --config "$BUILD_TYPE" --prefix "$prefix"

    local binary="$prefix/bin/mlvc_demo"
    [[ -x "$binary" ]] || { echo "error: packaged binary is missing: $binary" >&2; return 1; }
    local benchmark_binary="$prefix/bin/mlvc_backend_bench"
    [[ -x "$benchmark_binary" ]] || {
        echo "error: packaged benchmark binary is missing: $benchmark_binary" >&2
        return 1
    }
    local backend_list
    backend_list="$(env -u LD_LIBRARY_PATH "$binary" --backend-name)"
    [[ "$backend_list" == "$backend" ]] || {
        echo "error: backend isolation check failed: $backend_list" >&2
        return 1
    }
    local benchmark_backend
    benchmark_backend="$(env -u LD_LIBRARY_PATH "$benchmark_binary" --backend-name)"
    [[ "$benchmark_backend" == "$backend" ]] || {
        echo "error: benchmark backend isolation check failed: $benchmark_backend" >&2
        return 1
    }
    audit_linkage "$binary"
    audit_linkage "$benchmark_binary"
    local probe_pattern
    case "$backend" in
        onnxruntime) probe_pattern='libonnxruntime_providers_cuda.so*' ;;
        libtorch) probe_pattern='libtorch_cuda.so*' ;;
        tensorrt) probe_pattern='libnvinfer.so*' ;;
    esac
    local probe_library
    probe_library="$(find "$prefix/lib" -maxdepth 1 -type f -name "$probe_pattern" \
        -print -quit)"
    [[ -n "$probe_library" ]] || {
        echo "error: packaged GPU runtime library is missing: $probe_pattern" >&2
        return 1
    }
    audit_linkage "$probe_library" "$prefix/lib"

    {
        printf 'name=%s\n' "$product"
        printf 'version=%s\n' "$VERSION"
        printf 'backend=%s\n' "$backend"
        printf 'gpu_only=true\n'
        printf 'nvidia_driver=%s\n' "$DRIVER_VERSION"
        printf 'driver_cuda_capability=%s\n' "${DRIVER_CUDA_CAPABILITY:-unknown}"
        printf 'cuda_toolkit=%s\n' "${CUDA_TOOLKIT_VERSION:-not-detected}"
        printf 'build_gpu=%s\n' "$GPU_NAMES"
        printf 'sdk=%s\n' "$sdk_label"
    } > "$prefix/BUILD-MANIFEST.txt"
    (
        cd "$prefix"
        find bin lib share -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
    ) > "$prefix/SHA256SUMS"

    if [[ "$NO_TAR" -eq 1 ]]; then
        if [[ -e "$OUTPUT_DIR/$product" ]]; then
            echo "error: output already exists: $OUTPUT_DIR/$product" >&2
            return 1
        fi
        mv "$prefix" "$OUTPUT_DIR/$product"
        rmdir "$stage_root"
        STAGE_TO_CLEAN=""
        echo "package tree: $OUTPUT_DIR/$product"
    else
        local tarball="$OUTPUT_DIR/${product}.tar.gz"
        tar -C "$stage_root" -czf "$tarball" "$product"
        cmake -E remove_directory "$stage_root"
        STAGE_TO_CLEAN=""
        echo "package: $tarball"
    fi
}

STAGE_TO_CLEAN=""
cleanup() {
    if [[ -n "$STAGE_TO_CLEAN" && -d "$STAGE_TO_CLEAN" ]]; then
        cmake -E remove_directory "$STAGE_TO_CLEAN"
    fi
}
trap cleanup EXIT

for backend in "${BACKENDS[@]}"; do
    package_backend "$backend"
done
