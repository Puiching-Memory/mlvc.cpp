#!/usr/bin/env bash
# Build the Driver+cubin NVIDIA GPU release.
# Use --skip-gpu-tests --skip-models only for compile-only CI smoke packages;
# those packages are not deployable releases.
#
# Usage: ./tools/package.sh [--build-root DIR] [--output-dir DIR] [--model-root DIR]
#          [--jobs N] [--no-tar] [--skip-gpu-tests] [--skip-models]
set -euo pipefail

BUILD_ROOT="build/release"
OUTPUT_DIR="packages"
MODEL_ROOT="models/canonical"
BUILD_TYPE="Release"
JOBS=""
NO_TAR=0
SKIP_GPU_TESTS=0
SKIP_MODELS=0

usage() {
    sed -n '2,7p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-root) BUILD_ROOT="${2:?--build-root needs a value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir needs a value}"; shift 2 ;;
        --model-root) MODEL_ROOT="${2:?--model-root needs a value}"; shift 2 ;;
        --jobs) JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        --no-tar) NO_TAR=1; shift ;;
        --skip-gpu-tests) SKIP_GPU_TESTS=1; shift ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$SKIP_MODELS" -eq 1 ]]; then
    [[ "$SKIP_GPU_TESTS" -eq 1 ]] || {
        echo "error: --skip-models requires --skip-gpu-tests" >&2
        exit 2
    }
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tools/dependencies.env
source "$ROOT/tools/dependencies.env"
[[ "$BUILD_ROOT" = /* ]] || BUILD_ROOT="$ROOT/$BUILD_ROOT"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
[[ "$MODEL_ROOT" = /* ]] || MODEL_ROOT="$ROOT/$MODEL_ROOT"

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
for tool in ldd python3 sha256sum tar; do
    command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 1; }
done
if [[ "$SKIP_GPU_TESTS" -eq 0 ]]; then
    command -v nvidia-smi >/dev/null || { echo "error: nvidia-smi is required" >&2; exit 1; }
    nvidia-smi -L >/dev/null || {
        echo "error: no usable NVIDIA GPU was detected" >&2
        exit 1
    }
fi
[[ -f "$ROOT/third_party/mlvc/packages/msrtc_rans/CMakeLists.txt" &&
   -f "$ROOT/third_party/nlohmann_json/CMakeLists.txt" ]] || {
    echo "error: source submodules are missing; run tools/bootstrap.sh" >&2
    exit 1
}

VERSION="$(sed -n 's/^project(mlvc_cpp VERSION \([0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt")"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
DRIVER_VERSION="not-detected"
DRIVER_CUDA_CAPABILITY="unknown"
CUDA_TOOLKIT_VERSION="$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | head -n 1)"
[[ "$CUDA_TOOLKIT_VERSION" == "$MLVC_CUDA_VERSION" ]] || {
    echo "error: CUDA $MLVC_CUDA_VERSION is required, found $CUDA_TOOLKIT_VERSION" >&2
    exit 1
}
GPU_NAMES="not-detected"
if [[ "$SKIP_GPU_TESTS" -eq 0 ]]; then
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
    DRIVER_CUDA_CAPABILITY="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
    GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | LC_ALL=C sort -u | paste -sd ',' -)"
fi

CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
CUDA_STUB_DIR="$CUDA_ROOT/targets/x86_64-linux/lib/stubs"

run_on_host_or_cuda_stub() {
    if [[ "$SKIP_GPU_TESTS" -eq 1 && -f "$CUDA_STUB_DIR/libcuda.so" ]]; then
        LD_LIBRARY_PATH="$CUDA_STUB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$@"
    else
        "$@"
    fi
}

run_basic_tests() {
    if [[ "$SKIP_GPU_TESTS" -eq 1 ]]; then
        run_on_host_or_cuda_stub ctest "$@" --exclude-regex '^mlvc_driver_cubin_probe$' \
            --label-exclude gpu
    else
        ctest "$@"
    fi
}

mkdir -p "$BUILD_ROOT" "$OUTPUT_DIR"

finalize_package() {
    local stage_root="$1"
    local prefix="$2"
    local product="$3"
    (
        cd "$prefix"
        find bin include lib share -type f -print0 | LC_ALL=C sort -z |
            xargs -0 sha256sum
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

audit_driver_linkage() {
    local target="$1" report
    report="$(ldd "$target")"
    if [[ "$report" == *"not found"* ]]; then
        local unresolved
        unresolved="$(grep 'not found' <<<"$report" |
            grep -vE '^[[:space:]]*libcuda\.so\.1 => not found$' || true)"
        if [[ "$SKIP_GPU_TESTS" -eq 0 || -n "$unresolved" ]]; then
            echo "$report" >&2
            return 1
        fi
        echo "note: allowing missing NVIDIA driver library while GPU tests are skipped: $target"
    fi
    if [[ "$report" != *"libcuda.so.1"* ]]; then
        echo "error: $target does not require the NVIDIA driver" >&2
        return 1
    fi
    if grep -Eqi \
        'libcudart|libcudnn|libcublas|libnvinfer|libonnxruntime|libtorch' \
        <<< "$report"; then
        echo "error: $target links a forbidden CUDA Toolkit or inference runtime" >&2
        echo "$report" >&2
        return 1
    fi
}

package_driver_cubin() {
    local backend=driver-cubin
    local build_dir="$BUILD_ROOT/$backend"
    local product="mlvc_cpp-${VERSION}-${backend}-nvidia-${OS}-${ARCH}"
    if [[ "$SKIP_MODELS" -eq 1 ]]; then
        product+="-ci"
    fi

    echo "==> configuring $backend"
    cmake -S "$ROOT" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DMLVC_MODEL_ROOT="$MODEL_ROOT" \
        -DMLVC_TEST_ASSETS_DIR="$ROOT/models/fixtures" \
        -DMLVC_ENABLE_IPO=ON \
        -DMLVC_WARNINGS_AS_ERRORS=ON \
        -DMLVC_EMBED_MODELS="$([[ "$SKIP_MODELS" -eq 0 ]] && echo ON || echo OFF)" \
        -DCMAKE_INSTALL_BINDIR=bin \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_INSTALL_INCLUDEDIR=include \
        -DBUILD_TESTING=ON
    echo "==> building and testing $backend"
    cmake --build "$build_dir" --config "$BUILD_TYPE" --parallel "$JOBS"
    run_basic_tests --test-dir "$build_dir" --output-on-failure

    local stage_root prefix
    stage_root="$(mktemp -d "$OUTPUT_DIR/.stage-${backend}.XXXXXX")"
    STAGE_TO_CLEAN="$stage_root"
    prefix="$stage_root/$product"
    cmake --install "$build_dir" --config "$BUILD_TYPE" --prefix "$prefix"

    local demo="$prefix/bin/mlvc_demo"
    local benchmark="$prefix/bin/mlvc_backend_bench"
    local probe="$prefix/bin/mlvc_driver_probe"
    local codec_library
    codec_library="$(find "$prefix/lib" -maxdepth 1 -type f \
        -name 'libmlvc_codec.so.*' -print -quit)"
    [[ -n "$codec_library" ]] || {
        echo "error: packaged codec library is missing: libmlvc_codec.so.*" >&2
        return 1
    }
    local binary
    for binary in "$demo" "$benchmark" "$probe"; do
        [[ -x "$binary" ]] || {
            echo "error: packaged driver-cubin binary is missing: $binary" >&2
            return 1
        }
    done
    [[ -f "$codec_library" ]] || {
        echo "error: packaged driver-cubin library is missing: $codec_library" >&2
        return 1
    }
    for binary in "$demo" "$benchmark" "$probe" "$codec_library"; do
        audit_driver_linkage "$binary"
    done
    local bundled_models="none"
    if [[ "$SKIP_GPU_TESTS" -eq 0 ]]; then
        [[ "$("$demo" --backend-name)" == "$backend" ]] || {
            echo "error: packaged codec is not the $backend variant" >&2
            return 1
        }
        [[ "$("$benchmark" --backend-name)" == "$backend" ]] || {
            echo "error: packaged benchmark is not the $backend variant" >&2
            return 1
        }
        bundled_models="$("$demo" --list-model-profiles |
            LC_ALL=C sort | paste -sd ',' -)"
        [[ "$bundled_models" == "mlvc-psnr-v1,mlvc-s-psnr-v1" ]] || {
            echo "error: packaged embedded model discovery failed: $bundled_models" >&2
            return 1
        }
        [[ ! -e "$prefix/share/mlvc/models" ]] || {
            echo "error: driver-cubin models must be embedded" >&2
            return 1
        }
        "$probe" --iterations 100 >/dev/null
    else
        echo "note: skipping packaged runtime smoke checks for $backend"
    fi

    local fatbin="$build_dir/generated/driver_cubin/fatbin/mlvc_driver_kernels.fatbin"
    local embedded_models="$build_dir/generated/embedded-models/mlvc_driver_models.bin"
    [[ -f "$fatbin" ]] || {
        echo "error: generated driver fatbin is missing: $fatbin" >&2
        return 1
    }
    if [[ "$SKIP_MODELS" -eq 0 ]]; then
        [[ -f "$embedded_models" ]] || {
            echo "error: embedded model image is missing: $embedded_models" >&2
            return 1
        }
    fi
    {
        printf 'name=%s\n' "$product"
        printf 'version=%s\n' "$VERSION"
        printf 'backend=driver-cubin\n'
        if [[ "$SKIP_MODELS" -eq 0 ]]; then
            printf 'codec_pipeline=complete\n'
            printf 'aot_graphs=MLVCEncoder,MLVCDecoder\n'
        else
            printf 'codec_pipeline=compile-only\n'
            printf 'aot_graphs=omitted-ci\n'
        fi
        printf 'codec_library=%s\n' "$(basename "$codec_library")"
        printf 'codec_api=c-abi\n'
        printf 'codec_abi_version=2\n'
        printf 'aot_graph_schema=2\n'
        printf 'floating_point=fp16-only\n'
        printf 'runtime_gpu_dependency=libcuda.so.1\n'
        printf 'cuda_toolkit_runtime_dependency=false\n'
        printf 'build_driver=%s\n' "$DRIVER_VERSION"
        printf 'fatbin_sha256=%s\n' "$(sha256sum "$fatbin" | cut -d' ' -f1)"
        printf 'fatbin_targets=sm_75,sm_80,sm_86,sm_89,compute_89\n'
        if [[ "$SKIP_MODELS" -eq 0 ]]; then
            printf 'model_storage=embedded:lib/libmlvc_codec.so\n'
        else
            printf 'model_storage=omitted-ci\n'
        fi
        printf 'bundled_models=%s\n' "$bundled_models"
        printf 'gpu_runtime_tests=%s\n' "$([[ "$SKIP_GPU_TESTS" -eq 0 ]] && echo true || echo false)"
        printf 'models_embedded=%s\n' "$([[ "$SKIP_MODELS" -eq 0 ]] && echo true || echo false)"
        if [[ "$SKIP_MODELS" -eq 0 ]]; then
            printf 'embedded_models_bytes=%s\n' "$(stat -c '%s' "$embedded_models")"
            printf 'embedded_models_sha256=%s\n' \
                "$(sha256sum "$embedded_models" | cut -d' ' -f1)"
        fi
    } > "$prefix/BUILD-MANIFEST.txt"

    finalize_package "$stage_root" "$prefix" "$product"
}

STAGE_TO_CLEAN=""
cleanup() {
    if [[ -n "$STAGE_TO_CLEAN" && -d "$STAGE_TO_CLEAN" ]]; then
        cmake -E remove_directory "$STAGE_TO_CLEAN"
    fi
}
trap cleanup EXIT

package_driver_cubin
