#!/usr/bin/env bash
# Build isolated NVIDIA GPU releases for all supported backends.
# Use --skip-gpu-tests --skip-models only for compile-only CI smoke packages;
# those packages are not deployable releases.
#
# Usage: ./tools/package.sh [--backend all|onnxruntime|libtorch|tensorrt|driver-cubin]
#          [--build-root DIR] [--output-dir DIR] [--model-root DIR]
#          [--jobs N] [--no-tar] [--skip-gpu-tests] [--skip-models]
set -euo pipefail

BACKEND="all"
BUILD_ROOT="build/release"
OUTPUT_DIR="packages"
MODEL_ROOT="models/canonical"
BUILD_TYPE="Release"
JOBS=""
NO_TAR=0
SKIP_GPU_TESTS=0
SKIP_MODELS=0

usage() {
    sed -n '2,6p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="${2:?--backend needs a value}"; shift 2 ;;
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

case "$BACKEND" in
    all) BACKENDS=(onnxruntime libtorch tensorrt driver-cubin) ;;
    onnxruntime|libtorch|tensorrt|driver-cubin) BACKENDS=("$BACKEND") ;;
    *) echo "error: unsupported backend: $BACKEND" >&2; exit 2 ;;
esac

if [[ "$SKIP_MODELS" -eq 1 ]]; then
    [[ "$SKIP_GPU_TESTS" -eq 1 ]] || {
        echo "error: --skip-models requires --skip-gpu-tests" >&2
        exit 2
    }
    [[ "${#BACKENDS[@]}" -eq 1 && "${BACKENDS[0]}" == "driver-cubin" ]] || {
        echo "error: --skip-models is only supported for driver-cubin" >&2
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

has_library() {
    local root="$1"
    local name="$2"
    [[ -n "$root" ]] || return 1
    compgen -G "$root/lib/$name" >/dev/null ||
        compgen -G "$root/lib64/$name" >/dev/null
}

resolve_cudart() {
    # The GPU releases link libcudart (CUDA runtime), which the NVIDIA driver
    # does not provide.  Resolve it from the active CUDA 13 toolkit so it can
    # be bundled next to the backend libraries.
    local nvcc_real toolkit_root candidate path
    nvcc_real="$(readlink -f "$(command -v nvcc)")"
    toolkit_root="$(dirname "$(dirname "$nvcc_real")")"
    for candidate in \
        "$toolkit_root/targets/x86_64-linux/lib/libcudart.so.13" \
        "$toolkit_root/lib64/libcudart.so.13"; do
        [[ -e "$candidate" ]] || continue
        printf '%s\n' "$candidate"
        return 0
    done
    path="$(ldconfig -p 2>/dev/null |
        awk '/libcudart\.so\.13 /{print $NF; exit}')"
    if [[ -n "$path" && -e "$path" ]]; then
        printf '%s\n' "$path"
        return 0
    fi
    return 1
}

resolve_cuda_eula() {
    local nvcc_real toolkit_root
    nvcc_real="$(readlink -f "$(command -v nvcc)")"
    toolkit_root="$(dirname "$(dirname "$nvcc_real")")"
    [[ -f "$toolkit_root/EULA.txt" ]] || return 1
    printf '%s\n' "$toolkit_root/EULA.txt"
}

ONNXRUNTIME_SDK="$ROOT/third_party/onnxruntime/onnxruntime-linux-x64-gpu-$MLVC_ONNXRUNTIME_VERSION"
LIBTORCH_SDK="$ROOT/third_party/libtorch/${MLVC_LIBTORCH_VERSION}-${MLVC_LIBTORCH_CUDA_VARIANT}/libtorch"
TENSORRT_SDK=system

find_tensorrt_header() {
    local multiarch
    multiarch="$(gcc -print-multiarch)"
    [[ -f "/usr/include/$multiarch/NvInfer.h" ]] && {
        printf '%s\n' "/usr/include/$multiarch"
        return
    }
    [[ -f /usr/include/NvInfer.h ]] && printf '%s\n' /usr/include
}

tensorrt_is_complete() {
    local include_dir linker_cache major minor
    include_dir="$(find_tensorrt_header)" || return 1
    linker_cache="$(ldconfig -p)"
    grep -q 'libnvinfer\.so\.11' <<< "$linker_cache" || return 1
    grep -q 'libnvonnxparser\.so\.11' <<< "$linker_cache" || return 1
    major="$(sed -n 's/^#define TRT_MAJOR_ENTERPRISE \([0-9][0-9]*\).*/\1/p' \
        "$include_dir/NvInferVersion.h")"
    minor="$(sed -n 's/^#define TRT_MINOR_ENTERPRISE \([0-9][0-9]*\).*/\1/p' \
        "$include_dir/NvInferVersion.h")"
    [[ "$major.$minor" == 11.2 ]]
}

preflight() {
    case "$1" in
        onnxruntime)
            [[ -f "$ONNXRUNTIME_SDK/include/onnxruntime_cxx_api.h" ]] || {
                echo "error: ONNX Runtime GPU SDK not found; run tools/bootstrap.sh --backend onnxruntime" >&2
                return 1
            }
            has_library "$ONNXRUNTIME_SDK" 'libonnxruntime_providers_cuda.so*' || {
                echo "error: $ONNXRUNTIME_SDK is not an ONNX Runtime GPU package" >&2
                return 1
            }
            ;;
        libtorch)
            [[ -f "$LIBTORCH_SDK/share/cmake/Torch/TorchConfig.cmake" ]] || {
                echo "error: CUDA libtorch SDK not found; run tools/bootstrap.sh --backend libtorch" >&2
                return 1
            }
            has_library "$LIBTORCH_SDK" 'libtorch_cuda.so*' || {
                echo "error: $LIBTORCH_SDK is a CPU-only libtorch package" >&2
                return 1
            }
            ;;
        tensorrt)
            tensorrt_is_complete || {
                echo "error: TensorRT SDK not found; run tools/bootstrap.sh --backend tensorrt" >&2
                return 1
            }
            ;;
        driver-cubin) ;;
    esac
}

# Validate every requested SDK before producing any partial release.
for backend in "${BACKENDS[@]}"; do
    preflight "$backend"
done

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
        local unresolved
        unresolved="$(grep 'not found' <<<"$report" |
            grep -vE '^[[:space:]]*libcuda\.so\.1 => not found$' || true)"
        if [[ "$SKIP_GPU_TESTS" -eq 0 || -n "$unresolved" ]]; then
            echo "error: unresolved dependency in $target" >&2
            echo "$report" >&2
            return 1
        fi
        echo "note: allowing missing NVIDIA driver library while GPU tests are skipped: $target"
    fi
}

audit_bundled_cudart() {
    local target="$1"
    local package_lib="$2"
    local report actual expected
    report="$(LD_LIBRARY_PATH="$package_lib" ldd "$target")"
    actual="$(awk '$1 == "libcudart.so.13" && $2 == "=>" {print $3; exit}' \
        <<<"$report")"
    expected="$(readlink -f "$package_lib/libcudart.so.13")"
    if [[ -z "$actual" || "$(readlink -f "$actual")" != "$expected" ]]; then
        echo "error: $target does not resolve libcudart.so.13 from $package_lib" >&2
        echo "$report" >&2
        return 1
    fi
}

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

package_backend() {
    local backend="$1"
    local build_dir="$BUILD_ROOT/$backend"
    local product="mlvc_cpp-${VERSION}-${backend}-nvidia-${OS}-${ARCH}"
    local stage_root
    local prefix
    local sdk_root
    local sdk_label
    local bundled_models=""
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
        -DMLVC_SELECTED_BACKEND="$backend"
        -DMLVC_MODEL_ROOT="$MODEL_ROOT"
        -DMLVC_TEST_ASSETS_DIR="$ROOT/models/fixtures"
        -DMLVC_ENABLE_IPO=ON
        -DCMAKE_INSTALL_BINDIR=bin
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_INSTALL_INCLUDEDIR=include
        -DBUILD_TESTING=ON
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
    if [[ "$SKIP_GPU_TESTS" -eq 1 ]]; then
        echo "==> running basic tests $backend"
        run_basic_tests --test-dir "$build_dir" --output-on-failure
    fi

    stage_root="$(mktemp -d "$OUTPUT_DIR/.stage-${backend}.XXXXXX")"
    STAGE_TO_CLEAN="$stage_root"
    prefix="$stage_root/$product"
    cmake --install "$build_dir" --config "$BUILD_TYPE" --prefix "$prefix"

    echo "==> bundling CUDA runtime"
    local cudart_so
    if ! cudart_so="$(resolve_cudart)"; then
        echo "error: CUDA runtime libcudart.so.13 not found on the build host" >&2
        return 1
    fi
    cp -L "$cudart_so" "$prefix/lib/libcudart.so.13"
    local cuda_eula
    if ! cuda_eula="$(resolve_cuda_eula)"; then
        echo "error: CUDA EULA.txt not found beside the active toolkit" >&2
        return 1
    fi
    mkdir -p "$prefix/share/licenses/cuda"
    cp "$cuda_eula" "$prefix/share/licenses/cuda/EULA.txt"

    echo "==> bundling $backend models"
    python3 "$ROOT/tools/model_package.py" package \
        --backend "$backend" \
        --model-root "$MODEL_ROOT" \
        --output-root "$prefix/share/mlvc/models"
    bundled_models="$(find "$prefix/share/mlvc/models" -mindepth 2 -maxdepth 2 \
        -type d -printf '%P\n' | LC_ALL=C sort | paste -sd ',' -)"

    local binary="$prefix/bin/mlvc_demo"
    [[ -x "$binary" ]] || { echo "error: packaged binary is missing: $binary" >&2; return 1; }
    local codec_library
    codec_library="$(find "$prefix/lib" -maxdepth 1 -type f -name 'libmlvc_codec.so.*' -print -quit)"
    [[ -n "$codec_library" ]] || {
        echo "error: packaged codec library is missing: libmlvc_codec.so.*" >&2
        return 1
    }
    local benchmark_binary="$prefix/bin/mlvc_backend_bench"
    [[ -x "$benchmark_binary" ]] || {
        echo "error: packaged benchmark binary is missing: $benchmark_binary" >&2
        return 1
    }
    if [[ "$SKIP_GPU_TESTS" -eq 0 ]]; then
        local backend_list
        backend_list="$(env -u LD_LIBRARY_PATH "$binary" --backend-name)"
        [[ "$backend_list" == "$backend" ]] || {
            echo "error: backend isolation check failed: $backend_list" >&2
            return 1
        }
    else
        echo "note: skipping packaged runtime smoke checks for $backend"
    fi
    if [[ "$SKIP_GPU_TESTS" -eq 0 ]]; then
        local benchmark_backend
        benchmark_backend="$(env -u LD_LIBRARY_PATH "$benchmark_binary" --backend-name)"
        [[ "$benchmark_backend" == "$backend" ]] || {
            echo "error: benchmark backend isolation check failed: $benchmark_backend" >&2
            return 1
        }
    fi
    if [[ "$SKIP_GPU_TESTS" -eq 0 ]]; then
        local packaged_profiles
        packaged_profiles="$("$binary" --list-model-profiles |
            LC_ALL=C sort | paste -sd ',' -)"
        [[ "$packaged_profiles" == "mlvc-psnr-v1,mlvc-s-psnr-v1" ]] || {
            echo "error: packaged model discovery failed: $packaged_profiles" >&2
            return 1
        }
    fi
    audit_linkage "$binary"
    audit_linkage "$benchmark_binary"
    audit_linkage "$codec_library" "$prefix/lib"
    if [[ "$backend" == "tensorrt" ]]; then
        audit_bundled_cudart "$codec_library" "$prefix/lib"
    fi
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
        printf 'codec_library=%s\n' "$(basename "$codec_library")"
        printf 'codec_api=c-abi,cxx\n'
        printf 'nvidia_driver=%s\n' "$DRIVER_VERSION"
        printf 'driver_cuda_capability=%s\n' "${DRIVER_CUDA_CAPABILITY:-unknown}"
        printf 'cuda_toolkit=%s\n' "${CUDA_TOOLKIT_VERSION:-not-detected}"
        printf 'build_gpu=%s\n' "$GPU_NAMES"
        printf 'sdk=%s\n' "$sdk_label"
        printf 'bundled_cuda_runtime=%s\n' "libcudart.so.13"
        printf 'cuda_runtime_license=%s\n' "share/licenses/cuda/EULA.txt"
        printf 'model_root=share/mlvc/models\n'
        printf 'bundled_models=%s\n' "$bundled_models"
        printf 'gpu_runtime_tests=%s\n' "$([[ "$SKIP_GPU_TESTS" -eq 0 ]] && echo true || echo false)"
    } > "$prefix/BUILD-MANIFEST.txt"
    finalize_package "$stage_root" "$prefix" "$product"
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
        -DMLVC_SELECTED_BACKEND=driver_cubin \
        -DMLVC_MODEL_ROOT="$MODEL_ROOT" \
        -DMLVC_TEST_ASSETS_DIR="$ROOT/models/fixtures" \
        -DMLVC_ENABLE_IPO=ON \
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
        printf 'codec_api=c-abi,cxx\n'
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

for backend in "${BACKENDS[@]}"; do
    if [[ "$backend" == driver-cubin ]]; then
        package_driver_cubin
    else
        package_backend "$backend"
    fi
done
