# Build, Packaging & CLI Usage

This guide covers prerequisites, CMake configuration across desktop and mobile targets, release packaging, and `mlvc_demo` command-line operation.

---

## 1. Prerequisites

### Core Requirements (All Platforms / `cpu` & `none` Backends)

- CMake `>= 3.23`
- C++20 and C11 compiler (GCC, Clang, AppleClang, or MSVC)
- Git with initialized pinned submodules (`third_party/mlvc`, `third_party/cutlass`, `third_party/protobuf` v33.5, `third_party/abseil` 20250512.1):
  ```bash
  ./tools/scripts/bootstrap.sh
  # Or directly on non-Linux hosts:
  git submodule update --init --recursive --depth 1
  ```

### Additional Requirements for `driver-cubin` (Linux `x86_64` NVIDIA GPU)

- **Build host**: CUDA Toolkit **`>= 12.0`** (CI/packaging pins `13.3` in [`tools/scripts/dependencies.env`](../tools/scripts/dependencies.env); verified on CUDA 12.4 and 13.3). On Ubuntu 26.04, run `./tools/scripts/bootstrap.sh --with-cuda`.
- **Model embedding (`MLVC_EMBED_MODELS=ON`)**: Python 3 interpreter with `protobuf==6.33.5` (from `build/model-venv` or `tools/python/requirements-proto.txt`) and canonical bundles under `models/canonical/`.
- **Deployment host**: NVIDIA GPU (`sm_75` Turing or newer; PTX fallback for future architectures via `compute_89`) and a compatible NVIDIA display driver (`libcuda.so.1`). CUDA Toolkit is **not** required on deployment hosts.

---

## 2. CMake Options & Presets

### Key CMake Cache Variables

| Variable                  | Default                                        | Description                                                                                                                          |
| ------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `MLVC_BACKEND`            | `AUTO`                                         | Inference backend(s) to compile: `AUTO` (`driver_cubin` on Linux `x86_64`, `cpu` elsewhere), `cpu`, `driver_cubin`, `all`, or `none` |
| `MLVC_EMBED_MODELS`       | `ON` for `driver_cubin`/`all`, `OFF` otherwise | Embed canonical bundles into `mlvc_core` via ELF `.incbin`                                                                           |
| `MLVC_MODEL_ROOT`         | `${CMAKE_SOURCE_DIR}/models/canonical`         | Directory searched for canonical model bundles when embedding or testing                                                             |
| `MLVC_TEST_ASSETS_DIR`    | `${CMAKE_SOURCE_DIR}/models/fixtures`          | Directory containing official Python reference fixtures                                                                              |
| `MLVC_PROTOC_EXECUTABLE`  | *(auto)*                                       | Host `protoc` binary when cross-compiling (required if `CMAKE_CROSSCOMPILING=TRUE` and not in `PATH`)                                |
| `MLVC_BUILD_SHARED`       | `ON` (follows `BUILD_SHARED_LIBS`)             | Build `mlvc_codec` as a shared library (`ON`) or static library (`OFF`)                                                              |
| `MLVC_ENABLE_IPO`         | `ON`                                           | Enable LTO/IPO in Release builds                                                                                                     |
| `MLVC_WARNINGS_AS_ERRORS` | `OFF`                                          | Treat compiler warnings as errors (`-Werror` / `/WX`)                                                                                |
| `MLVC_ENABLE_SANITIZERS`  | `OFF`                                          | Enable ASan + UBSan on GCC/Clang (also instruments `libprotobuf-lite` & `mlvc_model_proto` with ASan)                                |
| `MLVC_ENABLE_TSAN`        | `OFF`                                          | Enable ThreadSanitizer on GCC/Clang (mutually exclusive with `MLVC_ENABLE_SANITIZERS`)                                               |
| `MLVC_ENABLE_FUZZERS`     | `OFF`                                          | Build libFuzzer harnesses under `tests/fuzz/` (requires Clang)                                                                       |
| `BUILD_TESTING`           | `ON`                                           | Register CTest unit and conformance targets                                                                                          |

### Built-In Presets (`CMakePresets.json`)

| Preset                  | Purpose                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `driver-cubin-release`  | Linux `x86_64` Release build with `IPO=ON`, `BUILD_TESTING=ON`, embedded models                                                       |
| `driver-cubin-sanitize` | Debug host build (`MLVC_EMBED_MODELS=OFF`, `MLVC_ENABLE_SANITIZERS=ON`, `MLVC_WARNINGS_AS_ERRORS=ON`) running all non-GPU unit tests  |
| `cpu-release`           | Cross-platform Release build for the `cpu` backend (`MLVC_WARNINGS_AS_ERRORS=ON`, Ninja)                                              |
| `cpu-sanitize`          | Debug `cpu` backend build with ASan + UBSan across all host unit tests                                                                |
| `cpu-tsan`              | RelWithDebInfo `cpu` backend build with ThreadSanitizer across all host unit tests                                                    |

---

## 3. Platform Build Recipes

### Linux `x86_64` — NVIDIA `driver-cubin` (or Both Backends)

```bash
# Default preset (driver-cubin + embedded models)
cmake --preset driver-cubin-release
cmake --build --preset driver-cubin-release -j
ctest --preset driver-cubin-release

# Compile both driver-cubin and cpu backends into one binary
cmake -S . -B build/all-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLVC_BACKEND=all \
  -DMLVC_EMBED_MODELS=ON
cmake --build build/all-release -j
```

### Linux / macOS / Windows — Portable `cpu` Backend

```bash
# Linux / macOS
cmake -S . -B build/cpu-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLVC_BACKEND=cpu
cmake --build build/cpu-release -j
ctest --test-dir build/cpu-release --output-on-failure

# Windows (PowerShell, Visual Studio / Ninja)
cmake -S . -B build/cpu-win -DCMAKE_BUILD_TYPE=Release -DMLVC_BACKEND=cpu
cmake --build build/cpu-win --config Release --parallel
ctest --test-dir build/cpu-win -C Release --output-on-failure
```

### Cross-Compiling for Android & OpenHarmony (`arm64-v8a`)

Cross-compiling uses the host `protoc` (built first on the host or supplied via `-DMLVC_PROTOC_EXECUTABLE`) so target builds skip compiling `protoc` and `upb`.

**Step 1 — Build host `protoc` once** (or reuse from a host build directory):

```bash
cmake -S . -B build/host-tools -DCMAKE_BUILD_TYPE=Release -DMLVC_BACKEND=none -DBUILD_TESTING=OFF
cmake --build build/host-tools --target protoc -j
HOST_PROTOC="$PWD/build/host-tools/_deps/protobuf/protoc"
```

**Step 2a — Android NDK (`arm64-v8a`)**:

```bash
cmake -S . -B build/android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLVC_BACKEND=cpu \
  -DMLVC_EMBED_MODELS=OFF \
  -DMLVC_PROTOC_EXECUTABLE="$HOST_PROTOC" \
  -DBUILD_TESTING=OFF
cmake --build build/android-arm64 -j
```

**Step 2b — OpenHarmony SDK (`arm64-v8a`)**:

```bash
cmake -S . -B build/ohos-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$OHOS_SDK_NATIVE/build/cmake/ohos.toolchain.cmake" \
  -DOHOS_ARCH=arm64-v8a \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLVC_BACKEND=cpu \
  -DMLVC_EMBED_MODELS=OFF \
  -DMLVC_PROTOC_EXECUTABLE="$HOST_PROTOC" \
  -DBUILD_TESTING=OFF
cmake --build build/ohos-arm64 -j
```

Both toolchains automatically enable `MLVC_CPU_ENABLE_ARM64=1` and compile the ARMv8.0 NEON + ARMv8.2 `+fp16`/`+fp16fml` assembly microkernels (with BTI/PAC `.note.gnu.property`).

---

## 4. Release Packaging (`driver-cubin` Linux `x86_64`)

[`tools/scripts/package.sh`](../tools/scripts/package.sh) builds, runs CTest, audits dynamic linkage with `ldd`, checks embedded model discovery and `mlvc_driver_probe`, writes `BUILD-MANIFEST.txt` and `SHA256SUMS`, and outputs a tarball:

```bash
# Deployable release (requires CUDA >= 12.0, NVIDIA GPU, canonical models, protobuf==6.33.5)
./tools/scripts/package.sh --jobs 8
# -> packages/mlvc_cpp-<version>-driver-cubin-nvidia-linux-x86_64.tar.gz

# Driverless compile-only CI package (not deployable; produces *-ci.tar.gz)
./tools/scripts/package.sh --jobs 2 --skip-gpu-tests --skip-models
```

### Installing & Auditing a Release Archive

```bash
tar -xzf mlvc_cpp-<version>-driver-cubin-nvidia-linux-x86_64.tar.gz -C /opt/mlvc
export MLVC_PREFIX=/opt/mlvc/mlvc_cpp-<version>-driver-cubin-nvidia-linux-x86_64

(cd "$MLVC_PREFIX" && sha256sum -c SHA256SUMS)
cat "$MLVC_PREFIX/BUILD-MANIFEST.txt"
ldd "$MLVC_PREFIX/lib/libmlvc_codec.so"          # must show only system C/C++ libs (libcuda.so.1 is dlopen'd at runtime)
"$MLVC_PREFIX/bin/mlvc_driver_probe" --iterations 100
"$MLVC_PREFIX/bin/mlvc_demo" --backend-name
"$MLVC_PREFIX/bin/mlvc_demo" --list-model-profiles
```

Keep the relative `bin/`, `lib/`, `include/`, and `share/` layout intact (`$ORIGIN` / `$ORIGIN/../lib` RPATHs rely on it). Verify target host glibc, `libstdc++`, and NVIDIA driver compatibility against `BUILD-MANIFEST.txt`.

---

## 5. CLI Reference (`mlvc_demo`)

### Discovery Commands

```bash
mlvc_demo --backend-name         # default compiled backend ("driver-cubin", "cpu", or "none")
mlvc_demo --list-backends        # all registered backends in this build
mlvc_demo --list-model-profiles  # ELF-embedded model profiles (empty when MLVC_EMBED_MODELS=OFF)
```

### Encode & Decode Syntax

```text
mlvc_demo encode --input IN.yuv|- --output OUT.mlvc|- --width W --height H
     [--model-dir DIR | --model-profile NAME] [--frames N]
     [--q-index N] [--backend NAME] [--device-id N] [--threads N]
     [--encode-device-id N] [--debug-dir DIR]

mlvc_demo decode --input IN.mlvc|- --output OUT.yuv|-
     [--width W --height H] [--model-dir DIR | --model-profile NAME]
     [--frames N] [--backend NAME] [--device-id N] [--threads N]
     [--decode-device-id N] [--debug-dir DIR]
```

### Key Options

- `--input` / `--output`: File path, `-` for stdin/stdout, or a POSIX named FIFO path. When `--output -` is used, human-readable throughput logs automatically route to `stderr` so binary stdout is never corrupted.
- `--width` / `--height`: Positive even integers. Required on `encode`; optional on `decode` (auto-detected from bitstream `SPS` when omitted, or validated against `SPS` when supplied).
- `--frames N`: Number of frames to process; `0` processes until EOF.
- `--q-index N`: Quantization index in `[0, 51]` (default `21`).
- `--model-profile NAME` vs. `--model-dir DIR`: Mutually exclusive. Default uses embedded `mlvc-psnr-v1`. Select the lightweight model with `--model-profile mlvc-s-psnr-v1`, or override with a canonical bundle directory via `--model-dir`.
- `--backend NAME`: Select a specific registered backend (`driver-cubin` or `cpu`) in multi-backend (`MLVC_BACKEND=all`) builds.
- `--device-id N` / `--encode-device-id N` / `--decode-device-id N`: GPU index for `driver-cubin`.
- `--threads N`: Worker thread count for `cpu` (`0` = auto-detect performance/prime cores).
- `--debug-dir DIR`: Dumps named model input/output tensors per frame (disables device-resident state binding; for diagnosis only).

### Common Workflows

**Basic file encode & decode (MLVC-S on GPU 0)**:

```bash
mlvc_demo encode --input input.yuv --output output.mlvc \
  --width 640 --height 360 --frames 60 --q-index 21 \
  --model-profile mlvc-s-psnr-v1 --device-id 0

mlvc_demo decode --input output.mlvc --output reconstructed.yuv \
  --frames 60 --model-profile mlvc-s-psnr-v1 --device-id 0
```

**CPU backend with explicit bundle and thread count**:

```bash
mlvc_demo encode --backend cpu --threads 4 \
  --model-dir models/canonical/mlvc-s-psnr-v1/640x368 \
  --input input.yuv --output output.mlvc \
  --width 640 --height 360 --frames 30 --q-index 21
```

**Streaming pipe across two GPUs**:

```bash
cat input.yuv \
  | mlvc_demo encode --input - --output - \
      --width 640 --height 360 --frames 0 --q-index 21 \
      --encode-device-id 0 2>encode.log \
  | mlvc_demo decode --input - --output - \
      --frames 0 --decode-device-id 1 2>decode.log \
  > reconstructed.yuv
```

**Live camera capture via FFmpeg (persistent single encoder process)**:

```bash
ffmpeg -f v4l2 -i /dev/video0 -vf scale=640:360 -pix_fmt yuv420p -f rawvideo - \
  | mlvc_demo encode --input - --output camera.mlvc \
      --width 640 --height 360 --frames 0 --q-index 21
```

> **GOP continuity rule**: Encoder and decoder maintain recurrent reference state across P-frames (reset every 64 frames by default). Always begin streaming from an IDR frame (`SPS`+`PPS`+`IDR_N_LP`) and do not drop arbitrary P-frames without resetting to the next IDR.
