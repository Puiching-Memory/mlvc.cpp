# mlvc.cpp

> English | [简体中文](README.zh-CN.md)

[![CI](https://github.com/Puiching-Memory/mlvc.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Puiching-Memory/mlvc.cpp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](CMakeLists.txt)
[![arXiv](https://img.shields.io/badge/Paper-MLVC%20(2606.28027)-b31b1b.svg)](https://arxiv.org/abs/2606.28027)
[![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows%20%7C%20Android%20%7C%20OHOS-2ea44f.svg)](#platform--backend-matrix)

**Standalone C++20 inference & codec engine for [Microsoft MLVC (`DMC-6.1sb`)](https://github.com/microsoft/mlvc) neural video coding — running at 1,100+ FPS on NVIDIA GPU or pure C++/ASM on mobile CPU, inside a 6.1 MiB library with zero ML-framework dependencies.**

No PyTorch. No ONNX Runtime. No TensorRT. No `libcudart` / `libcublas` / `libcudnn`. Just a self-contained C++20 library (`libmlvc_codec.so`) and streaming CLI (`mlvc_demo`) implementing the full `mlvc-nalu-v0` Annex-B bitstream container, recurrent GOP/DPB state, canonical PMF tables, and `msrtc_rans` entropy coding for both official models (**`mlvc-psnr-v1`** and **`mlvc-s-psnr-v1`**).

---

## Benchmarks & Footprint

Measured on **NVIDIA A30** (`sm_80`, 1440 MHz, 640×360 YUV420 padded to 640×368, `q_index = 21`, `nsys --cuda-graph-trace=node` uncontended frame windows):

| Model Profile        | Params | Direction | Kernels / Frame | GPU Busy / Frame | Inter-Frame Bubble |        Sustained GPU Pace |  Frame-0 AU Match  |     YUV vs Python Ref     |
| -------------------- | -----: | :-------: | --------------: | ---------------: | -----------------: | ------------------------: | :----------------: | :-----------------------: |
| **`mlvc-s-psnr-v1`** |   5.4M |  Encode   |             112 |      **0.89 ms** |            10.9 µs |  **1,106 fps** (904 µs/f) | Byte-exact (328 B) | max err ≤ 1, 63.6–65.4 dB |
| **`mlvc-s-psnr-v1`** |   5.4M |  Decode   |             106 |      **0.87 ms** |            11.2 µs |  **1,120 fps** (893 µs/f) |     Byte-exact     | max err ≤ 1, 63.6–65.4 dB |
| **`mlvc-psnr-v1`**   |  18.3M |  Encode   |             188 |      **1.91 ms** |            11.3 µs | **~521 fps** (~1.92 ms/f) | Byte-exact (137 B) | max err ≤ 1, 63.5–67.3 dB |
| **`mlvc-psnr-v1`**   |  18.3M |  Decode   |             153 |      **1.79 ms** |            12.4 µs | **~555 fps** (~1.80 ms/f) |     Byte-exact     | max err ≤ 1, 63.5–67.3 dB |

### Deployment Footprint (`driver-cubin` Linux `x86_64`)

```text
libmlvc_codec.so   6.1 MiB   (includes multi-arch fatbin: sm_75/80/86/89/90 + compute_80 PTX)
mlvc_demo           36 KiB   (CLI frontend)
DT_NEEDED          libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6   (libcuda.so.1 loaded via dlopen)
```

> **Why zero `DT_NEEDED libcuda.so.1`?** A multi-backend (`-DMLVC_BACKEND=all`) binary runs cleanly even on driverless Linux hosts and falls back automatically to `cpu` when no NVIDIA GPU is present.

---

## Why `mlvc.cpp`?

| Dimension                | Official Python / ONNX Runtime                       | `mlvc.cpp`                                                                                                                                |
| ------------------------ | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Runtime dependencies** | Python + PyTorch / ONNX Runtime / CUDA Runtime (GBs) | System C/C++ libs only (`6.1 MiB` `.so`, `dlopen("libcuda.so.1")`)                                                                        |
| **GPU execution**        | Generic op-by-op ORT/TRT graph                       | Custom CUTLASS Tensor Core EVT in-register ReGLU fusion (0 standalone ReGLU launches), 3-stream DMA/compute overlap, per-slot CUDA Graphs |
| **CPU / Mobile**         | ORT CPU / vendor NPU stacks                          | Hand-written ARM64 NEON & ARMv8.2 `+fp16fml` 4×16 double-buffered assembly (BTI/PAC), x86 AVX2/FMA/F16C, big.LITTLE thread pool           |
| **Integration**          | Python scripts                                       | Stable C ABI v2/v3 (`include/mlvc/codec.h`), CMake `find_package(mlvc_codec)`, stdin/stdout/FIFO streaming                                |
| **Compatibility**        | Reference implementation                             | Strict conformance contract: Frame-0 AU byte-identical, YUV max error ≤ 1, PSNR ≥ 60 dB vs Python reference                               |

---

## Try It in 60 Seconds

### 1. Clone, initialize submodules, and prepare models (one command)

`prepare_models.sh` automatically downloads Microsoft's official checkpoints (with SHA-256 verification), compiles AOT graphs, and assembles canonical bundles under `models/canonical/`:

```bash
git clone https://github.com/Puiching-Memory/mlvc.cpp.git
cd mlvc.cpp
./tools/scripts/bootstrap.sh
./tools/scripts/prepare_models.sh
```

<details>
<summary>Official Microsoft MLVC checkpoints & SHA-256 checksums</summary>

| Profile          | Params | Direct Download (Microsoft Azure Blob)                                                          | SHA-256                                                            |
| ---------------- | -----: | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `mlvc-psnr-v1`   |  18.3M | [mlvc-psnr-v1.ckpt](https://mlvideopub.blob.core.windows.net/mlvc/models/mlvc-psnr-v1.ckpt)     | `834adaab680837c4106a9bb62e5778b3f13729b4fe6f751797e0747214f5cca1` |
| `mlvc-s-psnr-v1` |   5.4M | [mlvc-s-psnr-v1.ckpt](https://mlvideopub.blob.core.windows.net/mlvc/models/mlvc-s-psnr-v1.ckpt) | `1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896` |

</details>

### 2. Build (or grab a prebuilt binary from [Releases](https://github.com/Puiching-Memory/mlvc.cpp/releases))

```bash
# Linux x86_64 — NVIDIA GPU (driver-cubin, requires CUDA Toolkit >= 12.0 on build host only)
cmake --preset driver-cubin-release
cmake --build --preset driver-cubin-release -j
ctest --preset driver-cubin-release

# Cross-platform CPU backend — Linux / macOS (Apple Silicon) / Windows (no CUDA required)
cmake -S . -B build/cpu-release -DCMAKE_BUILD_TYPE=Release -DMLVC_BACKEND=cpu
cmake --build build/cpu-release -j
ctest --test-dir build/cpu-release --output-on-failure
```

### 3. Encode any MP4 & compare side-by-side against H.264

```bash
# Run on your own MP4 (or omit the file to auto-generate a synthetic 640x360 clip)
./tools/scripts/demo_mp4.sh input.mp4 --profile mlvc-s-psnr-v1 --q-index 21
# Outputs: demo_output/encoded.mlvc, reconstructed_mlvc.mp4, and compare_mlvc_vs_x264.mp4 (left=mlvc.cpp, right=x264 @ matched bitrate)
```

**Or pipe live through FFmpeg / `ffplay` (zero temp files):**

```bash
ffmpeg -i input.mp4 -vf scale=640:360 -pix_fmt yuv420p -f rawvideo - 2>/dev/null \
  | ./build/driver-cubin-release/tools/mlvc_demo encode --input - --output - \
      --width 640 --height 360 --frames 0 --q-index 21 --model-profile mlvc-s-psnr-v1 \
  | ./build/driver-cubin-release/tools/mlvc_demo decode --input - --output - \
      --frames 0 --model-profile mlvc-s-psnr-v1 \
  | ffplay -f rawvideo -pixel_format yuv420p -video_size 640x360 -framerate 30 -
```

For CPU builds without embedded models, add `--backend cpu --model-dir models/canonical/mlvc-s-psnr-v1/640x368`.

---

## Key Technical Highlights

- **`driver-cubin` GPU backend** (`src/backends/driver_cubin/`):
  - Embedded multi-arch CUDA fatbin (`sm_75`/`sm_80`/`sm_86`/`sm_89`/`sm_90` + `compute_80` PTX) loaded via `dlopen("libcuda.so.1")`.
  - Weight-interleaved CUTLASS 4.8 EVT epilogue (`VisitorReGLUStore`) fusing ReGLU directly in registers — **100% eliminating standalone ReGLU kernels** and halving HBM write traffic.
  - SM-aware small-M tiles (`64×128`, `64×64`), hardware SMEM/SM occupancy guards, 4-warp Split-K `16×16` MMA kernels for `spatial < 920`, zero-SMEM 2-row sliding-window `oct2` 3×3 depthwise conv, and fused encoder/decoder quantization tails.
  - 3-stream (`htod` / `compute` / `dtoh`) DMA+compute overlap with per-slot full CUDA Graphs (reducing inter-frame bubbles from 73–90 µs to **~11 µs**) and two-pass batched CUTLASS initialization (~6 ms GPU setup).
- **`cpu` cross-platform backend** (`src/backends/cpu/`):
  - FP16 storage with FP32 accumulation and internal NHWC layout.
  - Runtime ISA dispatch: hand-written AArch64 NEON and ARMv8.2 `+fp16fml` 4×16 double-buffered assembly microkernels (BTI/PAC hardened, in-register ReGLU fusion), `x86_64` AVX2/FMA/F16C kernels, heterogeneous big.LITTLE thread pool, and process-wide packed weight cache.
- **Clean 4-layer architecture & C SDK**:
  - `src/core` (bitstream, rANS, YUV, `mmap` assets), `src/runtime` (AOT IR, backend registry, buffered codec I/O), `src/backends` (`driver-cubin`, `cpu`), and `src/codec` (`EncoderSession`/`DecoderSession` + stable C ABI v2/v3 in [`include/mlvc/codec.h`](include/mlvc/codec.h)).
  - Verified under ASan/UBSan, ThreadSanitizer, libFuzzer, and strict 2-frame reference parity.

## Platform & Backend Matrix

| Target Platform                       | `MLVC_BACKEND=AUTO` Default                    | Supported Backends                   | Model Storage                                                |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| Linux `x86_64` (NVIDIA GPU)           | `driver_cubin`                                 | `driver_cubin`, `cpu`, `all`, `none` | ELF-embedded (`.incbin`) or directory bundle                 |
| Linux `aarch64` / `x86_64` (CPU-only) | `cpu` (`x86_64` requires `-DMLVC_BACKEND=cpu`) | `cpu`, `none`                        | Directory bundle or ELF-embedded (`MLVC_EMBED_MODELS=ON`)    |
| Android / OpenHarmony (`arm64-v8a`)   | `cpu`                                          | `cpu`, `none`                        | Directory bundle, ELF-embedded, or custom `ModelAssetReader` |
| Windows (`x86_64`) / macOS (`arm64`)  | `cpu`                                          | `cpu`, `none`                        | Directory bundle or custom `ModelAssetReader`                |

## Documentation

All technical documentation lives in [`docs/`](docs/README.md):

| Document                                                   | Contents                                                                                                                                                          |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [docs/README.md](docs/README.md)                           | Documentation index organized by role                                                                                                                             |
| [docs/architecture.md](docs/architecture.md)               | 4-layer system architecture, decoupling boundaries, `ModelAssetReader`, `EncoderSession`/`DecoderSession`, and source map                                         |
| [docs/backends.md](docs/backends.md)                       | `driver-cubin` (NVIDIA Driver API, fatbin, CUTLASS EVT ReGLU, 3-stream graphs, empirical A30 lessons) and `cpu` (ARM64 assembly, AVX2, heterogeneous thread pool) |
| [docs/codec-and-bitstream.md](docs/codec-and-bitstream.md) | Split-model I/O, YUV420 padding/cropping, GOP/DPB/QP schedule, scale extraction, rANS order, and `mlvc-nalu-v0` Annex-B specification                             |
| [docs/model-pipeline.md](docs/model-pipeline.md)           | Offline checkpoint AOT compilation, `src/proto/mlvc_model.proto` (bundle v2 / AOT v3), arena planner, `profiles.json`, and ELF model embedding                    |
| [docs/build-and-usage.md](docs/build-and-usage.md)         | CMake options/presets, cross-compilation (Linux, Windows, Android, OpenHarmony), release packaging, and `mlvc_demo` CLI / streaming / FIFO                        |
| [docs/sdk-integration.md](docs/sdk-integration.md)         | Public C ABI v2/v3 (`include/mlvc/codec.h`), CMake `find_package`, symbol boundaries, internal C++ session/asset APIs, and service/mobile deployment              |
| [docs/verification.md](docs/verification.md)               | Unit test matrix, CI & sanitizers, AOT host conformance, `mlvc_codec_compatibility` tolerances, and acceptance limits                                             |

## Acknowledgements & License

- [Microsoft MLVC](https://github.com/microsoft/mlvc) supplies the neural video codec architecture, pretrained checkpoints, and `msrtc_rans` entropy coder (MIT License).
- [Microsoft libmlvc](https://github.com/microsoft/libmlvc) supplies the v0.x NALU Annex-B bitstream specification, bit I/O and NALU parser/builder design, and frozen conformance vectors (MIT License; see [third_party/libmlvc/LICENSE](third_party/libmlvc/LICENSE)).
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) supplies CUDA Tensor Core kernel building blocks under its upstream license.
- [Protobuf](https://github.com/protocolbuffers/protobuf) and [Abseil](https://github.com/abseil/abseil-cpp) provide statically linked Lite binary model serialization.
- Licensed under the [MIT License](LICENSE).
