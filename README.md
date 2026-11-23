# mlvc.cpp

> English | [简体中文](README.zh-CN.md)

A standalone C++20 implementation of **Microsoft MLVC (`DMC-6.1sb`)** neural video coding with a decoupled multi-backend runtime, Protobuf-serialized AOT model bundles, and the `mlvc-nalu-v0` Annex-B bitstream container.

Both official model profiles — **`mlvc-psnr-v1`** (`dmc61sbr_reglu`) and **`mlvc-s-psnr-v1`** (`dmc61sbr_mini_reglu`) — are supported end-to-end with 8-bit YUV420 I/O, recurrent GOP/DPB state, canonical PMF tables, and `msrtc_rans` entropy coding.

## Highlights

- **Zero inference-framework runtime dependency**: Offline PyTorch dispatch compiles official checkpoints directly into fixed-shape AOT graphs (`proto/mlvc_model.proto`) and aligned FP16 weight blobs. Deployed binaries link no PyTorch, ONNX Runtime, TensorRT, or CUDA Runtime.
- **Multi-backend, cross-platform execution**:
  - **`driver-cubin`** (Linux `x86_64` NVIDIA GPUs): Links only the NVIDIA Driver API (`libcuda.so.1`) with an embedded multi-architecture fatbin (`sm_75`/`sm_80`/`sm_86`/`sm_89` + `compute_89` PTX), CUTLASS Tensor Core kernels, CUDA Graphs, and fused recurrent/quantization-tail kernels.
  - **`cpu`** (Linux, Windows, macOS, Android, OpenHarmony): Portable FP16-storage / FP32-accumulation CPU runtime with runtime ISA dispatch — hand-written AArch64 NEON and ARMv8.2 `+fp16fml` 4×16 double-buffered assembly microkernels (BTI/PAC compatible, in-register ReGLU fusion), `x86_64` AVX2/FMA/F16C kernels, heterogeneous big.LITTLE-aware thread pool, and graph fusion.
- **Clean 4-layer architecture**: Decoupled `core` (bitstream, rANS, YUV, assets), `runtime` (AOT IR, backend registry, buffered codec I/O), `backends` (`driver-cubin`, `cpu`), and `codec` (`EncoderSession`/`DecoderSession` + stable C ABI v2 in `public/include/mlvc/codec.h`).
- **Standardized bitstream container**: Implements the MLVC v0.x NALU Annex-B container (`mlvc-nalu-v0`) derived from `microsoft/libmlvc` (`SPS`, `PPS`, `IDR_N_LP`, `TRAIL_R`, `TSA_R`, EBSP emulation prevention, `QP` in `[0, 51]`).
- **Streamable CLI & C SDK**: Supports file I/O, stdin/stdout (`-`), POSIX named FIFOs, dual-GPU encode/decode pipelines, and a self-contained CMake package (`mlvc::mlvc_codec`).

## Platform & Backend Matrix

| Target Platform                       | `MLVC_BACKEND` Default (`AUTO`)                | Supported Backends                   | Model Storage                                                |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| Linux `x86_64` (NVIDIA GPU)           | `driver_cubin`                                 | `driver_cubin`, `cpu`, `all`, `none` | ELF-embedded (`.incbin`) or directory bundle                 |
| Linux `aarch64` / `x86_64` (CPU-only) | `cpu` (`x86_64` requires `-DMLVC_BACKEND=cpu`) | `cpu`, `none`                        | Directory bundle or ELF-embedded (`MLVC_EMBED_MODELS=ON`)    |
| Android / OpenHarmony (`arm64-v8a`)   | `cpu`                                          | `cpu`, `none`                        | Directory bundle, ELF-embedded, or custom `ModelAssetReader` |
| Windows (`x86_64`) / macOS (`arm64`)  | `cpu`                                          | `cpu`, `none`                        | Directory bundle or custom `ModelAssetReader`                |

## Quick Start

### 1. Initialize submodules and prepare models

```bash
./tools/scripts/bootstrap.sh
# Optional on Ubuntu 26.04 for NVIDIA Driver+cubin builds:
./tools/scripts/bootstrap.sh --with-cuda

uv venv --python 3.12 build/model-venv
uv pip install --python build/model-venv/bin/python -r tools/python/requirements-models.txt
uv pip install --python build/model-venv/bin/python -e ./tools/python
uv pip install --python build/model-venv/bin/python ./third_party/mlvc/packages/msrtc_rans
source build/model-venv/bin/activate

# Compile and assemble canonical bundle (repeat for mlvc-s-psnr-v1)
python -m mlvc_tools.aot \
  --profile mlvc-psnr-v1 --checkpoint /path/to/mlvc-psnr-v1.ckpt \
  --model-width 640 --model-height 368 \
  --output-dir models/generated/checkpoint/mlvc-psnr-v1/640x368

python -m mlvc_tools.package assemble \
  --profile mlvc-psnr-v1 \
  --source-dir models/generated/checkpoint/mlvc-psnr-v1/640x368 \
  --output-dir models/canonical/mlvc-psnr-v1/640x368
```

### 2. Build

```bash
# Linux x86_64 NVIDIA Driver+cubin release (embeds both canonical bundles)
cmake --preset driver-cubin-release
cmake --build --preset driver-cubin-release -j
ctest --preset driver-cubin-release

# Cross-platform CPU backend build (no CUDA Toolkit required)
cmake -S . -B build/cpu-release -DCMAKE_BUILD_TYPE=Release -DMLVC_BACKEND=cpu
cmake --build build/cpu-release -j
ctest --test-dir build/cpu-release --output-on-failure
```

### 3. Encode & decode

```bash
./build/driver-cubin-release/tools/mlvc_demo --list-backends
./build/driver-cubin-release/tools/mlvc_demo --list-model-profiles

./build/driver-cubin-release/tools/mlvc_demo encode \
  --input in.yuv --output out.mlvc \
  --width 640 --height 360 --frames 60 --q-index 21

./build/driver-cubin-release/tools/mlvc_demo decode \
  --input out.mlvc --output rec.yuv --frames 60
```

For CPU builds without embedded models, pass `--model-dir models/canonical/mlvc-psnr-v1/640x368` (and optionally `--backend cpu --threads 4`).

## Documentation

All technical documentation lives in [`docs/`](docs/README.md) (written in English):

| Document                                                   | Contents                                                                                                                                                          |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [docs/README.md](docs/README.md)                           | Documentation index organized by role                                                                                                                             |
| [docs/architecture.md](docs/architecture.md)               | 4-layer system architecture, decoupling boundaries, `ModelAssetReader`, `EncoderSession`/`DecoderSession`, and source map                                         |
| [docs/backends.md](docs/backends.md)                       | `driver-cubin` (NVIDIA Driver API, fatbin, CUTLASS, CUDA Graphs, fusion lessons) and `cpu` (ARM64 NEON/FP16FML assembly, AVX2, heterogeneous thread pool, fusion) |
| [docs/codec-and-bitstream.md](docs/codec-and-bitstream.md) | Split-model I/O, YUV420 padding/cropping, GOP/DPB/QP schedule, scale extraction, rANS order, and `mlvc-nalu-v0` Annex-B specification                             |
| [docs/model-pipeline.md](docs/model-pipeline.md)           | Offline checkpoint AOT compilation, `proto/mlvc_model.proto` (bundle v2 / AOT v3), arena planner, `profiles.json`, and ELF model embedding                        |
| [docs/build-and-usage.md](docs/build-and-usage.md)         | CMake options/presets, cross-compilation (Linux, Windows, Android, OpenHarmony), release packaging, and `mlvc_demo` CLI / streaming / FIFO                        |
| [docs/sdk-integration.md](docs/sdk-integration.md)         | Public C ABI v2 (`include/mlvc/codec.h`), CMake `find_package`, symbol boundaries, internal C++ session/asset APIs, and service/mobile deployment                 |
| [docs/verification.md](docs/verification.md)               | Unit test matrix, CI & sanitizers, AOT host conformance, `mlvc_codec_compatibility` tolerances, and acceptance limits                                             |

## Acknowledgements

- [Microsoft MLVC](https://github.com/microsoft/mlvc) supplies the neural video codec architecture and `msrtc_rans` entropy coder (MIT License).
- [Microsoft libmlvc](https://github.com/microsoft/libmlvc) supplies the v0.x NALU Annex-B bitstream specification, bit I/O and NALU parser/builder design, and frozen conformance vectors incorporated into `core/` and `tests/` (MIT License; see [licenses/libmlvc/LICENSE](licenses/libmlvc/LICENSE)).
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) supplies CUDA Tensor Core kernel building blocks under its upstream license.
- [Protobuf](https://github.com/protocolbuffers/protobuf) and [Abseil](https://github.com/abseil/abseil-cpp) provide statically linked Lite binary model serialization.
