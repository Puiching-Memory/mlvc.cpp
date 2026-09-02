# mlvc.cpp

Portable C++ implementation of the **MLVC** (Microsoft Learned Video
Codec, DMC-6.1sb) encode/decode pipeline, targeting a single self-contained
command-line binary with maximum inference performance on PC devices.

The demo mirrors the behaviour of the official MLVC split-model deployment path
(`video/conversion`): the neural parts run through one of three interchangeable
inference backends — **ONNX Runtime**, **libtorch** (TorchScript) or
**TensorRT** — while quantization and entropy coding (rANS) run in portable C++
using the official `msrtc_rans` library.

> Status: **initial scaffolding**. Repository layout, build system, backend
> abstraction and design notes are in place; the pipeline implementation is in
> progress.

## Features (planned)

- Encode YUV420 (8-bit) to the MLVC bitstream format
- Decode MLVC bitstream back to YUV420
- Three inference backends selected at runtime via `--backend`:
  - `onnxruntime` — ONNX graphs on ONNX Runtime (CPU and CUDA EP)
  - `libtorch` — TorchScript exports on libtorch (CPU and CUDA)
  - `tensorrt` — ONNX graphs built into TensorRT engines (NVIDIA GPU)
- Portable binary: statically links `msrtc_rans`, dynamically loads backend
  SDKs with `$ORIGIN` rpath

## Dependencies

| Dependency                    | Role                                 | Notes                                                                 |
| ----------------------------- | ------------------------------------ | --------------------------------------------------------------------- |
| ONNX Runtime (>= 1.19)        | Inference backend (default)          | CPU or GPU package; NOT vendored (see `scripts/fetch_onnxruntime.sh`) |
| libtorch (>= 2.13, optional)  | Inference backend                    | Requires C++20; see `scripts/fetch_libtorch.sh`                       |
| TensorRT (>= 10, optional)    | Inference backend                    | NVIDIA GPU; see `scripts/fetch_tensorrt.sh`                           |
| `msrtc_rans`                  | rANS entropy coding                  | Vendored under `third_party/msrtc_rans` (MIT, from microsoft/mlvc)    |
| nlohmann/json                 | JSON parsing (PMF tables / metadata) | Fetched at build time                                                 |
| CMake >= 3.23, C++20 compiler | Build                                |                                                                       |

MLVC exported model artifacts (`MLVCEncoder.onnx`, `MLVCDecoder.onnx`,
`gaussian_pmf.json`, `bit_estimator_pmf.json`, `metadata.json`) are produced by
the official converter and supplied at runtime via `--model-dir`. The
`libtorch` backend additionally needs TorchScript exports
(`MLVCEncoder.ts` / `MLVCDecoder.ts`) in the same directory.

## Build

```bash
# 1. Fetch backend SDKs (all optional except ONNX Runtime for the default
#    build). Scripts auto-detect OS/arch and verify downloads against SHA256
#    sums pinned in the scripts, so every system fetches identical bytes.
./scripts/fetch_onnxruntime.sh           # CPU edition; add --gpu for CUDA
./scripts/fetch_libtorch.sh              # CPU edition; add --cuda cu130 for CUDA
./scripts/fetch_tensorrt.sh TensorRT-*.tar.gz   # tarball from nvidia.com

# 2. Configure and build. SDKs unpacked under third_party/ are discovered
#    automatically; backends beyond onnxruntime are opt-in via MLVC_WITH_*.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMLVC_WITH_LIBTORCH=ON
cmake --build build -j

# 3. (Optional) assemble a portable package
cmake --install build --prefix dist
```

GitHub Actions CI builds the onnxruntime backend on Ubuntu/Windows/macOS and
the libtorch backend on Ubuntu (TensorRT is excluded: NVIDIA requires a login
to download it).

## Usage (planned)

```bash
./mlvc_demo encode --input in.yuv --width 640 --height 360 --frames 60 \
                   --q-index 21 --model-dir models/640x368 --output out.mlvc \
                   --backend tensorrt --device cuda
./mlvc_demo decode --input out.mlvc --width 640 --height 360 --frames 60 \
                   --model-dir models/640x368 --output rec.yuv \
                   --backend onnxruntime
```

## Repository layout

```
CMakeLists.txt               Top-level build (MLVC_WITH_* backend options)
src/main.cpp                 CLI entry point
src/backends/                Inference backend implementations + factory
include/mlvc/                Public headers (backend abstraction, pipeline)
third_party/msrtc_rans/      Vendored rANS entropy coder (MIT)
third_party/json/            nlohmann/json (fetched by CMake)
third_party/onnxruntime/     ONNX Runtime (fetched by script, git-ignored)
third_party/libtorch/        libtorch (fetched by script, git-ignored)
third_party/tensorrt/        TensorRT (extracted by script, git-ignored)
docs/design.md               Pipeline design notes (ONNX IO, entropy coding, GOP)
examples/                    Sample scripts/data generation
scripts/                     Fetch/bootstrap scripts
```

## Design notes

See [docs/design.md](docs/design.md) for the reference pipeline analysis
(model split I/O, scale extraction, rANS message order, bitstream format, GOP
and DPB handling) and implementation decisions.

## Acknowledgements

- [Microsoft MLVC](https://github.com/microsoft/mlvc) — model, converter and
  `msrtc_rans` (MIT licensed).
- ONNX Runtime — Microsoft's cross-platform inference engine (MIT licensed).
