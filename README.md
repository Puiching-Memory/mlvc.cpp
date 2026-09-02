# mlvc.cpp

Linux x86_64 C++ implementation of the **MLVC** (Microsoft Learned Video
Codec, DMC-6.1sb) encode/decode pipeline, targeting three isolated NVIDIA GPU
release variants with maximum inference performance.

The demo mirrors the behaviour of the official MLVC split-model deployment path
(`video/conversion`): the neural parts run through one of three interchangeable
inference backends — **ONNX Runtime**, **libtorch** (TorchScript) or
**TensorRT** — while quantization and entropy coding (rANS) run in portable C++
using the official `msrtc_rans` library.

> Status: **initial scaffolding**. Repository layout, build system, backend
> abstraction and design notes are in place; the pipeline implementation is in
> progress.

## Release variants

Each release contains exactly one backend and its own runtime libraries. This
prevents CUDA/cuDNN libraries from different SDKs being mixed in one process:

- `mlvc_cpp-<version>-onnxruntime-nvidia-<os>-<arch>.tar.gz`
- `mlvc_cpp-<version>-libtorch-nvidia-<os>-<arch>.tar.gz`
- `mlvc_cpp-<version>-tensorrt-nvidia-<os>-<arch>.tar.gz`

The NVIDIA driver is not bundled. The packaging host and deployment host must
provide a driver compatible with the CUDA dependencies recorded in
`BUILD-MANIFEST.txt`.

## Features (planned)

- Encode YUV420 (8-bit) to the MLVC bitstream format
- Decode MLVC bitstream back to YUV420
- Three independently compiled inference backends:
  - `onnxruntime` — ONNX graphs on ONNX Runtime CUDA EP
  - `libtorch` — TorchScript exports on CUDA libtorch
  - `tensorrt` — ONNX graphs built into TensorRT engines (NVIDIA GPU)
- Portable binary: statically links `msrtc_rans`, dynamically loads backend
  SDKs with `$ORIGIN` rpath

## Dependencies

| Dependency                    | Role                                 | Notes                                                     |
| ----------------------------- | ------------------------------------ | --------------------------------------------------------- |
| CUDA 13.3                     | NVIDIA compute stack                 | System toolkit                                            |
| ONNX Runtime 1.26.0           | Inference backend                    | CUDA 13 GPU package + pinned cuDNN 9.20 runtime            |
| libtorch 2.13.0               | Inference backend                    | cu130 + pinned CUDA-X runtime closure; requires C++20      |
| TensorRT 11.2.1               | Inference backend                    | Minimal C++ packages from NVIDIA's Ubuntu 26.04 repository |
| `msrtc_rans`                  | rANS entropy coding                  | From the pinned `microsoft/mlvc` Git submodule            |
| nlohmann/json 3.11.3          | JSON parsing (PMF tables / metadata) | Pinned Git submodule                                      |
| CMake >= 3.23, C++20 compiler | Build                                |                                                           |

MLVC exported model artifacts (`MLVCEncoder.onnx`, `MLVCDecoder.onnx`,
`gaussian_pmf.json`, `bit_estimator_pmf.json`, `metadata.json`) are produced by
the official converter and supplied at runtime via `--model-dir`. The
`libtorch` backend additionally needs TorchScript exports
(`MLVCEncoder.ts` / `MLVCDecoder.ts`) in the same directory.

## NVIDIA release build

```bash
# Shallow-clone source submodules and install the pinned GPU SDKs.
./scripts/bootstrap.sh --backend all

# Preflight all SDKs, build three isolated trees, validate runtime linkage,
# verify that each binary contains only its intended backend, then create
# three archives under packages/.
./scripts/package.sh

# One release can also be built independently.
./scripts/package.sh --backend tensorrt --jobs 8
```

For source-only development setup, run `./scripts/bootstrap.sh`. This executes
`git submodule update --init --recursive --depth 1`; `.gitmodules` also marks
both source dependencies as shallow. ONNX Runtime and libtorch are downloaded
SDK inputs and remain git-ignored; TensorRT is installed from NVIDIA's system
repository. The one supported version matrix lives in `scripts/dependencies.env`.
GPU bootstrap installs missing Ubuntu build tools before acquiring the SDKs.

`package.sh` refuses CPU-only ONNX Runtime/libtorch SDKs. It uses clean staging
directories, enables Release IPO, writes file checksums, and audits both the
executable and the backend's GPU library for unresolved dependencies.

The runtime paths retain native fp16/int32 tensor types. ONNX Runtime enables
full graph optimization, exhaustive cuDNN algorithm search, and TunableOp;
libtorch uses inference mode, cuDNN benchmarking, and optional TF32; TensorRT
uses optimization level 5, a configurable workspace, pinned transfer buffers,
reused device allocations, and hardware-specific serialized engine caches.

For a direct development build, bootstrap and select one GPU backend:

```bash
./scripts/bootstrap.sh --backend onnxruntime
cmake -S . -B build -DMLVC_BACKEND=onnxruntime \
      -DONNXRUNTIME_ROOT="$(./scripts/fetch_onnxruntime.sh --print-dir)"
cmake --build build -j
```

CI and release builds run on a Linux x86_64 NVIDIA self-hosted runner.

## Usage (planned)

```bash
./mlvc_demo encode --input in.yuv --width 640 --height 360 --frames 60 \
                   --q-index 21 --model-dir models/640x368 --output out.mlvc
./mlvc_demo decode --input out.mlvc --width 640 --height 360 --frames 60 \
                   --model-dir models/640x368 --output rec.yuv
```

## Repository layout

```
CMakeLists.txt               Top-level single-backend build (MLVC_BACKEND)
src/main.cpp                 CLI entry point
src/backends/                One implementation selected per release
include/mlvc/                Public headers (backend abstraction, pipeline)
third_party/mlvc/            microsoft/mlvc shallow Git submodule
third_party/nlohmann_json/   nlohmann/json shallow Git submodule
third_party/onnxruntime/     ONNX Runtime (fetched by script, git-ignored)
third_party/libtorch/        libtorch (fetched by script, git-ignored)
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
  the pinned `msrtc_rans` source submodule (MIT licensed).
- ONNX Runtime — Microsoft's cross-platform inference engine (MIT licensed).
