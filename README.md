# mlvc.cpp

Linux x86_64 C++ implementation of the **MLVC** (Microsoft Learned Video
Codec, DMC-6.1sb) encode/decode pipeline, targeting four isolated NVIDIA GPU
release variants.

The demo mirrors the behaviour of the official MLVC split-model deployment path
(`video/conversion`): the neural parts run through **ONNX Runtime**,
**libtorch** (TorchScript), **TensorRT**, or an embedded **Driver+cubin** AOT
graph, while scale extraction and entropy coding (rANS) run in portable C++
using the official `msrtc_rans` library.

> Status: **complete fixed-shape codec path**. YUV420 preprocessing,
> padding/cropping, recurrent GOP/DPB state, PMF/rANS, frame containers, and
> `encode`/`decode` CLI are implemented for both MLVC and MLVC-S. All four
> backends pass the official-Python two-frame compatibility target.
> Driver+cubin uses WMMA dense convolutions and a specialized depthwise path;
> broader fusion and device-resident work remain optimization tasks, not codec
> gaps.

## Release variants

Each release contains exactly one backend and its own runtime libraries. This
prevents CUDA/cuDNN libraries from different SDKs being mixed in one process:

- `mlvc_cpp-<version>-onnxruntime-nvidia-<os>-<arch>.tar.gz`
- `mlvc_cpp-<version>-libtorch-nvidia-<os>-<arch>.tar.gz`
- `mlvc_cpp-<version>-tensorrt-nvidia-<os>-<arch>.tar.gz`
- `mlvc_cpp-<version>-driver-cubin-nvidia-linux-x86_64.tar.gz`

The NVIDIA driver is not bundled. The packaging host and deployment host must
provide a driver compatible with the CUDA dependencies recorded in
`BUILD-MANIFEST.txt`.

## Features

- Encode YUV420 (8-bit) to the MLVC bitstream format
- Decode MLVC bitstream back to YUV420
- MLVC (`dmc61sbr_reglu`) and MLVC-S (`dmc61sbr_mini_reglu`) profiles
- GOP QP shifts and separate encoder/decoder reference-feature DPBs
- Four independently compiled inference backends:
  - `onnxruntime` — ONNX graphs on ONNX Runtime CUDA EP
  - `libtorch` — TorchScript exports on CUDA libtorch
  - `tensorrt` — ONNX graphs built into TensorRT engines (NVIDIA GPU)
  - `driver_cubin` — fixed-shape AOT graph using embedded CUDA kernels and the
    Driver API only
- Portable binary: statically links `msrtc_rans`, dynamically loads backend
  SDKs with `$ORIGIN` rpath
- Canonical, hash-verified model bundles shared by every backend
- Official Python I/P-frame bitstream, reconstruction, and intermediate-tensor
  conformance tests

## Dependencies

| Dependency                    | Role                                 | Notes                                                      |
| ----------------------------- | ------------------------------------ | ---------------------------------------------------------- |
| CUDA 13.3                     | NVIDIA compute stack                 | System toolkit                                             |
| ONNX Runtime 1.26.0           | Inference backend                    | CUDA 13 GPU package + pinned cuDNN 9.20 runtime            |
| libtorch 2.13.0               | Inference backend                    | cu130 + pinned CUDA-X runtime closure; requires C++20      |
| TensorRT 11.2.1               | Inference backend                    | Minimal C++ packages from NVIDIA's Ubuntu 26.04 repository |
| NVIDIA Driver API             | Driver+cubin backend                 | Only `libcuda.so.1` is required in its deployed package    |
| `msrtc_rans`                  | rANS entropy coding                  | From the pinned `microsoft/mlvc` Git submodule             |
| nlohmann/json 3.11.3          | JSON parsing (PMF tables / metadata) | Pinned Git submodule                                       |
| CMake >= 3.23, C++20 compiler | Build                                |                                                            |

MLVC exported model artifacts (`MLVCEncoder.onnx`, `MLVCDecoder.onnx`,
`gaussian_pmf.json`, `bit_estimator_pmf.json`, `metadata.json`) are produced by
the official converter and supplied at runtime via `--model-dir`. The
`libtorch` backend additionally needs TorchScript exports
(`MLVCEncoder.ts` / `MLVCDecoder.ts`) in the same directory.

The repository provides thin wrappers around the upstream converter for the
two framework artifact formats. Run them from `third_party/mlvc`:

```bash
uv run ../../scripts/export_onnx.py export \
    --model-version dmc61sbr_reglu --model-type onnx \
    --target-device generic --torch-device cuda --precision fp16 \
    --weights-path /absolute/path/to/mlvc-psnr-v1.ckpt \
    --no-validate-conversion

uv run ../../scripts/export_torchscript.py export \
    --model-version dmc61sbr_reglu --model-type torch \
    --torch-device cuda --precision fp16 \
    --weights-path /absolute/path/to/mlvc-psnr-v1.ckpt \
    --no-validate-conversion
```

Both wrappers preserve upstream model construction and save the final captured
inputs/outputs as `benchmark-MLVC*.npz`. The ONNX wrapper uses a small NVIDIA
pass set and deliberately omits QNN workarounds and channel-splitting passes.
The TorchScript wrapper replaces the upstream eager `.torch` serializer with a
frozen `.ts` export consumable by libtorch C++.

Do not mix metadata or PMFs from independent exports. Build AOT graph data and
assemble all formats into one canonical package whose ONNX metadata and PMFs
are authoritative:

```bash
REPO="$(pwd)"
(cd third_party/mlvc && uv run ../../scripts/build_aot_model.py \
    --model-dir "$REPO/model-assets/models/mlvc-psnr-v1/onnx-generic/640x368" \
    --output-dir /tmp/mlvc-aot)
./scripts/assemble_model_package.py --profile mlvc-psnr-v1 \
    --onnx-dir model-assets/models/mlvc-psnr-v1/onnx-generic/640x368 \
    --torchscript-dir model-assets/models/mlvc-psnr-v1/torch-generic/640x368 \
    --aot-dir /tmp/mlvc-aot --output-dir models/mlvc-psnr-v1/640x368
./scripts/assemble_model_package.py \
    --output-dir models/mlvc-psnr-v1/640x368 --verify-only
```

Repeat with the `mlvc-s-psnr-v1` profile and corresponding export directories
for MLVC-S. `configs/model_profiles.json` is the source of profile identity;
runtime tensor dimensions always come from the verified `metadata.json`.

## NVIDIA release build

```bash
# Shallow-clone source submodules and install the pinned GPU SDKs.
./scripts/bootstrap.sh --backend all

# Preflight all SDKs, build three framework trees, validate runtime linkage,
# verify that each binary contains only its intended backend, then create
# three archives under packages/.
./scripts/package.sh

# One release can also be built independently.
./scripts/package.sh --backend tensorrt --jobs 8

# Build the fourth, inference-framework-free release.
./scripts/build_driver_fatbin.sh
./scripts/package_driver_cubin.sh
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

The runtime accepts FP16 floating-point tensors only; int32 remains available
for model control inputs. ONNX Runtime enables full graph optimization,
exhaustive cuDNN algorithm search, and TunableOp; libtorch uses inference mode,
cuDNN benchmarking, and explicitly disables TF32; TensorRT uses strongly typed
FP16 ONNX graphs, optimization level 5, a configurable workspace, pinned
transfer buffers, reused device allocations, and hardware-specific serialized
engine caches.

For a direct development build, bootstrap and select one GPU backend:

```bash
./scripts/bootstrap.sh --backend onnxruntime
cmake -S . -B build -DMLVC_BACKEND=onnxruntime \
      -DONNXRUNTIME_ROOT="$(./scripts/fetch_onnxruntime.sh --print-dir)"
cmake --build build -j
```

CI and release builds run on a Linux x86_64 NVIDIA self-hosted runner.

## Usage

```bash
./mlvc_demo encode --input in.yuv --width 640 --height 360 --frames 60 \
                   --q-index 21 --model-dir models/640x368 --output out.mlvc
./mlvc_demo decode --input out.mlvc --width 640 --height 360 --frames 60 \
                   --model-dir models/640x368 --output rec.yuv
```

`--frames 0` processes until EOF. TensorRT additionally accepts
`--engine-cache-dir`; every backend accepts `--debug-dir` to emit named model
inputs and outputs for compatibility diagnosis.

## Codec compatibility

Generate a two-frame official-Python reference (frame 0 is intra, frame 1 uses
the recurrent feature DPB), then run the formal compatibility target:

```bash
REPO="$(pwd)"
(cd third_party/mlvc && uv run ../../scripts/make_codec_reference.py \
    --model-dir "$REPO/models/mlvc-psnr-v1/640x368" \
    --input "$REPO/input.yuv" --width 640 --height 360 --frames 2 --q-index 21 \
    --output-dir "$REPO/model-assets/references/mlvc-psnr-v1/gray-q21-2f")
cmake --build build --target mlvc_codec_compatibility
```

The target runs both MLVC and MLVC-S when their reference directories are
present. It verifies bundle/reference hashes, canonical Python bitstream
decoding, backend round trips, YUV reconstruction, shifted QP inputs, and the
encoder/decoder DPB tensors. See
[`docs/codec-compatibility.md`](docs/codec-compatibility.md) for the exact
normative and tolerance rules.

## FP16 backend benchmark

The upstream conversion loop can save the exact inputs and Python reference
outputs used by each model part as `model_data_*.npz`. Convert one snapshot in
the same Python environment that runs MLVC:

```bash
./scripts/make_benchmark_case.py \
    --input output/debug/model_data_0.npz \
    --model MLVCEncoder \
    --output-dir benchmark-cases/encoder-frame-0
```

Run the resulting case with each isolated backend build:

```bash
./build-release/onnxruntime/mlvc_backend_bench \
    --model-dir models/640x368 \
    --case benchmark-cases/encoder-frame-0/case.json \
    --warmup 20 --iterations 100 \
    --result results/onnxruntime-encoder-frame-0.json
```

Use the same command from the libtorch, TensorRT, and Driver+cubin build
directories. Results contain model loading time, host-input-to-host-output
p50/p95/p99 latency, throughput, FP16 bit mismatches, maximum absolute/relative
error, RMSE, cosine similarity, and non-finite counts. Optional
`--max-abs-error` and `--max-rmse` arguments turn metric thresholds into a
non-zero process exit status.

Summarize comparable result files in one table:

```bash
./scripts/compare_backend_results.py \
    results/onnxruntime-encoder-frame-0.json \
    results/libtorch-encoder-frame-0.json \
    results/tensorrt-encoder-frame-0.json \
    results/driver-cubin-encoder-frame-0.json \
    --json results/encoder-frame-0-summary.json
```

The converter assigns a content-derived `case_id`; the comparison script
rejects results produced from different tensors or timing implementations.

The four-backend NVIDIA A30 baseline and its limitations are recorded in
[`docs/benchmarks/2026-09-02-a30-fp16.md`](docs/benchmarks/2026-09-02-a30-fp16.md).

This first benchmark intentionally measures the current public host tensor API,
including H2D/D2H copies and synchronization. A later device-resident benchmark
will isolate inference after I/O binding and CUDA pipeline work lands.

## Driver-only embedded cubin path

An experimental AOT path mirrors the deployment architecture observed in the
local NGX DLSS binary: model kernels are compiled into a multi-architecture
fatbin, embedded in the executable, and dispatched through the official CUDA
Driver API. The executable requires `libcuda.so.1` from the NVIDIA driver, but
does not link CUDA Runtime, cuDNN, cuBLAS, TensorRT, ONNX Runtime, or libtorch.

CUDA Toolkit is a build-host input only. The offline compiler converts each
fixed-shape ONNX graph into a static weights blob, operator schedule, tensor
lifetime arena, and integrity metadata. Rebuild the kernel fatbin, assemble the
model package as shown above, then build the isolated backend:

```bash
./scripts/build_driver_fatbin.sh
cmake -S . -B build-driver-cubin \
      -DMLVC_DRIVER_CUBIN_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-driver-cubin -j
./build-driver-cubin/mlvc_driver_probe --iterations 1000
./build-driver-cubin/mlvc_demo encode --input in.yuv --output out.mlvc \
    --width 640 --height 360 --frames 2 --q-index 21 \
    --model-dir models/mlvc-psnr-v1/640x368
```

The prebuilt fatbin contains `sm_75`, `sm_80`, `sm_86`, and `sm_89` cubins plus
a `compute_89` PTX fallback. Package and audit this path independently with:

```bash
./scripts/package_driver_cubin.sh
```

The embedded module implements every operator used by both complete
MLVCEncoder/MLVCDecoder graphs: grouped/depthwise convolution, pointwise
convolution, residual arithmetic, concat/slice/gather, activations, reciprocal,
rounding, and space/depth transforms. The runtime loads weights once, allocates
a statically planned arena, and dispatches the full graph with Driver API
streams. `mlvc_driver_probe` remains as a small fatbin/launch diagnostic.

Pointwise convolution uses four-warp WMMA tiles, dense spatial convolution uses
WMMA implicit GEMM, and depthwise convolution has a barrier-free specialized
kernel. FP32 accumulation preserves reference parity. Architecture-specific
autotuning, fusion, and CUDA Graph capture remain performance improvements.
See [`docs/research/nvngx-dlss-cubin-architecture.md`](docs/research/nvngx-dlss-cubin-architecture.md).

## Repository layout

```
CMakeLists.txt               Top-level single-backend build (MLVC_BACKEND)
src/main.cpp                 CLI entry point
src/backend_bench.cpp        FP16 model latency and parity benchmark
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
