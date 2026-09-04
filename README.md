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
  - `driver_cubin` — fixed-shape AOT graphs, weights, and CUDA kernels embedded
    in one shared library using the Driver API only
- Portable binary: statically links `msrtc_rans`, dynamically loads backend
  SDKs with `$ORIGIN` rpath
- Portable shared library: `libmlvc_codec.so` with separate `mlvc_encode` and
  `mlvc_decode` C ABI entry points, one stable C header, and a CMake package
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

MLVC exported model artifacts are produced by the official converter. The
ONNX Runtime and TensorRT release packages copy `MLVCEncoder.onnx` and
`MLVCDecoder.onnx`; the libtorch package copies `MLVCEncoder.ts` and
`MLVCDecoder.ts`. All three install both registered profiles, their metadata,
and their PMFs under `share/mlvc/models`. The runtime discovers that directory
relative to the installed `libmlvc_codec.so`, so packaged applications do not
need `--model-dir`. Driver+cubin validates the same canonical bundles at build
time, then embeds their metadata, PMFs, AOT schedules, and weights in
`libmlvc_codec.so`.

The unified conversion tool provides subcommands for both framework artifact
formats. Run it from `third_party/mlvc`:

```bash
uv run ../../tools/model_convert.py onnx export \
    --model-version dmc61sbr_reglu --model-type onnx \
    --target-device generic --torch-device cuda --precision fp16 \
    --weights-path /absolute/path/to/mlvc-psnr-v1.ckpt \
    --no-validate-conversion

uv run ../../tools/model_convert.py torchscript export \
    --model-version dmc61sbr_reglu --model-type torch \
    --torch-device cuda --precision fp16 \
    --weights-path /absolute/path/to/mlvc-psnr-v1.ckpt \
    --no-validate-conversion
```

Both subcommands preserve upstream model construction and save the final captured
inputs/outputs as `benchmark-MLVC*.npz`. The ONNX wrapper uses a small NVIDIA
pass set and deliberately omits QNN workarounds and channel-splitting passes.
The TorchScript wrapper replaces the upstream eager `.torch` serializer with a
frozen `.ts` export consumable by libtorch C++.

Do not mix metadata or PMFs from independent exports. Build AOT graph data and
assemble all formats into one canonical package whose ONNX metadata and PMFs
are authoritative:

```bash
REPO="$(pwd)"
(cd third_party/mlvc && uv run ../../tools/model_aot.py \
    --model-dir "$REPO/models/generated/exports/mlvc-psnr-v1/onnx-generic/640x368" \
    --output-dir /tmp/mlvc-aot)
./tools/model_package.py assemble --profile mlvc-psnr-v1 \
    --onnx-dir models/generated/exports/mlvc-psnr-v1/onnx-generic/640x368 \
    --torchscript-dir models/generated/exports/mlvc-psnr-v1/torch-generic/640x368 \
    --aot-dir /tmp/mlvc-aot --output-dir models/canonical/mlvc-psnr-v1/640x368
./tools/model_package.py verify \
    --model-dir models/canonical/mlvc-psnr-v1/640x368
```

Repeat with the `mlvc-s-psnr-v1` profile and corresponding export directories
for MLVC-S. `models/profiles/profiles.json` is the source of profile identity;
runtime tensor dimensions always come from the verified `metadata.json`.

## NVIDIA release build

```bash
# Shallow-clone source submodules and install the pinned GPU SDKs.
./tools/bootstrap.sh --backend all

# Preflight all SDKs, build all four isolated backends, validate
# runtime linkage, and create four archives under packages/.
./tools/package.sh

# One release can also be built independently.
./tools/package.sh --backend driver-cubin --jobs 8
```

The packaging command uses canonical bundles below `models/canonical` by default;
pass `--model-root DIR` to use another validated source tree. Every package
uses `mlvc-psnr-v1` by default; pass `--model-profile mlvc-s-psnr-v1` to select
the small model. Driver+cubin reads its embedded model, while ONNX Runtime,
libtorch, and TensorRT discover their copied model files inside the package.
An explicit `--model-dir` remains available as a development override. The
TensorRT archive also carries `libcudart.so.13` and its CUDA EULA, while the
Driver+cubin archive remains dependent only on the system NVIDIA driver. The
driver fatbin is generated in the build tree by CMake from
`backends/driver_cubin/kernels/module.cu`.

For source-only development setup, run `./tools/bootstrap.sh`. This executes
`git submodule update --init --recursive --depth 1`; `.gitmodules` also marks
both source dependencies as shallow. ONNX Runtime and libtorch are downloaded
SDK inputs and remain git-ignored; TensorRT is installed from NVIDIA's system
repository. The one supported version matrix lives in `tools/dependencies.env`.
GPU bootstrap installs missing Ubuntu build tools before acquiring the SDKs.

`tools/package.sh` refuses CPU-only ONNX Runtime/libtorch SDKs. It uses clean staging
directories, enables Release IPO, writes file checksums, and audits both the
executable and the backend's GPU library for unresolved dependencies. It also
checks that the staged codec resolves `libcudart.so.13` from the archive rather
than from the build host's CUDA toolkit.

The runtime accepts FP16 floating-point tensors only; int32 remains available
for model control inputs. ONNX Runtime enables full graph optimization,
exhaustive cuDNN algorithm search, and TunableOp; libtorch uses inference mode,
cuDNN benchmarking, and explicitly disables TF32; TensorRT uses strongly typed
FP16 ONNX graphs, optimization level 5, a configurable workspace, pinned
transfer buffers, reused device allocations, and hardware-specific serialized
engine caches.

For a direct development build, bootstrap and select one GPU backend:

```bash
./tools/bootstrap.sh --backend onnxruntime
cmake --preset onnxruntime-release
cmake --build --preset onnxruntime-release -j
```

All source builds are placed below the single top-level `build/` directory.
Equivalent presets are provided for `libtorch`, `tensorrt`, and
`driver-cubin`.

CI runs on a standard Linux x86_64 runner. It builds the driver-cubin code,
runs the portable tests, and creates a compile-only smoke package without
model assets or GPU runtime checks. The full release packages still require
the canonical model bundles and an NVIDIA build host; run the commands above
for those packages.

## Usage

```bash
./mlvc_demo encode --input in.yuv --width 640 --height 360 --frames 60 \
                   --q-index 21 --output out.mlvc
./mlvc_demo decode --input out.mlvc --width 640 --height 360 --frames 60 \
                   --output rec.yuv
```

`--frames 0` processes until EOF. TensorRT additionally accepts
`--engine-cache-dir`; engines are automatically placed below a profile-named
subdirectory so MLVC and MLVC-S can safely share one cache root. Every backend
accepts `--debug-dir` to emit named model inputs and outputs for compatibility
diagnosis. `--device-id` selects the CUDA ordinal for either command;
`--encode-device-id` and `--decode-device-id` are direction-specific aliases.
No release package requires `--model-dir`: all default to `mlvc-psnr-v1` and
accept `--model-profile` to select another packaged profile.

To use one GPU for encoding and another for decoding, run the two independent
entry points concurrently:

```bash
./bin/mlvc_demo encode --input in.yuv --output out.mlvc \
    --width 640 --height 360 --frames 60 --q-index 21 \
    --encode-device-id 0
./bin/mlvc_demo decode --input out.mlvc --output rec.yuv \
    --width 640 --height 360 --frames 60 \
    --decode-device-id 1
```

These file-based commands are intentionally ordered. For live overlap, use a
shell pipe or named pipe; the per-frame container is streamable and both
processes retain their own GPU ordinal. Human-readable command summaries go to
stderr whenever binary output is stdout:

```bash
cat in.yuv \
  | ./bin/mlvc_demo encode --input - --output - --width 640 --height 360 \
      --frames 60 --q-index 21 \
      --encode-device-id 0 2>encode.log \
  | ./bin/mlvc_demo decode --input - --output - --width 640 --height 360 \
      --frames 60 \
      --decode-device-id 1 2>decode.log \
  > rec.yuv
```

The same `-` convention is available through the C ABI (`input_path` and
`output_path`); named FIFO paths are detected automatically and flushed after
each frame.

The standard compatibility target uses the two-frame I/P fixture. For a
100-frame TensorRT audit, use `tests/conformance/run_codec_conformance.py --diagnostic`;
it continues after contract violations and writes every frame's PSNR, payload
hash, and intermediate-tensor error to the requested JSON result. This is
useful for measuring long-term DPB drift without changing the strict default
test behaviour.

Every release package installs `libmlvc_codec.so`, the stable
`include/mlvc/codec.h` C header, and `lib/cmake/mlvc_codec`. Core, runtime, and
backend C++ headers remain private to the source tree. The C ABI keeps the two
directions separate and returns an error code instead of allowing C++
exceptions to cross the boundary:

```c
#include <mlvc/codec.h>

mlvc_codec_options options = {
    .width = 640, .height = 360, .q_index = 21, .frames = 60,
    .device_id = 0, .workspace_mib = 4096,
    .input_path = "in.yuv", .output_path = "out.mlvc",
    .model_dir = NULL, /* Use the package's default mlvc-psnr-v1 profile. */
};
mlvc_codec_stats stats;
char error[512];
if (mlvc_encode(&options, &stats, error, sizeof(error)) != 0) {
    /* error contains the diagnostic */
}
```

Set `options.device_id` to `1` in a separate `mlvc_decode` call to bind the
decoder to the second GPU. A CMake consumer can use
`find_package(mlvc_codec CONFIG REQUIRED)` and link `mlvc::mlvc_codec`.

## Codec compatibility

Generate a two-frame official-Python reference (frame 0 is intra, frame 1 uses
the recurrent feature DPB), then run the formal compatibility target:

```bash
REPO="$(pwd)"
(cd third_party/mlvc && uv run ../../tools/model_reference.py \
    --model-dir "$REPO/models/canonical/mlvc-psnr-v1/640x368" \
    --input "$REPO/input.yuv" --width 640 --height 360 --frames 2 --q-index 21 \
    --output-dir "$REPO/models/fixtures/references/mlvc-psnr-v1/gray-q21-2f")
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
./tools/benchmark.py make-case \
    --input output/debug/model_data_0.npz \
    --model MLVCEncoder \
    --output-dir benchmark-cases/encoder-frame-0
```

Run the resulting case with each isolated backend build:

```bash
./build/onnxruntime-release/mlvc_backend_bench \
    --model-dir models/canonical/mlvc-psnr-v1/640x368 \
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
./tools/benchmark.py compare \
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

This model-level benchmark intentionally keeps all graph inputs and outputs
host-visible so backend results remain directly comparable. The codec path uses
device-resident `feature -> ref_feature` state and therefore omits that recurrent
tensor's per-frame D2H/H2D transfers.

## Driver-only monolithic AOT path

An experimental AOT path mirrors the deployment architecture observed in the
local NGX DLSS binary. Model metadata, entropy tables, graph schedules, FP16
weights, and a multi-architecture kernel fatbin are linked into read-only
sections of `libmlvc_codec.so` and dispatched through the official CUDA Driver
API. The library requires `libcuda.so.1` from the NVIDIA driver, but does not
link CUDA Runtime, cuDNN, cuBLAS, TensorRT, ONNX Runtime, or libtorch.

CUDA Toolkit and canonical model bundles are build-host inputs only. The
offline compiler converts each fixed-shape ONNX graph into a static weights
blob, operator schedule, tensor lifetime arena, and integrity metadata. CMake
validates and embeds both registered profiles when building the isolated
backend:

```bash
cmake --preset driver-cubin-release
cmake --build --preset driver-cubin-release -j
./build/driver-cubin-release/mlvc_driver_probe --iterations 1000
./build/driver-cubin-release/mlvc_demo encode --input in.yuv --output out.mlvc \
    --width 640 --height 360 --frames 2 --q-index 21
./build/driver-cubin-release/mlvc_demo --list-model-profiles
```

The prebuilt fatbin contains `sm_75`, `sm_80`, `sm_86`, and `sm_89` cubins plus
a `compute_89` PTX fallback. Package and audit the monolithic library
independently with:

```bash
./tools/package.sh --backend driver-cubin
```

The embedded module implements every operator used by both complete
MLVCEncoder/MLVCDecoder graphs: grouped/depthwise convolution, pointwise
convolution, residual arithmetic, concat/slice/gather, activations, reciprocal,
rounding, and space/depth transforms. The runtime loads weights once, allocates
them directly from the library's read-only model image, allocates a statically
planned arena, and dispatches the full graph with Driver API streams.
`mlvc_driver_probe` remains as a small fatbin/launch diagnostic.

Pointwise convolution uses four-warp WMMA tiles, dense spatial convolution uses
WMMA implicit GEMM, and depthwise convolution has a barrier-free specialized
kernel. FP32 accumulation preserves reference parity. The recurrent feature is
kept in its device input allocation and the fixed-shape steady-state schedule is
replayed with CUDA Graphs. Further optimization focuses on general transfer and
pipeline scheduling instead of per-model code generation or device-specific
autotuning tables.
See [`docs/nvngx-dlss-cubin-architecture.md`](docs/nvngx-dlss-cubin-architecture.md).

## Repository layout

```
public/include/mlvc/codec.h Stable C ABI header
core/                        Tensor, model, YUV, entropy and bitstream logic
runtime/                     Backend interface and runtime contracts
codec/                       C ABI and codec pipeline
backends/                    One backend implementation selected per release
backends/driver_cubin/src/   AOT load, validation, planning and execution modules
backends/driver_cubin/kernels/ Explicitly aggregated CUDA kernel modules
tools/cli/                   CLI entry point
tools/benchmark/             FP16 model latency and parity benchmark
tools/bootstrap.sh           Source and GPU SDK setup
tools/package.sh             Unified four-backend release packaging
tools/model_*.py             Model conversion, AOT, packaging, and references
tools/benchmark.py           Benchmark fixture and result utilities
tests/unit/                  Portable unit and C API tests
tests/conformance/           Official reference conformance runner
models/canonical/            Verified backend-independent model bundles
models/fixtures/             Test inputs, benchmark cases and references
models/generated/            Ignored converter/export/cache outputs
models/profiles/profiles.json Model profile registry
third_party/mlvc/            microsoft/mlvc shallow Git submodule
third_party/nlohmann_json/   nlohmann/json shallow Git submodule
third_party/onnxruntime/     ONNX Runtime (fetched by script, git-ignored)
third_party/libtorch/        libtorch (fetched by script, git-ignored)
docs/design.md               Pipeline design notes (ONNX IO, entropy coding, GOP)
```

## Design notes

See [docs/design.md](docs/design.md) for the reference pipeline analysis
(model split I/O, scale extraction, rANS message order, bitstream format, GOP
and DPB handling) and implementation decisions.

For external application integration, including CLI/C ABI/CMake usage and a
微信小程序 + Linux GPU service architecture, see
[`docs/integration-guide-zh.md`](docs/integration-guide-zh.md).

## Acknowledgements

- [Microsoft MLVC](https://github.com/microsoft/mlvc) — model, converter and
  the pinned `msrtc_rans` source submodule (MIT licensed).
- ONNX Runtime — Microsoft's cross-platform inference engine (MIT licensed).
