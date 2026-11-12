# mlvc.cpp

Linux x86_64 C++ implementation of Microsoft MLVC (DMC-6.1sb), using the
NVIDIA Driver API and embedded CUDA kernels. **Driver+cubin is the only runtime.**

The codec implements 8-bit YUV420 input/output, padding/cropping, recurrent GOP
and reference-feature state, canonical PMF tables, rANS entropy coding, and the
official per-frame bitstream container. Both MLVC and MLVC-S are supported.

The deployed `libmlvc_codec.so` embeds model metadata, entropy tables, fixed-shape
operator schedules, FP16 weights, and a multi-architecture fatbin. Its GPU
dependency is the system NVIDIA driver (`libcuda.so.1`). CUDA Toolkit is a
build-host dependency; Python/PyTorch are offline model-preparation dependencies.

## Requirements

- Linux x86_64, CMake >= 3.23, a C++20 compiler.
- CUDA Toolkit 13.3 on the build host.
- Pinned source submodules: Microsoft MLVC (`msrtc_rans` and model source),
  nlohmann/json, and NVIDIA CUTLASS.
- Original registered `mlvc-psnr-v1.ckpt` and `mlvc-s-psnr-v1.ckpt` checkpoints.
- An NVIDIA driver compatible with the generated GPU code on the deployment host.

The fatbin contains `sm_75`, `sm_80`, `sm_86`, and `sm_89` cubins, plus a
`compute_89` PTX fallback. Driver+cubin executes FP16 neural tensors with FP32
convolution accumulation; int32 is used for model control inputs.

## Prepare source and models

```bash
./tools/bootstrap.sh
# On Ubuntu 26.04, optionally install the pinned build toolkit:
./tools/bootstrap.sh --with-cuda

uv venv --python 3.12 build/model-venv
uv pip install --python build/model-venv/bin/python -r tools/requirements-models.txt
uv pip install --python build/model-venv/bin/python ./third_party/mlvc/packages/msrtc_rans
```

The model compiler constructs the official network from its registered original
checkpoint, applies upstream structural optimization, and captures one fixed-shape
eager invocation through PyTorch dispatch. It folds constant computation, removes
dead branches, lowers live operations to the CUDA runtime contract, plans tensor
lifetimes, and writes aligned weights. Unsupported live operations fail explicitly.
Only model preparation uses PyTorch; no framework model file is loaded at runtime.

```bash
build/model-venv/bin/python tools/model_aot.py \
  --profile mlvc-psnr-v1 --checkpoint /path/to/mlvc-psnr-v1.ckpt \
  --model-width 640 --model-height 368 \
  --output-dir models/generated/checkpoint/mlvc-psnr-v1/640x368

build/model-venv/bin/python tools/model_package.py assemble \
  --profile mlvc-psnr-v1 \
  --source-dir models/generated/checkpoint/mlvc-psnr-v1/640x368 \
  --output-dir models/canonical/mlvc-psnr-v1/640x368

build/model-venv/bin/python tools/model_package.py verify \
  --model-dir models/canonical/mlvc-psnr-v1/640x368
```

Repeat for `mlvc-s-psnr-v1` with its checkpoint. Both canonical bundles are
required by the default embedded build. Checkpoint SHA-256 and construction
parameters are registered in `models/profiles/profiles.json`. PMFs are generated
from FP32 checkpoint parameters before neural graphs are converted to FP16.
Each bundle contains metadata, two PMF files, and the AOT graphs/weights; model
dimensions always come from the metadata.

Native channel-first pixel shuffle/unshuffle uses AOT graph schema **2**. Earlier
graph packages must be regenerated from the checkpoints.

## Build and package

```bash
cmake --preset driver-cubin-release
cmake --build --preset driver-cubin-release -j
ctest --preset driver-cubin-release

./tools/package.sh --jobs 8
```

The release is `packages/mlvc_cpp-<version>-driver-cubin-nvidia-linux-x86_64.tar.gz`.
It contains the codec library, CLI, benchmark, driver probe, C header, CMake
package, licenses, `BUILD-MANIFEST.txt`, and `SHA256SUMS`. Models are embedded
in the library. The driver and system C/C++ runtime libraries are not bundled.

For driverless CI compilation without model assets:

```bash
./tools/package.sh --jobs 2 --skip-gpu-tests --skip-models
```

These archives have a `-ci` suffix and are compile-only artifacts. CI builds
with GCC and Clang under warnings-as-errors and runs host safety tests with
AddressSanitizer/UndefinedBehaviorSanitizer. Full GPU conformance requires models
and generated Python reference fixtures.

## Use

After extracting a release:

```bash
./bin/mlvc_demo --backend-name
./bin/mlvc_demo --list-model-profiles
./bin/mlvc_demo encode --input in.yuv --output out.mlvc \
  --width 640 --height 360 --frames 60 --q-index 21
./bin/mlvc_demo decode --input out.mlvc --output rec.yuv \
  --width 640 --height 360 --frames 60
```

Both commands default to embedded `mlvc-psnr-v1`. Select the smaller model with
`--model-profile mlvc-s-psnr-v1`. `--model-dir` overrides the embedded model with
a complete AOT bundle for development. `--frames 0` processes until EOF;
`--device-id` selects a GPU. Encoding and decoding can run in separate processes
on separate GPUs using `--encode-device-id` / `--decode-device-id`.

Standard input/output and named FIFOs support streaming. Human-readable status
is sent to stderr when binary output is stdout:

```bash
cat in.yuv \
  | ./bin/mlvc_demo encode --input - --output - --width 640 --height 360 \
      --frames 60 --q-index 21 --encode-device-id 0 2>encode.log \
  | ./bin/mlvc_demo decode --input - --output - --width 640 --height 360 \
      --frames 60 --decode-device-id 1 2>decode.log \
  > rec.yuv
```

The bitstream stores QP and payload length per frame. The decoder must be given
the same profile, original image dimensions, and GOP rules as the encoder.

## C API

The public API is `include/mlvc/codec.h`, with independent `mlvc_encode` and
`mlvc_decode` entry points. Options and statistics use **ABI version 2**. The old
options layout and its unused fields have been removed; rebuild applications.

```c
#include <mlvc/codec.h>

mlvc_codec_options options;
mlvc_codec_options_init(&options);
options.width = 640;
options.height = 360;
options.frames = 60;
options.q_index = 21;
options.device_id = 0;
options.input_path = "in.yuv";
options.output_path = "out.mlvc";

mlvc_codec_stats stats;
mlvc_codec_stats_init(&stats);
char error[512];
int result = mlvc_encode(&options, &stats, error, sizeof(error));
```

Use `find_package(mlvc_codec CONFIG REQUIRED)` and link `mlvc::mlvc_codec`.
Only the C header is installed; core, runtime, CUDA, CUTLASS, and AOT headers
remain private. Errors return `-1` and a diagnostic instead of crossing the ABI
as C++ exceptions. See [the Chinese integration guide](docs/integration-guide-zh.md).

## Verification and benchmarks

Run the host compiler checks in the model environment:

```bash
build/model-venv/bin/python tests/unit/aot_tests.py
build/model-venv/bin/python tests/unit/model_package_tests.py
```

To compare serialized AOT schedules with official eager execution on varied
inputs and Q indices 0/21/63, run
`tests/conformance/run_aot_conformance.py --checkpoint-root /path/to/checkpoints`.
This host check validates capture/lowering separately from CUDA arithmetic.

Generate official-Python reference fixtures directly from each checkpoint,
using the corresponding canonical bundle to identify the codec contract:

```bash
build/model-venv/bin/python tools/model_reference.py \
  --model-dir models/canonical/mlvc-psnr-v1/640x368 \
  --checkpoint /path/to/mlvc-psnr-v1.ckpt \
  --input input.yuv --width 640 --height 360 --frames 2 --q-index 21 \
  --output-dir models/fixtures/references/mlvc-psnr-v1/gray-q21-2f
```

Repeat for MLVC-S, then configure and build the formal target:

```bash
cmake --preset driver-cubin-release
cmake --build build/driver-cubin-release --target mlvc_codec_compatibility
```

`--debug-dir` dumps model inputs and outputs and disables device-state binding
for diagnosis. The required gate exercises two I/P frames; it does not establish
long-sequence, arbitrary-content, or rate-distortion acceptance. See the
[compatibility contract](docs/codec-compatibility.md).

`model_aot.py` also writes `benchmark-MLVC*.npz` snapshots. Prepare a benchmark
with `tools/benchmark.py make-case`, run `mlvc_backend_bench --model-dir
embedded:<profile> --case <case.json>`, and compare results using
`tools/benchmark.py compare`. Model host-roundtrip timing and full codec timing
measure different workloads and must be reported separately.

## Source map

```text
public/include/mlvc/codec.h  Public C API
codec/                      C API implementation and frame pipeline
core/                       Model metadata, tensors, YUV, PMF/rANS, containers
runtime/                    Graph execution and buffered codec contracts
backends/driver_cubin/      AOT loader, planner, Driver API, CUDA kernels
tools/model_source.py       Official checkpoint loading and split-model inputs
tools/model_aot.py          Direct checkpoint-to-AOT compiler
tools/model_package.py      Canonical bundle verification and embedding
tools/model_reference.py    Official eager-Python reference generation
tools/cli/                  Encode/decode command-line entry point
tools/benchmark/            FP16 graph latency and tensor parity measurement
tests/                      Host safety, C API, AOT, and codec conformance tests
models/profiles/            Registered checkpoint identities and model parameters
models/canonical/           Generated, ignored canonical model bundles
models/fixtures/            Generated, ignored test fixtures and references
```

See [design notes](docs/design.md) and
[Driver+cubin architecture](docs/nvngx-dlss-cubin-architecture.md).

The [driver-only validation record](docs/validation/2026-11-12-driver-only.md)
documents the exercised hardware, fixtures, checks, and acceptance limits.

## Acknowledgements

Microsoft MLVC supplies the model implementation and `msrtc_rans` source (MIT).
NVIDIA CUTLASS supplies CUDA kernel building blocks under its upstream license.
