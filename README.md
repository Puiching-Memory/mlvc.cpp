# mlvc-cpp-demo

Portable C++/ONNX Runtime implementation of the **MLVC** (Microsoft Learned Video
Codec, DMC-6.1sb) encode/decode pipeline, targeting a single self-contained
command-line binary.

The demo mirrors the behaviour of the official MLVC split-model deployment path
(`video/conversion`): the neural parts run as exported ONNX graphs via ONNX
Runtime, while quantization and entropy coding (rANS) run in portable C++ using
the official `msrtc_rans` library.

> Status: **initial scaffolding**. Repository layout, build system and design
> notes are in place; the pipeline implementation is in progress.

## Features (planned)

- Encode YUV420 (8-bit) to the MLVC bitstream format
- Decode MLVC bitstream back to YUV420
- Runs with ONNX Runtime CPU and CUDA execution providers
- Portable binary: statically links `msrtc_rans`, dynamically loads
  `libonnxruntime.so` with `$ORIGIN` rpath

## Dependencies

| Dependency | Role | Notes |
|---|---|---|
| ONNX Runtime (>= 1.19) | Inference | CPU or GPU package; NOT vendored (see `scripts/fetch_onnxruntime.sh`) |
| `msrtc_rans` | rANS entropy coding | Vendored under `third_party/msrtc_rans` (MIT, from microsoft/mlvc) |
| nlohmann/json | JSON parsing (PMF tables / metadata) | Fetched at build time |
| CMake >= 3.23, C++17 compiler | Build | |

MLVC exported model artifacts (`MLVCEncoder.onnx`, `MLVCDecoder.onnx`,
`gaussian_pmf.json`, `bit_estimator_pmf.json`, `metadata.json`) are produced by
the official converter and supplied at runtime via `--model-dir`.

## Build

```bash
# 1. Fetch ONNX Runtime (CPU edition by default; add --gpu for the CUDA build)
./scripts/fetch_onnxruntime.sh

# 2. Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT="$PWD/third_party/onnxruntime/onnxruntime-linux-x64-1.19.2"
cmake --build build -j

# 3. (Optional) assemble a portable package
cmake --install build --prefix dist
```

## Usage (planned)

```bash
./mlvc_demo encode --input in.yuv --width 640 --height 360 --frames 60 \
                   --q-index 21 --model-dir models/640x368 --output out.mlvc
./mlvc_demo decode --input out.mlvc --width 640 --height 360 --frames 60 \
                   --model-dir models/640x368 --output rec.yuv
```

## Repository layout

```
CMakeLists.txt               Top-level build
src/                         C++ sources (main + pipeline modules)
include/                     Public headers of the demo library
third_party/msrtc_rans/      Vendored rANS entropy coder (MIT)
third_party/json/            nlohmann/json (fetched by CMake)
third_party/onnxruntime/     ONNX Runtime (fetched by script, git-ignored)
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
