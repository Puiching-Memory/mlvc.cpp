# Model Pipeline & Binary Format

`mlvc.cpp` compiles official Microsoft MLVC PyTorch checkpoints **offline** directly into fixed-shape Protobuf AOT schedules and aligned FP16 weight blobs. No intermediate ONNX/TorchScript model file is produced, and no ML framework is linked into the deployed C++ codec.

---

## 1. Profile Registry (`models/profiles/profiles.json`)

[`models/profiles/profiles.json`](../models/profiles/profiles.json) is the single source of truth binding each profile name to its official checkpoint SHA-256 hash and upstream network construction parameters:

| Profile          | `model_version`       | `split_type`    | Checkpoint SHA-256 Prefix |
| ---------------- | --------------------- | --------------- | ------------------------- |
| `mlvc-psnr-v1`   | `dmc61sbr_reglu`      | `dmc61sbr_e1d1` | `834adaab680837c4...`     |
| `mlvc-s-psnr-v1` | `dmc61sbr_mini_reglu` | `dmc61sbr_e1d1` | `1b86b757ddb11534...`     |

`profiles.json` is human-readable JSON consumed **only** by offline Python tooling (`mlvc_tools`). No C++ target links a JSON library.

---

## 2. Offline Compilation Workflow

### Environment Setup

```bash
uv venv --python 3.12 build/model-venv
uv pip install --python build/model-venv/bin/python -r tools/python/requirements-models.txt
uv pip install --python build/model-venv/bin/python -e ./tools/python
uv pip install --python build/model-venv/bin/python ./third_party/mlvc/packages/msrtc_rans
source build/model-venv/bin/activate
```

Python model tools require **`protobuf==6.33.5`** (matching C++ Protobuf `v33.5`) and PyTorch (`2.10.0` CPU is sufficient).

### Step-by-Step Pipeline

1. **Direct Checkpoint AOT Compilation (`mlvc_tools.aot`)**:
   - `mlvc_tools/source.py` verifies the checkpoint SHA-256 against `profiles.json`, constructs the official `microsoft/mlvc` split network (`MLVCEncoder` and `MLVCDecoder`), applies upstream structural optimization, and extracts canonical PMF tables (`gaussian_pmf.pb`, `bit_estimator_pmf.pb`) from **FP32** parameters before converting weights to FP16.
   - `mlvc_tools/aot.py` runs one fixed-shape eager FP16 invocation inside a PyTorch `TorchDispatchMode` tracer:
     - Folds constant-only subgraphs offline.
     - Prunes dead branches not reachable from graph outputs.
     - Lowers every live op to the 14-operator runtime contract (unsupported live ops fail explicitly).
     - Runs an interval-based static **arena lifetime planner** with 256-byte alignment (`ALIGNMENT = 256`), reusing memory slots once tensors die (`birth`..`death`).
     - Writes deterministic binary Protobuf graphs (`graph.pb`) and aligned FP16 weight blobs (`weights.bin`).

   ```bash
   python -m mlvc_tools.aot \
     --profile mlvc-psnr-v1 \
     --checkpoint /path/to/mlvc-psnr-v1.ckpt \
     --model-width 640 --model-height 368 \
     --output-dir models/generated/checkpoint/mlvc-psnr-v1/640x368
   ```

2. **Canonical Bundle Assembly (`mlvc_tools.package assemble`)**:
   - Copies the 8 required artifacts into a canonical bundle directory, validates cross-file identity and checkpoint provenance, computes SHA-256 digests and sizes, and writes `model_bundle.pb`.

   ```bash
   python -m mlvc_tools.package assemble \
     --profile mlvc-psnr-v1 \
     --source-dir models/generated/checkpoint/mlvc-psnr-v1/640x368 \
     --output-dir models/canonical/mlvc-psnr-v1/640x368
   ```

3. **Integrity Verification (`mlvc_tools.package verify`)**:

   ```bash
   python -m mlvc_tools.package verify \
     --model-dir models/canonical/mlvc-psnr-v1/640x368
   ```

Repeat steps 1–3 for `mlvc-s-psnr-v1`. Both canonical bundles (`models/canonical/mlvc-psnr-v1/640x368` and `models/canonical/mlvc-s-psnr-v1/640x368`) are required when building with `MLVC_EMBED_MODELS=ON`.

---

## 3. Canonical Bundle Layout

A verified canonical bundle directory contains `model_bundle.pb` plus 8 hashed payload files:

| File                          | Protobuf Message / Format              | Role                                                                                                                                                 |
| ----------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_bundle.pb`             | `mlvc.wire.ModelBundle` (schema **2**) | Profile identity, dimensions, checkpoint & entropy hashes, `CodecContract`, SHA-256 + byte size of all 8 artifacts                                   |
| `metadata.pb`                 | `mlvc.wire.ModelMetadata` (schema 1)   | Profile, model version, canvas dimensions, `pixel_range`, `qp_num`, `frame_index_map`, `qp_shift`, channel counts, downsampling factors, GOP periods |
| `gaussian_pmf.pb`             | `mlvc.wire.GaussianPmf` (schema 1)     | Quantized Gaussian PMF table for `y` coding (`scale_min`, `scale_max`, `scale_levels`, `index_space`)                                                |
| `bit_estimator_pmf.pb`        | `mlvc.wire.BitEstimatorPmf` (schema 1) | Quantized bit-estimator PMF table for `z` coding (`qp_num`, `channels`)                                                                              |
| `aot/manifest.pb`             | `mlvc.wire.AotManifest` (schema 1)     | Checkpoint SHA-256 and per-graph summary (`node_count`, `weights_bytes`, `arena_bytes`)                                                              |
| `aot/MLVCEncoder/graph.pb`    | `mlvc.wire.Graph` (schema **3**)       | Encoder tensor descriptors, integer tensor IDs, operator nodes, arena size & alignment, weight SHA-256                                               |
| `aot/MLVCEncoder/weights.bin` | Raw binary (256-byte aligned)          | Encoder FP16/int32/int64 constant & weight payloads                                                                                                  |
| `aot/MLVCDecoder/graph.pb`    | `mlvc.wire.Graph` (schema **3**)       | Decoder tensor descriptors, operator nodes, arena layout, weight SHA-256                                                                             |
| `aot/MLVCDecoder/weights.bin` | Raw binary (256-byte aligned)          | Decoder constant & weight payloads                                                                                                                   |

---

## 4. Protobuf Schema & Operator Contract (`proto/mlvc_model.proto`)

[`proto/mlvc_model.proto`](../proto/mlvc_model.proto) (`syntax = "proto3"`, `package mlvc.wire`, `option optimize_for = LITE_RUNTIME`) defines the shared wire contract:

- **Schema versions**:
  - `ModelBundle.schema_version = 2`
  - `Graph.schema_version = 3` (native channel-first `DEPTH_TO_SPACE` / `SPACE_TO_DEPTH` CRD ordering retaining original PyTorch weight channel order; earlier schemas and JSON bundles are rejected)
- **Data types (`DataType`)**: `FP16`, `INT32`, `INT64` (neural activations/weights use FP16; model control inputs use `INT32`; slice/index constants may use `INT64`).
- **Tensor addressing**: Graph edges use dense `uint32` `TensorId`s (`0` reserved for absent optional operands such as bias-free convolutions). Each `TensorDescriptor` has storage `oneof`: `ArenaAllocation { offset, size_bytes, birth, death }` or `WeightLocation { offset, size_bytes }`.
- **Supported Operator Set (`OpKind`, 14 ops)**:
  - Arithmetic / Elementwise: `ADD`, `SUB`, `MUL`, `CLIP`, `RECIPROCAL`, `ROUND`
  - Activations: `LEAKY_RELU` (`ActivationAttributes.alpha`), `SIGMOID`
  - Convolutions: `CONV` (`ConvAttributes { strides, pads, dilations, groups }`)
  - Tensor / Layout: `CONCAT` (`AxisAttributes.axis`), `SLICE`, `GATHER` (`AxisAttributes.axis`), `DEPTH_TO_SPACE` (`PixelAttributes.block_size`), `SPACE_TO_DEPTH` (`PixelAttributes.block_size`)
- **Zero Protobuf in hot path**: `runtime/src/aot_graph.cpp` parses `graph.pb` once during model load into native `mlvc::aot::LoadedGraph` structs and validates weight SHA-256 and message size limits (64 MiB max). No Protobuf objects remain during frame inference.

### Protobuf Build & Python Binding Regeneration

- CMake (`cmake/ModelProto.cmake`) compiles both build-host `protoc` and `libprotobuf-lite` + `Abseil` from the pinned submodules (`third_party/protobuf` v33.5, `third_party/abseil` 20250512.1) with hidden visibility and static linkage.
- Installed SDK consumers need **no** system Protobuf library or headers.
- Whenever `proto/mlvc_model.proto` is modified, regenerate checked-in Python bindings using the exact same compiler:

```bash
cmake --build build/<preset> --target mlvc_generate_python_proto
```

---

## 5. ELF Model Embedding (`MLVC_EMBED_MODELS`)

When `MLVC_EMBED_MODELS=ON` (default for `driver_cubin` and `all`; optional on ELF `cpu` targets):

1. `core/CMakeLists.txt` discovers every `${MLVC_MODEL_ROOT}/*/*/model_bundle.pb`.
2. `python tools/python/mlvc_tools/package.py embed` verifies both registered profiles and packs all assets (16-byte aligned) into `mlvc_driver_models.bin` plus a compile-time index table `mlvc_embedded_models_index.inc` and dependency file.
3. `cmake/EmbedModelData.S.in` embeds the binary blob into `.rodata` via GNU/Clang `.incbin` (`mlvc_embedded_models_start`..`mlvc_embedded_models_end`).
4. `core/src/model_assets.cpp` exposes embedded profiles through `embedded:<profile>` (default `embedded:mlvc-psnr-v1`) and `list_embedded_model_profiles()`.

> **Platform requirement**: `MLVC_EMBED_MODELS=ON` requires an ELF target (`Linux`, `Android`, or `OHOS`) with GNU/Clang ASM `.incbin`. On Windows/macOS or when packaging lean mobile libraries, leave `MLVC_EMBED_MODELS=OFF` and load models from a directory bundle or custom `ModelAssetReader`.
