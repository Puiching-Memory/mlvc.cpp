# Verification & Compatibility Contract

Passing a model microbenchmark alone does not establish codec compatibility. This document defines the repository's test matrix, the normative end-to-end compatibility contract (`mlvc_codec_compatibility`), and explicit acceptance boundaries.

---

## 1. Test Suite Matrix

### C++ / C Tests (`CTest`, registered in `tests/CMakeLists.txt`)

| CTest Target               | Labels                                | Scope                                                                                                                                            |
| -------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mlvc_core_safety`         | *(portable)*                          | Bounds checking, malformed inputs, YUV/scale/rANS safety; runs under ASan + UBSan via `driver-cubin-sanitize`                                    |
| `mlvc_bitstream`           | *(portable)*                          | `mlvc-nalu-v0` NALU Annex-B builder/parser, EBSP `0x03`, SPS/PPS/VCL syntax, frozen `microsoft/libmlvc` conformance vectors, QP `[0, 51]` checks |
| `mlvc_model_proto`         | *(portable)*                          | Protobuf Lite serialization/deserialization, schema validation, corrupt/oversized message rejection                                              |
| `mlvc_runtime_decoupling`  | *(portable)*                          | `ModelAssetReader`, `EncoderSession`/`DecoderSession`, backend registry, `ExecutionPreference`, mock backend isolation                           |
| `mlvc_c_api`               | *(portable)*                          | Public C ABI v2 struct sizing, version rejection, error buffer handling, null guards                                                             |
| `mlvc_public_header`       | *(portable)*                          | Pure C11 compilation check of `public/include/mlvc/codec.h`                                                                                      |
| `mlvc_cpu_backend`         | *(when `cpu`/`all`)*                  | CPU execution plan, ISA microkernels, fusion vs. unfused equivalence, `CpuThreadPool`                                                            |
| `mlvc_driver_cubin_probe`  | `gpu`                                 | CUDA Driver API init, fatbin load, probe kernel launch (`--iterations 100`)                                                                      |
| `mlvc_cutlass_kernels`     | `gpu`                                 | CUTLASS Tensor Core convolution correctness on GPU                                                                                               |
| `mlvc_model_package`       | *(when Python protobuf ok)*           | Runs `tests/unit/model_package_tests.py` (bundle assembly, SHA-256 verification, embedding)                                                      |
| `mlvc_codec_reference`     | *(when fixtures exist)*               | Portable C++ reference parity (`mlvc_codec_tests`): PMF/rANS, 64-frame GOP wrap, I-frame reset schedule                                          |
| `mlvc_codec_conformance_*` | `gpu;conformance` / `cpu;conformance` | End-to-end `mlvc_demo` conformance against official Python references for `mlvc-psnr-v1` and `mlvc-s-psnr-v1`                                    |

### Offline Python Compiler Tests

```bash
build/model-venv/bin/python tests/unit/aot_tests.py
build/model-venv/bin/python tests/unit/model_package_tests.py
```

### Host AOT-vs-Eager Graph Conformance

Compares serialized AOT graph execution on the host directly against official PyTorch eager execution across varied inputs and Q indices `0`, `21`, and `63` — isolating capture/lowering correctness from CUDA/CPU kernel arithmetic:

```bash
build/model-venv/bin/python tests/conformance/run_aot_conformance.py \
  --checkpoint-root /path/to/checkpoints
```

### CI Pipeline (`.github/workflows/ci.yml`)

GitHub Actions runs on `ubuntu-26.04` for every push and PR to `main`:

1. **`model-tools`**: Python 3.12 + PyTorch 2.10.0 CPU + `protobuf==6.33.5` running `aot_tests.py` and `model_package_tests.py`.
2. **`package` (matrix: GCC and Clang)**: Installs CUDA Toolkit 13.3, runs `tools/scripts/package.sh --skip-gpu-tests --skip-models` with `-Werror`, audits ELF linkage, and uploads compile-only `-ci.tar.gz` artifacts.
3. **Host sanitizers (GCC)**: Builds and executes `driver-cubin-sanitize` (`mlvc_core_safety` with ASan + UBSan).

---

## 2. Normative Codec Compatibility Contract

Both `driver-cubin` and `cpu` runtimes must pass `mlvc_codec_compatibility` for **both** registered profiles (`mlvc-psnr-v1` and `mlvc-s-psnr-v1`).

### Identity & Integrity Rules

- Canonical bundles must be assembled by `python -m mlvc_tools.package assemble` (bundle schema **2**, AOT graph schema **3**) from the original checkpoints registered in `models/profiles/profiles.json`.
- `metadata.pb`, `gaussian_pmf.pb`, and `bit_estimator_pmf.pb` must originate from FP32 parameters of the same checkpoint.
- `tests/conformance/run_codec_conformance.py` rejects any profile identity, checkpoint hash, or bundle artifact-hash mismatch before running inference.

### Bitstream & Entropy Rules

- Container identifier: **`mlvc-nalu-v0`** (Annex-B 4-byte start codes, EBSP `0x03`, `SPS` v0.1 + `PPS` before every `IDR_N_LP`, `TRAIL_R`/`TSA_R` inter frames, QP strictly in `[0, 51]`).
- Entropy identifier: **`canonical-pmf-v1`**. Given identical integer-valued `z_raw`, `y_raw_0`, and `y_raw_1`, C++ and Python must produce identical rANS payloads and decode identically.
- **I-frame bit-exactness**: Frame 0's complete Access Unit (`SPS` + `PPS` + `IDR_N_LP` bytes) must be **byte-for-byte identical** to the official Python reference.
- **Recurrent P-frame rule**: P-frame bitstream bytes are not required to match Python byte-for-byte because legal FP16 rounding differences in the preceding DPB feature can cross an integer quantization boundary. A P-frame stream conforms iff it decodes cleanly with the canonical PMFs and meets all reconstruction and tensor tolerances below.

### Reconstruction & Tensor Tolerances

The standard conformance fixture consists of two `640×360` YUV420 frames padded to `640×368` at `q_index = 21`: frame 0 (I-frame, zero reference) and frame 1 (P-frame, recurrent DPB).

Every conforming runtime must satisfy **all** of the following:

1. Decode the canonical Python bitstream and self-round-trip its own encoded bitstream (both with explicit `--width`/`--height` and auto-detected from `SPS`).
2. Preserve exact frame count, unshifted frame QP in `[0, 51]`, display dimensions, `frame_type`, and `cur_frame_idx`.
3. **YUV reconstruction** (both canonical decode and self round-trip vs. Python):
   - Maximum YUV sample error: **`<= 1`** (on `[0, 255]`)
   - Per-frame PSNR: **`>= 60 dB`**
4. **Exact integer & input tensors**: All `int32` control tensors (`q_index_shifted`), preprocessed `x` tensors, and frame-0 quantized latents (`z_raw`, `y_raw_0`, `y_raw_1`) must match exactly.
5. **FP16 intermediate & output tensors** (`feature`, `ref_feature`, `x_hat`, including frame-1 recurrent DPB consumption):
   - Maximum absolute error: **`<= 0.25`**
   - RMSE: **`<= 0.007`**

P-frame quantized latents are logged for diagnostics rather than tolerance-gated (a single rounding-boundary flip is integer-valued); their correctness is enforced by rANS decodability and round-trip reconstruction quality.

---

## 3. Running Conformance & Diagnostics

### Step 1 — Generate Official Python Reference Fixtures

Generate references for both `mlvc-psnr-v1` and `mlvc-s-psnr-v1` directly from the registered checkpoints:

```bash
build/model-venv/bin/python -m mlvc_tools.reference \
  --model-dir models/canonical/mlvc-psnr-v1/640x368 \
  --checkpoint /path/to/mlvc-psnr-v1.ckpt \
  --input input.yuv --width 640 --height 360 --frames 2 --q-index 21 \
  --output-dir models/fixtures/references/mlvc-psnr-v1/gray-q21-2f

build/model-venv/bin/python -m mlvc_tools.reference \
  --model-dir models/canonical/mlvc-s-psnr-v1/640x368 \
  --checkpoint /path/to/mlvc-s-psnr-v1.ckpt \
  --input input.yuv --width 640 --height 360 --frames 2 --q-index 21 \
  --output-dir models/fixtures/references/mlvc-s-psnr-v1/gray-q21-2f
```

### Step 2 — Run `mlvc_codec_compatibility`

```bash
# GPU (driver-cubin) build
cmake --preset driver-cubin-release
cmake --build build/driver-cubin-release --target mlvc_codec_compatibility
# Or directly via CTest labels:
ctest --preset driver-cubin-release -L conformance

# CPU build (or cross-compiled with CMAKE_CROSSCOMPILING_EMULATOR)
cmake -S . -B build/cpu-release -DCMAKE_BUILD_TYPE=Release -DMLVC_BACKEND=cpu -DBUILD_TESTING=ON
cmake --build build/cpu-release --target mlvc_codec_compatibility
```

### Diagnostics & Auditing

- **Audit JSON report**: Run `tests/conformance/run_codec_conformance.py --result result.json ...` to record payload sizes, per-frame YUV metrics, and per-tensor errors.
- **Long-sequence diagnostic mode**: Pass `--diagnostic` with a longer generated fixture to record every frame's metrics without stopping on the first violation (`contract_passed=false` is still emitted; this mode never relaxes the normal gate).
- **Per-frame tensor dumps (`--debug-dir`)**: Pass `--debug-dir DIR` to `mlvc_demo` to write named input/output tensors per frame. Remember that `--debug-dir` disables device/backend-resident state binding and must not be used for performance measurements.
- **CPU A/B verification**: Combine `MLVC_CPU_ISA=scalar|neon_fp32|fp16fml|avx2` and `MLVC_CPU_DISABLE_FUSION=1` to isolate SIMD microkernel or graph-fusion discrepancies.

---

## 4. Explicit Acceptance Boundaries

Passing the required two-frame I/P compatibility gate validates the full codec pipeline (preprocessing, QP shift, neural graphs, DPB recurrence, scale extraction, rANS, NALU Annex-B container, and reconstruction) on the exercised hardware and fixture. It does **not** by itself establish:

- Long-sequence drift stability across diverse real-world video content
- Visual quality or rate-distortion (BD-rate) acceptance across arbitrary datasets
- Throughput or latency guarantees on untested GPU architectures or mobile SoCs
- Unmodified validity of historical timing numbers after AOT graph or kernel changes — always measure complete end-to-end codec latency on the target hardware after verifying conformance
