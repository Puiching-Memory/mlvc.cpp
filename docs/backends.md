# Inference Backends

`mlvc.cpp` ships two first-class inference backends that both consume the exact same Protobuf AOT graph (schema 3) and FP16 weights, and both implement `InferenceBackend` and `BufferedCodecBackend`:

| Backend        | Target                                      | Runtime Dependency                  | Accumulation / Storage                    | Best For                                                                 |
| -------------- | ------------------------------------------- | ----------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| `driver-cubin` | Linux `x86_64` NVIDIA GPUs (`sm_75+`)       | System `libcuda.so.1` only          | FP16 tensors, FP32 conv accumulation      | High-throughput GPU servers & workstations                               |
| `cpu`          | Linux, Windows, macOS, Android, OpenHarmony | Standard C++20 + `Threads::Threads` | FP16 tensors, FP32 GEMM/conv accumulation | Mobile (`arm64-v8a`), laptops, driverless servers, portable verification |

---

## 1. `driver-cubin` Backend (NVIDIA GPU)

### Deployment Architecture

`driver-cubin` follows an ahead-of-time monolithic kernel-runtime boundary (the same deployment pattern used by standalone GPU inference libraries such as NGX/DLSS):

1. **Build-host-only CUDA toolchain**: CUDA Toolkit (`13.3`) and NVIDIA CUTLASS are used exclusively during the build to compile fixed-shape kernels into a multi-architecture fatbin.
2. **Precompiled fatbin embedded in `libmlvc_codec.so`**: [`backends/driver_cubin/tools/build_fatbin.sh`](../backends/driver_cubin/tools/build_fatbin.sh) compiles [`backends/driver_cubin/kernels/module.cu`](../backends/driver_cubin/kernels/module.cu) into cubins for `sm_75`, `sm_80`, `sm_86`, and `sm_89` plus a `compute_89` PTX forward-compatibility image, embedded as a read-only C array via `cmake/EmbedBinary.cmake`.
3. **Driver API only**: At runtime, the library links `CUDA::cuda_driver` (`libcuda.so.1`) — never `libcudart`, `libcudnn`, `libcublas`, `libnvinfer`, `libonnxruntime`, or `libtorch`. Modules are loaded with `cuModuleLoadData`, kernels resolved via `cuModuleGetFunction`, and launched via `cuLaunchKernel`.
4. **Device-resident state & CUDA Graphs**: Weights, the static tensor arena, and bound recurrent DPB features stay permanently on the GPU. After loading and planning, the execution schedule is captured once into a CUDA Graph and replayed per frame.
5. **Double-buffered GPU codec I/O**: Through `BufferedCodecBackend`, `driver-cubin` allocates two pinned host slots and runs YUV420↔FP16-YUV444 conversion, edge padding, cropping, and optional rotation directly on the GPU (`kernels/codec_io.cu`), overlapping CPU rANS coding with GPU inference.

> **Note**: Unlike libraries that dynamically fall back when `libcuda.so.1` is absent, a `driver-cubin`-linked binary intentionally declares `DT_NEEDED libcuda.so.1`. Use `-DMLVC_BACKEND=cpu` for driverless deployments or `-DMLVC_BACKEND=all` if you want both backends available on Linux `x86_64`.

### Internal Module Boundaries

All implementation details under `backends/driver_cubin/` stay private behind `aot_graph.hpp` and `backend.cpp`:

| Module                    | Responsibility                                                                                 |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| `backend.cpp`             | Registers `"driver-cubin"` and implements `InferenceBackend` / `BufferedCodecBackend`          |
| `driver.cpp`              | RAII wrappers for CUDA Driver contexts, streams, events, modules, allocations, and CUDA Graphs |
| `model_loader.cpp`        | Loads Protobuf AOT graphs/weights via `ModelAssetReader`, uploads weights, allocates arena     |
| `graph_ir.cpp`            | Typed graph value, dtype, shape, and scalar accessors                                          |
| `graph_validator.cpp`     | Validates state bindings, slice bounds, consumer counts, and arena alias ranges                |
| `memory_planner.cpp`      | Plans input slice aliases and reusable epilogue, ReGLU, spatial, and direct-concat workspaces  |
| `kernel_registry.cpp`     | Resolves fatbin kernel entry points and resource attributes                                    |
| `fusion_registry.cpp`     | Validates exact multi-node dependency chains and dispatches fused kernels                      |
| `execution_plan.cpp`      | Lowers matched chains and unmatched single nodes to validated kernel launches                  |
| `cuda_graph_executor.cpp` | Owns pinned staging, state zeroing, CUDA Graph capture/replay, and double-buffered codec slots |

The lifecycle is strictly `load -> validate/plan -> capture -> replay`; no graph loading, validation, or memory planning occurs during steady-state frame execution.

### Kernel Suite (`backends/driver_cubin/kernels/`)

`module.cu` aggregates focused compilation units tracked as CMake dependencies:

- **Convolutions**: `cutlass.cu` (SM80+ Tensor Core spatial & pointwise GEMM via vendored CUTLASS), `pointwise_conv.cu`, `spatial_conv.cu`, `depthwise_conv.cu`, `generic_conv.cu`
- **Codec & Tensor Ops**: `codec_io.cu` (GPU YUV420/444 padding/crop), `layout.cu` (NCHW/NHWC/KRSC transposes), `tensor_ops.cu` (native channel-first `DepthToSpace`/`SpaceToDepth`, `Slice`, `Concat`, `Gather`), `elementwise.cu`, `activation.cu`
- **Fused Subgraphs**: `recurrent_feature.cu`, `quantization_tail.cu`
- **Diagnostics**: `probe.cu` (exercised by `mlvc_driver_probe`)

### Fusion Schedule & Empirical Design Lessons

In `execution_plan.cpp`, `AotGraph::execute_schedule()` matches fused subgraphs in priority order before falling back to single-node dispatch:

1. **`mlvc_y0_tail_fp16`** (19 nodes, encoder): Fuses first prior split, `y0` normalization/quantization, `y0` prior update, and two trailing `Concat`s, while writing normalized and clipped side outputs kept live for `y1`. Gated on `quant_channels >= 32` so `mlvc-s-psnr-v1` (24 `y` channels) stays on the generic schedule where its smaller workload does not amortize the fused kernel footprint.
2. **`mlvc_y1_tail_fp16`** (15 nodes, encoder): Fuses `y1` quantization, `y1` prior update, residual add, and final scale multiply, writing `y_raw_1` and the decoder input directly.
3. **`mlvc_feature_update_outputs_fp16`** (11 nodes, decoder): Fuses the `Slice`/`Sigmoid`/binary recurrent update chain and writes all three outputs (`scaled`, `blended`, `feature`) in one launch, eliding the deferred `Concat` and its two copy launches. When `feature` is bound to `ref_feature`, it writes directly to the persistent state buffer; host-visible debug runs use a dedicated output buffer to avoid clobbering still-live reconstruction inputs.
4. **`mlvc_feature_update_fp16`** (12 nodes, encoder): Fuses the encoder recurrent-feature tail (`Slice`/`Sigmoid`/binary/`Concat`) in place into the next recurrent state.
5. **Pointwise + ReGLU** (5 nodes), **Pointwise + Epilogue** (2 nodes), and **Standalone ReGLU** (4 nodes).

**Numeric & performance rules verified on NVIDIA A30**:

- **Exact FP16 materialization**: Every fused kernel preserves intermediate FP16 rounding points (`Clip`, `Reciprocal`, `Add`, `Mul`, `Sub` before `Round`), keeping fused outputs byte-identical to the unfused graph.
- **Measure before keeping fusions**: Several plausible optimizations regressed on A30 and were intentionally rejected:
  - Replacing CUTLASS pointwise + vectorized ReGLU with a single monolithic Conv+ReGLU kernel regressed encoder latency by ~4%.
  - Reducing the hottest CUTLASS pointwise pipeline from 4 stages to 2 increased register pressure (228 → 248 regs/thread) and regressed latency by ~2.5%.
  - Fusing `DepthToSpace(block=8)+Clip` yielded only ~0.09% gain while requiring an extra 1.35 MiB persistent buffer due to lifetime overlap, so it was removed.

#### Environment Tuning

| Variable                | Effect                                                                                                   |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| `MLVC_CUTLASS_LOG_TILE` | Opt-in L2 threadblock swizzle `log_tile` (clamped to `[0, 3]`, default `0`) for CUTLASS pointwise GEMMs. |

---

## 2. `cpu` Backend (Cross-Platform FP16 CPU)

The `cpu` backend executes the same Protobuf AOT graphs across Linux, Windows, macOS, Android, and OpenHarmony with zero external BLAS or runtime library dependencies.

### Layout, Precision & Execution Model

- **FP16 storage, FP32 accumulation**: All neural activations and weights remain FP16 in memory (`Float16Storage`) while GEMM and convolution microkernels accumulate in FP32 and round back to FP16 at node boundaries — matching GPU precision and passing the same strict codec conformance thresholds.
- **Internal NHWC layout**: 4D FP16 activations (`C > 1`, `H*W > 1`) are executed in 64-byte-aligned NHWC inside the CPU arena with boundary NCHW↔NHWC transposes, enabling contiguous channel vectorization and direct GEMM lowering for 1×1 pointwise convolutions and `im2col`/direct spatial convolutions.
- **Weight pre-packing**: During `load()`, convolution weights and biases are packed once into microkernel-friendly tile layouts (for example, 16-column `N`-blocks for ARM64 4×16 kernels).
- **Buffered Codec I/O**: Implements `BufferedCodecBackend` with persistent YUV420 buffers and vectorized pre/post-processing so `EncoderSession` / `DecoderSession` and `encode_video` / `decode_video` share the same fast path as `driver-cubin`.

### Runtime ISA Dispatch & Microkernels

`backends/cpu/src/cpu_features.cpp` probes hardware capabilities once at startup (via `getauxval(AT_HWCAP)` on Linux/Android/OHOS ARM64, `__builtin_cpu_supports` or `__cpuid`/`_xgetbv` on `x86_64`, and compile-time guarantees on Apple Silicon):

| Mode (`gemm_mode_name`) | Hardware Requirement                                                                             | Source Units                                                                         | Key Characteristics                                                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `arm64-fp16fml`         | AArch64 `HWCAP_FPHP \| HWCAP_ASIMDHP \| HWCAP_ASIMDFHM` (ARMv8.2-A `+fp16fml`, or Apple Silicon) | `kernels_arm64_fp16fml.cpp`, `kernels_arm64_fp16fml_asm.S`, `kernels_arm64_fp16.cpp` | Hand-written 4×16 `FMLAL`/`FMLAL2` assembly microkernel with FP32 accumulators, ping-pong double-buffered weight loads (`v0..v1`/`v2..v3`), and zero-buffer in-register ReGLU fusion (`v24..v31`) |
| `arm64-neon-fp32`       | Baseline ARMv8.0-A AArch64 NEON (no `+fp16`/`+fp16fml` required)                                 | `kernels_arm64_neon.cpp`, `kernels_arm64_neon_asm.S`                                 | Hand-written 4×16 FP16→FP32 assembly microkernel with ping-pong double-buffered weight loads/conversions and exact arbitrary `(M, N, K)` tail handling                                            |
| `x86-avx2-f16c`         | `x86_64` AVX2 + FMA + F16C                                                                       | `kernels_x86_avx2.cpp`                                                               | AVX2 FP16↔FP32 (`_mm256_cvtph_ps` / `_mm256_cvtps_ph`) GEMM, depthwise/spatial conv, and fast polynomial `exp`/`sigmoid` kernels                                                                  |
| `scalar-fp32`           | Any C++20 CPU                                                                                    | `kernels_scalar.cpp`                                                                 | Portable reference implementation and fallback                                                                                                                                                    |

**ARM64 security hardening (BTI / PAC)**: Both AArch64 assembly files (`kernels_arm64_neon_asm.S` and `kernels_arm64_fp16fml_asm.S`) emit pointer authentication hints (`paciasp`/`autiasp` encoded as pre-ARMv8.3-compatible `hint` instructions), `.note.GNU-stack`, and ELF `.note.gnu.property` (`GNU_PROPERTY_AARCH64_FEATURE_1_AND = BTI | PAC`) so shared libraries link cleanly on modern Android, OpenHarmony, and hardened Linux toolchains.

### Heterogeneous Big.LITTLE Thread Pool (`thread_pool.cpp`)

Mobile ARM64 SoCs mix prime, performance, and low-clock efficiency (LITTLE) cores where uniform static work partitioning causes severe barrier stalls. `CpuThreadPool` addresses this in three ways:

1. **Auto thread count (`num_threads <= 0`)**: On Linux/Android/OHOS, reads `/sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq` and counts only fast cores (`freq >= 60%` of peak frequency, capped at 16), avoiding spawning workers that get pinned to slow LITTLE cores.
2. **Topology-aware dynamic chunking**: When `min_freq < 85% * max_freq`, `probe_cpu_topology()` classifies cores as LITTLE (`< 80%` peak) or prime (`>= 95%` peak). During parallel loops, `sched_getcpu()` lets big/prime cores claim 2×–3× larger chunks when ample work remains, and reserves the final work tail (`min(total/3, grain*12)`) exclusively for fast threads whenever any fast thread is still active.
3. **Low-latency spin/sleep barrier**: The caller thread participates directly as worker 0; workers spin briefly (`16000` relax iterations using `yield` on ARM64 / `_mm_pause` on `x86_64`) before sleeping on a condition variable, eliminating futex wakeups across consecutive layers within a frame.

### CPU Graph Fusion (`fusion_matcher.cpp`)

`build_fused_schedule()` runs a topological pattern matcher before execution plan lowering, matching in priority order:

1. `kY0Tail` and `kY1Tail` (quantization & dual-prior tails)
2. `kFeatureUpdate` / `kFeatureUpdateOutputs` (encoder/decoder recurrent feature tails, eliding deferred `Concat`)
3. `kConvWithEpilogue` with `ReGLU` (pointwise conv + ReGLU fused directly in registers on ARM64 `+fp16fml`)
4. Standalone `kReglu`
5. `kConvWithEpilogue` (`Clamp`, `LeakyReLU`, `ResidualAdd`)
6. `kSingleNode` fallback

#### Environment Tuning & Testing

| Variable                  | Values                                            | Effect                                                                                                         |
| ------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `MLVC_CPU_ISA`            | `scalar`, `fp32` / `neon_fp32`, `fp16fml`, `avx2` | Forces a specific GEMM/kernel ISA path (if supported by hardware) for A/B conformance testing and benchmarking |
| `MLVC_CPU_DISABLE_FUSION` | `1`                                               | Emits 1:1 unfused `kSingleNode` steps to verify fused-vs-unfused numerical equivalence                         |
