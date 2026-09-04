# NVIDIA A30 FP16 backend benchmark - updated 2026-09-04

This report separates model-level host-roundtrip latency from codec-level
throughput. The two measurements exercise different boundaries and must not be
compared as if they were the same workload.

## Setup

- GPU: NVIDIA A30, SM80, 24 GiB
- NVIDIA driver: 595.71.05
- CUDA toolkit: 13.3
- Backends: ONNX Runtime 1.26.0, libtorch 2.13.0 cu130, TensorRT 11.2.1,
  Driver+cubin
- Model: `dmc61sbr_reglu`, official `mlvc-psnr-v1.ckpt`
- Model input size: 640x368 for a 640x360 frame
- Q index: 21
- Floating-point precision: FP16; control input: int32
- GPU clocks: unlocked
- Concurrent GPU load: none

The model-level results are the median result from three independent processes.
Each process performed 20 warm-up iterations followed by 100 measured
iterations. Timing covers host input through host output, including all H2D and
D2H transfers and required synchronization. Each inference transfers 5,420,804
bytes (5.17 MiB) of combined host input and output data across the backend
boundary.

TensorRT used the existing SM80 engine cache. Engine construction time is
excluded from the steady-state and load measurements below. Because clocks were
not locked, tail latency should be treated as indicative rather than as a
production bound.

## Model-Level Performance

### Encoder

| Backend      | Load ms | Mean ms | p50 ms | p95 ms | Inferences/s | Relative latency |
| ------------ | ------: | ------: | -----: | -----: | -----------: | ---------------: |
| TensorRT     | 150.256 |  3.3490 | 3.3462 | 3.3673 |       298.60 |           1.000x |
| Driver+cubin |  50.048 |  3.5521 | 3.4517 | 4.7236 |       281.52 |           1.061x |
| ONNX Runtime | 116.394 |  7.0856 | 6.9556 | 8.0699 |       141.13 |           2.116x |
| libtorch     | 358.795 |  7.1898 | 7.1378 | 7.4892 |       139.09 |           2.147x |

Driver+cubin encoder means were 3.5521, 3.7695, and 3.4481 ms. Two runs had a
p95 around 4.72-4.75 ms, so the current encoder path has a visible high-latency
tail even though its median is close to TensorRT.

### Decoder

| Backend      | Load ms | Mean ms | p50 ms | p95 ms | Inferences/s | Relative latency |
| ------------ | ------: | ------: | -----: | -----: | -----------: | ---------------: |
| Driver+cubin |  46.861 |  3.2853 | 3.2842 | 3.2955 |       304.39 |           1.000x |
| TensorRT     | 207.567 |  3.5094 | 3.4461 | 3.4865 |       284.95 |           1.068x |
| libtorch     | 351.028 |  6.7828 | 6.6689 | 8.0239 |       147.43 |           2.065x |
| ONNX Runtime | 108.930 |  7.0390 | 6.9623 | 7.0905 |       142.07 |           2.143x |

## Driver+cubin Progression

The current path combines the accumulated kernel improvements with static
memory planning, direct concat/slice placement, persistent pinned staging,
fused adjacent operations, and CUDA Graph replay. It does not select kernels
from a per-GPU timing table or an offline autotune database.

| Model   | Generic mean ms | 2026-09-02 mean ms | Current mean ms | vs. 2026-09-02 | vs. generic |
| ------- | --------------: | -----------------: | --------------: | -------------: | ----------: |
| Encoder |        244.8031 |            12.7472 |          3.5521 |         3.589x |     68.917x |
| Decoder |        251.4018 |            12.6680 |          3.2853 |         3.856x |     76.523x |

### 2026-09-04 recurrent-feature fusion

The encoder tail now fuses a fixed 12-node recurrent-feature update into one
kernel while preserving the original FP16 rounding boundaries. A controlled
A/B on an A30 used six interleaved processes per variant, 50 warm-up iterations
and 300 measured iterations per process. GPU clocks remained unlocked.

| Variant                          | Median-of-runs mean ms | Median-of-runs p50 ms |
| -------------------------------- | ---------------------: | --------------------: |
| Unfused 12-node tail             |                 3.4646 |                3.4635 |
| Fused `mlvc_feature_update_fp16` |                 3.3992 |                3.3981 |

The fused path reduced both measures by 1.9%. The standard and small embedded
model profiles produced byte-identical output tensor files with the fusion
enabled and disabled. This focused A/B is additive to, rather than a
replacement for, the multi-backend table above.

### 2026-09-04 decoder recurrent-feature output fusion

The decoder now recognizes the corresponding 11-node recurrent-feature chain
and fuses it with the later feature Concat. The kernel materializes the scaled
and blended tensors required by reconstruction and writes the concatenated
feature directly, replacing 11 pointwise kernels and both Concat copy launches.

A controlled A/B on the same A30 alternated eight runs per variant. Each run
used 50 warm-up iterations and 500 measured iterations.

| Profile | Unfused mean of run medians ms | Fused mean of run medians ms | Improvement |
| ------- | -----------------------------: | ---------------------------: | ----------: |
| MLVC    |                       3.288624 |                     3.224106 |       1.96% |
| MLVC-S  |                       5.845329 |                     5.812108 |       0.57% |

Nsight Systems measured the new kernel at approximately 21.4 us on the
standard profile and an aggregate GPU kernel-time reduction of roughly 0.09 ms
per inference. Fused and unfused `x_hat` and `feature` files were byte-identical
for both profiles.

A `DepthToSpace(block=8)+Clip` fusion was also implemented and measured, then
removed. It improved endpoint latency by only about 0.09% while requiring an
additional 1.35 MiB persistent buffer to avoid an existing arena overlap.

### 2026-09-04 y-latent quantization and prior-tail fusion

The standard encoder profile now fuses the y0 and y1 quantization/prior-update
tails. `mlvc_y0_tail_fp16` replaces 19 graph nodes through the input to the
spatial-prior stack; `mlvc_y1_tail_fp16` replaces 15 nodes through the final
decoder-input scale. The kernels preserve the intermediate FP16 rounding
boundaries and materialize `z_raw`, `y_raw_0`, and `y_raw_1` identically to the
generic schedule across two-frame conformance runs.

The matcher is shape- and alias-aware. It accepts only the fixed dependency
chain with the expected consumer counts and permits only the exact arena
aliases used by the standard profile. MLVC-S is intentionally left on the
generic tail because its 24-channel workload is too small for a stable fused
latency win on the A30.

## Accuracy

The reference is the upstream PyTorch conversion-loop output rounded to FP16.
All four backends produced numerically exact `z_raw`, `y_raw_0`, and `y_raw_1`
for this case. Raw half bits differ only for signed zero, which does not change
the numerical latent value or entropy symbol.

### Encoder feature output

| Backend      | Max absolute error |        RMSE | Cosine similarity |
| ------------ | -----------------: | ----------: | ----------------: |
| Driver+cubin |        0.005859375 | 0.000376355 |       0.999999349 |
| libtorch     |        0.005859375 | 0.000416811 |       0.999999181 |
| ONNX Runtime |        0.005859375 | 0.000420948 |       0.999999161 |
| TensorRT     |        0.014160156 | 0.000945649 |       0.999995603 |

### Decoder reconstruction

| Backend      | `x_hat` max absolute error | `x_hat` RMSE | Feature RMSE |
| ------------ | -------------------------: | -----------: | -----------: |
| ONNX Runtime |                0.000488281 |  0.000196233 |  0.000420948 |
| libtorch     |                0.000488281 |  0.000196570 |  0.000416811 |
| Driver+cubin |                0.000488281 |  0.000197143 |  0.000376355 |
| TensorRT     |                0.000488281 |  0.000212725 |  0.000998282 |

## Codec-Level Performance

The codec benchmark used a deterministic 48-frame 640x360 30 fps YUV420
sequence generated with:

```bash
ffmpeg -f lavfi -i testsrc2=size=640x360:rate=30 -frames:v 48 \
  -pix_fmt yuv420p -f rawvideo input.yuv
```

The input SHA-256 is
`2b268c3ac28af93eeec35cd59e6c9b79b5faa98c582488de21d8488e065c092c`.
Each result below is the median of three independent processes. Codec timing
excludes model or engine loading and includes YUV conversion, entropy coding,
file I/O, inference, and required synchronization. The codec keeps
`feature -> ref_feature` state on the device between frames.

| Backend      | Encode fps (three runs)  | Median | Decode fps (three runs)  | Median | Stream bytes |
| ------------ | ------------------------ | -----: | ------------------------ | -----: | -----------: |
| Driver+cubin | 248.90 / 356.62 / 355.75 | 355.75 | 398.02 / 395.69 / 396.37 | 396.37 |       13,959 |
| TensorRT     | 117.68 / 106.84 / 108.13 | 108.13 | 125.01 / 104.32 / 107.03 | 107.03 |       14,032 |

The Driver+cubin codec path now uses two persistent pinned host slots, direct
non-owning tensor and YUV views, GPU YUV conversion, direct entropy decode into
backend input storage, cached entropy indices and scratch buffers, and reused
rANS streams. On this sequence it is 229.0% faster than TensorRT for encode and
270.3% faster for decode. Relative to the previous Driver+cubin medians of
112.52 and 120.62 fps, the complete host-pipeline work improved encode by
216.2% and decode by 228.6%.

A separate 480-frame run, which amortizes first-use scheduling and graph
capture, measured 386.05 fps encode and 407.52 fps decode. The optimized paths
crossed all GOP resets in that run without changing the deterministic stream.

### Determinism and codec checks

All three runs of each backend produced byte-identical outputs within that
backend:

| Backend      | Encoded stream SHA-256                                             | Decoded YUV SHA-256                                                |
| ------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Driver+cubin | `9eaa181552fc13497125e040b8ff64fafdb37b76e691d219193ffae7ee6d48e3` | `9ca7d3112fb5019fdc4ef0aa51615dabb4d0f264006d1057367a3e118b2afd28` |
| TensorRT     | `996acc996cb48ea70633e6f5221f263d056ee6d92bcb3d669c35c86f913389d2` | `eece579befde7a9dd38743e941c44a1ae5c5eabda157d24073ee2f708cb4f9b4` |

Driver+cubin also passed a separate 65-frame fast-path versus debug-path
comparison with byte-identical bitstream and decoded YUV, including the GOP
reset at frame 64.
The backend-specific streams are different because their non-entropy feature
calculations are not bit-exact. In this test, the Driver+cubin stream is 1.6%
smaller than the TensorRT stream; there is no evidence of entropy divergence or
collapse in this sequence. This single synthetic sequence is not sufficient to
make that claim for all content, Q indices, or long-running reference states.

## Conclusions

- Driver+cubin and TensorRT are now in the same model-latency class. At the
  host-roundtrip model boundary, Driver+cubin encoder mean latency is 6.1%
  higher, while decoder throughput is 6.8% higher (mean latency is 6.4% lower).
- ONNX Runtime and libtorch are approximately 2.1x slower than the fastest
  backend in this model-level test.
- At the full-codec boundary, Driver+cubin is 229.0% faster for encode and
  270.3% faster for decode than TensorRT on the measured 48-frame sequence.
- The exact entropy latents and deterministic codec output address the observed
  test cases, but do not replace multi-sequence, multi-Q, multi-GPU conformance
  and rate-distortion testing.
- These results apply to one A30, resolution, Q index, model, and short synthetic
  sequence. They should not be generalized to other profiles or GPUs without
  measurement.

## General Optimization Headroom

1. Optimize or replace the remaining rANS encode/decode core, now the largest
   named CPU hotspot after YUV conversion and staging moved off the host path.
2. Reduce scale-extraction work further or fuse it with entropy index
   preparation; it accounts for approximately 6.5% of encode and 8.8% of
   decode CPU samples in the current `perf` profiles.
3. Expand benchmarks across real sequences, Q indices, model profiles, frame
   types, longer GOP histories, and supported GPU architectures.
4. Gate future changes on codec byte determinism, latent correctness, decoded
   metrics, bitrate, and end-to-end throughput.

Per-layer code generation, large families of specialized kernel variants, and
offline per-device autotuning are intentionally outside the current direction;
their maintenance cost is not justified by this benchmark.
