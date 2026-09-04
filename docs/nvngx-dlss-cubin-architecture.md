# NVNGX DLSS cubin architecture notes - 2026-09-02

This note records local binary observations from `nvngx_dlss.dll`. The DLL is
proprietary input supplied for analysis and is intentionally not committed.

## Binary observations

- File size: approximately 57 MiB
- SHA-256: `be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e`
- PE imports: `VERSION.dll`, `ADVAPI32.dll`, `USER32.dll`, `KERNEL32.dll`
- No static import of `nvcuda.dll`, CUDA Runtime, cuDNN, or an inference SDK
- Imports `LoadLibraryW` and `GetProcAddress`
- Contains strings for `nvcuda.dll`, `cuGetProcAddress`, `cuModuleLoadData`,
  `cuModuleGetFunction`, and `cuLaunchKernel`
- Contains 169 valid CUDA fatbin containers totaling 34,723,000 bytes
- Contains 670 cubins: 169 each for `sm_75`, `sm_86`, and `sm_89`, plus 163
  for `sm_80`
- Contains 169 PTX fallbacks: 163 targeting `sm_89` and six targeting `sm_80`
- Each fatbin exposes one entry kernel; observed names include fused convolution,
  upsample, anisotropic reconstruction, padded-window, and ray-reconstruction
  layers

The network-kernel families expose both NHWC and channel-packed NCHW8 variants.
Several entry names describe whole subgraphs rather than single operators, for
example `bilinear_upsample_conv_1x1_conv_1x1`, `conv_3x3_pool`, and
`conv_3x3_pool_conv_1x1_pool`. The SM80 images inspected for these families use
roughly 16-49 KiB of shared memory per block. A representative 128x128 FP16
pointwise image uses 194 registers per thread and 24 KiB of shared memory.

Runtime strings also identify recycled buffer resources, three tensor
containers, an alias tensor container, a shared weights buffer, and fused
engine input/output kernels. These observations reinforce that layout,
lifetime planning, and pre/post-processing are part of the compiled schedule;
they are not left to a generic operator interpreter.

The evidence shows an ahead-of-time kernel runtime, not an embedded general
inference framework. Model layers are represented by fixed-shape, heavily
fused cubin kernels and selected through an internal kernel map.

## mlvc.cpp interpretation

The equivalent deployment boundary is:

1. Use CUDA Toolkit, CUTLASS/CuTe, and model conversion tools only on the build
   host.
2. Produce architecture-specific cubins plus a PTX forward-compatibility image.
3. Embed the fatbin, graph schedules, PMFs, metadata, and FP16 weights in the
   ELF shared library, matching the DLL's monolithic deployment boundary.
4. Link only NVIDIA's official CUDA Driver API.
5. Load modules with `cuModuleLoadData`, resolve entries with
   `cuModuleGetFunction`, and dispatch with `cuLaunchKernel`.
6. Keep weights, DPB/history, and intermediate tensor arenas resident on the
   device.

## Applied optimization lesson

The encoder's recurrent-feature tail previously replayed 12 graph nodes: three
channel slices, two sigmoids, six binary operations, and a concat. Following
the DLL's fused input/output-kernel pattern, mlvc.cpp now recognizes this exact
dependency chain and dispatches one `mlvc_feature_update_fp16` kernel. The
kernel writes the next recurrent state in place and retains every intermediate
FP16 rounding point, so it remains byte-identical to the unfused graph for both
embedded model profiles.

Two other ideas were measured and rejected on A30 rather than retained on
architectural similarity alone. Forcing the existing single-kernel
Conv+ReGLU implementation instead of CUTLASS plus the vectorized ReGLU kernel
regressed encoder latency by about 4%. Reducing the hottest CUTLASS pointwise
pipeline from four stages to two increased register use from 228 to 248 and
regressed latency by about 2.5%. The DLL's resource figures therefore serve as
directional evidence, not portable launch parameters.

Unlike the inspected DLL, mlvc.cpp intentionally requires the NVIDIA driver at
program load time. It does not dynamically fall back when the driver is absent.
The release package must not contain or depend on CUDA Runtime, cuDNN, cuBLAS,
TensorRT, ONNX Runtime, or libtorch.

NVIDIA CUDA headers remain subject to the NVIDIA software license. They are
used from the build host's CUDA Toolkit and are not copied into the package.

## 2026-09-04 y-latent tail fusion

The encoder's y-latent tail now has two additional fixed-shape fusion kernels:

- `mlvc_y0_tail_fp16` replaces the first prior split, y0 normalization and
  quantization, y0 prior update, and the two following concats. It also writes
  the normalized tensor and clipped prior as side outputs because both remain
  live for the y1 pass.
- `mlvc_y1_tail_fp16` replaces y1 quantization, y1 prior update, the residual
  add, and the final scale multiply. It writes `y_raw_1` and the decoder input
  directly.

Both kernels retain the graph's FP16 materialization points: Clip and
Reciprocal are rounded before normalization, every Add/Mul/Sub is rounded in
FP16, and Round is applied after the same FP16 sum. The backend only enables
the match after validating the complete node dependency chain, consumer counts,
channel Slice ranges, static shapes, and permitted arena aliases. The MLVC-S
profile has 24 y channels, so it deliberately uses the generic schedule; on
the A30 its smaller workload did not amortize the fused kernels' footprint.
