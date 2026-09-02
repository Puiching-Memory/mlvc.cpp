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

The evidence shows an ahead-of-time kernel runtime, not an embedded general
inference framework. Model layers are represented by fixed-shape, heavily
fused cubin kernels and selected through an internal kernel map.

## mlvc.cpp interpretation

The equivalent deployment boundary is:

1. Use CUDA Toolkit, CUTLASS/CuTe, and model conversion tools only on the build
   host.
2. Produce architecture-specific cubins plus a PTX forward-compatibility image.
3. Embed the fatbin and FP16 weights in the executable or model package.
4. Link only NVIDIA's official CUDA Driver API.
5. Load modules with `cuModuleLoadData`, resolve entries with
   `cuModuleGetFunction`, and dispatch with `cuLaunchKernel`.
6. Keep weights, DPB/history, and intermediate tensor arenas resident on the
   device.

Unlike the inspected DLL, mlvc.cpp intentionally requires the NVIDIA driver at
program load time. It does not dynamically fall back when the driver is absent.
The release package must not contain or depend on CUDA Runtime, cuDNN, cuBLAS,
TensorRT, ONNX Runtime, or libtorch.

NVIDIA CUDA headers remain subject to the NVIDIA software license. They are
used from the build host's CUDA Toolkit and are not copied into the package.
