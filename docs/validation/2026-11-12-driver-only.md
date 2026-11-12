# Driver-only checkpoint validation — 2026-11-12

The project uses Driver+cubin only. Model preparation loads the registered
original checkpoints and pinned official network code directly, without an
intermediate framework model file. C ABI 2 and AOT graph schema 2 replace the
previous interfaces and graph packages.

## Environment

- Model tooling: Windows, Python 3.12, PyTorch 2.10.0 CPU; no ONNX, ONNX Runtime,
  or TensorRT Python package installed.
- GPU: NVIDIA A30, driver 595.91.07, isolated Ubuntu 26.04 container,
  GCC 15.2 and CUDA Toolkit 13.3.
- Both registered profiles: `mlvc-psnr-v1` and `mlvc-s-psnr-v1`.
- Codec fixture: two 640x360 YUV420 frames filled with sample value 128,
  padded to 640x368, q-index 21.
- Reference: pinned official eager-Python split model, FP16 neural execution,
  and entropy tables constructed from FP32 checkpoint parameters.

## Results

| Check | Result |
| --- | --- |
| AOT capture/lowering unit tests | 2/2 passed |
| Model bundle integrity and embedding tests | 2/2 passed |
| Full-size AOT host replay against eager execution | 12/12 passed with zero numeric tolerance: both profiles, encoder/decoder, Q 0/21/63 |
| GCC build with project warnings as errors | Passed |
| GPU and portable CTest suite | 9/9 passed |
| Host ASan/UBSan safety test | 1/1 passed |
| Release packaging with embedded models | Passed, including 9/9 package-build tests |
| Installed package checksums | Passed |
| Installed CMake/C consumer | Configured, compiled, linked, and executed |
| Installed library dependencies | NVIDIA driver plus system C/C++ libraries only |

The GPU tests cover the driver probe, native channel-first pixel shuffle and
unshuffle, CUTLASS kernels, core safety, C ABI, public header, model bundles,
portable reference parity, and strict two-frame codec conformance for both
profiles. The existing codec thresholds were not relaxed.

The package records `codec_abi_version=2`, `aot_graph_schema=2`, both embedded
profiles, `gpu_runtime_tests=true`, and `runtime_gpu_dependency=libcuda.so.1`.
Its embedded image contains 18 indexed assets totaling 81,025,506 bytes.

## Reproduction and limits

Follow the checkpoint and reference preparation steps in the
[README](../../README.md), then run the release preset, CTest, the sanitizer
preset, and `tools/package.sh`. The repeatable host graph comparison is
`tests/conformance/run_aot_conformance.py --checkpoint-root <directory>`.

Local logs are under the ignored `build/validation-driver-only/` directory.
Model bundles and reference tensors are also generated, ignored artifacts.
This evidence covers the stated A30 fixture and host checks. It does not establish
long-sequence quality, arbitrary-content conformance, rate-distortion results,
throughput guarantees, other GPU architectures, or a hosted CI run.
