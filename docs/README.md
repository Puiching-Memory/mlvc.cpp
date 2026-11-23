# mlvc.cpp Documentation

Welcome to the `mlvc.cpp` technical documentation. Every document has a single responsibility and stays aligned with the current C++20 multi-backend codebase (`core` / `runtime` / `backends` / `codec` / `public`).

## Start by Role

- **Evaluating or running the CLI** → [Build, Packaging & CLI Usage](build-and-usage.md)
- **Integrating `libmlvc_codec` into an app or service** → [SDK Integration Guide](sdk-integration.md)
- **Compiling official checkpoints into canonical bundles** → [Model Pipeline & Binary Format](model-pipeline.md)
- **Understanding the system or contributing a backend / kernel** → [System Architecture](architecture.md) → [Inference Backends](backends.md) → [Codec Pipeline & Bitstream](codec-and-bitstream.md)
- **Running conformance tests or auditing compatibility** → [Verification & Compatibility Contract](verification.md)

## Document Map

| Document                                         | Scope                                                                                                                                                                                                           |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [architecture.md](architecture.md)               | Layered module design (`core`, `runtime`, `backends`, `codec`, `public`), asset/session/backend abstractions, and source tree map                                                                               |
| [backends.md](backends.md)                       | Internal design of `driver-cubin` (NVIDIA Driver API, fatbin, CUTLASS, CUDA Graphs, kernel fusion lessons) and `cpu` (ARM64 NEON/FP16FML assembly, AVX2, heterogeneous thread pool, graph fusion)               |
| [codec-and-bitstream.md](codec-and-bitstream.md) | Split-model I/O contract, YUV420 pre/post-processing, GOP/DPB/QP schedule, scale extraction, `msrtc_rans` coding order, and `mlvc-nalu-v0` Annex-B bitstream specification                                      |
| [model-pipeline.md](model-pipeline.md)           | Offline PyTorch dispatch AOT compiler, Protobuf schemas (`proto/mlvc_model.proto`), static arena planning, canonical bundle structure, `profiles.json`, and ELF model embedding                                 |
| [build-and-usage.md](build-and-usage.md)         | Requirements, CMake options & presets, cross-compilation (Linux GPU/CPU, Windows, Android, OpenHarmony), `package.sh` release packaging, and `mlvc_demo` streaming/FIFO/FFmpeg usage                            |
| [sdk-integration.md](sdk-integration.md)         | C ABI v2 (`public/include/mlvc/codec.h`), CMake `find_package`, exported symbol policy, internal C++ `EncoderSession`/`DecoderSession` & `ModelAssetReader` boundaries, and service/mobile integration guidance |
| [verification.md](verification.md)               | Unit test matrix, CI & ASan/UBSan checks, AOT eager conformance, `mlvc_codec_compatibility` tolerances, diagnostic tooling, and explicit acceptance boundaries                                                  |
