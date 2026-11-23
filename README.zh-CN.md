# mlvc.cpp

> [English](README.md) | 简体中文

**Microsoft MLVC (`DMC-6.1sb`)** 神经视频编解码器的独立 C++20 实现，采用解耦的多后端推理运行时、Protobuf 二进制 AOT 模型包与 `mlvc-nalu-v0` NALU Annex-B 码流容器。

项目端到端支持官方两个模型档位——**`mlvc-psnr-v1`**（`dmc61sbr_reglu`）与 **`mlvc-s-psnr-v1`**（`dmc61sbr_mini_reglu`），完整实现 8-bit YUV420 输入输出、循环 GOP/DPB 参考特征状态、规范 PMF 概率表与 `msrtc_rans` 熵编码。

## 核心特性

- **运行时零推理框架依赖**：离线通过 PyTorch dispatch 将官方 checkpoint 直接编译为定形 AOT 计算图（`src/proto/mlvc_model.proto`）与对齐的 FP16 权重文件；部署产物不依赖 PyTorch、ONNX Runtime、TensorRT 或 CUDA Runtime。
- **跨平台多后端推理**：
  - **`driver-cubin`**（Linux `x86_64` NVIDIA GPU）：仅动态链接系统 NVIDIA 驱动（`libcuda.so.1`），内嵌多架构 CUDA fatbin（`sm_75`/`sm_80`/`sm_86`/`sm_89` + `compute_89` PTX）、CUTLASS Tensor Core 卷积、CUDA Graphs 与循环/量化尾融合内核。
  - **`cpu`**（Linux、Windows、macOS、Android、OpenHarmony）：FP16 存储 / FP32 累加的通用 CPU 推理后端，支持运行时 ISA 分发——自研 AArch64 NEON 与 ARMv8.2 `+fp16fml` 4×16 双缓冲汇编微内核（兼容 BTI/PAC、寄存器内融合 ReGLU）、`x86_64` AVX2/FMA/F16C 内核、异构大小核感知线程池与图融合调度。
- **清晰的四层解耦架构**：`src/core`（码流、rANS、YUV、资产读取）、`src/runtime`（AOT 图 IR、后端注册表、双缓冲编解码接口）、`src/backends`（`driver-cubin`、`cpu`）与 `src/codec`（无文件 I/O 逐帧会话 `EncoderSession`/`DecoderSession` 及 `include/mlvc/codec.h` 稳定 C ABI v2）。
- **标准化 NALU 码流容器**：实现源自 `microsoft/libmlvc` 的 MLVC v0.x NALU Annex-B 格式（`mlvc-nalu-v0`，含 `SPS`、`PPS`、`IDR_N_LP`、`TRAIL_R`、`TSA_R`、EBSP 防竞争字节，码流 `QP` 范围 `[0, 51]`）。
- **流式 CLI 与 C SDK**：支持文件、标准输入输出（`-`）、POSIX 命名管道（FIFO）、双 GPU 编解码流水线，以及独立 CMake 消费包（`mlvc::mlvc_codec`）。

## 平台与后端支持矩阵

| 目标平台                              | `MLVC_BACKEND=AUTO` 默认值                    | 可选后端                             | 模型加载方式                                |
| ------------------------------------- | --------------------------------------------- | ------------------------------------ | ------------------------------------------- |
| Linux `x86_64`（NVIDIA GPU）          | `driver_cubin`                                | `driver_cubin`, `cpu`, `all`, `none` | ELF 内嵌（`.incbin`）或目录包               |
| Linux `aarch64` / `x86_64`（纯 CPU）  | `cpu`（`x86_64` 需指定 `-DMLVC_BACKEND=cpu`） | `cpu`, `none`                        | 目录包或 ELF 内嵌（`MLVC_EMBED_MODELS=ON`） |
| Android / OpenHarmony（`arm64-v8a`）  | `cpu`                                         | `cpu`, `none`                        | 目录包、ELF 内嵌或自定义 `ModelAssetReader` |
| Windows（`x86_64`）/ macOS（`arm64`） | `cpu`                                         | `cpu`, `none`                        | 目录包或自定义 `ModelAssetReader`           |

## 快速上手

### 1. 初始化子模块并准备模型

```bash
./tools/scripts/bootstrap.sh
# Ubuntu 26.04 如需构建 NVIDIA Driver+cubin，可附加 --with-cuda：
./tools/scripts/bootstrap.sh --with-cuda

uv venv --python 3.12 build/model-venv
uv pip install --python build/model-venv/bin/python -r tools/python/requirements-models.txt
uv pip install --python build/model-venv/bin/python -e ./tools/python
uv pip install --python build/model-venv/bin/python ./third_party/mlvc/packages/msrtc_rans
source build/model-venv/bin/activate

# 从官方 checkpoint 编译并组装标准模型包（mlvc-s-psnr-v1 同理）
python -m mlvc_tools.aot \
  --profile mlvc-psnr-v1 --checkpoint /path/to/mlvc-psnr-v1.ckpt \
  --model-width 640 --model-height 368 \
  --output-dir models/generated/checkpoint/mlvc-psnr-v1/640x368

python -m mlvc_tools.package assemble \
  --profile mlvc-psnr-v1 \
  --source-dir models/generated/checkpoint/mlvc-psnr-v1/640x368 \
  --output-dir models/canonical/mlvc-psnr-v1/640x368
```

### 2. 编译构建

```bash
# Linux x86_64 NVIDIA Driver+cubin 发布构建（默认内嵌两个标准模型包）
cmake --preset driver-cubin-release
cmake --build --preset driver-cubin-release -j
ctest --preset driver-cubin-release

# 跨平台 CPU 后端构建（无需 CUDA Toolkit）
cmake -S . -B build/cpu-release -DCMAKE_BUILD_TYPE=Release -DMLVC_BACKEND=cpu
cmake --build build/cpu-release -j
ctest --test-dir build/cpu-release --output-on-failure
```

### 3. 编解码运行

```bash
./build/driver-cubin-release/tools/mlvc_demo --list-backends
./build/driver-cubin-release/tools/mlvc_demo --list-model-profiles

./build/driver-cubin-release/tools/mlvc_demo encode \
  --input in.yuv --output out.mlvc \
  --width 640 --height 360 --frames 60 --q-index 21

./build/driver-cubin-release/tools/mlvc_demo decode \
  --input out.mlvc --output rec.yuv --frames 60
```

在未内嵌模型的 CPU 构建中，通过 `--model-dir models/canonical/mlvc-psnr-v1/640x368` 指定模型目录（可选 `--backend cpu --threads 4`）。

## 文档导航

全部技术文档统一维护于 [`docs/`](docs/README.md)（英文主文档）：

| 文档                                                       | 内容概览                                                                                                                  |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| [docs/README.md](docs/README.md)                           | 按读者角色组织的文档总索引                                                                                                |
| [docs/architecture.md](docs/architecture.md)               | 四层解耦架构、`ModelAssetReader`、`EncoderSession`/`DecoderSession` 与源码目录地图                                        |
| [docs/backends.md](docs/backends.md)                       | `driver-cubin`（驱动 API、Fatbin、CUTLASS、CUDA Graphs 与融合经验）与 `cpu`（ARM64 汇编微内核、AVX2、异构线程池与图融合） |
| [docs/codec-and-bitstream.md](docs/codec-and-bitstream.md) | 模型输入输出、YUV420 补边/裁剪、GOP/DPB/QP 调度、尺度提取、rANS 顺序与 `mlvc-nalu-v0` Annex-B 码流规范                    |
| [docs/model-pipeline.md](docs/model-pipeline.md)           | 离线 AOT 编译、`src/proto/mlvc_model.proto`（bundle v2 / AOT v3）、静态 Arena 规划、`profiles.json` 与 ELF 内嵌机制       |
| [docs/build-and-usage.md](docs/build-and-usage.md)         | CMake 选项与 Preset、跨平台编译（Linux/Windows/Android/OHOS）、发布打包与 `mlvc_demo` 流式/FIFO 用法                      |
| [docs/sdk-integration.md](docs/sdk-integration.md)         | 公开 C ABI v2（`include/mlvc/codec.h`）、CMake 消费、符号导出边界、内部 C++ 会话扩展与服务端/移动端部署建议               |
| [docs/verification.md](docs/verification.md)               | 单元测试矩阵、CI 与 Sanitizer、AOT 主机一致性校验、`mlvc_codec_compatibility` 数值容差与验收边界                          |

## 致谢与许可

- [Microsoft MLVC](https://github.com/microsoft/mlvc)：提供神经视频编解码模型架构与 `msrtc_rans` 熵编码实现（MIT License）。
- [Microsoft libmlvc](https://github.com/microsoft/libmlvc)：提供 v0.x NALU Annex-B 码流规范、位流 I/O 与 NALU 解析/构造设计及一致性测试向量（MIT License，见 [third_party/libmlvc/LICENSE](third_party/libmlvc/LICENSE)）。
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)：提供 CUDA Tensor Core 卷积算子构件。
- [Protobuf](https://github.com/protocolbuffers/protobuf) 与 [Abseil](https://github.com/abseil/abseil-cpp)：提供私有静态链接的 Lite 二进制模型序列化能力。
