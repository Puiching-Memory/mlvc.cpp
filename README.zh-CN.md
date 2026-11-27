# mlvc.cpp

> [English](README.md) | 简体中文

[![CI](https://github.com/Puiching-Memory/mlvc.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Puiching-Memory/mlvc.cpp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](CMakeLists.txt)
[![arXiv](https://img.shields.io/badge/Paper-MLVC%20(2606.28027)-b31b1b.svg)](https://arxiv.org/abs/2606.28027)
[![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows%20%7C%20Android%20%7C%20OHOS-2ea44f.svg)](#平台与后端支持矩阵)

**[Microsoft MLVC (`DMC-6.1sb`)](https://github.com/microsoft/mlvc) 神经视频编解码器的独立 C++20 推理与编解码引擎——单卡 NVIDIA GPU 实测突破 1,100+ FPS，移动端支持手写 ARM64 汇编纯 CPU 运行，核心动态库仅 6.1 MiB，运行时零深度学习框架依赖。**

不依赖 PyTorch，不依赖 ONNX Runtime，不依赖 TensorRT，甚至不硬链接 `libcudart` / `libcublas` / `libcudnn`。一个独立的 `libmlvc_codec.so` 与流式命令行工具 `mlvc_demo`，端到端完整实现官方 **`mlvc-psnr-v1`** 与 **`mlvc-s-psnr-v1`** 双档位模型的 `mlvc-nalu-v0` Annex-B 码流容器、GOP/DPB 循环状态、规范 PMF 概率表与 `msrtc_rans` 熵编解码。

---

## 实测性能与产物体积

以下数据实测于 **NVIDIA A30**（`sm_80`，1440 MHz，640×360 YUV420 补边至 640×368，`q_index = 21`，`nsys --cuda-graph-trace=node` 无抢占连续帧窗口）：

| 模型档位             | 参数量 | 方向  | Kernel 数 / 帧 | 单帧 GPU Busy | 帧间 GPU 气泡 |           连续帧 GPU 吞吐 |    首帧 AU 码流    |  重建 YUV 对比官方 Python  |
| -------------------- | -----: | :---: | -------------: | ------------: | ------------: | ------------------------: | :----------------: | :------------------------: |
| **`mlvc-s-psnr-v1`** |   5.4M | 编码  |            112 |   **0.89 ms** |       10.9 µs |  **1,106 fps** (904 µs/f) | 逐字节一致 (328 B) | 最大误差 ≤ 1，63.6–65.4 dB |
| **`mlvc-s-psnr-v1`** |   5.4M | 解码  |            106 |   **0.87 ms** |       11.2 µs |  **1,120 fps** (893 µs/f) |     逐字节一致     | 最大误差 ≤ 1，63.6–65.4 dB |
| **`mlvc-psnr-v1`**   |  18.3M | 编码  |            188 |   **1.91 ms** |       11.3 µs | **~521 fps** (~1.92 ms/f) | 逐字节一致 (137 B) | 最大误差 ≤ 1，63.5–67.3 dB |
| **`mlvc-psnr-v1`**   |  18.3M | 解码  |            153 |   **1.79 ms** |       12.4 µs | **~555 fps** (~1.80 ms/f) |     逐字节一致     | 最大误差 ≤ 1，63.5–67.3 dB |

### 部署产物体积（Linux `x86_64` `driver-cubin`）

```text
libmlvc_codec.so   6.1 MiB   （内含多架构 Fatbin：sm_75/80/86/89/90 + compute_80 PTX）
mlvc_demo           36 KiB   （命令行前端）
DT_NEEDED          libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6   （libcuda.so.1 运行时 dlopen 动态加载）
```

> **为什么没有 `DT_NEEDED libcuda.so.1`？** 启用多后端（`-DMLVC_BACKEND=all`）编出的同一个二进制，在完全没装 NVIDIA 驱动的机器上也能正常启动，并自动回退到 `cpu` 后端运行。

---

## 为什么做 `mlvc.cpp`？

| 维度             | 官方 Python / ONNX Runtime 方案                         | `mlvc.cpp`                                                                                                       |
| ---------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **运行时依赖**   | Python + PyTorch / ONNX Runtime / CUDA Runtime（数 GB） | 仅系统 C/C++ 基础库（`6.1 MiB` `.so`，`dlopen("libcuda.so.1")`）                                                 |
| **GPU 执行引擎** | 通用框架逐算子调度                                      | 权重交错 + CUTLASS 寄存器级 ReGLU 融合（零独立 ReGLU 核）、三流 DMA/计算重叠、Per-Slot 完整 CUDA Graph           |
| **CPU / 移动端** | ORT CPU / 厂商私有 NPU 栈                               | 自研 ARM64 NEON 与 ARMv8.2 `+fp16fml` 4×16 双缓冲汇编微内核（兼容 BTI/PAC）、x86 AVX2/FMA/F16C、大小核感知线程池 |
| **工程集成**     | Python 脚本                                             | 稳定 C ABI v2/v3（`include/mlvc/codec.h`）、CMake `find_package(mlvc_codec)`、stdin/stdout/FIFO 实时流式管道     |
| **数值一致性**   | 官方参考实现                                            | 严格一致性契约：首帧 AU 与官方 Python 逐字节一致，YUV 最大像素误差 ≤ 1，PSNR ≥ 60 dB                             |

---

## 60 秒快速体验

### 1. 拉取代码并一键下载编译官方模型

`prepare_models.sh` 会自动从微软官方地址下载 checkpoint 并校验 SHA-256，编译 AOT 计算图并组装标准模型包到 `models/canonical/`：

```bash
git clone https://github.com/Puiching-Memory/mlvc.cpp.git
cd mlvc.cpp
./tools/scripts/bootstrap.sh
./tools/scripts/prepare_models.sh
```

<details>
<summary>官方 Microsoft MLVC Checkpoint 直链与 SHA-256 校验值</summary>

| 档位             | 参数量 | 官方直链（Microsoft Azure Blob）                                                                | SHA-256                                                            |
| ---------------- | -----: | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `mlvc-psnr-v1`   |  18.3M | [mlvc-psnr-v1.ckpt](https://mlvideopub.blob.core.windows.net/mlvc/models/mlvc-psnr-v1.ckpt)     | `834adaab680837c4106a9bb62e5778b3f13729b4fe6f751797e0747214f5cca1` |
| `mlvc-s-psnr-v1` |   5.4M | [mlvc-s-psnr-v1.ckpt](https://mlvideopub.blob.core.windows.net/mlvc/models/mlvc-s-psnr-v1.ckpt) | `1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896` |

</details>

### 2. 编译构建（或直接从 [Releases](https://github.com/Puiching-Memory/mlvc.cpp/releases) 下载预编译包）

```bash
# Linux x86_64 — NVIDIA GPU（仅构建机需要 CUDA Toolkit >= 12.0，部署机无需装 CUDA）
cmake --preset driver-cubin-release
cmake --build --preset driver-cubin-release -j
ctest --preset driver-cubin-release

# 跨平台纯 CPU 后端 — Linux / macOS (Apple Silicon) / Windows（无需 CUDA）
cmake -S . -B build/cpu-release -DCMAKE_BUILD_TYPE=Release -DMLVC_BACKEND=cpu
cmake --build build/cpu-release -j
ctest --test-dir build/cpu-release --output-on-failure
```

### 3. 一键实测任意 MP4，并与同码率 H.264 同屏对比

```bash
# 传入任意 MP4（不传路径会自动用 ffmpeg 生成一段 640x360 测试视频）
./tools/scripts/demo_mp4.sh input.mp4 --profile mlvc-s-psnr-v1 --q-index 21
# 自动输出：demo_output/encoded.mlvc、reconstructed_mlvc.mp4，以及同码率双屏对比 compare_mlvc_vs_x264.mp4（左=mlvc.cpp，右=x264）
```

**或者直接用 FFmpeg / `ffplay` 零临时文件实时串联播放：**

```bash
ffmpeg -i input.mp4 -vf scale=640:360 -pix_fmt yuv420p -f rawvideo - 2>/dev/null \
  | ./build/driver-cubin-release/tools/mlvc_demo encode --input - --output - \
      --width 640 --height 360 --frames 0 --q-index 21 --model-profile mlvc-s-psnr-v1 \
  | ./build/driver-cubin-release/tools/mlvc_demo decode --input - --output - \
      --frames 0 --model-profile mlvc-s-psnr-v1 \
  | ffplay -f rawvideo -pixel_format yuv420p -video_size 640x360 -framerate 30 -
```

在未内嵌模型的 CPU 构建中，只需附加 `--backend cpu --model-dir models/canonical/mlvc-s-psnr-v1/640x368`。

---

## 核心技术亮点

- **`driver-cubin` NVIDIA GPU 后端**（`src/backends/driver_cubin/`）：
  - 内嵌多架构 CUDA Fatbin（`sm_75`/`sm_80`/`sm_86`/`sm_89`/`sm_90` + `compute_80` PTX），运行时通过 `dlopen("libcuda.so.1")` 动态加载。
  - 加载期权重行交错 + CUTLASS 4.8 EVT 寄存器级融合（`VisitorReGLUStore`），**100% 消除独立 ReGLU 核函数**并减半写回带宽。
  - SM 感知的 Small-M Tile（`64×128`、`64×64`）、硬件 SMEM/SM 占载守卫、面向 `spatial < 920` 的 4-warp Split-K `16×16` MMA、零 SMEM 双行滑窗 `oct2` 3×3 深度卷积，以及完整的编解码量化尾部融合。
  - 三流（`htod` / `compute` / `dtoh`）DMA 与计算全重叠 + Per-Slot 完整 CUDA Graph（帧间气泡从 73–90 µs 压至 **~11 µs**）与两阶段批量 CUTLASS 初始化（GPU 端仅 ~6 ms）。
- **`cpu` 跨平台后端**（`src/backends/cpu/`）：
  - FP16 存储 / FP32 累加 + 内部 NHWC 布局。
  - 运行时 ISA 分发：自研 AArch64 NEON 与 ARMv8.2 `+fp16fml` 4×16 双缓冲汇编微内核（兼容 BTI/PAC、寄存器内融合 ReGLU）、`x86_64` AVX2/FMA/F16C、大小核感知线程池与进程级预打包权重缓存。
- **四层解耦架构与 C SDK**：
  - `src/core`（码流、rANS、YUV、`mmap` 资产读取）、`src/runtime`（AOT 图 IR、后端注册表、双缓冲编解码接口）、`src/backends`（`driver-cubin`、`cpu`）与 `src/codec`（`EncoderSession`/`DecoderSession` 及 [`include/mlvc/codec.h`](include/mlvc/codec.h) 稳定 C ABI v2/v3）。
  - 全链路覆盖 ASan/UBSan、ThreadSanitizer、libFuzzer 与官方 2 帧数值一致性门禁。

## 平台与后端支持矩阵

| 目标平台                              | `MLVC_BACKEND=AUTO` 默认值                    | 可选后端                             | 模型加载方式                                |
| ------------------------------------- | --------------------------------------------- | ------------------------------------ | ------------------------------------------- |
| Linux `x86_64`（NVIDIA GPU）          | `driver_cubin`                                | `driver_cubin`, `cpu`, `all`, `none` | ELF 内嵌（`.incbin`）或目录包               |
| Linux `aarch64` / `x86_64`（纯 CPU）  | `cpu`（`x86_64` 需指定 `-DMLVC_BACKEND=cpu`） | `cpu`, `none`                        | 目录包或 ELF 内嵌（`MLVC_EMBED_MODELS=ON`） |
| Android / OpenHarmony（`arm64-v8a`）  | `cpu`                                         | `cpu`, `none`                        | 目录包、ELF 内嵌或自定义 `ModelAssetReader` |
| Windows（`x86_64`）/ macOS（`arm64`） | `cpu`                                         | `cpu`, `none`                        | 目录包或自定义 `ModelAssetReader`           |

## 文档导航

全部技术文档统一维护于 [`docs/`](docs/README.md)（英文主文档）：

| 文档                                                       | 内容概览                                                                                                                     |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [docs/README.md](docs/README.md)                           | 按读者角色组织的文档总索引                                                                                                   |
| [docs/architecture.md](docs/architecture.md)               | 四层解耦架构、`ModelAssetReader`、`EncoderSession`/`DecoderSession` 与源码目录地图                                           |
| [docs/backends.md](docs/backends.md)                       | `driver-cubin`（驱动 API、Fatbin、CUTLASS EVT ReGLU、三流建图与 A30 实测经验）与 `cpu`（ARM64 汇编微内核、AVX2、异构线程池） |
| [docs/codec-and-bitstream.md](docs/codec-and-bitstream.md) | 模型输入输出、YUV420 补边/裁剪、GOP/DPB/QP 调度、尺度提取、rANS 顺序与 `mlvc-nalu-v0` Annex-B 码流规范                       |
| [docs/model-pipeline.md](docs/model-pipeline.md)           | 离线 AOT 编译、`src/proto/mlvc_model.proto`（bundle v2 / AOT v3）、静态 Arena 规划、`profiles.json` 与 ELF 内嵌机制          |
| [docs/build-and-usage.md](docs/build-and-usage.md)         | CMake 选项与 Preset、跨平台编译（Linux/Windows/Android/OHOS）、发布打包与 `mlvc_demo` 流式/FIFO 用法                         |
| [docs/sdk-integration.md](docs/sdk-integration.md)         | 公开 C ABI v2/v3（`include/mlvc/codec.h`）、CMake 消费、符号导出边界、内部 C++ 会话扩展与服务端/移动端部署建议               |
| [docs/verification.md](docs/verification.md)               | 单元测试矩阵、CI 与 Sanitizer、AOT 主机一致性校验、`mlvc_codec_compatibility` 数值容差与验收边界                             |

## 致谢与许可

- [Microsoft MLVC](https://github.com/microsoft/mlvc)：提供神经视频编解码模型架构、预训练权重与 `msrtc_rans` 熵编码实现（MIT License）。
- [Microsoft libmlvc](https://github.com/microsoft/libmlvc)：提供 v0.x NALU Annex-B 码流规范、位流 I/O 与 NALU 解析/构造设计及一致性测试向量（MIT License，见 [third_party/libmlvc/LICENSE](third_party/libmlvc/LICENSE)）。
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)：提供 CUDA Tensor Core 卷积算子构件。
- [Protobuf](https://github.com/protocolbuffers/protobuf) 与 [Abseil](https://github.com/abseil/abseil-cpp)：提供私有静态链接的 Lite 二进制模型序列化能力。
- 本项目采用 [MIT License](LICENSE) 开源。
