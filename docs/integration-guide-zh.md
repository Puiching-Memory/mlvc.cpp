# MLVC NVIDIA Linux SDK 使用与集成手册

本文面向将 MLVC 编解码能力接入 Linux 服务、命令行工具或摄像头演示的开发者。
当前版本为 `0.1.0` 性能调优版本，C ABI 和码流外层协议仍可能调整，不承诺后续
版本保持二进制兼容。

SDK 是 Linux x86_64 NVIDIA GPU 原生软件，不能直接加载到 Android、iOS、浏览器
或微信小程序中。客户端演示需要通过 HTTP、WebSocket 或其他业务协议调用 Linux
GPU 服务。

## 1. 选择发布包

本次发布提供四个可部署归档：

| 后端         | 归档                                                     | 模型存储                                        | 主要运行时依赖                                               |
| ------------ | -------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| Driver+cubin | `mlvc_cpp-0.1.0-driver-cubin-nvidia-linux-x86_64.tar.gz` | AOT 图、权重与 fatbin 内嵌在 `libmlvc_codec.so` | NVIDIA 驱动和系统 C/C++ 运行库                               |
| ONNX Runtime | `mlvc_cpp-0.1.0-onnxruntime-nvidia-linux-x86_64.tar.gz`  | 包内 ONNX model bundle                          | NVIDIA 驱动；包内 ONNX Runtime、cuDNN 和 CUDA-X 运行库       |
| libtorch     | `mlvc_cpp-0.1.0-libtorch-nvidia-linux-x86_64.tar.gz`     | 包内 TorchScript model bundle                   | NVIDIA 驱动；包内 libtorch、cuDNN 和 CUDA-X 运行库           |
| TensorRT     | `mlvc_cpp-0.1.0-tensorrt-nvidia-linux-x86_64.tar.gz`     | 包内 ONNX model bundle，首次运行构建 engine     | NVIDIA 驱动；包内 TensorRT、ONNX parser 和 `libcudart.so.13` |

Driver+cubin 包更小、启动更快，且不依赖 CUDA Toolkit、CUDA Runtime、TensorRT、
ONNX Runtime 或 libtorch。其 fatbin 包含 `sm_75`、`sm_80`、`sm_86`、`sm_89`
原生代码和 `compute_89` PTX。

三个框架后端包都携带对应模型和运行所需的动态库。TensorRT engine 与 GPU 架构、
TensorRT 版本、模型和 workspace 配置相关，不是跨机器通用模型文件。四种后端共用
codec、GOP、熵编码和 YUV pipeline；推理实现、依赖、性能及 FP16 数值结果不同。

### 1.1 归档校验值

| 归档         | 大小（bytes） | SHA-256                                                            |
| ------------ | ------------: | ------------------------------------------------------------------ |
| Driver+cubin |   145,604,536 | `d61fd32eb7eabd6e82bce0402f9ac49b255139faed3fb31e12797cb4849ae6f1` |
| ONNX Runtime |   634,265,317 | `e7392d550e5ee56ea77db303d96b829658dc364d7a2d62905a0b830e966f2cb9` |
| libtorch     | 1,344,965,552 | `d535a6ccd55d27ac0251414ebe9b6a60eecf0eb00c15583c94ea9790173f89a8` |
| TensorRT     | 2,129,064,826 | `d823d2b8dd0bb5494fb3b2aae4e036b55e079acc15cf7647e047dde755e4ba60` |

部署前应从可信渠道获得归档，并校验完整 SHA-256。归档内的 `SHA256SUMS` 用于校验
解包后的文件。

## 2. 安装与运行条件

以 Driver+cubin 为例：

```bash
MLVC_ARCHIVE=mlvc_cpp-0.1.0-driver-cubin-nvidia-linux-x86_64.tar.gz
MLVC_PACKAGE=mlvc_cpp-0.1.0-driver-cubin-nvidia-linux-x86_64

sha256sum "packages/$MLVC_ARCHIVE"
mkdir -p /opt/mlvc
tar -xzf "packages/$MLVC_ARCHIVE" -C /opt/mlvc
export MLVC_PREFIX="/opt/mlvc/$MLVC_PACKAGE"
```

使用其他后端时，把变量中的 `driver-cubin` 替换为 `onnxruntime`、`libtorch` 或
`tensorrt`。例如 TensorRT：

```bash
MLVC_ARCHIVE=mlvc_cpp-0.1.0-tensorrt-nvidia-linux-x86_64.tar.gz
MLVC_PACKAGE=mlvc_cpp-0.1.0-tensorrt-nvidia-linux-x86_64
```

部署主机必须满足：

- Linux x86_64；
- 一块可用且受当前 SDK 支持的 NVIDIA GPU；
- 能加载包内 CUDA 代码和运行库的 NVIDIA 驱动；
- 系统提供 glibc、`libstdc++.so.6`、`libgcc_s.so.1`、`libm.so.6` 等基础运行库；
- 有权限创建输出文件；TensorRT 还需要可写的 engine cache 目录。

发布包在 NVIDIA A30、驱动 595.71.05 环境完成构建与当前验证。NVIDIA 驱动本身
不随包分发，`libcuda.so.1` 必须来自部署主机或容器的 NVIDIA runtime。

Driver+cubin 不要求安装 CUDA Toolkit。TensorRT 包也不要求另装 CUDA Toolkit 或
TensorRT，但仍需要系统 C/C++ 运行库和 NVIDIA 驱动。

本次二进制的实际动态符号要求最高达到 `GLIBC_2.38`、`GLIBCXX_3.4.32`；
Driver+cubin 的 codec library 还要求 `CXXABI_1.3.15`。目标系统必须提供这些版本
或更新版本，仅仅满足“Linux x86_64”并不够。可在部署前检查：

```bash
readelf --version-info "$MLVC_PREFIX/lib/libmlvc_codec.so" \
  | grep -Eo 'GLIBC(X{2})?_[0-9.]+|CXXABI_[0-9.]+' | sort -Vu
ldd "$MLVC_PREFIX/lib/libmlvc_codec.so"
```

### 2.1 解包后检查

```bash
(cd "$MLVC_PREFIX" && sha256sum -c SHA256SUMS)
"$MLVC_PREFIX/bin/mlvc_demo" --backend-name
```

第二条命令应输出 `driver-cubin`、`onnxruntime`、`libtorch` 或 `tensorrt`，并与
所选归档一致。四种归档都可以列出可用模型：

```bash
"$MLVC_PREFIX/bin/mlvc_demo" --list-model-profiles
```

当前应输出：

```text
mlvc-psnr-v1
mlvc-s-psnr-v1
```

归档主要内容如下：

```text
bin/mlvc_demo               CLI 编解码器
bin/mlvc_backend_bench      模型级 benchmark 工具
lib/libmlvc_codec.so        C ABI/C++ codec 共享库
include/mlvc/codec.h        C ABI 头文件
lib/cmake/mlvc_codec/       relocatable CMake package
BUILD-MANIFEST.txt          后端、构建环境和模型清单
SHA256SUMS                  包内文件校验值
```

Driver+cubin 另外包含 `bin/mlvc_driver_probe`。ONNX Runtime、libtorch 和 TensorRT
归档另外包含后端动态库、运行库许可和 `share/mlvc/models/`。

## 3. 模型、输入和输出约束

四个归档都包含两个互相独立的模型 profile：

- `mlvc-psnr-v1`：默认主模型；
- `mlvc-s-psnr-v1`：较小模型。

编码端和解码端必须使用相同 profile、画布尺寸和 GOP 规则。不要把两个 profile
的模型文件或状态混在同一会话中。

所有归档默认选择 `mlvc-psnr-v1`，也可以显式选择：

```bash
--model-profile mlvc-psnr-v1
--model-profile mlvc-s-psnr-v1
```

Driver+cubin 从共享库读取内嵌模型。其余三个后端自动从归档中的对应目录读取模型：

```text
$MLVC_PREFIX/share/mlvc/models/mlvc-psnr-v1/640x368/
$MLVC_PREFIX/share/mlvc/models/mlvc-s-psnr-v1/640x368/
```

ONNX Runtime 和 TensorRT 目录包含 `MLVCEncoder.onnx`、`MLVCDecoder.onnx`；libtorch
目录包含 `MLVCEncoder.ts`、`MLVCDecoder.ts`。三个后端的目录都包含两个 PMF、
`metadata.json` 和 `model_bundle.json`。无需传 `--model-dir`；该参数只保留作自定义
模型的显式覆盖。不要只替换其中某个文件；运行库会校验 profile、模型版本、固定
尺寸和协议标识。

### 3.1 YUV 输入

当前模型画布是 `640x368`。输入必须是 8-bit planar YUV420/I420：完整 Y 平面，
随后完整 U、V 平面。每帧字节数为：

```text
width * height * 3 / 2
```

宽高必须是正偶数，并满足以下任一条件：

```text
width <= 640 && height <= 368
height <= 640 && width <= 368
```

第二种情况会在模型输入和输出处转置方向。未占满的画布区域使用边缘值在右侧和
底部填充，输出恢复为调用方指定的原始宽高。

当前两个 profile 的原始 `q-index` 范围都是 `0..63`。数值越大并不应在没有实际
率失真测量时简单解释为固定的“质量等级”。

## 4. CLI 文件模式

以下示例假设已有 640x360 I420 文件。可以用 FFmpeg 生成测试输入：

```bash
ffmpeg -f lavfi -i testsrc2=size=640x360:rate=30 -frames:v 48 \
  -pix_fmt yuv420p -f rawvideo input.yuv
```

### 4.1 四种发布包

编码：

```bash
"$MLVC_PREFIX/bin/mlvc_demo" encode \
  --input input.yuv --output output.mlvc \
  --width 640 --height 360 --frames 48 --q-index 21 \
  --model-profile mlvc-psnr-v1 --device-id 0
```

解码：

```bash
"$MLVC_PREFIX/bin/mlvc_demo" decode \
  --input output.mlvc --output reconstructed.yuv \
  --width 640 --height 360 --frames 48 \
  --model-profile mlvc-psnr-v1 --device-id 0
```

省略 `--model-profile` 时使用包内的 `mlvc-psnr-v1`。上述命令适用于全部四种归档。

### 4.2 TensorRT engine cache

```bash
MLVC_ENGINE_CACHE=/srv/mlvc-engine-cache

"$MLVC_PREFIX/bin/mlvc_demo" encode \
  --input input.yuv --output output.mlvc \
  --width 640 --height 360 --frames 48 --q-index 21 \
  --engine-cache-dir "$MLVC_ENGINE_CACHE" --device-id 0

"$MLVC_PREFIX/bin/mlvc_demo" decode \
  --input output.mlvc --output reconstructed.yuv \
  --width 640 --height 360 --frames 48 \
  --engine-cache-dir "$MLVC_ENGINE_CACHE" --device-id 0
```

TensorRT 首次运行某个 GPU、模型、TensorRT 版本和 workspace 组合时会构建 engine，
启动可能明显更慢。应持久化并复用 cache 根目录，不要跨模型或 GPU 手工复制 engine。
运行库会在 cache 根目录下继续按 profile 和 engine 参数隔离文件。

### 4.3 常用参数

| 参数                   | 含义                                  |
| ---------------------- | ------------------------------------- |
| `--frames N`           | 处理 N 帧；`0` 表示持续处理到输入 EOF |
| `--q-index N`          | 编码原始 Q index；解码从每帧头读取    |
| `--device-id N`        | 当前命令使用的 CUDA ordinal           |
| `--encode-device-id N` | encode 专用的 `--device-id` 别名      |
| `--decode-device-id N` | decode 专用的 `--device-id` 别名      |
| `--model-profile NAME` | 选择包内模型 profile                  |
| `--model-dir DIR`      | 显式覆盖包内模型，仅用于自定义部署    |
| `--workspace-mib N`    | 覆盖默认 4096 MiB workspace 上限      |
| `--debug-dir DIR`      | 写出逐帧模型输入输出，供兼容性诊断    |

指定 `--debug-dir` 会关闭设备驻留状态绑定，使 `ref_feature` 和 `feature` 能被写到
host 调试目录。这条路径不代表正常运行性能。

## 5. 管道和 FIFO

`--input -` 和 `--output -` 分别使用 stdin 和 stdout。二进制数据写 stdout 时，
CLI 摘要和错误写 stderr，不会污染码流。下面是 Driver+cubin 的实时管道示例：

```bash
set -o pipefail
cat input.yuv \
  | "$MLVC_PREFIX/bin/mlvc_demo" encode \
      --input - --output - --width 640 --height 360 \
      --frames 0 --q-index 21 --model-profile mlvc-psnr-v1 \
      --encode-device-id 0 2>encode.log \
  | "$MLVC_PREFIX/bin/mlvc_demo" decode \
      --input - --output - --width 640 --height 360 \
      --frames 0 --model-profile mlvc-psnr-v1 \
      --decode-device-id 1 2>decode.log \
  > reconstructed.yuv
```

TensorRT 的两个命令应使用各自可访问的 `--engine-cache-dir`。选择非默认模型时，
编码和解码命令还必须传入相同的 `--model-profile`。

也可以使用 POSIX named FIFO。运行库会在每个 MLVC 帧写出后 flush，因此管道和
FIFO 能提供操作系统级背压。不要每帧重启 CLI；每次重启都会重建 backend，并丢失
连续 DPB 状态。

标准管道适合从会话起点开始、可靠、有序的两个长驻进程。它不是网络协议，也不
携带 profile、宽高、帧率、时间戳或会话生命周期信息。

## 6. GOP 和参考状态

`mlvc-psnr-v1` 与 `mlvc-s-psnr-v1` 的模型 metadata 都指定：

```text
iframe_period = 64
reset_period = null
ltr_period = null
```

公共 codec pipeline 按以下方式编排所有推理后端：

```text
绝对帧号    0    1    2  ...  63    64    65 ...
GOP 帧号    0    1    2  ...  63     0     1 ...
帧类型      I    P    P  ...   P     I     P ...
参考特征   zero  F0   F1 ...  F62   zero  F64 ...
```

官方称帧 0、64、128 等为 `I_FRAME/IDR`。这里没有切换到另一套图像编码网络；
公共 pipeline 清空参考状态，同一个模型以全零 `ref_feature` 处理该帧。它是可以
脱离前一个 GOP 解码的零参考刷新帧。随后 63 个 P 帧递归使用上一帧 feature。

Encoder DPB 保存编码器产生的 feature，Decoder DPB 保存解码器自己重建的 feature。
两端必须在相同帧执行 reset，但不能把编码器内部 feature 直接作为解码器状态传输。

用户提供的原始 q-index 会根据 GOP 位置生成模型使用的 shifted q-index；熵容器仍
保存原始 q-index。因此帧计数失步不仅会使用错误参考，也会使用错误的 QP phase。

正常路径将 `feature -> ref_feature` 保留在 GPU：

- Driver+cubin 使用持久设备状态；
- TensorRT 使用设备 I/O state binding；
- ONNX Runtime 使用 CUDA I/O binding；
- libtorch 保留 CUDA tensor。

GOP 判断和 QP 调度属于公共 codec pipeline，后端的 `reset_state()` 只负责执行设备
状态清零。四个后端遵循相同状态语义，但不保证产生跨后端 bit-exact 的 FP16 feature
或码流。

## 7. MLVC 帧格式及限制

当前每个 MLVC 帧的原生容器格式为：

```text
int32_le q_index | uint32_le payload_size | payload_size bytes rANS payload
```

每个 rANS payload 可以独立解析，但 P 帧重建依赖前一帧 Decoder DPB。当前帧头不
包含以下信息：

- magic、格式版本或 profile；
- width、height 或像素格式；
- session ID、frame ID 或时间戳；
- I/P 类型或 reset 标志；
- 校验和、丢包信息或结束标记。

因此原生 `.mlvc` 流只适用于双方预先约定 profile、尺寸和像素格式，并从会话第
一帧开始按顺序可靠传输的场景。EOF 或业务层结束消息用于结束 `--frames 0` 会话；
双方不需要预先知道最终帧数。

不能把日志或其他数据写入 encoder stdout。读取端必须先获得完整 8 字节头，再按
`payload_size` 读取完整 payload。

## 8. 摄像头与长时间直播

### 8.1 当前版本能做什么

摄像头可以通过 FFmpeg 转成连续 I420，再交给一个长驻 encoder：

```bash
ffmpeg -f v4l2 -framerate 30 -video_size 640x360 -i /dev/video0 \
  -an -pix_fmt yuv420p -f rawvideo - \
  | "$MLVC_PREFIX/bin/mlvc_demo" encode \
      --input - --output camera.mlvc \
      --width 640 --height 360 --frames 0 --q-index 21 \
      --model-profile mlvc-psnr-v1 --device-id 0
```

具体 V4L2 输入格式取决于摄像头；可用 `ffmpeg -f v4l2 -list_formats all
-i /dev/video0` 检查。对于 TensorRT，替换模型参数并配置 engine cache。
上述示例把结果录制到普通文件；向直播封装器逐帧发送时，应把输出改为 stdout
或 FIFO。只有 stdout/FIFO 路径有显式逐帧 flush 语义。

只要输入不断开，这个进程会继续运行，并在每 64 帧自动产生零参考刷新帧。模型
状态大小固定，不会随帧数增长。

当前 `frames` 和统计帧数使用 32 位有符号整数；它不适合作为真正永久运行服务的
长期计数。30 fps 下达到上限约需 2.27 年。生产 session 应改用 64 位 frame ID，
并定期受控轮换进程。

### 8.2 三种“丢帧”不是同一问题

| 情况                           | 后果                                               | 处理                                      |
| ------------------------------ | -------------------------------------------------- | ----------------------------------------- |
| 摄像头在编码前丢帧             | 编码器仍引用上一已编码帧，画面时间间隔增大         | 可以继续编码，保留正确时间戳              |
| 编码器主动跳过尚未编码的输入帧 | 与上一项相同                                       | 由业务层记录 frame ID/PTS，不伪造 MLVC 帧 |
| 编码后的 MLVC P 帧在传输中丢失 | Decoder DPB 与 Encoder DPB 分叉，后续 P 帧不再可靠 | 丢弃后续 P 帧，等待或请求零参考刷新       |

原生帧头没有 reset 标志，所以接收端无法仅凭 payload 判断下一帧是不是刷新帧。
也不能在任意 P 帧中途启动新 decoder。

### 8.3 当前版本的可行直播方案

在不修改 codec ABI 的前提下，业务层应为 MLVC payload 增加自己的传输 envelope：

```text
SessionHeader:
  protocol_version, session_id, profile, width, height, pixel_format,
  fps_num, fps_den, gop_period

FrameEnvelope:
  session_id, uint64 sequence_number, capture_frame_id, pts,
  flags(KEYFRAME/RESET), mlvc_frame_size,
  mlvc_frame_bytes, optional_checksum
```

`sequence_number` 是从 0 开始的 MLVC session 内序号，不能用可能跨 session 连续
增长的摄像头帧号代替。`mlvc_frame_bytes` 包含完整的原生 8 字节帧头和 rANS
payload。这层 envelope 不是当前 `.mlvc` 文件格式的一部分。

发送端应在 `sequence_number % gop_period == 0` 的帧上设置 `KEYFRAME|RESET`；
当前两个 profile 的 `gop_period` 都是 64。接收端只能从该标志帧创建或恢复
decoder，并把该帧作为 decoder 的本地 frame 0。

当前 CLI/C ABI 没有运行中的 `force_idr()`：

- 新会话启动时，第一个输入帧自然是零参考刷新帧；
- 需要立即刷新时，停止旧 encoder，创建新 session ID，并用新长驻 encoder 编码
  下一摄像头帧；
- 只重启 decoder 而继续发送旧 encoder 的 P 帧是错误的；
- 丢包后可以等到业务层标记的下一个周期刷新帧，再从该帧启动新 decoder；
- profile、分辨率或设备切换必须创建新 session。

后续生产接口应提供显式 Encoder/Decoder session、逐帧 `push/pop`、`force_idr()`、
64 位 frame ID、取消和 flush；在这些接口实现前，本文不把路径型 ABI 描述为完整
直播 SDK。

### 8.4 推荐进程拓扑

```text
摄像头 / FFmpeg / GStreamer
          |
          | I420 + PTS
          v
长驻 MLVC encoder session
          |
          | 业务 FrameEnvelope + MLVC payload
          v
可靠有序传输 / 录制文件
          |
          v
长驻 MLVC decoder session
          |
          | I420
          v
FFmpeg 转 H.264/H.265/MP4 或服务端分析
```

若传输不能保证可靠有序，必须实现丢包检测、等待刷新和 session 重建。简单地把原生
MLVC 字节连续塞进 WebSocket，并不能自动获得直播恢复能力。

## 9. C ABI 集成

当前 C ABI 是阻塞、路径型的整段流接口。一次 `mlvc_encode()` 或 `mlvc_decode()`
调用会创建一个 backend，持续处理到指定帧数或 EOF，然后销毁会话。不要每帧调用
一次，也不要把它误当作逐帧 API。

### 9.1 最小编码示例

```c
#include <mlvc/codec.h>
#include <stdio.h>

int main(void) {
    mlvc_codec_options options;
    mlvc_codec_options_init(&options);

    options.width = 640;
    options.height = 360;
    options.q_index = 21;
    options.frames = 0;              /* 一直处理到 EOF */
    options.device_id = 0;
    options.input_path = "input.yuv";
    options.output_path = "output.mlvc";

    /* 所有发布包：NULL 使用包内默认 mlvc-psnr-v1。 */
    options.model_dir = NULL;

    /* TensorRT 可设置可写 cache：
    options.engine_cache_dir = "/srv/mlvc-engine-cache";
    */

    mlvc_codec_stats stats;
    char error[1024];
    if (mlvc_encode(&options, &stats, error, sizeof(error)) != 0) {
        fprintf(stderr, "MLVC encode failed: %s\n", error);
        return 1;
    }

    fprintf(stderr, "encoded %d frames in %.3f s\n",
            stats.frames, stats.elapsed_seconds);
    return 0;
}
```

Driver+cubin 的 C ABI 可用 `"embedded:mlvc-s-psnr-v1"` 选择小模型；其他三个后端
可用 `"packaged:mlvc-s-psnr-v1"`。传入结构体的字符串指针必须在函数返回前保持
有效。失败时函数返回 `-1`，错误写入调用方 buffer；C++ 异常不会跨过 C ABI。

`workspace_mib == 0` 使用库默认的 4096 MiB。`input_path` 或 `output_path` 可以是
`"-"`，也可以是普通文件或 FIFO 路径。

### 9.2 链接

直接使用编译器：

```bash
c++ app.cpp -I"$MLVC_PREFIX/include" \
  -L"$MLVC_PREFIX/lib" -Wl,-rpath,"$MLVC_PREFIX/lib" \
  -lmlvc_codec -o app
```

使用归档内的 CMake package：

```cmake
cmake_minimum_required(VERSION 3.23)
project(my_mlvc_app LANGUAGES CXX)

find_package(mlvc_codec CONFIG REQUIRED)
add_executable(my_mlvc_app main.cpp)
target_link_libraries(my_mlvc_app PRIVATE mlvc::mlvc_codec)
```

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$MLVC_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

动态加载时可以对 `$MLVC_PREFIX/lib/libmlvc_codec.so` 使用
`dlopen(..., RTLD_NOW | RTLD_LOCAL)`，再通过 `dlsym` 获取 C ABI 符号。不要依赖
未列入 `codec.h` 的 C++ 符号保持版本兼容。

### 9.3 多 GPU

编码与解码是两个独立入口，可以分别设置：

```text
encoder options.device_id = 0
decoder options.device_id = 1
```

推荐为每条方向和会话使用独立长驻 worker。若使用 stdin/stdout，一个进程只能安全
拥有对应的标准流；多会话应使用独立进程、FIFO 或应用管理的文件描述符封装。

## 10. 浏览器和小程序演示

浏览器和微信小程序的 `<video>` 不能直接播放 MLVC 帧流或裸 I420，也不能加载本
归档的 `.so`。可采用以下服务端链路：

```text
客户端采集/上传
      |
      v
Linux GPU 服务：解复用或转换为 I420
      |
      v
MLVC encode -> 可选 MLVC decode 回环
      |
      v
FFmpeg 将重建 I420 编码为 H.264/H.265/MP4
      |
      v
客户端播放 URL 或实时媒体流
```

上传文件的演示可以使用普通 HTTP 任务。摄像头直播应使用有 session 生命周期和
FrameEnvelope 的双向连接；服务端需要自己实现鉴权、限流、超时、GPU 调度、对象
存储和媒体封装，这些组件不在 SDK 中。

不要让客户端提交任意服务器模型路径、输出路径或 engine cache 路径。服务端应把
profile、最大尺寸、QP 范围、GPU 和资源配额配置为白名单。

## 11. 常见问题

| 现象                                               | 检查和处理                                                              |
| -------------------------------------------------- | ----------------------------------------------------------------------- |
| `libmlvc_codec.so: cannot open shared object file` | 使用归档 CMake target、链接时设置 rpath，或检查应用的动态库搜索路径     |
| `libcuda.so.1` 或 CUDA 初始化失败                  | 检查 NVIDIA 驱动、容器 GPU runtime、设备权限和 `nvidia-smi`             |
| TensorRT/`libcudart.so.13` 找不到                  | 不要只复制单个 `.so`；保持归档 `bin/`、`lib/` 的相对布局                |
| `packaged model profile` 相关错误                  | 保持归档的 `lib/` 与 `share/mlvc/models/` 相对布局，并检查 profile 名称 |
| `embedded model asset not found`                   | Driver+cubin 使用 `--list-model-profiles` 检查名称，不要混用外部 bundle |
| 首次 TensorRT 启动很慢                             | 正常的 engine build；持久化并复用匹配的 cache                           |
| `YUV420 dimensions` 或 plane 截断                  | 确认偶数宽高、I420 plane 顺序和每帧 `width*height*3/2` bytes            |
| `rANS EOF`、payload 截断                           | 按 8 字节小端头读取完整 payload；检查传输是否截断或插入日志             |
| 解码不报错但画面从某帧开始异常                     | 检查 MLVC 帧丢失、乱序、profile 不同或 Encoder/Decoder reset 失步       |
| 新 decoder 从中间帧无法恢复                        | 只能从业务层标记的零参考刷新帧启动，并重建 decoder session              |
| 管道停止读取后 encoder 卡住                        | 这是正常背压；检查消费者、超时和取消策略                                |
| 小程序不能播放结果                                 | 服务端先把解码 I420 转成客户端支持的媒体格式                            |

## 12. 集成验收

部署到目标机器前至少完成：

1. 校验归档 SHA-256 和包内 `SHA256SUMS`。
2. 确认 `mlvc_demo --backend-name` 与归档一致。
3. 分别用 `mlvc-psnr-v1` 和计划使用的其他 profile 做 encode/decode 回环。
4. 覆盖帧 `0、1、63、64、65`，确认第 64 帧两端同时 reset。
5. 运行至少 100 帧，检查 payload 完整、输出帧数、视觉质量和 GPU 内存是否稳定。
6. 用管道或 FIFO 验证逐帧输出、背压、EOF 和异常退出。
7. 对业务 FrameEnvelope 测试 P 帧丢失、乱序、重复、断线重连和中途加入。
8. 确认接收端只在 `KEYFRAME|RESET` 帧创建新 decoder。
9. 分别验证目标 GPU、driver、profile、分辨率、QP 和并发数。
10. 对发布候选执行码流确定性、重建指标、码率和端到端吞吐门禁。

模型和码流兼容性证据见 [codec-compatibility.md](codec-compatibility.md)，公共 pipeline
设计见 [design.md](design.md)，当前性能数据见 [benchmarks.md](benchmarks.md)。
