# MLVC TensorRT 可移植库集成指南

本文面向拿到以下归档、需要将 MLVC 编解码能力接入应用的开发者：

```text
packages/mlvc_cpp-0.1.0-tensorrt-nvidia-linux-x86_64.tar.gz
```

归档是 Linux x86_64 NVIDIA GPU 原生库，不是微信小程序前端 SDK。推荐先
在一台 Linux GPU 机器上把 CLI 或 C ABI 服务跑通，再由小程序通过 HTTPS
或 WebSocket 调用该服务。

## 1. 归档与运行条件

发布归档的 SHA-256 为：

```text
50699e721a6ca9a82ec6113840e81447e53bb993af4fb1bfe4a4750529580a60
```

归档大小为 `2,057,313,608` bytes（约 1.92 GiB），解包和部署时请预留足够磁盘空间。

下载后先校验并解包：

```bash
sha256sum mlvc_cpp-0.1.0-tensorrt-nvidia-linux-x86_64.tar.gz
mkdir -p /opt/mlvc
tar -xzf mlvc_cpp-0.1.0-tensorrt-nvidia-linux-x86_64.tar.gz -C /opt/mlvc
export MLVC_PREFIX=/opt/mlvc/mlvc_cpp-0.1.0-tensorrt-nvidia-linux-x86_64
```

归档主要内容：

```text
bin/mlvc_demo                         CLI 编解码器
bin/mlvc_backend_bench                TensorRT 后端基准工具
lib/libmlvc_codec.so                  稳定 C ABI 共享库
lib/libnvinfer*.so*                   TensorRT 运行时/构建资源
lib/libnvonnxparser*.so*              ONNX 解析器
lib/libcudart.so.13                   CUDA 13.3 运行时（随包分发，无需另装 CUDA）
include/mlvc/codec.h                  C ABI 头文件
include/mlvc/*.hpp                    C++ 头文件
lib/cmake/mlvc_codec/                 CMake package
BUILD-MANIFEST.txt                    构建环境与版本清单
SHA256SUMS                            归档内文件校验和
```

部署主机必须满足：

- Linux x86_64 和可用的 NVIDIA GPU；
- NVIDIA 驱动能够运行 CUDA 13.3 代码。发布包是在 NVIDIA A30、驱动
  595.71.05 环境上构建和验证的，实际部署应以 `nvidia-smi` 和驱动发布说明
  为准；
- 包内已经带 TensorRT 11.2.1 运行库、ONNX parser 和 CUDA 13.3 运行时
  （`libcudart.so.13`），不需要另装 TensorRT 或 CUDA 工具包；
- NVIDIA 驱动本身不在归档内，`libcuda.so.1` 必须由系统驱动提供；
- 首次运行某个 GPU、模型和 workspace 组合时，TensorRT 可能需要构建 engine，
  这一步可能较慢。之后使用 engine cache 启动会快很多。

启动前可检查：

```bash
nvidia-smi
"$MLVC_PREFIX/bin/mlvc_demo" --backend-name
# 预期输出：tensorrt
```

不要把这个归档直接放进 Android/iOS、微信小程序包或浏览器。`.so` 是
Linux x86_64 原生动态库，小程序端既不能加载它，也不能直接访问 NVIDIA GPU。

## 2. 准备模型目录

模型权重没有放进 2 GB 级别的运行归档，需要由部署方单独分发一个与
profile 对应的 canonical model bundle。TensorRT 需要以下文件：

```text
models/mlvc-psnr-v1/640x368/
  MLVCEncoder.onnx
  MLVCDecoder.onnx
  gaussian_pmf.json
  bit_estimator_pmf.json
  metadata.json
  model_bundle.json       # 强烈建议保留，用于校验 profile/协议身份
```

MLVC-S 使用另一套 profile，不可只替换文件名或尺寸：

```text
models/mlvc-s-psnr-v1/640x368/
  MLVCEncoder.onnx
  MLVCDecoder.onnx
  gaussian_pmf.json
  bit_estimator_pmf.json
  metadata.json
  model_bundle.json
```

所有文件必须来自同一次官方 MLVC split-model 导出。不要把一个导出的
`metadata.json` 与另一个导出的 PMF 或 ONNX 混用。`model_bundle.json` 存在
时，运行库会检查 profile、模型版本、固定尺寸以及 `mlvc-frame-le-v1` /
`canonical-pmf-v1` 协议标识。

当前发布包针对固定模型形状 `640x368`，最常用的输入是偶数尺寸
`640x360`。输入是 8-bit、planar YUV420（I420：完整 Y 平面，随后完整 U、V
平面），每帧字节数为 `width * height * 3 / 2`。宽高必须为正偶数，且按原方向
或交换宽高后必须能放入模型画布；运行库会对后者执行现有的旋转尺寸处理。

## 3. CLI：文件模式

编码 YUV420 到 MLVC 帧流：

```bash
"$MLVC_PREFIX/bin/mlvc_demo" encode \
  --input input.yuv \
  --output output.mlvc \
  --width 640 --height 360 \
  --frames 60 --q-index 21 \
  --model-dir /srv/models/mlvc-psnr-v1/640x368 \
  --engine-cache-dir /srv/mlvc-engine-cache \
  --device-id 0
```

解码 MLVC 到 YUV420：

```bash
"$MLVC_PREFIX/bin/mlvc_demo" decode \
  --input output.mlvc \
  --output reconstructed.yuv \
  --width 640 --height 360 \
  --frames 60 \
  --model-dir /srv/models/mlvc-psnr-v1/640x368 \
  --engine-cache-dir /srv/mlvc-engine-cache \
  --device-id 1
```

`--frames 0` 表示一直处理到输入 EOF。编码时 `--q-index` 必须在模型支持的
范围内；解码会从每帧容器头读取 q-index。`--workspace-mib` 可覆盖默认的
4096 MiB TensorRT workspace，例如 `--workspace-mib 8192`。调试模型输入、
输出张量时可加 `--debug-dir /tmp/mlvc-debug`。

`--engine-cache-dir` 是 cache 根目录，运行库会自动按 `metadata.json` 的
profile 名称隔离 MLVC 与 MLVC-S，并按 GPU 架构、TensorRT 版本、workspace 和
FP16 配置命名 engine。不要在不同模型或不同 GPU 间手工复制 engine 文件。

## 4. CLI：stdin/stdout 流式模式

`--input -` 和 `--output -` 分别选择标准输入和标准输出。输出为二进制时，
CLI 的摘要和错误会写到 stderr，不会污染 stdout。下面的命令让 GPU 0 编码、
GPU 1 解码，并且边编码边解码：

```bash
set -o pipefail
cat input.yuv \
  | "$MLVC_PREFIX/bin/mlvc_demo" encode \
      --input - --output - --width 640 --height 360 --frames 0 --q-index 21 \
      --model-dir /srv/models/mlvc-psnr-v1/640x368 \
      --engine-cache-dir /srv/mlvc-engine-cache --device-id 0 2>encode.log \
  | "$MLVC_PREFIX/bin/mlvc_demo" decode \
      --input - --output - --width 640 --height 360 --frames 0 \
      --model-dir /srv/models/mlvc-psnr-v1/640x368 \
      --engine-cache-dir /srv/mlvc-engine-cache --device-id 1 2>decode.log \
  > reconstructed.yuv
```

也可以用 POSIX named FIFO；运行库会在每帧写出后 flush，适合实时生产者和
消费者之间做背压。标准管道适合两个长驻进程的串联，避免中间临时文件；但
它不是 HTTP 协议，也不携带视频帧率、色彩空间、profile 等业务元数据，应用
需要在管道外维护这些信息。

每个 MLVC 帧的容器格式是小端序：

```text
int32_le q_index | uint32_le payload_size | payload_size bytes rANS payload
```

解码端必须从帧边界开始读取，不能把日志写入编码器 stdout。输入 YUV 则是
连续的裸 I420 帧，没有长度头；双方必须预先约定宽、高和帧数。

### stdin/stdout 是否是理想调用方式？

对于 Linux 上的两个独立 worker，答案是“适合但不是所有场景的最佳选择”：

- 适合：快速集成、进程隔离、自然背压、无需落盘、可直接把编码 GPU 和解码
  GPU 分开；
- 不足：每个 CLI 进程只有一条会话，业务元数据需另传，异常恢复和多路会话
  由上层负责；
- 不建议：每帧重新启动 CLI 或每帧调用一次 C ABI，这会重复初始化 TensorRT
  backend，并失去 DPB 的连续状态。

生产环境通常保持一个编码 worker 和一个解码 worker 长驻，通过管道、FIFO 或
应用内队列传输连续帧。需要更低延迟时，应在一个长驻 C++ 进程中复用设备、
TensorRT context、DPB 和 pinned buffer；当前归档的稳定 C ABI 是“路径型、整段
流调用”，还没有逐帧 `push/pop` session ABI。

## 5. C ABI 集成

### 5.1 直接链接

最小 C/C++ 调用流程如下。`mlvc_encode` 和 `mlvc_decode` 是两个独立入口；
`options.device_id` 只影响当前方向，因此可以在两个 worker 中分别设置为 0
和 1。`workspace_mib` 的单位是 MiB，设置为 0 使用库默认的 4096 MiB。

```c
#include <mlvc/codec.h>
#include <stdio.h>

int main(void) {
    mlvc_codec_options options;
    mlvc_codec_options_init(&options);
    options.width = 640;
    options.height = 360;
    options.q_index = 21;
    options.frames = 0;                 /* 读到 EOF */
    options.device_id = 0;              /* 编码 GPU */
    options.input_path = "input.yuv";  /* 也可以是 "-" */
    options.output_path = "output.mlvc";
    options.model_dir = "/srv/models/mlvc-psnr-v1/640x368";
    options.engine_cache_dir = "/srv/mlvc-engine-cache";
    /* workspace_mib == 0 使用库默认 4096 MiB */

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

编译时使用 `c++` 或 CMake 链接共享库，并让运行时找到同一归档的 `lib`：

```bash
c++ app.cpp -I"$MLVC_PREFIX/include" \
  -L"$MLVC_PREFIX/lib" -Wl,-rpath,"$MLVC_PREFIX/lib" \
  -lmlvc_codec -o app
```

如果使用动态加载而不是链接，可以对
`$MLVC_PREFIX/lib/libmlvc_codec.so` 调用 `dlopen(..., RTLD_NOW | RTLD_LOCAL)`，
再用 `dlsym` 获取 `mlvc_codec_options_init`、`mlvc_encode` 或 `mlvc_decode`。
无论哪种方式，都不要让 C++ 异常跨过 C ABI；失败时函数返回 `-1`，错误文本
写入调用方提供的 buffer，并保证在容量大于零时 NUL 结尾。

### 5.2 CMake

归档自带 relocatable CMake package：

```cmake
cmake_minimum_required(VERSION 3.23)
project(my_mlvc_app LANGUAGES CXX)

find_package(mlvc_codec CONFIG REQUIRED)
add_executable(my_mlvc_app main.cpp)
target_link_libraries(my_mlvc_app PRIVATE mlvc::mlvc_codec)
```

配置时把归档根目录加入 `CMAKE_PREFIX_PATH`：

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$MLVC_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

导出的 target 会自动添加 `include`，共享库自身使用 `$ORIGIN` 查找归档内
TensorRT 库。容器或 systemd 部署时仍建议把 `MLVC_PREFIX` 固定为只读版本
目录，并在启动探针中执行一次 `mlvc_demo --backend-name`。

### 5.3 两块 GPU 的编解码

当前 ABI 没有把“编码 GPU”和“解码 GPU”放在同一个 options 结构中的复合调用，
这是有意设计：方向是独立入口。创建两个 worker，各自构造 options 并调用：

```text
encoder options.device_id = 0;  mlvc_encode(...)
decoder options.device_id = 1;  mlvc_decode(...)
```

两个调用可以由两个线程或两个进程并行，但每个方向都应保持自己的连续输入
流，以便 DPB 顺序与帧号一致。

## 6. 微信小程序演示架构

### 6.1 推荐拓扑

微信小程序侧负责 UI、采集和网络；Linux GPU 服务负责 MLVC。典型链路是：

```text
小程序 wx.chooseVideo / 相机
        │ HTTPS 上传 MP4/HEVC 或预处理后的 I420
        ▼
Linux x86_64 GPU API 服务
  FFmpeg 解复用/转 YUV420
  mlvc_demo encode 或 libmlvc_codec.so
  (可选) mlvc_decode 做回环质量演示
  FFmpeg 将重建 YUV 编码为 H.264/H.265/MP4
        ▼
小程序通过 HTTPS URL 播放或下载
```

小程序的 `<video>` 组件不能直接播放 MLVC 帧流或裸 YUV。因此演示“重建视频”
时，服务端需要把 `mlvc_decode` 的 YUV 输出交给 FFmpeg，生成小程序可播放的
MP4/H.264（或业务允许的 H.265）再返回 URL。若要在客户端直接播放 MLVC，
需要另行开发 WebAssembly/JavaScript 解码器；本归档不提供该部分，也不应把
`.so` 转成小程序资源来尝试加载。

### 6.2 服务 API 建议

不要让客户端传入任意服务器路径。服务端为每个 profile 预配置固定的模型
目录、分辨率上限、QP 范围、engine cache 和 GPU。一个最小 API 可以是：

```text
POST /v1/encode
  headers: X-MLVC-Profile: mlvc-psnr-v1
  body: application/octet-stream (I420 连续帧，或服务端约定的上传视频)
  query: width=640&height=360&frames=0&q_index=21
  response: application/octet-stream (mlvc-frame-le-v1)

POST /v1/decode
  headers: X-MLVC-Profile: mlvc-psnr-v1
  body: application/octet-stream (MLVC 帧流)
  query: width=640&height=360&frames=0
  response: application/octet-stream (I420)，或异步任务 URL
```

实时演示可以使用 WebSocket 二进制消息承载连续 I420/MLVC 数据，但建议在
应用协议中增加会话 ID、profile、width、height、帧率、结束标记和错误码。
服务内部仍可把一条会话绑定到一个长驻 CLI worker 的 stdin/stdout；不要为每
一帧创建进程。

小程序上传原始视频并等待服务端返回可播放结果时，客户端代码可以保持很薄：

```js
wx.chooseVideo({
  sourceType: ['album', 'camera'],
  success: (pick) => {
    wx.uploadFile({
      url: 'https://api.example.com/v1/reconstruct',
      filePath: pick.tempFilePath,
      name: 'file',
      formData: {
        profile: 'mlvc-psnr-v1',
        width: '640',
        height: '360',
        q_index: '21'
      },
      success: (res) => {
        const result = JSON.parse(res.data);
        // result.url 指向服务端生成的 MP4/H.264 文件
        this.setData({ videoSrc: result.url });
      }
    });
  }
});
```

`/v1/reconstruct` 是业务服务需要自行实现的包装接口，不是归档中的可执行
文件。服务端可以选择“编码后立即解码再转 MP4”来展示回环结果，也可以只编码
并把 MLVC 归档到对象存储。FFmpeg、HTTP/WebSocket 框架、鉴权和对象存储均不
随本归档提供。

### 6.3 服务端 CLI 包装示意

离线任务可以由服务端先落临时文件，再调用 CLI；实时任务则用管道。下面只
展示进程边界，不是现成的 HTTP 服务：

```bash
# 服务端已经把上传内容转换成 I420，并完成权限/尺寸校验
ffmpeg -i upload.mp4 -f rawvideo -pix_fmt yuv420p - \
  | "$MLVC_PREFIX/bin/mlvc_demo" encode --input - --output - \
      --width 640 --height 360 --frames 0 --q-index 21 \
      --model-dir /srv/models/mlvc-psnr-v1/640x368 \
      --engine-cache-dir /srv/mlvc-engine-cache --device-id 0 \
      > result.mlvc
```

如果要生成可播放的重建视频：

```bash
"$MLVC_PREFIX/bin/mlvc_demo" decode --input result.mlvc --output - \
  --width 640 --height 360 --frames 0 \
  --model-dir /srv/models/mlvc-psnr-v1/640x368 \
  --engine-cache-dir /srv/mlvc-engine-cache --device-id 1 \
  | ffmpeg -f rawvideo -pix_fmt yuv420p -s 640x360 -r 30 -i - \
      -c:v libx264 -movflags +faststart reconstructed.mp4
```

生产服务应使用独立 worker 进程或长驻进程、请求超时、最大帧数/字节数限制、
磁盘配额和鉴权。stdout 只放二进制数据，日志统一写 stderr 或服务日志系统。

## 7. 常见错误排查

| 现象                                               | 处理                                                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `libmlvc_codec.so: cannot open shared object file` | 设置 `LD_LIBRARY_PATH="$MLVC_PREFIX/lib"`，或使用上面的 rpath/CMake target。                      |
| `libcuda.so.1`、CUDA 初始化失败                    | 检查 NVIDIA 驱动、容器的 GPU runtime 和 `nvidia-smi`；驱动不随包提供。                            |
| `cannot open model metadata` / ONNX 找不到         | `--model-dir` 必须指向包含 `metadata.json`、两个 ONNX 和两个 PMF 的目录。                         |
| `model bundle manifest does not match metadata`    | 不要混用不同导出的模型文件；重新部署同一个 canonical bundle。                                     |
| `YUV420 dimensions` 或输出帧数不足                 | 输入必须是偶数宽高的 planar I420；确认 `width * height * 3 / 2` 字节是否完整。                    |
| 首次启动很慢                                       | 正常的 TensorRT engine build；保留并复用 `--engine-cache-dir`，且为每个 profile 隔离 cache。      |
| 管道下游无法解码                                   | 检查上游 stdout 是否混入日志；CLI 摘要应在 stderr，且 encode/decode 的宽高、profile、帧顺序一致。 |
| rANS EOF、payload 截断或失步                       | 不要改写 8 字节帧头；按 `int32_le q_index + uint32_le payload_size` 读取完整 payload。            |
| 小程序无法播放结果                                 | MLVC/YUV 不是 `<video>` 支持的容器；服务端用 FFmpeg 转为 MP4/H.264/H.265。                        |

## 8. 集成验收清单

部署到其他机器前建议至少完成：

1. 校验归档 SHA-256，确认 `mlvc_demo --backend-name` 输出 `tensorrt`。
2. 用一段已知 I420 输入完成 2 帧 encode/decode 回环，检查输出帧数和文件大小。
3. 用 `--frames 100` 做长流测试，确认没有 truncated payload、rANS EOF、帧错位或
   解码中止，并抽帧转成 PNG/MP4 做视觉检查。
4. 分别在 `device_id=0` 和 `device_id=1` 启动编码/解码 worker，确认两块 GPU
   都有负载且 engine cache 没有串 profile。
5. 在服务层加入输入尺寸、帧数、字节数、QP、超时和并发限制，再开放给小程序。
