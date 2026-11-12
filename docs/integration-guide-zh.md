# MLVC Driver+cubin 使用与集成

SDK 面向 Linux x86_64 NVIDIA GPU。发布包内含模型、AOT 调度和 CUDA 内核，
运行时需要 NVIDIA 驱动及匹配的系统 C/C++ 运行库。CUDA Toolkit 和 Python
只用于构建及模型准备。

当前只提供 Driver+cubin。公开 C 接口使用 ABI 2，不维护旧结构体兼容。
原始 checkpoint 直接生成的 AOT 图采用 schema 3，旧模型包需要重新生成。

## 安装

```bash
tar -xzf mlvc_cpp-<version>-driver-cubin-nvidia-linux-x86_64.tar.gz -C /opt/mlvc
export MLVC_PREFIX=/opt/mlvc/mlvc_cpp-<version>-driver-cubin-nvidia-linux-x86_64
(cd "$MLVC_PREFIX" && sha256sum -c SHA256SUMS)
ldd "$MLVC_PREFIX/lib/libmlvc_codec.so"
"$MLVC_PREFIX/bin/mlvc_driver_probe" --iterations 100
"$MLVC_PREFIX/bin/mlvc_demo" --list-model-profiles
```

检查 `BUILD-MANIFEST.txt` 中的构建信息和 GPU 架构。仅满足 Linux x86_64 并不足以
保证二进制兼容，还要检查归档所需的 glibc、libstdc++ 符号版本和 NVIDIA 驱动。
保持 `bin/`、`lib/`、`include/`、`share/` 的相对目录布局。

正式包内嵌 `mlvc-psnr-v1` 和 `mlvc-s-psnr-v1`，默认使用前者。带 `-ci` 后缀
的包不含模型，只能用于编译检查。

## 输入和输出

输入为 8 位 planar YUV420/I420：完整 Y 平面后依次放置 U、V 平面。
宽高必须为正偶数，每帧大小为 `width * height * 3 / 2` 字节。
模型使用固定画布，常用配置为输入 640×360、画布 640×368；程序补边并在输出时裁剪。

```bash
"$MLVC_PREFIX/bin/mlvc_demo" encode \
  --input input.yuv --output output.mlvc \
  --width 640 --height 360 --frames 60 --q-index 21 \
  --model-profile mlvc-s-psnr-v1 --device-id 0
"$MLVC_PREFIX/bin/mlvc_demo" decode \
  --input output.mlvc --output reconstructed.yuv \
  --width 640 --height 360 --frames 60 \
  --model-profile mlvc-s-psnr-v1 --device-id 0
```

`--frames 0` 处理到 EOF。码流每帧只保存原始 q-index、payload 长度和 rANS 数据，
解码端必须另行获得相同 profile、原始宽高和 GOP 规则。
`--model-dir` 可指定完整的 AOT 模型目录，适合开发时覆盖内嵌模型。

## 连续视频与双 GPU

每个 CLI 进程保持自己的编码或解码状态。普通文件应先编码再解码；需要边编码边
解码时使用管道或 FIFO。`-` 代表标准输入或标准输出，二进制输出时状态日志走 stderr。

```bash
cat input.yuv \
  | "$MLVC_PREFIX/bin/mlvc_demo" encode --input - --output - \
      --width 640 --height 360 --frames 0 --q-index 21 \
      --encode-device-id 0 2>encode.log \
  | "$MLVC_PREFIX/bin/mlvc_demo" decode --input - --output - \
      --width 640 --height 360 --frames 0 \
      --decode-device-id 1 2>decode.log \
  > reconstructed.yuv
```

编码器和解码器各自维护参考特征，默认每 64 帧清零。需要从同一个 GOP 起点开始，
不能丢弃任意 P 帧后继续使用旧状态。FIFO 提供背压，但不等于完整的实时传输协议。

摄像头可由 FFmpeg 转成 YUV420 后送入编码器，避免每帧重启进程：

```bash
ffmpeg -f v4l2 -i /dev/video0 -vf scale=640:360 -pix_fmt yuv420p -f rawvideo - \
  | "$MLVC_PREFIX/bin/mlvc_demo" encode --input - --output camera.mlvc \
      --width 640 --height 360 --frames 0 --q-index 21
```

## C 接口与 CMake

```c
#include <mlvc/codec.h>

mlvc_codec_options options;
mlvc_codec_options_init(&options);
options.width = 640;
options.height = 360;
options.frames = 60;
options.device_id = 0;
options.q_index = 21;
options.input_path = "input.yuv";
options.output_path = "output.mlvc";
options.model_dir = "embedded:mlvc-s-psnr-v1";

mlvc_codec_stats stats;
mlvc_codec_stats_init(&stats);
char error[512];
int rc = mlvc_encode(&options, &stats, error, sizeof(error));
```

`mlvc_decode()` 使用相同结构体。返回 0 表示成功，-1 表示失败，错误信息写入
`error`。必须用初始化函数设置 `struct_size` 和 `abi_version`。
一次调用处理一段输入，直到帧数或 EOF；目前公开接口不提供逐帧内存会话。
路径 `-` 和 FIFO 在 C 接口中同样有效。

```cmake
find_package(mlvc_codec CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE mlvc::mlvc_codec)
```

配置消费者时把安装前缀传给 `CMAKE_PREFIX_PATH`。对外只安装 C 头文件，
内部 C++、CUDA 和 AOT 类型不属于 SDK 接口。

## 服务集成

浏览器、手机和微信小程序通过业务协议调用 Linux GPU 服务。服务端负责会话、
原始宽高、profile、GOP 起点及码流传输；预览客户端需要可播放的输出格式。
当前仓库提供编解码库和 CLI，HTTP/WebSocket 服务需由应用实现。

连续会话应复用同一个编解码进程，限制队列长度并明确背压和断流后的重启规则。
服务端选择模型与文件路径，不接受客户端提交任意服务器路径。

## 验证边界

`--debug-dir` 输出命名输入输出张量，用于定位模型、QP 和参考特征差异。
启用它会关闭设备常驻参考状态，性能不能代表正常路径。

两帧 I/P 兼容性测试检查官方 Python 码流、编码后自解码、重建图像和中间张量。
它不代表任意视频内容、长时间运行或压缩率/画质已经验收。详细规则见
[兼容性契约](codec-compatibility.md)，模型生成与构建步骤见 [README](../README.md)。
