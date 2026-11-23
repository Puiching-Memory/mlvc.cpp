# SDK Integration Guide

`mlvc.cpp` installs a stable, minimal C ABI (`libmlvc_codec` + `<mlvc/codec.h>`) with a CMake package export, while keeping internal C++ session and asset-reader APIs cleanly separated for in-tree or embedded mobile builds.

---

## 1. Public C ABI v2 / v3 (`include/mlvc/codec.h`)

Only [`include/mlvc/codec.h`](../include/mlvc/codec.h) is installed as part of the public SDK. All internal C++, runtime, CUDA, CUTLASS, Protobuf, and Abseil headers remain private.

### ABI Contract

- **Versioning**:
  - **ABI v2 (`MLVC_CODEC_ABI_VERSION == 2`)**: Preserved for full backward compatibility (`mlvc_codec_options`, `mlvc_codec_stats`, `mlvc_encode`, `mlvc_decode`, `mlvc_backend_name`).
  - **ABI v3 (`MLVC_CODEC_ABI_V3_VERSION == 3`)**: Adds structured status codes (`mlvc_status`), process-wide logging (`mlvc_set_log_callback`), runtime backend/profile discovery (`mlvc_list_backends`, `mlvc_list_model_profiles`), extended video options (`mlvc_video_options`, `mlvc_encode_video`, `mlvc_decode_video`), custom model asset callbacks (`mlvc_asset_reader`), and stateful file-I/O-free per-frame sessions (`mlvc_encoder_*`, `mlvc_decoder_*`) with strided YUV420 frames and `force_idr`.
- **Initialization**: Always call the matching `*_init()` helper (`mlvc_codec_options_init`, `mlvc_codec_stats_init`, `mlvc_video_options_init`, `mlvc_session_options_init`, `mlvc_encode_params_init`) so `struct_size` and `abi_version` are populated.
- **Error handling**: C++ exceptions never cross the ABI boundary. v2 entry points return `0` / `-1`; v3 entry points return typed `mlvc_status` codes (`MLVC_STATUS_OK`, `MLVC_STATUS_INVALID_ARGUMENT`, `MLVC_STATUS_ABI_MISMATCH`, `MLVC_STATUS_IO_ERROR`, `MLVC_STATUS_BITSTREAM_ERROR`, `MLVC_STATUS_BACKEND_ERROR`, etc.) and write a NUL-terminated diagnostic into `error_buffer`.
- **Cross-platform DLL / visibility macros**: Windows consumers automatically receive `__declspec(dllimport)` (or empty when compiling against a static `MLVC_BUILD_SHARED=OFF` target with `MLVC_CODEC_STATIC=1`).
- **Model selection via `model_dir` / `asset_reader`**:
  - `NULL` or `""`: Uses embedded `mlvc-psnr-v1` (in embedded builds).
  - `"embedded:<profile>"`: Selects a specific embedded profile (e.g. `"embedded:mlvc-s-psnr-v1"`).
  - `"/path/to/bundle"`: Loads and SHA-256-verifies a canonical directory bundle (zero-copy `mmap` for large weights).
  - `asset_reader` (v3): Custom C callbacks (`exists`, `read_binary`, `free_buffer`) for in-memory or mobile asset-pack loading without unpacking to disk.
- **Streamable paths**: `input_path` / `output_path` accept regular files, `"-"` (stdin/stdout, automatically switched to binary mode on Windows), and POSIX named FIFOs. On decode, setting `width = 0` and `height = 0` auto-detects display dimensions from bitstream `SPS`.

### Minimal C Example

```c
#include <mlvc/codec.h>
#include <stdio.h>

int encode_file(void) {
    mlvc_codec_options options;
    mlvc_codec_options_init(&options);
    options.width = 640;
    options.height = 360;
    options.frames = 60;
    options.q_index = 21;
    options.device_id = 0;
    options.input_path = "input.yuv";
    options.output_path = "output.mlvc";
    options.model_dir = "embedded:mlvc-s-psnr-v1";

    mlvc_codec_stats stats;
    mlvc_codec_stats_init(&stats);
    char error[512];

    if (mlvc_encode(&options, &stats, error, sizeof(error)) != 0) {
        fprintf(stderr, "mlvc_encode failed: %s\n", error);
        return -1;
    }
    printf("Encoded %d frames (%llu -> %llu bytes) in %.3f s\n",
           stats.frames,
           (unsigned long long)stats.input_bytes,
           (unsigned long long)stats.output_bytes,
           stats.elapsed_seconds);
    return 0;
}
```

### CMake Consumption

Pass the installed release prefix to `CMAKE_PREFIX_PATH` and link the imported target `mlvc::mlvc_codec`:

```cmake
cmake_minimum_required(VERSION 3.23)
project(my_mlvc_consumer LANGUAGES C)

find_package(mlvc_codec CONFIG REQUIRED)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE mlvc::mlvc_codec)
```

See [`tests/integration/cmake_consumer/`](../tests/integration/cmake_consumer/) for the complete verified consumer example.

### Exported Symbol Policy

All targets compile with hidden default visibility (`-fvisibility=hidden` on Unix/macOS) and explicit `MLVC_CODEC_API` export annotations (no `WINDOWS_EXPORT_ALL_SYMBOLS` on MSVC). On Linux/Android/OpenHarmony ELF targets, [`cmake/codec.exports.map`](../cmake/codec.exports.map) further locks the dynamic symbol table strictly to the pure C ABI `mlvc_*` functions (`local: *;`). `mlvc_demo` itself consumes only the public C ABI. All C++ internals, Protobuf Lite, Abseil, `msrtc_rans`, CUTLASS, and backend symbols remain strictly private, preventing ODR or version conflicts across Linux, Windows, and macOS hosts.

---

## 2. Internal C++ Session & Asset APIs (In-Tree / Mobile Integration)

The public C ABI currently operates on file/pipe/FIFO streams (`mlvc_encode` / `mlvc_decode`). Applications building `mlvc.cpp` from source (for example, Android JNI, OpenHarmony NAPI, or in-process RTC media engines) can use the internal file-I/O-free C++ session API in [`src/codec/pipeline.hpp`](../src/codec/pipeline.hpp) directly:

### Per-Frame `EncoderSession` & `DecoderSession`

```cpp
#include "codec/pipeline.hpp"

mlvc::SessionOptions opts;
opts.width = 640;
opts.height = 360;
opts.q_index = 21;
opts.backend_name = "cpu";   // or "driver-cubin", or "" for default
opts.num_threads = 4;        // 0 = auto big/prime cores on CPU
opts.model_dir = "/data/local/tmp/mlvc-s-psnr-v1/640x368";

mlvc::EncoderSession encoder(opts);
mlvc::DecoderSession decoder(opts);

mlvc::EncodedFrame pkt = encoder.encode_frame(yuv_view);
// Or override QP per frame within [0, 51]:
// mlvc::EncodedFrame pkt = encoder.encode_frame(yuv_view, /*q_index=*/24);

mlvc::Yuv420Frame decoded = decoder.decode_frame(pkt);
// Or decode directly into a caller-owned buffer:
// decoder.decode_frame_into(pkt, mutable_yuv_view);
```

- Calling `reset()` clears GOP/DPB state and forces the next frame to start a fresh IDR Access Unit.
- `ExecutionPreference` (`kAuto`, `kCpu`, `kGpu`, `kNpu`) and `allow_cpu_fallback` are exposed on `SessionOptions` and `CodecOptions` for explicit device routing policies.

### Custom `ModelAssetReader` (e.g. Android `AAssetManager` / In-Memory)

Instead of unpacking model bundles to temporary files on mobile devices, subclass `mlvc::ModelAssetReader` (`src/core/assets.hpp`) and attach it to `SessionOptions::asset_reader`:

```cpp
class MemoryAssetReader final : public mlvc::ModelAssetReader {
public:
    std::string location_label() const override { return "memory:mlvc-s-psnr-v1"; }
    bool exists(const std::filesystem::path& rel) const override { /* ... */ }
    std::vector<std::byte> read_binary(const std::filesystem::path& rel) const override { /* ... */ }
};

opts.asset_reader = std::make_shared<MemoryAssetReader>(/* ... */);
```

Both `core` (metadata & PMF loaders) and `runtime` / `backends` (AOT graph & weight loaders) read exclusively through `ModelAssetReader`, so custom asset stores work seamlessly across both `cpu` and `driver-cubin`.

---

## 3. Production Deployment Guidance

### GPU Media Services (Web / Mobile / Mini-Program Backends)

`mlvc.cpp` provides the codec library and CLI; transport protocols (HTTP, WebSocket, WebRTC, gRPC) belong in the application service layer:

- **Reuse long-lived sessions or processes**: Never spawn a new process or re-create a backend per frame — model loading, weight upload, and CUDA Graph capture are one-time setup costs amortized over continuous video streams.
- **Multi-GPU isolation**: Each `mlvc_encode` / `mlvc_decode` call or session binds to its own `device_id`, allowing concurrent encoder and decoder instances on separate GPUs.
- **Backpressure & recovery**: Bound input/output queues explicitly. On stream drop or unrecoverable packet loss, reset the session and restart from a fresh IDR Access Unit (`SPS` + `PPS` + `IDR_N_LP`).
- **Path & profile hygiene**: Resolve model profiles and filesystem paths strictly on the server; never pass untrusted client strings into `options.model_dir`, `input_path`, or `output_path`.
- **Client preview**: `mlvc-nalu-v0` is a custom neural codec bitstream; browser/mini-program preview clients require either transcoding to a playable format or an in-client decoder.

### Mobile (`cpu` Backend on Android / OpenHarmony)

- Prefer **`mlvc-s-psnr-v1`** for real-time mobile workloads (`96/48/48` channels vs. `256/128/128`).
- Leave `num_threads = 0` unless profiling shows otherwise: `CpuThreadPool` automatically avoids saturating slow LITTLE cores on heterogeneous SoCs.
- Keep `MLVC_EMBED_MODELS=OFF` if distributing models via dynamic asset packs or CDN downloads, and load via `ModelAssetReader` or a canonical directory bundle.
