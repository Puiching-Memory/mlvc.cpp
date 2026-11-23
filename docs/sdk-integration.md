# SDK Integration Guide

`mlvc.cpp` installs a stable, minimal C ABI (`libmlvc_codec` + `<mlvc/codec.h>`) with a CMake package export, while keeping internal C++ session and asset-reader APIs cleanly separated for in-tree or embedded mobile builds.

---

## 1. Public C ABI v2 (`public/include/mlvc/codec.h`)

Only [`public/include/mlvc/codec.h`](../public/include/mlvc/codec.h) is installed as part of the public SDK. All internal C++, runtime, CUDA, CUTLASS, Protobuf, and Abseil headers remain private.

### ABI Contract

- **Version**: `MLVC_CODEC_ABI_VERSION == 2` (`MLVC_CODEC_OPTIONS_V2_SIZE`, `MLVC_CODEC_STATS_V2_SIZE`).
- **Initialization**: Always call `mlvc_codec_options_init(&options)` and `mlvc_codec_stats_init(&stats)` before populating fields so `struct_size` and `abi_version` are set. Incompatible callers are rejected cleanly.
- **Error handling**: C++ exceptions never cross the ABI boundary. `mlvc_encode()` and `mlvc_decode()` return `0` on success and `-1` on failure, writing a NUL-terminated diagnostic message into `error_buffer` (when non-null and `error_capacity > 0`).
- **Entry points**:
  - `mlvc_backend_name()`: Returns the default compiled backend name (e.g. `"driver-cubin"`, `"cpu"`, or `"none"`).
  - `mlvc_encode(options, stats, error_buffer, error_capacity)`
  - `mlvc_decode(options, stats, error_buffer, error_capacity)`
- **Model selection via `options.model_dir`**:
  - `NULL` or `""`: Uses embedded `mlvc-psnr-v1` (in embedded builds).
  - `"embedded:<profile>"`: Selects a specific embedded profile (e.g. `"embedded:mlvc-s-psnr-v1"`).
  - `"/path/to/bundle"`: Loads a canonical directory bundle.
- **Streamable paths**: `input_path` / `output_path` accept regular files, `"-"` (stdin/stdout), and POSIX named FIFOs. On `mlvc_decode()`, setting `width = 0` and `height = 0` auto-detects display dimensions from bitstream `SPS`.

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

On Linux/Android/OpenHarmony ELF targets, [`cmake/codec.exports.map`](../cmake/codec.exports.map) exports only the 5 C ABI symbols plus the internal `mlvc::` entry points consumed by `mlvc_demo` (`encode_video`, `decode_video`, `EncoderSession`, `DecoderSession`, `available_model_profiles`, `available_backends`, `default_backend_name`, `compiled_backend_name_c_str`). All Protobuf Lite, Abseil, `msrtc_rans`, CUTLASS, and backend symbols are hidden (`local: *;`), preventing ODR or version conflicts in host applications that link their own Protobuf or Abseil versions.

---

## 2. Internal C++ Session & Asset APIs (In-Tree / Mobile Integration)

The public C ABI currently operates on file/pipe/FIFO streams (`mlvc_encode` / `mlvc_decode`). Applications building `mlvc.cpp` from source (for example, Android JNI, OpenHarmony NAPI, or in-process RTC media engines) can use the internal file-I/O-free C++ session API in [`codec/include/mlvc/codec/pipeline.hpp`](../codec/include/mlvc/codec/pipeline.hpp) directly:

### Per-Frame `EncoderSession` & `DecoderSession`

```cpp
#include "mlvc/codec/pipeline.hpp"

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

Instead of unpacking model bundles to temporary files on mobile devices, subclass `mlvc::ModelAssetReader` (`core/include/mlvc/core/assets.hpp`) and attach it to `SessionOptions::asset_reader`:

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
