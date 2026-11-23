#pragma once

// C ABI version 2 for the MLVC Driver+cubin codec library.
//
// The encoder and decoder are intentionally separate entry points. Each call
// creates a Driver+cubin runtime on options->device_id, so applications can run
// one encoder on one GPU and one decoder on another GPU in parallel.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#  define MLVC_CODEC_NOEXCEPT noexcept
#else
#  define MLVC_CODEC_NOEXCEPT
#endif

#define MLVC_CODEC_ABI_VERSION 2u

#if defined(_WIN32)
#  define MLVC_CODEC_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define MLVC_CODEC_API __attribute__((visibility("default")))
#else
#  define MLVC_CODEC_API
#endif

typedef struct mlvc_codec_options {
    uint32_t struct_size;
    uint32_t abi_version;
    int width;
    int height;
    int q_index;
    int frames;
    int device_id;
    const char* input_path;
    const char* output_path;
    const char* model_dir;
    const char* debug_dir;
} mlvc_codec_options;

#define MLVC_CODEC_OPTIONS_V2_SIZE \
    ((uint32_t)(offsetof(mlvc_codec_options, debug_dir) + \
                sizeof(((mlvc_codec_options*)0)->debug_dir)))

typedef struct mlvc_codec_stats {
    uint32_t struct_size;
    uint32_t abi_version;
    int frames;
    uint64_t input_bytes;
    uint64_t output_bytes;
    double elapsed_seconds;
} mlvc_codec_stats;

#define MLVC_CODEC_STATS_V2_SIZE \
    ((uint32_t)(offsetof(mlvc_codec_stats, elapsed_seconds) + \
                sizeof(((mlvc_codec_stats*)0)->elapsed_seconds)))

// Fill an options struct with the same defaults as the CLI. Input/output path
// pointers remain null and must be supplied by the caller. Packaged releases
// use their bundled mlvc-psnr-v1 model when model_dir remains null.
// "embedded:<profile>" selects an embedded profile; a directory selects an AOT bundle.
MLVC_CODEC_API void mlvc_codec_options_init(
    mlvc_codec_options* options) MLVC_CODEC_NOEXCEPT;

// Initialize the output structure before passing it to encode/decode. The
// size and ABI fields let newer libraries reject incompatible callers without
// reading or writing beyond the caller's allocation.
MLVC_CODEC_API void mlvc_codec_stats_init(
    mlvc_codec_stats* stats) MLVC_CODEC_NOEXCEPT;

// Returns "driver-cubin".
MLVC_CODEC_API const char* mlvc_backend_name(void) MLVC_CODEC_NOEXCEPT;

// Return 0 on success.  On failure return -1 and write a human-readable error
// into error_buffer when it is non-null.  The error is always NUL terminated
// when error_capacity is greater than zero.
MLVC_CODEC_API int mlvc_encode(const mlvc_codec_options* options,
                               mlvc_codec_stats* stats,
                               char* error_buffer,
                               size_t error_capacity) MLVC_CODEC_NOEXCEPT;
MLVC_CODEC_API int mlvc_decode(const mlvc_codec_options* options,
                               mlvc_codec_stats* stats,
                               char* error_buffer,
                               size_t error_capacity) MLVC_CODEC_NOEXCEPT;

#ifdef __cplusplus
}  // extern "C"
#endif

#undef MLVC_CODEC_NOEXCEPT
