#pragma once

// Stable C ABI for the portable MLVC codec library.
//
// The encoder and decoder are intentionally separate entry points. Each call
// creates its selected backend on options->device_id, so applications can run
// one encoder on one GPU and one decoder on another GPU in parallel.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define MLVC_CODEC_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define MLVC_CODEC_API __attribute__((visibility("default")))
#else
#  define MLVC_CODEC_API
#endif

typedef struct mlvc_codec_options {
    int width;
    int height;
    int q_index;
    int frames;
    int device_id;
    // Zero selects the library default (4096 MiB).
    uint64_t workspace_mib;
    const char* input_path;
    const char* output_path;
    const char* model_dir;
    const char* engine_cache_dir;
    const char* debug_dir;
} mlvc_codec_options;

typedef struct mlvc_codec_stats {
    int frames;
    uint64_t input_bytes;
    uint64_t output_bytes;
    double elapsed_seconds;
} mlvc_codec_stats;

// Fill an options struct with the same defaults as the CLI. Input/output path
// pointers remain null and must be supplied by the caller. The driver-cubin
// release uses its embedded mlvc-psnr-v1 model when model_dir remains null;
// "embedded:<profile>" selects another embedded profile.
MLVC_CODEC_API void mlvc_codec_options_init(mlvc_codec_options* options);

// Returns the backend compiled into this package ("tensorrt" for the
// TensorRT release).
MLVC_CODEC_API const char* mlvc_backend_name(void);

// Return 0 on success.  On failure return -1 and write a human-readable error
// into error_buffer when it is non-null.  The error is always NUL terminated
// when error_capacity is greater than zero.
MLVC_CODEC_API int mlvc_encode(const mlvc_codec_options* options,
                               mlvc_codec_stats* stats,
                               char* error_buffer, size_t error_capacity);
MLVC_CODEC_API int mlvc_decode(const mlvc_codec_options* options,
                               mlvc_codec_stats* stats,
                               char* error_buffer, size_t error_capacity);

#ifdef __cplusplus
}  // extern "C"
#endif
