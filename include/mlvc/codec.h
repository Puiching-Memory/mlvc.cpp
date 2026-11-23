#ifndef MLVC_CODEC_H
#define MLVC_CODEC_H

/*
 * Public C ABI for the mlvc.cpp neural video codec library.
 *
 * - ABI v2 (mlvc_encode / mlvc_decode) is preserved for full backward
 *   compatibility with file, stdin/stdout ("-"), and FIFO workflows.
 * - ABI v3 adds structured status codes, log callbacks, backend/profile
 *   discovery, extended file options, custom model asset readers, and
 *   file-I/O-free per-frame Encoder/Decoder sessions with strided YUV420 I/O.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#  define MLVC_CODEC_NOEXCEPT noexcept
#else
#  define MLVC_CODEC_NOEXCEPT
#endif

#define MLVC_CODEC_ABI_VERSION 2u
#define MLVC_CODEC_ABI_V3_VERSION 3u

#if defined(MLVC_CODEC_STATIC)
#  define MLVC_CODEC_API
#elif defined(_WIN32)
#  if defined(MLVC_CODEC_BUILDING)
#    define MLVC_CODEC_API __declspec(dllexport)
#  else
#    define MLVC_CODEC_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define MLVC_CODEC_API __attribute__((visibility("default")))
#else
#  define MLVC_CODEC_API
#endif

/* Structured return codes for v3 APIs (0 = success, negative = failure). */
typedef enum mlvc_status {
    MLVC_STATUS_OK = 0,
    MLVC_STATUS_ERROR = -1,
    MLVC_STATUS_INVALID_ARGUMENT = -2,
    MLVC_STATUS_ABI_MISMATCH = -3,
    MLVC_STATUS_IO_ERROR = -4,
    MLVC_STATUS_BITSTREAM_ERROR = -5,
    MLVC_STATUS_BACKEND_ERROR = -6,
    MLVC_STATUS_OUT_OF_MEMORY = -7,
    MLVC_STATUS_UNSUPPORTED = -8
} mlvc_status;

typedef enum mlvc_log_level {
    MLVC_LOG_TRACE = 0,
    MLVC_LOG_DEBUG = 1,
    MLVC_LOG_INFO = 2,
    MLVC_LOG_WARN = 3,
    MLVC_LOG_ERROR = 4,
    MLVC_LOG_OFF = 5
} mlvc_log_level;

typedef enum mlvc_execution_preference {
    MLVC_EXEC_AUTO = 0,
    MLVC_EXEC_CPU = 1,
    MLVC_EXEC_GPU = 2,
    MLVC_EXEC_NPU = 3
} mlvc_execution_preference;

typedef enum mlvc_frame_type {
    MLVC_FRAME_I = 0,
    MLVC_FRAME_P = 1,
    MLVC_FRAME_LTR_RECOVERY = 2
} mlvc_frame_type;

typedef void (*mlvc_log_callback)(mlvc_log_level level,
                                  const char* component,
                                  const char* message,
                                  void* user_data);

/* -------------------------------------------------------------------------
 * ABI v2 (backward-compatible file/stream batch API)
 * ------------------------------------------------------------------------- */

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

/* Fill an options struct with default values. Input/output path pointers remain
 * null and must be supplied by the caller. Packaged builds with embedded models
 * default to "embedded:mlvc-psnr-v1" when model_dir remains null. */
MLVC_CODEC_API void mlvc_codec_options_init(
    mlvc_codec_options* options) MLVC_CODEC_NOEXCEPT;

/* Initialize the output stats structure before passing it to encode/decode. */
MLVC_CODEC_API void mlvc_codec_stats_init(
    mlvc_codec_stats* stats) MLVC_CODEC_NOEXCEPT;

/* Returns the default compiled backend name (e.g. "driver-cubin", "cpu", "none"). */
MLVC_CODEC_API const char* mlvc_backend_name(void) MLVC_CODEC_NOEXCEPT;

/* Return 0 on success. On failure return -1 and write a NUL-terminated
 * diagnostic into error_buffer when non-null and error_capacity > 0. */
MLVC_CODEC_API int mlvc_encode(const mlvc_codec_options* options,
                               mlvc_codec_stats* stats,
                               char* error_buffer,
                               size_t error_capacity) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API int mlvc_decode(const mlvc_codec_options* options,
                               mlvc_codec_stats* stats,
                               char* error_buffer,
                               size_t error_capacity) MLVC_CODEC_NOEXCEPT;

/* -------------------------------------------------------------------------
 * ABI v3: Observability & Runtime Discovery
 * ------------------------------------------------------------------------- */

/* Register a process-wide log callback. Passing NULL disables logging. */
MLVC_CODEC_API void mlvc_set_log_callback(
    mlvc_log_callback callback,
    void* user_data,
    mlvc_log_level min_level) MLVC_CODEC_NOEXCEPT;

/* Copy up to capacity pointers to NUL-terminated backend names into out_names
 * (when non-null) and return the total number of registered backends. Returned
 * string pointers remain valid for the lifetime of the library. */
MLVC_CODEC_API size_t mlvc_list_backends(
    const char** out_names,
    size_t capacity) MLVC_CODEC_NOEXCEPT;

/* Copy up to capacity pointers to NUL-terminated embedded model profile names
 * into out_names (when non-null) and return the total number of profiles. */
MLVC_CODEC_API size_t mlvc_list_model_profiles(
    const char** out_names,
    size_t capacity) MLVC_CODEC_NOEXCEPT;

/* -------------------------------------------------------------------------
 * ABI v3: Custom Model Asset Reader
 * ------------------------------------------------------------------------- */

typedef struct mlvc_asset_reader {
    uint32_t struct_size;
    uint32_t abi_version;
    void* user_data;
    /* Optional human-readable source label (e.g. "android-assets:mlvc-s"). */
    const char* (*location_label)(void* user_data);
    /* Return non-zero if relative_path exists in the bundle, 0 otherwise. */
    int (*exists)(void* user_data, const char* relative_path);
    /* Read relative_path into *out_data / *out_size. Return 0 on success. */
    int (*read_binary)(void* user_data,
                       const char* relative_path,
                       uint8_t** out_data,
                       size_t* out_size);
    /* Optional buffer deallocator called after the library copies the bytes. */
    void (*free_buffer)(void* user_data, uint8_t* data, size_t size);
} mlvc_asset_reader;

#define MLVC_ASSET_READER_V3_SIZE \
    ((uint32_t)(offsetof(mlvc_asset_reader, free_buffer) + \
                sizeof(((mlvc_asset_reader*)0)->free_buffer)))

/* -------------------------------------------------------------------------
 * ABI v3: Extended File/Stream Video API (multi-backend, threads, callbacks)
 * ------------------------------------------------------------------------- */

typedef struct mlvc_video_options {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t width;
    int32_t height;
    int32_t q_index;
    int32_t frames;
    int32_t device_id;
    int32_t num_threads;
    int32_t execution_preference; /* mlvc_execution_preference */
    int32_t allow_cpu_fallback;   /* non-zero = allow fallback */
    int32_t share_thread_pool;    /* non-zero = share CPU thread pool */
    const char* backend_name;
    const char* input_path;
    const char* output_path;
    const char* model_dir;
    const char* debug_dir;
    const mlvc_asset_reader* asset_reader;
} mlvc_video_options;

#define MLVC_VIDEO_OPTIONS_V3_SIZE \
    ((uint32_t)(offsetof(mlvc_video_options, asset_reader) + \
                sizeof(((mlvc_video_options*)0)->asset_reader)))

MLVC_CODEC_API void mlvc_video_options_init(
    mlvc_video_options* options) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API mlvc_status mlvc_encode_video(
    const mlvc_video_options* options,
    mlvc_codec_stats* stats,
    char* error_buffer,
    size_t error_capacity) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API mlvc_status mlvc_decode_video(
    const mlvc_video_options* options,
    mlvc_codec_stats* stats,
    char* error_buffer,
    size_t error_capacity) MLVC_CODEC_NOEXCEPT;

/* -------------------------------------------------------------------------
 * ABI v3: Stateful File-I/O-Free Per-Frame Encoder / Decoder Sessions
 * ------------------------------------------------------------------------- */

typedef struct mlvc_encoder mlvc_encoder;
typedef struct mlvc_decoder mlvc_decoder;

typedef struct mlvc_session_options {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t width;
    int32_t height;
    int32_t q_index;
    int32_t device_id;
    int32_t num_threads;
    int32_t execution_preference; /* mlvc_execution_preference */
    int32_t allow_cpu_fallback;   /* non-zero = allow fallback */
    int32_t share_thread_pool;    /* non-zero = share CPU thread pool */
    const char* backend_name;
    const char* model_dir;
    const char* debug_dir;
    const mlvc_asset_reader* asset_reader;
} mlvc_session_options;

#define MLVC_SESSION_OPTIONS_V3_SIZE \
    ((uint32_t)(offsetof(mlvc_session_options, asset_reader) + \
                sizeof(((mlvc_session_options*)0)->asset_reader)))

typedef struct mlvc_yuv420_const_frame {
    int32_t width;
    int32_t height;
    const uint8_t* y;
    size_t y_stride; /* 0 defaults to width */
    size_t y_size;   /* 0 defaults to (height - 1) * y_stride + width */
    const uint8_t* u;
    size_t u_stride; /* 0 defaults to width / 2 */
    size_t u_size;
    const uint8_t* v;
    size_t v_stride; /* 0 defaults to width / 2 */
    size_t v_size;
} mlvc_yuv420_const_frame;

typedef struct mlvc_yuv420_frame {
    int32_t width;
    int32_t height;
    uint8_t* y;
    size_t y_stride; /* 0 defaults to width */
    size_t y_size;
    uint8_t* u;
    size_t u_stride; /* 0 defaults to width / 2 */
    size_t u_size;
    uint8_t* v;
    size_t v_stride; /* 0 defaults to width / 2 */
    size_t v_size;
} mlvc_yuv420_frame;

typedef struct mlvc_encode_params {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t q_index;   /* < 0 uses the session's default q_index */
    int32_t force_idr; /* non-zero forces an IDR Access Unit */
} mlvc_encode_params;

#define MLVC_ENCODE_PARAMS_V3_SIZE \
    ((uint32_t)(offsetof(mlvc_encode_params, force_idr) + \
                sizeof(((mlvc_encode_params*)0)->force_idr)))

typedef struct mlvc_encoded_frame {
    uint8_t* data;
    size_t size;
    int32_t q_index;
    mlvc_frame_type frame_type;
} mlvc_encoded_frame;

MLVC_CODEC_API void mlvc_session_options_init(
    mlvc_session_options* options) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API void mlvc_encode_params_init(
    mlvc_encode_params* params) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API void mlvc_encoded_frame_free(
    mlvc_encoded_frame* frame) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API mlvc_status mlvc_encoder_create(
    const mlvc_session_options* options,
    mlvc_encoder** out_encoder,
    char* error_buffer,
    size_t error_capacity) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API void mlvc_encoder_destroy(
    mlvc_encoder* encoder) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API mlvc_status mlvc_encoder_reset(
    mlvc_encoder* encoder,
    char* error_buffer,
    size_t error_capacity) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API int32_t mlvc_encoder_frame_index(
    const mlvc_encoder* encoder) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API mlvc_status mlvc_encoder_encode_frame(
    mlvc_encoder* encoder,
    const mlvc_yuv420_const_frame* input_frame,
    const mlvc_encode_params* params,
    mlvc_encoded_frame* out_encoded,
    char* error_buffer,
    size_t error_capacity) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API mlvc_status mlvc_decoder_create(
    const mlvc_session_options* options,
    mlvc_decoder** out_decoder,
    char* error_buffer,
    size_t error_capacity) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API void mlvc_decoder_destroy(
    mlvc_decoder* decoder) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API mlvc_status mlvc_decoder_reset(
    mlvc_decoder* decoder,
    char* error_buffer,
    size_t error_capacity) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API int32_t mlvc_decoder_frame_index(
    const mlvc_decoder* decoder) MLVC_CODEC_NOEXCEPT;

MLVC_CODEC_API mlvc_status mlvc_decoder_decode_frame(
    mlvc_decoder* decoder,
    const uint8_t* access_unit_data,
    size_t access_unit_size,
    const mlvc_yuv420_frame* output_frame,
    char* error_buffer,
    size_t error_capacity) MLVC_CODEC_NOEXCEPT;

#ifdef __cplusplus
}  // extern "C"
#endif

#undef MLVC_CODEC_NOEXCEPT

#endif /* MLVC_CODEC_H */
