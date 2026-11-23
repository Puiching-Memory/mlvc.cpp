#include "mlvc/codec.h"

#include "codec/pipeline.hpp"
#include "core/assets.hpp"
#include "core/bitstream.hpp"
#include "core/logging.hpp"
#include "runtime/backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

struct mlvc_encoder {
    mlvc::EncoderSession session;
    int default_q_index = 21;

    explicit mlvc_encoder(const mlvc::SessionOptions& opts)
        : session(opts), default_q_index(opts.q_index)
    {
    }
};

struct mlvc_decoder {
    mlvc::DecoderSession session;

    explicit mlvc_decoder(const mlvc::SessionOptions& opts)
        : session(opts)
    {
    }
};

namespace {

class StatusError final : public std::runtime_error {
public:
    StatusError(mlvc_status code, const std::string& message)
        : std::runtime_error(message), code_(code)
    {
    }

    mlvc_status code() const noexcept { return code_; }

private:
    mlvc_status code_;
};

const char* nonnull(const char* value) noexcept
{
    return value ? value : "";
}

void write_error(std::string_view message, char* error_buffer,
                 size_t error_capacity) noexcept
{
    if (error_buffer && error_capacity != 0) {
        const size_t count = std::min(error_capacity - 1, message.size());
        if (count > 0)
            std::memcpy(error_buffer, message.data(), count);
        error_buffer[count] = '\0';
    }
}

void clear_error(char* error_buffer, size_t error_capacity) noexcept
{
    if (error_buffer && error_capacity != 0)
        error_buffer[0] = '\0';
}

mlvc_status classify_exception(const std::exception& error) noexcept
{
    if (const auto* status_err = dynamic_cast<const StatusError*>(&error))
        return status_err->code();
    if (dynamic_cast<const mlvc::BitstreamError*>(&error))
        return MLVC_STATUS_BITSTREAM_ERROR;
    if (dynamic_cast<const std::bad_alloc*>(&error))
        return MLVC_STATUS_OUT_OF_MEMORY;
    if (dynamic_cast<const std::invalid_argument*>(&error))
        return MLVC_STATUS_INVALID_ARGUMENT;
    const std::string_view what(error.what());
    if (what.find("cannot open") != std::string_view::npos ||
        what.find("cannot create") != std::string_view::npos ||
        what.find("failed to write") != std::string_view::npos ||
        what.find("failed to flush") != std::string_view::npos ||
        what.find("failed to finalize") != std::string_view::npos) {
        return MLVC_STATUS_IO_ERROR;
    }
    if (what.find("unsupported") != std::string_view::npos)
        return MLVC_STATUS_UNSUPPORTED;
    if (what.find("driver-cubin:") != std::string_view::npos ||
        what.find("cpu:") != std::string_view::npos ||
        what.find("backend") != std::string_view::npos) {
        return MLVC_STATUS_BACKEND_ERROR;
    }
    return MLVC_STATUS_ERROR;
}

template <typename Operation>
mlvc_status run_guarded(char* error_buffer, size_t error_capacity,
                        Operation&& operation) noexcept
{
    try {
        operation();
        clear_error(error_buffer, error_capacity);
        return MLVC_STATUS_OK;
    } catch (const std::exception& error) {
        write_error(error.what(), error_buffer, error_capacity);
        return classify_exception(error);
    } catch (...) {
        write_error("unknown MLVC codec error", error_buffer, error_capacity);
        return MLVC_STATUS_ERROR;
    }
}

mlvc::ExecutionPreference convert_exec_preference(int32_t value)
{
    switch (value) {
    case MLVC_EXEC_AUTO:
        return mlvc::ExecutionPreference::kAuto;
    case MLVC_EXEC_CPU:
        return mlvc::ExecutionPreference::kCpu;
    case MLVC_EXEC_GPU:
        return mlvc::ExecutionPreference::kGpu;
    case MLVC_EXEC_NPU:
        return mlvc::ExecutionPreference::kNpu;
    default:
        throw StatusError(MLVC_STATUS_INVALID_ARGUMENT,
                          "invalid execution_preference value");
    }
}

mlvc_frame_type convert_frame_type(mlvc::FrameType type) noexcept
{
    switch (type) {
    case mlvc::FrameType::kIFrame:
        return MLVC_FRAME_I;
    case mlvc::FrameType::kPFrame:
        return MLVC_FRAME_P;
    case mlvc::FrameType::kLtrRecovery:
        return MLVC_FRAME_LTR_RECOVERY;
    }
    return MLVC_FRAME_I;
}

void validate_abi_header(size_t struct_size, size_t min_size,
                         uint32_t abi_version, uint32_t expected_version,
                         const char* struct_name)
{
    if (struct_size < min_size) {
        throw StatusError(MLVC_STATUS_ABI_MISMATCH,
                          std::string(struct_name) +
                              " structure is smaller than ABI v" +
                              std::to_string(expected_version));
    }
    if (abi_version != expected_version) {
        throw StatusError(MLVC_STATUS_ABI_MISMATCH,
                          "unsupported " + std::string(struct_name) +
                              " ABI version");
    }
}

class CallbackAssetReader final : public mlvc::ModelAssetReader {
public:
    explicit CallbackAssetReader(const mlvc_asset_reader& callbacks)
        : callbacks_(callbacks)
    {
        validate_abi_header(callbacks_.struct_size, MLVC_ASSET_READER_V3_SIZE,
                            callbacks_.abi_version, MLVC_CODEC_ABI_V3_VERSION,
                            "asset_reader");
        if (!callbacks_.exists || !callbacks_.read_binary) {
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT,
                              "asset_reader must provide exists and read_binary callbacks");
        }
    }

    std::string location_label() const override
    {
        if (callbacks_.location_label) {
            const char* label = callbacks_.location_label(callbacks_.user_data);
            if (label && label[0] != '\0')
                return label;
        }
        return "custom-c-asset-reader";
    }

    bool exists(const std::filesystem::path& relative_path) const override
    {
        const std::string key = relative_path.generic_string();
        return callbacks_.exists(callbacks_.user_data, key.c_str()) != 0;
    }

    std::vector<std::byte> read_binary(
        const std::filesystem::path& relative_path) const override
    {
        const std::string key = relative_path.generic_string();
        uint8_t* data = nullptr;
        size_t size = 0;
        const int rc = callbacks_.read_binary(
            callbacks_.user_data, key.c_str(), &data, &size);
        struct BufferGuard {
            const mlvc_asset_reader* cb = nullptr;
            uint8_t* ptr = nullptr;
            size_t len = 0;
            ~BufferGuard()
            {
                if (ptr && cb && cb->free_buffer)
                    cb->free_buffer(cb->user_data, ptr, len);
            }
        } guard{&callbacks_, data, size};

        if (rc != 0) {
            throw StatusError(MLVC_STATUS_IO_ERROR,
                              "custom asset_reader failed to read: " + key);
        }
        if (size > 0 && !data) {
            throw StatusError(MLVC_STATUS_IO_ERROR,
                              "custom asset_reader returned null data for: " + key);
        }
        std::vector<std::byte> result(size);
        if (size > 0)
            std::memcpy(result.data(), data, size);
        return result;
    }

private:
    mlvc_asset_reader callbacks_{};
};

std::shared_ptr<const mlvc::ModelAssetReader> wrap_asset_reader(
    const mlvc_asset_reader* reader)
{
    if (!reader)
        return nullptr;
    return std::make_shared<CallbackAssetReader>(*reader);
}

mlvc::CodecOptions convert_options_v2(const mlvc_codec_options& source)
{
    validate_abi_header(source.struct_size, MLVC_CODEC_OPTIONS_V2_SIZE,
                        source.abi_version, MLVC_CODEC_ABI_VERSION, "options");
    mlvc::CodecOptions result;
    result.width = source.width;
    result.height = source.height;
    result.q_index = source.q_index;
    result.frames = source.frames;
    result.device_id = source.device_id;
    result.input_path = nonnull(source.input_path);
    result.output_path = nonnull(source.output_path);
    result.model_dir = nonnull(source.model_dir);
    result.debug_dir = nonnull(source.debug_dir);
    return result;
}

mlvc::CodecOptions convert_video_options(const mlvc_video_options& source)
{
    validate_abi_header(source.struct_size, MLVC_VIDEO_OPTIONS_V3_SIZE,
                        source.abi_version, MLVC_CODEC_ABI_V3_VERSION,
                        "video_options");
    mlvc::CodecOptions result;
    result.width = source.width;
    result.height = source.height;
    result.q_index = source.q_index;
    result.frames = source.frames;
    result.device_id = source.device_id;
    result.num_threads = source.num_threads;
    result.execution_preference = convert_exec_preference(source.execution_preference);
    result.allow_cpu_fallback = source.allow_cpu_fallback != 0;
    result.share_thread_pool = source.share_thread_pool != 0;
    result.backend_name = nonnull(source.backend_name);
    result.input_path = nonnull(source.input_path);
    result.output_path = nonnull(source.output_path);
    result.model_dir = nonnull(source.model_dir);
    result.debug_dir = nonnull(source.debug_dir);
    result.asset_reader = wrap_asset_reader(source.asset_reader);
    return result;
}

mlvc::SessionOptions convert_session_options(const mlvc_session_options& source)
{
    validate_abi_header(source.struct_size, MLVC_SESSION_OPTIONS_V3_SIZE,
                        source.abi_version, MLVC_CODEC_ABI_V3_VERSION,
                        "session_options");
    mlvc::SessionOptions result;
    result.width = source.width;
    result.height = source.height;
    result.q_index = source.q_index;
    result.device_id = source.device_id;
    result.num_threads = source.num_threads;
    result.execution_preference = convert_exec_preference(source.execution_preference);
    result.allow_cpu_fallback = source.allow_cpu_fallback != 0;
    result.share_thread_pool = source.share_thread_pool != 0;
    result.backend_name = nonnull(source.backend_name);
    result.model_dir = nonnull(source.model_dir);
    result.debug_dir = nonnull(source.debug_dir);
    result.asset_reader = wrap_asset_reader(source.asset_reader);
    return result;
}

void validate_stats_target(const mlvc_codec_stats* target)
{
    if (!target)
        return;
    validate_abi_header(target->struct_size, MLVC_CODEC_STATS_V2_SIZE,
                        target->abi_version, MLVC_CODEC_ABI_VERSION, "stats");
}

void copy_stats(const mlvc::CodecStats& source, mlvc_codec_stats* target) noexcept
{
    if (!target)
        return;
    target->frames = source.frames;
    target->input_bytes = static_cast<uint64_t>(source.input_bytes);
    target->output_bytes = static_cast<uint64_t>(source.output_bytes);
    target->elapsed_seconds = source.elapsed_seconds;
}

std::size_t resolve_plane_size(std::size_t explicit_size, std::size_t row_width,
                               std::size_t rows, std::size_t stride)
{
    if (explicit_size != 0U)
        return explicit_size;
    if (rows == 0U)
        return 0U;
    const std::size_t tail_rows = rows - 1U;
    if (tail_rows > 0U &&
        stride > (std::numeric_limits<std::size_t>::max() - row_width) / tail_rows) {
        throw StatusError(MLVC_STATUS_INVALID_ARGUMENT,
                          "YUV420 plane stride overflows size_t");
    }
    return tail_rows * stride + row_width;
}

template <typename ViewType, typename CFrame>
ViewType to_yuv_view(const CFrame& frame, const char* role)
{
    using Elem = std::conditional_t<
        std::is_same_v<ViewType, mlvc::Yuv420FrameView>,
        const std::uint8_t, std::uint8_t>;
    if (frame.width <= 0 || frame.height <= 0 ||
        frame.width % 2 != 0 || frame.height % 2 != 0) {
        throw StatusError(MLVC_STATUS_INVALID_ARGUMENT,
                          std::string(role) +
                              " YUV420 dimensions must be positive even integers");
    }
    if (!frame.y || !frame.u || !frame.v) {
        throw StatusError(MLVC_STATUS_INVALID_ARGUMENT,
                          std::string(role) +
                              " YUV420 plane pointers must not be null");
    }
    const std::size_t w = static_cast<std::size_t>(frame.width);
    const std::size_t h = static_cast<std::size_t>(frame.height);
    const std::size_t y_stride = frame.y_stride != 0U ? frame.y_stride : w;
    const std::size_t u_stride = frame.u_stride != 0U ? frame.u_stride : (w / 2U);
    const std::size_t v_stride = frame.v_stride != 0U ? frame.v_stride : (w / 2U);
    const std::size_t y_size = resolve_plane_size(frame.y_size, w, h, y_stride);
    const std::size_t u_size = resolve_plane_size(frame.u_size, w / 2U, h / 2U, u_stride);
    const std::size_t v_size = resolve_plane_size(frame.v_size, w / 2U, h / 2U, v_stride);
    return ViewType(frame.width, frame.height,
                    std::span<Elem>(frame.y, y_size), y_stride,
                    std::span<Elem>(frame.u, u_size), u_stride,
                    std::span<Elem>(frame.v, v_size), v_stride);
}

struct CLogBridgeState {
    std::mutex mutex;
    mlvc_log_callback callback = nullptr;
    void* user_data = nullptr;
};

CLogBridgeState& c_log_bridge() noexcept
{
    static CLogBridgeState state;
    return state;
}

void c_log_trampoline(mlvc::LogLevel level, const char* component,
                      const char* message, void* /*unused*/) noexcept
{
    mlvc_log_callback cb = nullptr;
    void* ud = nullptr;
    {
        auto& bridge = c_log_bridge();
        std::lock_guard<std::mutex> lock(bridge.mutex);
        cb = bridge.callback;
        ud = bridge.user_data;
    }
    if (cb) {
        cb(static_cast<mlvc_log_level>(static_cast<int>(level)),
           component, message, ud);
    }
}

const std::vector<std::string>& cached_model_profiles()
{
    static const std::vector<std::string> profiles = mlvc::available_model_profiles();
    return profiles;
}

}  // namespace

extern "C" {

MLVC_CODEC_API void mlvc_codec_options_init(mlvc_codec_options* options) noexcept
{
    if (!options)
        return;
    *options = mlvc_codec_options{};
    options->struct_size = MLVC_CODEC_OPTIONS_V2_SIZE;
    options->abi_version = MLVC_CODEC_ABI_VERSION;
    options->width = 640;
    options->height = 360;
    options->q_index = 21;
    options->device_id = 0;
}

MLVC_CODEC_API void mlvc_codec_stats_init(mlvc_codec_stats* stats) noexcept
{
    if (!stats)
        return;
    *stats = mlvc_codec_stats{};
    stats->struct_size = MLVC_CODEC_STATS_V2_SIZE;
    stats->abi_version = MLVC_CODEC_ABI_VERSION;
}

MLVC_CODEC_API const char* mlvc_backend_name(void) noexcept
{
    return mlvc::compiled_backend_name_c_str();
}

MLVC_CODEC_API int mlvc_encode(
    const mlvc_codec_options* options, mlvc_codec_stats* stats,
    char* error_buffer, size_t error_capacity) noexcept
{
    const mlvc_status status = run_guarded(error_buffer, error_capacity, [&]() {
        if (!options)
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT, "options must not be null");
        validate_stats_target(stats);
        copy_stats(mlvc::encode_video(convert_options_v2(*options)), stats);
    });
    return status == MLVC_STATUS_OK ? 0 : -1;
}

MLVC_CODEC_API int mlvc_decode(
    const mlvc_codec_options* options, mlvc_codec_stats* stats,
    char* error_buffer, size_t error_capacity) noexcept
{
    const mlvc_status status = run_guarded(error_buffer, error_capacity, [&]() {
        if (!options)
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT, "options must not be null");
        validate_stats_target(stats);
        copy_stats(mlvc::decode_video(convert_options_v2(*options)), stats);
    });
    return status == MLVC_STATUS_OK ? 0 : -1;
}

MLVC_CODEC_API void mlvc_set_log_callback(
    mlvc_log_callback callback, void* user_data,
    mlvc_log_level min_level) noexcept
{
    try {
        auto& bridge = c_log_bridge();
        {
            std::lock_guard<std::mutex> lock(bridge.mutex);
            bridge.callback = callback;
            bridge.user_data = callback ? user_data : nullptr;
        }
        if (callback) {
            const int clamped = std::clamp(
                static_cast<int>(min_level),
                static_cast<int>(mlvc::LogLevel::kTrace),
                static_cast<int>(mlvc::LogLevel::kOff));
            mlvc::set_log_callback(
                &c_log_trampoline, nullptr, static_cast<mlvc::LogLevel>(clamped));
        } else {
            mlvc::set_log_callback(nullptr, nullptr, mlvc::LogLevel::kOff);
        }
    } catch (...) {
    }
}

MLVC_CODEC_API size_t mlvc_list_backends(
    const char** out_names, size_t capacity) noexcept
{
    try {
        return mlvc::copy_available_backend_c_strs(out_names, capacity);
    } catch (...) {
        return 0;
    }
}

MLVC_CODEC_API size_t mlvc_list_model_profiles(
    const char** out_names, size_t capacity) noexcept
{
    try {
        const auto& profiles = cached_model_profiles();
        if (out_names && capacity > 0U) {
            const size_t count = std::min(capacity, profiles.size());
            for (size_t i = 0; i < count; ++i)
                out_names[i] = profiles[i].c_str();
        }
        return profiles.size();
    } catch (...) {
        return 0;
    }
}

MLVC_CODEC_API void mlvc_video_options_init(mlvc_video_options* options) noexcept
{
    if (!options)
        return;
    *options = mlvc_video_options{};
    options->struct_size = MLVC_VIDEO_OPTIONS_V3_SIZE;
    options->abi_version = MLVC_CODEC_ABI_V3_VERSION;
    options->width = 640;
    options->height = 360;
    options->q_index = 21;
    options->frames = 0;
    options->device_id = 0;
    options->num_threads = 0;
    options->execution_preference = MLVC_EXEC_AUTO;
    options->allow_cpu_fallback = 1;
    options->share_thread_pool = 0;
}

MLVC_CODEC_API mlvc_status mlvc_encode_video(
    const mlvc_video_options* options, mlvc_codec_stats* stats,
    char* error_buffer, size_t error_capacity) noexcept
{
    return run_guarded(error_buffer, error_capacity, [&]() {
        if (!options)
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT, "options must not be null");
        validate_stats_target(stats);
        copy_stats(mlvc::encode_video(convert_video_options(*options)), stats);
    });
}

MLVC_CODEC_API mlvc_status mlvc_decode_video(
    const mlvc_video_options* options, mlvc_codec_stats* stats,
    char* error_buffer, size_t error_capacity) noexcept
{
    return run_guarded(error_buffer, error_capacity, [&]() {
        if (!options)
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT, "options must not be null");
        validate_stats_target(stats);
        copy_stats(mlvc::decode_video(convert_video_options(*options)), stats);
    });
}

MLVC_CODEC_API void mlvc_session_options_init(
    mlvc_session_options* options) noexcept
{
    if (!options)
        return;
    *options = mlvc_session_options{};
    options->struct_size = MLVC_SESSION_OPTIONS_V3_SIZE;
    options->abi_version = MLVC_CODEC_ABI_V3_VERSION;
    options->width = 640;
    options->height = 360;
    options->q_index = 21;
    options->device_id = 0;
    options->num_threads = 0;
    options->execution_preference = MLVC_EXEC_AUTO;
    options->allow_cpu_fallback = 1;
    options->share_thread_pool = 0;
}

MLVC_CODEC_API void mlvc_encode_params_init(mlvc_encode_params* params) noexcept
{
    if (!params)
        return;
    *params = mlvc_encode_params{};
    params->struct_size = MLVC_ENCODE_PARAMS_V3_SIZE;
    params->abi_version = MLVC_CODEC_ABI_V3_VERSION;
    params->q_index = -1;
    params->force_idr = 0;
}

MLVC_CODEC_API void mlvc_encoded_frame_free(mlvc_encoded_frame* frame) noexcept
{
    if (!frame)
        return;
    std::free(frame->data);
    frame->data = nullptr;
    frame->size = 0;
    frame->q_index = 0;
    frame->frame_type = MLVC_FRAME_I;
}

MLVC_CODEC_API mlvc_status mlvc_encoder_create(
    const mlvc_session_options* options, mlvc_encoder** out_encoder,
    char* error_buffer, size_t error_capacity) noexcept
{
    if (out_encoder)
        *out_encoder = nullptr;
    return run_guarded(error_buffer, error_capacity, [&]() {
        if (!options || !out_encoder) {
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT,
                              "options and out_encoder must not be null");
        }
        auto encoder = std::make_unique<mlvc_encoder>(
            convert_session_options(*options));
        *out_encoder = encoder.release();
    });
}

MLVC_CODEC_API void mlvc_encoder_destroy(mlvc_encoder* encoder) noexcept
{
    delete encoder;
}

MLVC_CODEC_API mlvc_status mlvc_encoder_reset(
    mlvc_encoder* encoder, char* error_buffer, size_t error_capacity) noexcept
{
    return run_guarded(error_buffer, error_capacity, [&]() {
        if (!encoder)
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT, "encoder must not be null");
        encoder->session.reset();
    });
}

MLVC_CODEC_API int32_t mlvc_encoder_frame_index(
    const mlvc_encoder* encoder) noexcept
{
    return encoder ? static_cast<int32_t>(encoder->session.frame_index()) : 0;
}

MLVC_CODEC_API mlvc_status mlvc_encoder_encode_frame(
    mlvc_encoder* encoder,
    const mlvc_yuv420_const_frame* input_frame,
    const mlvc_encode_params* params,
    mlvc_encoded_frame* out_encoded,
    char* error_buffer, size_t error_capacity) noexcept
{
    if (out_encoded) {
        out_encoded->data = nullptr;
        out_encoded->size = 0;
        out_encoded->q_index = 0;
        out_encoded->frame_type = MLVC_FRAME_I;
    }
    return run_guarded(error_buffer, error_capacity, [&]() {
        if (!encoder || !input_frame || !out_encoded) {
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT,
                              "encoder, input_frame, and out_encoded must not be null");
        }
        int q_index = -1;
        bool force_idr = false;
        if (params) {
            validate_abi_header(params->struct_size, MLVC_ENCODE_PARAMS_V3_SIZE,
                                params->abi_version, MLVC_CODEC_ABI_V3_VERSION,
                                "encode_params");
            q_index = params->q_index;
            force_idr = params->force_idr != 0;
        }

        const mlvc::Yuv420FrameView view =
            to_yuv_view<mlvc::Yuv420FrameView>(*input_frame, "input");
        const int effective_q = q_index < 0 ? encoder->default_q_index : q_index;
        const mlvc::EncodedFrame encoded =
            encoder->session.encode_frame(view, effective_q, force_idr);

        uint8_t* copy = nullptr;
        if (!encoded.bitstream.empty()) {
            copy = static_cast<uint8_t*>(std::malloc(encoded.bitstream.size()));
            if (!copy)
                throw std::bad_alloc();
            std::memcpy(copy, encoded.bitstream.data(), encoded.bitstream.size());
        }
        out_encoded->data = copy;
        out_encoded->size = encoded.bitstream.size();
        out_encoded->q_index = encoded.q_index;
        out_encoded->frame_type = convert_frame_type(encoded.frame_type);
    });
}

MLVC_CODEC_API mlvc_status mlvc_decoder_create(
    const mlvc_session_options* options, mlvc_decoder** out_decoder,
    char* error_buffer, size_t error_capacity) noexcept
{
    if (out_decoder)
        *out_decoder = nullptr;
    return run_guarded(error_buffer, error_capacity, [&]() {
        if (!options || !out_decoder) {
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT,
                              "options and out_decoder must not be null");
        }
        auto decoder = std::make_unique<mlvc_decoder>(
            convert_session_options(*options));
        *out_decoder = decoder.release();
    });
}

MLVC_CODEC_API void mlvc_decoder_destroy(mlvc_decoder* decoder) noexcept
{
    delete decoder;
}

MLVC_CODEC_API mlvc_status mlvc_decoder_reset(
    mlvc_decoder* decoder, char* error_buffer, size_t error_capacity) noexcept
{
    return run_guarded(error_buffer, error_capacity, [&]() {
        if (!decoder)
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT, "decoder must not be null");
        decoder->session.reset();
    });
}

MLVC_CODEC_API int32_t mlvc_decoder_frame_index(
    const mlvc_decoder* decoder) noexcept
{
    return decoder ? static_cast<int32_t>(decoder->session.frame_index()) : 0;
}

MLVC_CODEC_API mlvc_status mlvc_decoder_decode_frame(
    mlvc_decoder* decoder,
    const uint8_t* access_unit_data,
    size_t access_unit_size,
    const mlvc_yuv420_frame* output_frame,
    char* error_buffer, size_t error_capacity) noexcept
{
    return run_guarded(error_buffer, error_capacity, [&]() {
        if (!decoder || !output_frame || (!access_unit_data && access_unit_size > 0U)) {
            throw StatusError(MLVC_STATUS_INVALID_ARGUMENT,
                              "decoder, access_unit_data, and output_frame must be valid");
        }
        const std::span<const std::byte> au(
            reinterpret_cast<const std::byte*>(access_unit_data), access_unit_size);
        decoder->session.decode_frame_into(
            au, to_yuv_view<mlvc::MutableYuv420FrameView>(*output_frame, "output"));
    });
}

}  // extern "C"
