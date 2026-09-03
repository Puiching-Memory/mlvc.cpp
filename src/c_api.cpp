#include "mlvc/codec.h"

#include "mlvc/pipeline.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

const char* nonnull(const char* value)
{
    return value ? value : "";
}

mlvc::CodecOptions convert_options(const mlvc_codec_options& source)
{
    mlvc::CodecOptions result;
    result.width = source.width;
    result.height = source.height;
    result.q_index = source.q_index;
    result.frames = source.frames;
    result.device_id = source.device_id;
    if (source.workspace_mib != 0) {
        constexpr uint64_t kMaxMiB =
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() >> 20);
        if (source.workspace_mib > kMaxMiB)
            throw std::runtime_error("workspace_mib is too large");
        result.workspace_size = static_cast<size_t>(source.workspace_mib) << 20;
    }
    result.input_path = nonnull(source.input_path);
    result.output_path = nonnull(source.output_path);
    result.model_dir = nonnull(source.model_dir);
    result.engine_cache_dir = nonnull(source.engine_cache_dir);
    result.debug_dir = nonnull(source.debug_dir);
    return result;
}

void copy_stats(const mlvc::CodecStats& source, mlvc_codec_stats* target)
{
    if (!target)
        return;
    target->frames = source.frames;
    target->input_bytes = static_cast<uint64_t>(source.input_bytes);
    target->output_bytes = static_cast<uint64_t>(source.output_bytes);
    target->elapsed_seconds = source.elapsed_seconds;
}

int fail(const std::string& message, char* error_buffer, size_t error_capacity)
{
    if (error_buffer && error_capacity != 0) {
        const size_t count = std::min(error_capacity - 1, message.size());
        std::memcpy(error_buffer, message.data(), count);
        error_buffer[count] = '\0';
    }
    return -1;
}

template <typename Operation>
int invoke(const mlvc_codec_options* options, mlvc_codec_stats* stats,
           char* error_buffer, size_t error_capacity, Operation&& operation)
{
    try {
        if (!options)
            throw std::runtime_error("options must not be null");
        copy_stats(operation(convert_options(*options)), stats);
        if (error_buffer && error_capacity != 0)
            error_buffer[0] = '\0';
        return 0;
    } catch (const std::exception& error) {
        return fail(error.what(), error_buffer, error_capacity);
    } catch (...) {
        return fail("unknown MLVC codec error", error_buffer, error_capacity);
    }
}

}  // namespace

extern "C" MLVC_CODEC_API void mlvc_codec_options_init(mlvc_codec_options* options)
{
    if (!options)
        return;
    *options = mlvc_codec_options{};
    options->width = 640;
    options->height = 360;
    options->q_index = 21;
    options->device_id = 0;
}

extern "C" MLVC_CODEC_API const char* mlvc_backend_name(void)
{
    static const std::string name(mlvc::compiled_backend_name());
    return name.c_str();
}

extern "C" MLVC_CODEC_API int mlvc_encode(
    const mlvc_codec_options* options, mlvc_codec_stats* stats,
    char* error_buffer, size_t error_capacity)
{
    return invoke(options, stats, error_buffer, error_capacity,
                  [](const mlvc::CodecOptions& converted) {
                      return mlvc::encode_video(converted);
                  });
}

extern "C" MLVC_CODEC_API int mlvc_decode(
    const mlvc_codec_options* options, mlvc_codec_stats* stats,
    char* error_buffer, size_t error_capacity)
{
    return invoke(options, stats, error_buffer, error_capacity,
                  [](const mlvc::CodecOptions& converted) {
                      return mlvc::decode_video(converted);
                  });
}
