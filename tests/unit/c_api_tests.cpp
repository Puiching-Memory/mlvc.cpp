#include "mlvc/codec.h"

#include <cstring>

int main()
{
    if (!mlvc_backend_name() || std::strlen(mlvc_backend_name()) == 0)
        return 1;

    mlvc_codec_options options;
    mlvc_codec_options_init(&options);
    if (options.struct_size != MLVC_CODEC_OPTIONS_V1_SIZE ||
        options.abi_version != MLVC_CODEC_ABI_VERSION ||
        options.width != 640 || options.height != 360 ||
        options.q_index != 21 || options.device_id != 0 ||
        options.workspace_mib != 0)
        return 2;

    mlvc_codec_stats stats;
    mlvc_codec_stats_init(&stats);
    if (stats.struct_size != MLVC_CODEC_STATS_V1_SIZE ||
        stats.abi_version != MLVC_CODEC_ABI_VERSION || stats.frames != 0)
        return 3;

    char error[128] = {};
    if (mlvc_encode(nullptr, nullptr, error, sizeof(error)) == 0 ||
        std::strlen(error) == 0)
        return 4;

    mlvc_codec_options invalid_options = options;
    invalid_options.struct_size = 0;
    if (mlvc_encode(&invalid_options, nullptr, error, sizeof(error)) == 0 ||
        std::strstr(error, "smaller than ABI v1") == nullptr)
        return 5;

    mlvc_codec_stats invalid_stats = stats;
    invalid_stats.abi_version = MLVC_CODEC_ABI_VERSION + 1;
    if (mlvc_encode(&options, &invalid_stats, error, sizeof(error)) == 0 ||
        std::strstr(error, "stats ABI version") == nullptr)
        return 6;

    options.input_path = "missing.yuv";
    options.output_path = "missing.mlvc";
    options.model_dir = "missing-model";
    if (mlvc_encode(&options, nullptr, error, sizeof(error)) == 0 ||
        std::strlen(error) == 0)
        return 7;
    return 0;
}
