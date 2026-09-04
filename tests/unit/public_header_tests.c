#include <mlvc/codec.h>

int main(void)
{
    mlvc_codec_options options = {0};
    mlvc_codec_stats stats = {0};
    options.struct_size = MLVC_CODEC_OPTIONS_V1_SIZE;
    options.abi_version = MLVC_CODEC_ABI_VERSION;
    stats.struct_size = MLVC_CODEC_STATS_V1_SIZE;
    stats.abi_version = MLVC_CODEC_ABI_VERSION;
    return options.struct_size == MLVC_CODEC_OPTIONS_V1_SIZE &&
                   options.abi_version == MLVC_CODEC_ABI_VERSION &&
                   stats.struct_size == MLVC_CODEC_STATS_V1_SIZE &&
                   stats.abi_version == MLVC_CODEC_ABI_VERSION
        ? 0 : 1;
}
