#include <mlvc/codec.h>

int main(void)
{
    mlvc_codec_options options = {0};
    mlvc_codec_stats stats = {0};
    mlvc_video_options v3_options = {0};
    mlvc_session_options session_options = {0};
    mlvc_encode_params encode_params = {0};
    mlvc_asset_reader asset_reader = {0};

    options.struct_size = MLVC_CODEC_OPTIONS_V2_SIZE;
    options.abi_version = MLVC_CODEC_ABI_VERSION;
    stats.struct_size = MLVC_CODEC_STATS_V2_SIZE;
    stats.abi_version = MLVC_CODEC_ABI_VERSION;
    v3_options.struct_size = MLVC_VIDEO_OPTIONS_V3_SIZE;
    v3_options.abi_version = MLVC_CODEC_ABI_V3_VERSION;
    session_options.struct_size = MLVC_SESSION_OPTIONS_V3_SIZE;
    session_options.abi_version = MLVC_CODEC_ABI_V3_VERSION;
    encode_params.struct_size = MLVC_ENCODE_PARAMS_V3_SIZE;
    encode_params.abi_version = MLVC_CODEC_ABI_V3_VERSION;
    asset_reader.struct_size = MLVC_ASSET_READER_V3_SIZE;
    asset_reader.abi_version = MLVC_CODEC_ABI_V3_VERSION;

    return options.struct_size == MLVC_CODEC_OPTIONS_V2_SIZE &&
                   options.abi_version == MLVC_CODEC_ABI_VERSION &&
                   stats.struct_size == MLVC_CODEC_STATS_V2_SIZE &&
                   stats.abi_version == MLVC_CODEC_ABI_VERSION &&
                   v3_options.struct_size == MLVC_VIDEO_OPTIONS_V3_SIZE &&
                   session_options.struct_size == MLVC_SESSION_OPTIONS_V3_SIZE &&
                   encode_params.struct_size == MLVC_ENCODE_PARAMS_V3_SIZE &&
                   asset_reader.struct_size == MLVC_ASSET_READER_V3_SIZE
        ? 0 : 1;
}
