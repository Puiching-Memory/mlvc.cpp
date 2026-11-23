#include <mlvc/codec.h>

#include <stdio.h>

int main(void)
{
    mlvc_codec_options options;
    mlvc_codec_stats stats;
    mlvc_session_options session;
    mlvc_codec_options_init(&options);
    mlvc_codec_stats_init(&stats);
    mlvc_session_options_init(&session);
    puts(mlvc_backend_name());
    return options.width == 640 && options.height == 360 &&
                   stats.struct_size == MLVC_CODEC_STATS_V2_SIZE &&
                   session.struct_size == MLVC_SESSION_OPTIONS_V3_SIZE &&
                   mlvc_list_backends(NULL, 0) > 0
        ? 0 : 1;
}
