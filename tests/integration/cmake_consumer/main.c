#include <mlvc/codec.h>

#include <stdio.h>

int main(void)
{
    mlvc_codec_options options;
    mlvc_codec_stats stats;
    mlvc_codec_options_init(&options);
    mlvc_codec_stats_init(&stats);
    puts(mlvc_backend_name());
    return options.width == 640 && options.height == 360 &&
                   stats.struct_size == MLVC_CODEC_STATS_V1_SIZE
        ? 0 : 1;
}
