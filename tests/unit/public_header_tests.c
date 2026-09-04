#include <mlvc/codec.h>

int main(void)
{
    mlvc_codec_options options = {0};
    mlvc_codec_stats stats = {0};
    return options.width == 0 && stats.frames == 0 ? 0 : 1;
}
