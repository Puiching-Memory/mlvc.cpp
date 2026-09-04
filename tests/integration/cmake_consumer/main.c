#include <mlvc/codec.h>

#include <stdio.h>

int main(void)
{
    mlvc_codec_options options;
    mlvc_codec_options_init(&options);
    puts(mlvc_backend_name());
    return options.width == 640 && options.height == 360 ? 0 : 1;
}
