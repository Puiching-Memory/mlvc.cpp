#include "mlvc/codec.h"

#include <cstring>

int main()
{
    if (!mlvc_backend_name() || std::strlen(mlvc_backend_name()) == 0)
        return 1;

    mlvc_codec_options options;
    mlvc_codec_options_init(&options);
    if (options.width != 640 || options.height != 360 ||
        options.q_index != 21 || options.device_id != 0 ||
        options.workspace_mib != 0)
        return 2;

    char error[128] = {};
    if (mlvc_encode(nullptr, nullptr, error, sizeof(error)) == 0 ||
        std::strlen(error) == 0)
        return 3;

    options.input_path = "missing.yuv";
    options.output_path = "missing.mlvc";
    options.model_dir = "missing-model";
    if (mlvc_encode(&options, nullptr, error, sizeof(error)) == 0 ||
        std::strlen(error) == 0)
        return 4;
    return 0;
}

