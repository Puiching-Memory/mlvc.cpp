#include "mlvc/codec.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

std::atomic<int> g_log_count{0};

void test_log_cb(mlvc_log_level /*level*/, const char* /*component*/,
                 const char* /*message*/, void* user_data)
{
    if (user_data)
        static_cast<std::atomic<int>*>(user_data)->fetch_add(1);
    g_log_count.fetch_add(1);
}

int fake_exists(void* /*ud*/, const char* /*rel*/)
{
    return 0;
}

int fake_read_binary(void* /*ud*/, const char* /*rel*/,
                     uint8_t** out_data, size_t* out_size)
{
    *out_data = nullptr;
    *out_size = 0;
    return -1;
}

}  // namespace

int main()
{
    // --- ABI v2 backward compatibility checks ---
    if (!mlvc_backend_name() || std::strlen(mlvc_backend_name()) == 0)
        return 1;

    mlvc_codec_options options;
    mlvc_codec_options_init(&options);
    if (options.struct_size != MLVC_CODEC_OPTIONS_V2_SIZE ||
        options.abi_version != MLVC_CODEC_ABI_VERSION ||
        options.width != 640 || options.height != 360 ||
        options.q_index != 21 || options.device_id != 0)
        return 2;

    mlvc_codec_stats stats;
    mlvc_codec_stats_init(&stats);
    if (stats.struct_size != MLVC_CODEC_STATS_V2_SIZE ||
        stats.abi_version != MLVC_CODEC_ABI_VERSION || stats.frames != 0)
        return 3;

    char error[256] = {};
    if (mlvc_encode(nullptr, nullptr, error, sizeof(error)) == 0 ||
        std::strlen(error) == 0)
        return 4;

    mlvc_codec_options invalid_options = options;
    invalid_options.struct_size = 0;
    if (mlvc_encode(&invalid_options, nullptr, error, sizeof(error)) == 0 ||
        std::strstr(error, "smaller than ABI v2") == nullptr)
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

    // --- ABI v3 discovery, logging, video options, and stateful session guards ---
    const size_t backend_count = mlvc_list_backends(nullptr, 0);
    std::vector<const char*> backends(backend_count);
    if (backend_count > 0) {
        if (mlvc_list_backends(backends.data(), backends.size()) != backend_count)
            return 8;
        if (!backends[0] || std::strlen(backends[0]) == 0)
            return 9;
    }

    const size_t profile_count = mlvc_list_model_profiles(nullptr, 0);
    std::vector<const char*> profiles(profile_count);
    if (profile_count > 0) {
        if (mlvc_list_model_profiles(profiles.data(), profiles.size()) != profile_count)
            return 10;
    }

    std::atomic<int> local_logs{0};
    mlvc_set_log_callback(&test_log_cb, &local_logs, MLVC_LOG_DEBUG);

    mlvc_video_options v3_opts;
    mlvc_video_options_init(&v3_opts);
    if (v3_opts.struct_size != MLVC_VIDEO_OPTIONS_V3_SIZE ||
        v3_opts.abi_version != MLVC_CODEC_ABI_V3_VERSION ||
        v3_opts.allow_cpu_fallback != 1)
        return 11;

    mlvc_video_options bad_v3 = v3_opts;
    bad_v3.struct_size = 0;
    if (mlvc_encode_video(&bad_v3, nullptr, error, sizeof(error)) !=
        MLVC_STATUS_ABI_MISMATCH)
        return 12;

    if (mlvc_encode_video(nullptr, nullptr, error, sizeof(error)) !=
        MLVC_STATUS_INVALID_ARGUMENT)
        return 13;

    mlvc_session_options sess_opts;
    mlvc_session_options_init(&sess_opts);
    if (sess_opts.struct_size != MLVC_SESSION_OPTIONS_V3_SIZE ||
        sess_opts.abi_version != MLVC_CODEC_ABI_V3_VERSION)
        return 14;

    mlvc_encoder* enc = nullptr;
    if (mlvc_encoder_create(nullptr, &enc, error, sizeof(error)) !=
            MLVC_STATUS_INVALID_ARGUMENT ||
        enc != nullptr)
        return 15;

    mlvc_asset_reader custom_reader = {};
    custom_reader.struct_size = MLVC_ASSET_READER_V3_SIZE;
    custom_reader.abi_version = MLVC_CODEC_ABI_V3_VERSION;
    custom_reader.exists = &fake_exists;
    custom_reader.read_binary = &fake_read_binary;
    sess_opts.asset_reader = &custom_reader;
    const mlvc_status create_rc =
        mlvc_encoder_create(&sess_opts, &enc, error, sizeof(error));
    if (create_rc != MLVC_STATUS_IO_ERROR || enc != nullptr || std::strlen(error) == 0)
        return 16;

    mlvc_decoder* dec = nullptr;
    if (mlvc_decoder_create(&sess_opts, &dec, error, sizeof(error)) !=
            MLVC_STATUS_IO_ERROR ||
        dec != nullptr)
        return 17;

    mlvc_encode_params params;
    mlvc_encode_params_init(&params);
    if (params.struct_size != MLVC_ENCODE_PARAMS_V3_SIZE ||
        params.abi_version != MLVC_CODEC_ABI_V3_VERSION ||
        params.q_index != -1 || params.force_idr != 0)
        return 18;

    mlvc_encoded_frame empty_frame = {};
    mlvc_encoded_frame_free(&empty_frame);
    mlvc_encoder_destroy(nullptr);
    mlvc_decoder_destroy(nullptr);
    if (mlvc_encoder_frame_index(nullptr) != 0 ||
        mlvc_decoder_frame_index(nullptr) != 0)
        return 19;

    mlvc_set_log_callback(nullptr, nullptr, MLVC_LOG_OFF);
    return 0;
}
