#include "mlvc/codec.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0)
{
    std::fprintf(stderr,
        "usage:\n"
        "  %s --backend-name\n"
        "  %s --list-backends\n"
        "  %s --list-model-profiles\n"
        "  %s encode --input IN.yuv|- --output OUT.mlvc|- --width W --height H\n"
        "       [--model-dir DIR | --model-profile NAME] [--frames N]\n"
        "       [--q-index N] [--backend NAME] [--device-id N] [--threads N]\n"
        "       [--encode-device-id N] [--share-thread-pool]\n"
        "  %s decode --input IN.mlvc|- --output OUT.yuv|-\n"
        "       [--width W --height H] [--model-dir DIR | --model-profile NAME]\n"
        "       [--frames N] [--backend NAME] [--device-id N] [--threads N]\n"
        "       [--decode-device-id N] [--share-thread-pool]\n"
        "  common: [--debug-dir DIR] [--log-level trace|debug|info|warn|error|off]\n"
        "  stream: use '-' for stdin/stdout; named FIFO paths are also supported\n",
        argv0, argv0, argv0, argv0, argv0);
}

int parse_integer(const char* text, const char* option, bool allow_zero)
{
    try {
        std::size_t parsed = 0;
        const long value = std::stol(text, &parsed);
        if (text[parsed] == '\0' && (allow_zero ? value >= 0 : value > 0) &&
            value <= std::numeric_limits<int>::max())
            return static_cast<int>(value);
    } catch (const std::exception&) {
    }
    throw std::runtime_error(std::string(option) + " has an invalid value");
}

mlvc_log_level parse_log_level(const char* text)
{
    if (std::strcmp(text, "trace") == 0)
        return MLVC_LOG_TRACE;
    if (std::strcmp(text, "debug") == 0)
        return MLVC_LOG_DEBUG;
    if (std::strcmp(text, "info") == 0)
        return MLVC_LOG_INFO;
    if (std::strcmp(text, "warn") == 0)
        return MLVC_LOG_WARN;
    if (std::strcmp(text, "error") == 0)
        return MLVC_LOG_ERROR;
    if (std::strcmp(text, "off") == 0)
        return MLVC_LOG_OFF;
    throw std::runtime_error(std::string("unknown --log-level value: ") + text);
}

const char* log_level_tag(mlvc_log_level level) noexcept
{
    switch (level) {
    case MLVC_LOG_TRACE:
        return "TRACE";
    case MLVC_LOG_DEBUG:
        return "DEBUG";
    case MLVC_LOG_INFO:
        return "INFO";
    case MLVC_LOG_WARN:
        return "WARN";
    case MLVC_LOG_ERROR:
        return "ERROR";
    case MLVC_LOG_OFF:
        return "OFF";
    }
    return "LOG";
}

void stderr_log_callback(mlvc_log_level level, const char* component,
                         const char* message, void* /*user_data*/)
{
    std::fprintf(stderr, "[mlvc:%s:%s] %s\n",
                 log_level_tag(level), component ? component : "core",
                 message ? message : "");
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::strcmp(argv[1], "--backend-name") == 0) {
            std::printf("%s\n", mlvc_backend_name());
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--list-backends") == 0) {
            const size_t count = mlvc_list_backends(nullptr, 0);
            std::vector<const char*> names(count);
            if (count > 0)
                mlvc_list_backends(names.data(), names.size());
            for (const char* name : names)
                std::printf("%s\n", name);
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--list-model-profiles") == 0) {
            const size_t count = mlvc_list_model_profiles(nullptr, 0);
            std::vector<const char*> profiles(count);
            if (count > 0)
                mlvc_list_model_profiles(profiles.data(), profiles.size());
            for (const char* profile : profiles)
                std::printf("%s\n", profile);
            return 0;
        }
        if (argc < 2 || std::strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return argc < 2 ? 2 : 0;
        }
        const std::string command = argv[1];
        if (command != "encode" && command != "decode")
            throw std::runtime_error("command must be encode or decode");

        mlvc_video_options options;
        mlvc_video_options_init(&options);
        if (command == "decode") {
            options.width = 0;
            options.height = 0;
        }

        std::string backend_name;
        std::string input_path;
        std::string output_path;
        std::string model_dir;
        std::string debug_dir;
        std::string model_profile;

        for (int i = 2; i < argc; ++i) {
            auto value = [&]() -> const char* {
                if (i + 1 >= argc)
                    throw std::runtime_error(std::string(argv[i]) + " requires a value");
                return argv[++i];
            };
            if (std::strcmp(argv[i], "--backend") == 0) {
                backend_name = value();
            } else if (std::strcmp(argv[i], "--device-id") == 0) {
                options.device_id = parse_integer(value(), "--device-id", true);
            } else if (std::strcmp(argv[i], "--threads") == 0) {
                options.num_threads = parse_integer(value(), "--threads", true);
            } else if (std::strcmp(argv[i], "--share-thread-pool") == 0) {
                options.share_thread_pool = 1;
            } else if (std::strcmp(argv[i], "--log-level") == 0) {
                const mlvc_log_level lvl = parse_log_level(value());
                mlvc_set_log_callback(
                    lvl == MLVC_LOG_OFF ? nullptr : &stderr_log_callback,
                    nullptr, lvl);
            } else if (std::strcmp(argv[i], "--encode-device-id") == 0) {
                if (command != "encode")
                    throw std::runtime_error("--encode-device-id is only valid for encode");
                options.device_id = parse_integer(value(), "--encode-device-id", true);
            } else if (std::strcmp(argv[i], "--decode-device-id") == 0) {
                if (command != "decode")
                    throw std::runtime_error("--decode-device-id is only valid for decode");
                options.device_id = parse_integer(value(), "--decode-device-id", true);
            } else if (std::strcmp(argv[i], "--model-dir") == 0) {
                model_dir = value();
            } else if (std::strcmp(argv[i], "--model-profile") == 0) {
                model_profile = value();
            } else if (std::strcmp(argv[i], "--debug-dir") == 0) {
                debug_dir = value();
            } else if (std::strcmp(argv[i], "--input") == 0) {
                input_path = value();
            } else if (std::strcmp(argv[i], "--output") == 0) {
                output_path = value();
            } else if (std::strcmp(argv[i], "--width") == 0) {
                options.width = parse_integer(value(), "--width", false);
            } else if (std::strcmp(argv[i], "--height") == 0) {
                options.height = parse_integer(value(), "--height", false);
            } else if (std::strcmp(argv[i], "--frames") == 0) {
                options.frames = parse_integer(value(), "--frames", true);
            } else if (std::strcmp(argv[i], "--q-index") == 0) {
                options.q_index = parse_integer(value(), "--q-index", true);
            } else if (std::strcmp(argv[i], "--help") == 0) {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + std::string(argv[i]));
            }
        }
        if (!model_profile.empty()) {
            if (!model_dir.empty())
                throw std::runtime_error(
                    "--model-dir and --model-profile are mutually exclusive");
            model_dir = "embedded:" + model_profile;
        }

        options.backend_name = backend_name.empty() ? nullptr : backend_name.c_str();
        options.input_path = input_path.empty() ? nullptr : input_path.c_str();
        options.output_path = output_path.empty() ? nullptr : output_path.c_str();
        options.model_dir = model_dir.empty() ? nullptr : model_dir.c_str();
        options.debug_dir = debug_dir.empty() ? nullptr : debug_dir.c_str();

        mlvc_codec_stats stats;
        mlvc_codec_stats_init(&stats);
        char error[1024] = {};
        const mlvc_status status = command == "encode"
            ? mlvc_encode_video(&options, &stats, error, sizeof(error))
            : mlvc_decode_video(&options, &stats, error, sizeof(error));
        if (status != MLVC_STATUS_OK) {
            std::fprintf(stderr, "error: %s\n", error[0] ? error : "MLVC codec error");
            return 1;
        }

        const double fps = stats.elapsed_seconds > 0.0
            ? static_cast<double>(stats.frames) / stats.elapsed_seconds : 0.0;
        // Binary output may be stdout.  Keep the human-readable summary on
        // stderr so it cannot corrupt a downstream decoder or muxer.
        FILE* status_out = output_path == "-" ? stderr : stdout;
        const char* active_backend = !backend_name.empty()
            ? backend_name.c_str()
            : mlvc_backend_name();
        std::fprintf(status_out,
                     "%s complete: backend=%s frames=%d input=%llu output=%llu "
                     "elapsed=%.3f s throughput=%.2f fps\n",
                     command.c_str(), active_backend, stats.frames,
                     static_cast<unsigned long long>(stats.input_bytes),
                     static_cast<unsigned long long>(stats.output_bytes),
                     stats.elapsed_seconds, fps);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
