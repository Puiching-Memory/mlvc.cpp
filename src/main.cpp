#include "mlvc/backend.hpp"
#include "mlvc/pipeline.hpp"

#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* argv0)
{
    std::fprintf(stderr,
        "usage:\n"
        "  %s --backend-name\n"
        "  %s encode --input IN.yuv|- --output OUT.mlvc|- --width W --height H\n"
        "       --model-dir DIR [--frames N] [--q-index N] [--device-id N]\n"
        "       [--encode-device-id N]\n"
        "  %s decode --input IN.mlvc|- --output OUT.yuv|- --width W --height H\n"
        "       --model-dir DIR [--frames N] [--device-id N]\n"
        "       [--decode-device-id N]\n"
        "  common: [--engine-cache-dir DIR] [--workspace-mib N] [--debug-dir DIR]\n"
        "  stream: use '-' for stdin/stdout; named FIFO paths are also supported\n",
        argv0, argv0, argv0);
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

std::size_t parse_workspace_size(const char* text)
{
    try {
        std::size_t parsed = 0;
        const unsigned long long mib = std::stoull(text, &parsed);
        if (text[parsed] == '\0' && mib > 0 &&
            mib <= (std::numeric_limits<std::size_t>::max() >> 20))
            return static_cast<std::size_t>(mib) << 20;
    } catch (const std::exception&) {
    }
    throw std::runtime_error("--workspace-mib must be a positive integer");
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::strcmp(argv[1], "--backend-name") == 0) {
            std::printf("%.*s\n", static_cast<int>(mlvc::compiled_backend_name().size()),
                        mlvc::compiled_backend_name().data());
            return 0;
        }
        if (argc < 2 || std::strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return argc < 2 ? 2 : 0;
        }
        const std::string command = argv[1];
        if (command != "encode" && command != "decode")
            throw std::runtime_error("command must be encode or decode");

        mlvc::CodecOptions options;
        for (int i = 2; i < argc; ++i) {
            auto value = [&]() -> const char* {
                if (i + 1 >= argc)
                    throw std::runtime_error(std::string(argv[i]) + " requires a value");
                return argv[++i];
            };
            if (std::strcmp(argv[i], "--device-id") == 0) {
                options.device_id = parse_integer(value(), "--device-id", true);
            } else if (std::strcmp(argv[i], "--encode-device-id") == 0) {
                if (command != "encode")
                    throw std::runtime_error("--encode-device-id is only valid for encode");
                options.device_id = parse_integer(value(), "--encode-device-id", true);
            } else if (std::strcmp(argv[i], "--decode-device-id") == 0) {
                if (command != "decode")
                    throw std::runtime_error("--decode-device-id is only valid for decode");
                options.device_id = parse_integer(value(), "--decode-device-id", true);
            } else if (std::strcmp(argv[i], "--model-dir") == 0) {
                options.model_dir = value();
            } else if (std::strcmp(argv[i], "--engine-cache-dir") == 0) {
                options.engine_cache_dir = value();
            } else if (std::strcmp(argv[i], "--debug-dir") == 0) {
                options.debug_dir = value();
            } else if (std::strcmp(argv[i], "--workspace-mib") == 0) {
                options.workspace_size = parse_workspace_size(value());
            } else if (std::strcmp(argv[i], "--input") == 0) {
                options.input_path = value();
            } else if (std::strcmp(argv[i], "--output") == 0) {
                options.output_path = value();
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

        const mlvc::CodecStats stats = command == "encode"
            ? mlvc::encode_video(options) : mlvc::decode_video(options);
        const double fps = stats.elapsed_seconds > 0.0
            ? static_cast<double>(stats.frames) / stats.elapsed_seconds : 0.0;
        // Binary output may be stdout.  Keep the human-readable summary on
        // stderr so it cannot corrupt a downstream decoder or muxer.
        FILE* status = options.output_path == "-" ? stderr : stdout;
        std::fprintf(status,
                     "%s complete: backend=%.*s frames=%d input=%zu output=%zu "
                     "elapsed=%.3f s throughput=%.2f fps\n",
                     command.c_str(), static_cast<int>(mlvc::compiled_backend_name().size()),
                     mlvc::compiled_backend_name().data(), stats.frames,
                     stats.input_bytes, stats.output_bytes, stats.elapsed_seconds, fps);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
