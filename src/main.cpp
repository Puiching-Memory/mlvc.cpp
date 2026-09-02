// mlvc.cpp: NVIDIA GPU MLVC (DMC-6.1sb) encode/decode scaffold.
//
// Placeholder entry point; the pipeline implementation (see docs/design.md)
// will land in subsequent commits.

#include "mlvc/backend.hpp"

#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* argv0)
{
    std::fprintf(stderr,
        "usage: %s [--device-id N] [--model-dir DIR] [--engine-cache-dir DIR]\n"
        "           [--workspace-mib N] [--no-tf32] [--backend-name]\n",
        argv0);
}

int parse_device_id(const char* text)
{
    try {
        std::size_t parsed = 0;
        const long value = std::stol(text, &parsed);
        if (text[parsed] == '\0' && value >= 0 &&
            value <= std::numeric_limits<int>::max())
            return static_cast<int>(value);
    } catch (const std::exception&) {
    }
    throw std::runtime_error("--device-id must be a non-negative integer");
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
    mlvc::BackendOptions options;
    bool print_backend_name = false;

    try {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--device-id") == 0 && i + 1 < argc) {
                options.device_id = parse_device_id(argv[++i]);
            } else if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
                options.model_dir = argv[++i];
            } else if (std::strcmp(argv[i], "--engine-cache-dir") == 0 && i + 1 < argc) {
                options.engine_cache_dir = argv[++i];
            } else if (std::strcmp(argv[i], "--workspace-mib") == 0 && i + 1 < argc) {
                options.workspace_size = parse_workspace_size(argv[++i]);
            } else if (std::strcmp(argv[i], "--no-tf32") == 0) {
                options.allow_tf32 = false;
            } else if (std::strcmp(argv[i], "--backend-name") == 0) {
                print_backend_name = true;
            } else if (std::strcmp(argv[i], "--help") == 0) {
                print_usage(argv[0]);
                return 0;
            } else {
                std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
                print_usage(argv[0]);
                return 2;
            }
        }

        if (print_backend_name) {
            std::printf("%.*s\n", static_cast<int>(mlvc::compiled_backend_name().size()),
                        mlvc::compiled_backend_name().data());
            return 0;
        }

        std::unique_ptr<mlvc::InferenceBackend> backend = mlvc::create_backend(options);
        std::printf("mlvc.cpp (scaffold): backend '%.*s' ready on CUDA device %d; "
                    "pipeline implementation is under construction\n",
                    static_cast<int>(backend->name().size()), backend->name().data(),
                    options.device_id);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
