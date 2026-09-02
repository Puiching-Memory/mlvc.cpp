// mlvc.cpp: portable MLVC (DMC-6.1sb) encode/decode via pluggable inference
// backends (onnxruntime / libtorch / tensorrt).
//
// Placeholder entry point; the pipeline implementation (see docs/design.md)
// will land in subsequent commits.

#include "mlvc/backend.hpp"

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

namespace {

void print_usage(const char* argv0)
{
    std::fprintf(stderr,
        "usage: %s --backend <onnxruntime|libtorch|tensorrt> [--device cpu|cuda]\n"
        "           [--model-dir DIR] [--list-backends]\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv)
{
    mlvc::BackendOptions options;
    std::string backend_name = "onnxruntime";
    bool list_backends = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            options.device = argv[++i];
        } else if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            options.model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--list-backends") == 0) {
            list_backends = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (list_backends) {
        std::printf("compiled-in backends:");
        for (mlvc::BackendKind kind : mlvc::available_backends())
            std::printf(" %s", mlvc::to_string(kind));
        std::printf("\n");
        return 0;
    }

    try {
        std::unique_ptr<mlvc::InferenceBackend> backend =
            mlvc::create_backend(backend_name, options);
        std::printf("mlvc.cpp (scaffold): backend '%s' ready on %s; "
                    "pipeline implementation is under construction\n",
                    mlvc::to_string(backend->kind()), options.device.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
