#pragma once

#include "mlvc/backend.hpp"

#include <cstddef>
#include <string>

namespace mlvc {

struct CodecOptions {
    int width = 640;
    int height = 360;
    int q_index = 21;
    int frames = 0;
    int device_id = 0;
    std::size_t workspace_size = std::size_t{4} << 30;
    std::string input_path;
    std::string output_path;
    std::string model_dir;
    std::string engine_cache_dir;
    std::string debug_dir;
};

struct CodecStats {
    int frames = 0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    double elapsed_seconds = 0.0;
};

CodecStats encode_video(const CodecOptions& options);
CodecStats decode_video(const CodecOptions& options);

}  // namespace mlvc
