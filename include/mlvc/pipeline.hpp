#pragma once
// Mercurial placeholder for the MLVC pipeline module.
// The full encode/decode pipeline (see docs/design.md) lands in this header
// and src/pipeline.cpp in a follow-up commit.

namespace mlvc {

struct CodecOptions {
    int width = 640;
    int height = 360;
    int model_width = 640;
    int model_height = 368;
    int q_index = 21;
    int frames = 0;
    const char* model_dir = nullptr;
};

}  // namespace mlvc
