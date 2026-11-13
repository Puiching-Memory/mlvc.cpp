#pragma once

#include "mlvc/core/assets.hpp"
#include "mlvc/core/bitstream.hpp"
#include "mlvc/core/model.hpp"
#include "mlvc/core/yuv.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace mlvc {

struct CodecOptions {
    int width = 640;
    int height = 360;
    int q_index = 21;
    int frames = 0;
    int device_id = 0;
    // A path of "-" selects stdin/stdout.  POSIX named FIFOs are also
    // supported for live frame-by-frame operation.
    std::string input_path;
    std::string output_path;
    std::string model_dir;
    std::string debug_dir;
    std::string backend_name;
    std::shared_ptr<const ModelAssetReader> asset_reader;
};

struct CodecStats {
    int frames = 0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    double elapsed_seconds = 0.0;
};

// File-I/O-free session options for per-frame encoding and decoding.
struct SessionOptions {
    int width = 640;
    int height = 360;
    int q_index = 21;
    int device_id = 0;
    std::string model_dir;
    std::string debug_dir;
    std::string backend_name;
    std::shared_ptr<const ModelAssetReader> asset_reader;
};

class EncoderSession final {
public:
    explicit EncoderSession(const SessionOptions& options);
    ~EncoderSession();

    EncoderSession(const EncoderSession&) = delete;
    EncoderSession& operator=(const EncoderSession&) = delete;
    EncoderSession(EncoderSession&&) noexcept;
    EncoderSession& operator=(EncoderSession&&) noexcept;

    void reset();
    int frame_index() const noexcept;
    const ModelConfig& model_config() const noexcept;

    EncodedFrame encode_frame(Yuv420FrameView frame);
    EncodedFrame encode_frame(Yuv420FrameView frame, int q_index);

private:
    friend CodecStats encode_video(const CodecOptions& options);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DecoderSession final {
public:
    explicit DecoderSession(const SessionOptions& options);
    ~DecoderSession();

    DecoderSession(const DecoderSession&) = delete;
    DecoderSession& operator=(const DecoderSession&) = delete;
    DecoderSession(DecoderSession&&) noexcept;
    DecoderSession& operator=(DecoderSession&&) noexcept;

    void reset();
    int frame_index() const noexcept;
    const ModelConfig& model_config() const noexcept;

    Yuv420Frame decode_frame(const EncodedFrame& encoded);
    void decode_frame_into(const EncodedFrame& encoded,
                           MutableYuv420FrameView output);

private:
    friend CodecStats decode_video(const CodecOptions& options);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

CodecStats encode_video(const CodecOptions& options);
CodecStats decode_video(const CodecOptions& options);
std::vector<std::string> available_model_profiles();

}  // namespace mlvc
