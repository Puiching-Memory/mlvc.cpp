#include "mlvc/core/yuv.hpp"

#include "mlvc/core/half.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mlvc {
namespace {

void validate_dimensions(int width, int height)
{
    if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0)
        throw std::runtime_error("YUV420 dimensions must be positive and even");
}

std::uint8_t quantize_byte(float value)
{
    const float clipped = std::clamp(value, 0.0F, 1.0F);
    return static_cast<std::uint8_t>(std::nearbyint(clipped * 255.0F));
}

}  // namespace

std::size_t yuv420_frame_bytes(int width, int height)
{
    validate_dimensions(width, height);
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U / 2U;
}

bool read_yuv420_frame(std::istream& input, int width, int height,
                       Yuv420Frame& frame)
{
    validate_dimensions(width, height);
    const std::size_t y_size = static_cast<std::size_t>(width) * height;
    const std::size_t uv_size = y_size / 4;
    frame = Yuv420Frame{width, height, std::vector<std::uint8_t>(y_size),
                        std::vector<std::uint8_t>(uv_size),
                        std::vector<std::uint8_t>(uv_size)};

    if (!read_yuv420_frame(input, mutable_yuv420_view(frame))) {
        frame = {};
        return false;
    }
    return true;
}

bool read_yuv420_frame(std::istream& input, MutableYuv420FrameView frame)
{
    validate_dimensions(frame.width, frame.height);
    if (!frame.y || !frame.u || !frame.v)
        throw std::runtime_error("YUV420 destination planes must not be null");
    const std::size_t y_size =
        static_cast<std::size_t>(frame.width) * frame.height;
    const std::size_t uv_size = y_size / 4;

    input.read(reinterpret_cast<char*>(frame.y),
               static_cast<std::streamsize>(y_size));
    if (input.gcount() == 0) {
        return false;
    }
    if (input.gcount() != static_cast<std::streamsize>(y_size))
        throw std::runtime_error("truncated Y plane in input YUV file");
    input.read(reinterpret_cast<char*>(frame.u),
               static_cast<std::streamsize>(uv_size));
    if (input.gcount() != static_cast<std::streamsize>(uv_size))
        throw std::runtime_error("truncated U plane in input YUV file");
    input.read(reinterpret_cast<char*>(frame.v),
               static_cast<std::streamsize>(uv_size));
    if (input.gcount() != static_cast<std::streamsize>(uv_size))
        throw std::runtime_error("truncated V plane in input YUV file");
    return true;
}

void write_yuv420_frame(std::ostream& output, const Yuv420Frame& frame)
{
    write_yuv420_frame(output, yuv420_view(frame));
}

void write_yuv420_frame(std::ostream& output, Yuv420FrameView frame)
{
    validate_dimensions(frame.width, frame.height);
    if (!frame.y || !frame.u || !frame.v)
        throw std::runtime_error("YUV420 source planes must not be null");
    const std::size_t y_size =
        static_cast<std::size_t>(frame.width) * frame.height;
    const std::size_t uv_size = y_size / 4;
    output.write(reinterpret_cast<const char*>(frame.y),
                 static_cast<std::streamsize>(y_size));
    output.write(reinterpret_cast<const char*>(frame.u),
                 static_cast<std::streamsize>(uv_size));
    output.write(reinterpret_cast<const char*>(frame.v),
                 static_cast<std::streamsize>(uv_size));
    if (!output)
        throw std::runtime_error("failed to write YUV output frame");
}

Yuv420FrameView yuv420_view(const Yuv420Frame& frame) noexcept
{
    return Yuv420FrameView{frame.width, frame.height, frame.y.data(),
                           frame.u.data(), frame.v.data()};
}

MutableYuv420FrameView mutable_yuv420_view(Yuv420Frame& frame) noexcept
{
    return MutableYuv420FrameView{frame.width, frame.height, frame.y.data(),
                                  frame.u.data(), frame.v.data()};
}

FramePadding frame_padding(int width, int height,
                           int model_width, int model_height)
{
    validate_dimensions(width, height);
    const bool native = width <= model_width && height <= model_height;
    const bool rotated = height <= model_width && width <= model_height;
    if (!native && !rotated)
        throw std::runtime_error("input frame is larger than the model dimensions");
    const int active_width = native ? width : height;
    const int active_height = native ? height : width;
    return FramePadding{0, model_width - active_width,
                        0, model_height - active_height, !native};
}

Tensor prepare_model_input(const Yuv420Frame& frame,
                           int model_width, int model_height,
                           float pixel_range, FramePadding& padding)
{
    padding = frame_padding(frame.width, frame.height, model_width, model_height);
    std::vector<Float16Storage> values(
        static_cast<std::size_t>(3) * model_height * model_width);
    const int active_width = padding.rotated ? frame.height : frame.width;
    const int active_height = padding.rotated ? frame.width : frame.height;
    const int uv_width = frame.width / 2;

    auto source_coordinate = [&](int y, int x) {
        if (padding.rotated)
            return std::pair{x, y};
        return std::pair{y, x};
    };
    for (int channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < model_height; ++y) {
            const int edge_y = std::min(y, active_height - 1);
            for (int x = 0; x < model_width; ++x) {
                const int edge_x = std::min(x, active_width - 1);
                const auto [source_y, source_x] = source_coordinate(edge_y, edge_x);
                std::uint8_t sample = 0;
                if (channel == 0) {
                    sample = frame.y[static_cast<std::size_t>(source_y) * frame.width + source_x];
                } else {
                    const auto& plane = channel == 1 ? frame.u : frame.v;
                    sample = plane[static_cast<std::size_t>(source_y / 2) * uv_width + source_x / 2];
                }
                const float normalized =
                    (static_cast<float>(sample) / 255.0F) * pixel_range;
                const std::size_t index =
                    (static_cast<std::size_t>(channel) * model_height + y) * model_width + x;
                values[index] = float_to_half(normalized);
            }
        }
    }
    return Tensor{"x", {1, 3, model_height, model_width}, std::move(values)};
}

Yuv420Frame prepare_model_output(const Tensor& tensor, int width, int height,
                                 float pixel_range,
                                 const FramePadding& padding)
{
    validate_dimensions(width, height);
    if (tensor.data_type() != TensorDataType::kFloat16)
        throw std::runtime_error("model output x_hat is not FP16");
    if (tensor.shape != std::vector<int64_t>{1, 3,
            height + padding.top + padding.bottom,
            width + padding.left + padding.right} && !padding.rotated) {
        throw std::runtime_error("model output x_hat shape mismatch");
    }
    if (tensor.shape.size() != 4 || tensor.shape[0] != 1 || tensor.shape[1] != 3)
        throw std::runtime_error("model output x_hat must be NCHW with three channels");

    const int model_height = static_cast<int>(tensor.shape[2]);
    const int model_width = static_cast<int>(tensor.shape[3]);
    const auto& values = std::get<std::vector<Float16Storage>>(tensor.data);
    Yuv420Frame result{width, height,
        std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height),
        std::vector<std::uint8_t>(static_cast<std::size_t>(width / 2) * (height / 2)),
        std::vector<std::uint8_t>(static_cast<std::size_t>(width / 2) * (height / 2))};

    auto sample = [&](int channel, int y, int x) {
        int model_y = y;
        int model_x = x;
        if (padding.rotated) {
            model_y = x;
            model_x = y;
        }
        model_y += padding.top;
        model_x += padding.left;
        const std::size_t index =
            (static_cast<std::size_t>(channel) * model_height + model_y) * model_width + model_x;
        return half_to_float(values[index]) / pixel_range;
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x)
            result.y[static_cast<std::size_t>(y) * width + x] = quantize_byte(sample(0, y, x));
    }
    for (int y = 0; y < height / 2; ++y) {
        for (int x = 0; x < width / 2; ++x) {
            float u = 0.0F;
            float v = 0.0F;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    u += sample(1, y * 2 + dy, x * 2 + dx);
                    v += sample(2, y * 2 + dy, x * 2 + dx);
                }
            }
            const std::size_t index = static_cast<std::size_t>(y) * (width / 2) + x;
            result.u[index] = quantize_byte(u * 0.25F);
            result.v[index] = quantize_byte(v * 0.25F);
        }
    }
    return result;
}

}  // namespace mlvc
