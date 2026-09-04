#include "mlvc/core/yuv.hpp"

#include "mlvc/core/half.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mlvc {
namespace {

void validate_dimensions(int width, int height)
{
    if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0)
        throw std::runtime_error("YUV420 dimensions must be positive and even");
}

struct PlaneSizes {
    std::size_t y;
    std::size_t uv;
};

PlaneSizes plane_sizes(int width, int height)
{
    validate_dimensions(width, height);
    const std::size_t y =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    return {y, y / 4U};
}

void validate_pixel_range(float pixel_range)
{
    if (!std::isfinite(pixel_range) || pixel_range <= 0.0F)
        throw std::runtime_error("pixel range must be finite and positive");
}

std::pair<int, int> padded_dimensions(int width, int height,
                                      const FramePadding& padding)
{
    if (padding.left < 0 || padding.right < 0 || padding.top < 0 ||
        padding.bottom < 0) {
        throw std::runtime_error("frame padding must be non-negative");
    }
    const int active_height = padding.rotated ? width : height;
    const int active_width = padding.rotated ? height : width;
    const std::int64_t padded_height =
        static_cast<std::int64_t>(active_height) + padding.top + padding.bottom;
    const std::int64_t padded_width =
        static_cast<std::int64_t>(active_width) + padding.left + padding.right;
    if (padded_height > std::numeric_limits<int>::max() ||
        padded_width > std::numeric_limits<int>::max()) {
        throw std::runtime_error("padded frame dimensions are too large");
    }
    return {static_cast<int>(padded_height), static_cast<int>(padded_width)};
}

std::uint8_t quantize_byte(float value)
{
    const float clipped = std::clamp(value, 0.0F, 1.0F);
    return static_cast<std::uint8_t>(std::nearbyint(clipped * 255.0F));
}

}  // namespace

Yuv420FrameView::Yuv420FrameView(
    int width, int height, std::span<const std::uint8_t> y,
    std::span<const std::uint8_t> u, std::span<const std::uint8_t> v)
    : width_(width), height_(height), y_(y), u_(u), v_(v)
{
    const PlaneSizes expected = plane_sizes(width, height);
    if (y_.size() != expected.y || u_.size() != expected.uv ||
        v_.size() != expected.uv) {
        throw std::runtime_error("YUV420 plane sizes do not match dimensions");
    }
}

MutableYuv420FrameView::MutableYuv420FrameView(
    int width, int height, std::span<std::uint8_t> y,
    std::span<std::uint8_t> u, std::span<std::uint8_t> v)
    : width_(width), height_(height), y_(y), u_(u), v_(v)
{
    const PlaneSizes expected = plane_sizes(width, height);
    if (y_.size() != expected.y || u_.size() != expected.uv ||
        v_.size() != expected.uv) {
        throw std::runtime_error("YUV420 plane sizes do not match dimensions");
    }
}

MutableYuv420FrameView::operator Yuv420FrameView() const
{
    return Yuv420FrameView(width_, height_, y_, u_, v_);
}

std::size_t yuv420_frame_bytes(int width, int height)
{
    const PlaneSizes sizes = plane_sizes(width, height);
    return sizes.y + sizes.uv * 2U;
}

bool read_yuv420_frame(std::istream& input, int width, int height,
                       Yuv420Frame& frame)
{
    const PlaneSizes sizes = plane_sizes(width, height);
    frame = Yuv420Frame{width, height, std::vector<std::uint8_t>(sizes.y),
                        std::vector<std::uint8_t>(sizes.uv),
                        std::vector<std::uint8_t>(sizes.uv)};

    if (!read_yuv420_frame(input, mutable_yuv420_view(frame))) {
        frame = {};
        return false;
    }
    return true;
}

bool read_yuv420_frame(std::istream& input, MutableYuv420FrameView frame)
{
    input.read(reinterpret_cast<char*>(frame.y().data()),
               static_cast<std::streamsize>(frame.y().size()));
    if (input.gcount() == 0) {
        return false;
    }
    if (input.gcount() != static_cast<std::streamsize>(frame.y().size()))
        throw std::runtime_error("truncated Y plane in input YUV file");
    input.read(reinterpret_cast<char*>(frame.u().data()),
               static_cast<std::streamsize>(frame.u().size()));
    if (input.gcount() != static_cast<std::streamsize>(frame.u().size()))
        throw std::runtime_error("truncated U plane in input YUV file");
    input.read(reinterpret_cast<char*>(frame.v().data()),
               static_cast<std::streamsize>(frame.v().size()));
    if (input.gcount() != static_cast<std::streamsize>(frame.v().size()))
        throw std::runtime_error("truncated V plane in input YUV file");
    return true;
}

void write_yuv420_frame(std::ostream& output, const Yuv420Frame& frame)
{
    write_yuv420_frame(output, yuv420_view(frame));
}

void write_yuv420_frame(std::ostream& output, Yuv420FrameView frame)
{
    output.write(reinterpret_cast<const char*>(frame.y().data()),
                 static_cast<std::streamsize>(frame.y().size()));
    output.write(reinterpret_cast<const char*>(frame.u().data()),
                 static_cast<std::streamsize>(frame.u().size()));
    output.write(reinterpret_cast<const char*>(frame.v().data()),
                 static_cast<std::streamsize>(frame.v().size()));
    if (!output)
        throw std::runtime_error("failed to write YUV output frame");
}

Yuv420FrameView yuv420_view(const Yuv420Frame& frame)
{
    return Yuv420FrameView(frame.width, frame.height, frame.y, frame.u, frame.v);
}

MutableYuv420FrameView mutable_yuv420_view(Yuv420Frame& frame)
{
    return MutableYuv420FrameView(
        frame.width, frame.height, frame.y, frame.u, frame.v);
}

FramePadding frame_padding(int width, int height,
                           int model_width, int model_height)
{
    validate_dimensions(width, height);
    validate_dimensions(model_width, model_height);
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
    validate_pixel_range(pixel_range);
    const Yuv420FrameView source = yuv420_view(frame);
    std::vector<std::int64_t> shape{1, 3, model_height, model_width};
    std::vector<Float16Storage> values(checked_tensor_element_count(shape));
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
                    sample = source.y()[
                        static_cast<std::size_t>(source_y) *
                            static_cast<std::size_t>(frame.width) +
                        static_cast<std::size_t>(source_x)];
                } else {
                    const std::span<const std::uint8_t> plane =
                        channel == 1 ? source.u() : source.v();
                    sample = plane[
                        static_cast<std::size_t>(source_y / 2) *
                            static_cast<std::size_t>(uv_width) +
                        static_cast<std::size_t>(source_x / 2)];
                }
                const float normalized =
                    (static_cast<float>(sample) / 255.0F) * pixel_range;
                const std::size_t index =
                    (static_cast<std::size_t>(channel) *
                         static_cast<std::size_t>(model_height) +
                     static_cast<std::size_t>(y)) *
                        static_cast<std::size_t>(model_width) +
                    static_cast<std::size_t>(x);
                values[index] = float_to_half(normalized);
            }
        }
    }
    return Tensor{"x", std::move(shape), std::move(values)};
}

Yuv420Frame prepare_model_output(const Tensor& tensor, int width, int height,
                                 float pixel_range,
                                 const FramePadding& padding)
{
    validate_dimensions(width, height);
    validate_pixel_range(pixel_range);
    tensor.validate();
    if (tensor.data_type() != TensorDataType::kFloat16)
        throw std::runtime_error("model output x_hat is not FP16");
    const auto [model_height, model_width] =
        padded_dimensions(width, height, padding);
    const std::vector<std::int64_t> expected_shape{
        1, 3, model_height, model_width};
    if (tensor.shape != expected_shape) {
        throw std::runtime_error("model output x_hat shape mismatch");
    }

    const auto& values = std::get<std::vector<Float16Storage>>(tensor.data);
    Yuv420Frame result{width, height,
        std::vector<std::uint8_t>(static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height)),
        std::vector<std::uint8_t>(static_cast<std::size_t>(width / 2) *
                                  static_cast<std::size_t>(height / 2)),
        std::vector<std::uint8_t>(static_cast<std::size_t>(width / 2) *
                                  static_cast<std::size_t>(height / 2))};

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
            (static_cast<std::size_t>(channel) *
                 static_cast<std::size_t>(model_height) +
             static_cast<std::size_t>(model_y)) *
                static_cast<std::size_t>(model_width) +
            static_cast<std::size_t>(model_x);
        return half_to_float(values[index]) / pixel_range;
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x)
            result.y[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(width) +
                     static_cast<std::size_t>(x)] =
                quantize_byte(sample(0, y, x));
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
            const std::size_t index =
                static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(width / 2) +
                static_cast<std::size_t>(x);
            result.u[index] = quantize_byte(u * 0.25F);
            result.v[index] = quantize_byte(v * 0.25F);
        }
    }
    return result;
}

}  // namespace mlvc
