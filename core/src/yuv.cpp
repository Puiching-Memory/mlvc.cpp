#include "mlvc/core/yuv.hpp"

#include "mlvc/core/half.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

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

#if defined(__aarch64__)
inline float32x4_t load_fp16x4_to_f32(const Float16Storage* src) noexcept
{
    std::uint16_t raw[4];
    std::memcpy(raw, src, sizeof(raw));
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(raw)));
}

void write_y_row_neon(const Float16Storage* src, std::uint8_t* dst, int width,
                      float pixel_range) noexcept
{
    const float32x4_t zero = vdupq_n_f32(0.0F);
    const float32x4_t one = vdupq_n_f32(1.0F);
    const float32x4_t v_range = vdupq_n_f32(pixel_range);
    const bool divide_range = pixel_range != 1.0F;

    int x = 0;
    for (; x + 8 <= width; x += 8) {
        float32x4_t v0 = load_fp16x4_to_f32(src + x);
        float32x4_t v1 = load_fp16x4_to_f32(src + x + 4);
        if (divide_range) {
            v0 = vdivq_f32(v0, v_range);
            v1 = vdivq_f32(v1, v_range);
        }
        v0 = vmaxq_f32(vminq_f32(v0, one), zero);
        v1 = vmaxq_f32(vminq_f32(v1, one), zero);
        const uint32x4_t q0 = vcvtq_u32_f32(vrndiq_f32(vmulq_n_f32(v0, 255.0F)));
        const uint32x4_t q1 = vcvtq_u32_f32(vrndiq_f32(vmulq_n_f32(v1, 255.0F)));
        const uint16x8_t q16 = vcombine_u16(vqmovn_u32(q0), vqmovn_u32(q1));
        vst1_u8(dst + x, vqmovn_u16(q16));
    }
    for (; x < width; ++x)
        dst[x] = quantize_byte(half_to_float(src[x]) / pixel_range);
}

void write_uv_row_neon(const Float16Storage* r0, const Float16Storage* r1,
                       std::uint8_t* dst, int uv_width,
                       float pixel_range) noexcept
{
    const float32x4_t zero = vdupq_n_f32(0.0F);
    const float32x4_t one = vdupq_n_f32(1.0F);
    const float32x4_t v_range = vdupq_n_f32(pixel_range);
    const bool divide_range = pixel_range != 1.0F;

    int ux = 0;
    for (; ux + 4 <= uv_width; ux += 4) {
        const int col = ux * 2;
        float32x4_t a0 = load_fp16x4_to_f32(r0 + col);
        float32x4_t a1 = load_fp16x4_to_f32(r0 + col + 4);
        float32x4_t b0 = load_fp16x4_to_f32(r1 + col);
        float32x4_t b1 = load_fp16x4_to_f32(r1 + col + 4);
        if (divide_range) {
            a0 = vdivq_f32(a0, v_range);
            a1 = vdivq_f32(a1, v_range);
            b0 = vdivq_f32(b0, v_range);
            b1 = vdivq_f32(b1, v_range);
        }
        // Preserve ((s00 + s01) + s10) + s11 order for bit-exact parity.
        const float32x4_t row0_pair = vpaddq_f32(a0, a1);
        const float32x4_t row1_even = vuzp1q_f32(b0, b1);
        const float32x4_t row1_odd = vuzp2q_f32(b0, b1);
        const float32x4_t sum = vaddq_f32(vaddq_f32(row0_pair, row1_even), row1_odd);
        const float32x4_t avg = vmulq_n_f32(sum, 0.25F);
        const float32x4_t clipped = vmaxq_f32(vminq_f32(avg, one), zero);
        const uint32x4_t q = vcvtq_u32_f32(vrndiq_f32(vmulq_n_f32(clipped, 255.0F)));
        const uint16x4_t q16 = vqmovn_u32(q);
        const uint8x8_t q8 = vqmovn_u16(vcombine_u16(q16, q16));
        const std::uint32_t packed4 = vget_lane_u32(vreinterpret_u32_u8(q8), 0);
        std::memcpy(dst + ux, &packed4, sizeof(packed4));
    }
    for (; ux < uv_width; ++ux) {
        const int col = ux * 2;
        const float sum =
            half_to_float(r0[col]) / pixel_range +
            half_to_float(r0[col + 1]) / pixel_range +
            half_to_float(r1[col]) / pixel_range +
            half_to_float(r1[col + 1]) / pixel_range;
        dst[ux] = quantize_byte(sum * 0.25F);
    }
}
#endif

void replicate_trailing_rows(Float16Storage* dst, int height,
                             int model_height, int model_width) noexcept
{
    if (height >= model_height)
        return;
    const std::size_t row_elems = static_cast<std::size_t>(model_width);
    const std::size_t row_bytes = row_elems * sizeof(Float16Storage);
    const Float16Storage* last_row = dst + static_cast<std::size_t>(height - 1) * row_elems;
    for (int y = height; y < model_height; ++y)
        std::memcpy(dst + static_cast<std::size_t>(y) * row_elems, last_row, row_bytes);
}

void convert_luma_plane_input(std::span<const std::uint8_t> src,
                              Float16Storage* dst,
                              int width, int height,
                              int model_width, int model_height,
                              const std::array<Float16Storage, 256>& lut) noexcept
{
    const std::uint8_t* src_ptr = src.data();
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* srow =
            src_ptr + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        Float16Storage* drow =
            dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(model_width);
        int x = 0;
        for (; x + 4 <= width; x += 4) {
            drow[x] = lut[srow[x]];
            drow[x + 1] = lut[srow[x + 1]];
            drow[x + 2] = lut[srow[x + 2]];
            drow[x + 3] = lut[srow[x + 3]];
        }
        for (; x < width; ++x)
            drow[x] = lut[srow[x]];
        if (width < model_width) {
            const Float16Storage edge = drow[width - 1];
            std::fill(drow + width, drow + model_width, edge);
        }
    }
    replicate_trailing_rows(dst, height, model_height, model_width);
}

void convert_chroma_plane_input(std::span<const std::uint8_t> src,
                                Float16Storage* dst,
                                int width, int height,
                                int model_width, int model_height,
                                const std::array<Float16Storage, 256>& lut) noexcept
{
    const int uv_width = width / 2;
    const int uv_height = height / 2;
    const std::size_t row_bytes =
        static_cast<std::size_t>(model_width) * sizeof(Float16Storage);
    const std::uint8_t* src_ptr = src.data();
    for (int uy = 0; uy < uv_height; ++uy) {
        const std::uint8_t* srow =
            src_ptr + static_cast<std::size_t>(uy) * static_cast<std::size_t>(uv_width);
        Float16Storage* drow0 =
            dst + static_cast<std::size_t>(uy * 2) * static_cast<std::size_t>(model_width);
        for (int ux = 0; ux < uv_width; ++ux) {
            const Float16Storage val = lut[srow[ux]];
            drow0[ux * 2] = val;
            drow0[ux * 2 + 1] = val;
        }
        if (width < model_width) {
            const Float16Storage edge = drow0[width - 1];
            std::fill(drow0 + width, drow0 + model_width, edge);
        }
        std::memcpy(drow0 + model_width, drow0, row_bytes);
    }
    replicate_trailing_rows(dst, height, model_height, model_width);
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

Tensor prepare_model_input(Yuv420FrameView source,
                           int model_width, int model_height,
                           float pixel_range, FramePadding& padding)
{
    const int width = source.width();
    const int height = source.height();
    padding = frame_padding(width, height, model_width, model_height);
    validate_pixel_range(pixel_range);
    std::vector<std::int64_t> shape{1, 3, model_height, model_width};
    std::vector<Float16Storage> values(checked_tensor_element_count(shape));

    std::array<Float16Storage, 256> lut{};
    for (int s = 0; s < 256; ++s) {
        const float normalized =
            (static_cast<float>(s) / 255.0F) * pixel_range;
        lut[static_cast<std::size_t>(s)] = float_to_half(normalized);
    }

    const std::size_t plane_elems =
        static_cast<std::size_t>(model_height) * static_cast<std::size_t>(model_width);
    if (!padding.rotated) {
        convert_luma_plane_input(source.y(), values.data(), width, height,
                                 model_width, model_height, lut);
        convert_chroma_plane_input(source.u(), values.data() + plane_elems,
                                   width, height, model_width, model_height, lut);
        convert_chroma_plane_input(source.v(), values.data() + plane_elems * 2U,
                                   width, height, model_width, model_height, lut);
        return Tensor{"x", std::move(shape), std::move(values)};
    }

    const int active_width = height;
    const int active_height = width;
    const int uv_width = width / 2;
    for (int channel = 0; channel < 3; ++channel) {
        Float16Storage* dst_plane =
            values.data() + static_cast<std::size_t>(channel) * plane_elems;
        for (int y = 0; y < model_height; ++y) {
            const int source_x = std::min(y, active_height - 1);
            Float16Storage* dst_row =
                dst_plane + static_cast<std::size_t>(y) * static_cast<std::size_t>(model_width);
            for (int x = 0; x < model_width; ++x) {
                const int source_y = std::min(x, active_width - 1);
                std::uint8_t sample = 0;
                if (channel == 0) {
                    sample = source.y()[
                        static_cast<std::size_t>(source_y) *
                            static_cast<std::size_t>(width) +
                        static_cast<std::size_t>(source_x)];
                } else {
                    const std::span<const std::uint8_t> plane =
                        channel == 1 ? source.u() : source.v();
                    sample = plane[
                        static_cast<std::size_t>(source_y / 2) *
                            static_cast<std::size_t>(uv_width) +
                        static_cast<std::size_t>(source_x / 2)];
                }
                dst_row[x] = lut[sample];
            }
        }
    }
    return Tensor{"x", std::move(shape), std::move(values)};
}

Tensor prepare_model_input(const Yuv420Frame& frame,
                           int model_width, int model_height,
                           float pixel_range, FramePadding& padding)
{
    return prepare_model_input(yuv420_view(frame), model_width, model_height,
                               pixel_range, padding);
}

void prepare_model_output_into(const Tensor& tensor,
                               MutableYuv420FrameView output,
                               float pixel_range,
                               const FramePadding& padding)
{
    const int width = output.width();
    const int height = output.height();
    validate_dimensions(width, height);
    validate_pixel_range(pixel_range);
    tensor.validate();
    if (tensor.data_type() != TensorDataType::kFloat16)
        throw std::runtime_error("model output x_hat is not FP16");
    const auto model_dims =
        padded_dimensions(width, height, padding);
    const int model_height = model_dims.first;
    const int model_width = model_dims.second;
    const std::vector<std::int64_t> expected_shape{
        1, 3, model_height, model_width};
    if (tensor.shape != expected_shape) {
        throw std::runtime_error("model output x_hat shape mismatch");
    }

    const auto& values = std::get<std::vector<Float16Storage>>(tensor.data);
    const std::span<std::uint8_t> y_plane = output.y();
    const std::span<std::uint8_t> u_plane = output.u();
    const std::span<std::uint8_t> v_plane = output.v();

    if (!padding.rotated) {
        const std::size_t plane_elems =
            static_cast<std::size_t>(model_height) * static_cast<std::size_t>(model_width);
        const Float16Storage* y_base = values.data();
        const Float16Storage* u_base = y_base + plane_elems;
        const Float16Storage* v_base = u_base + plane_elems;
        const std::size_t model_row_stride = static_cast<std::size_t>(model_width);
        const std::size_t active_offset =
            static_cast<std::size_t>(padding.top) * model_row_stride +
            static_cast<std::size_t>(padding.left);

        for (int y = 0; y < height; ++y) {
            const Float16Storage* src_row =
                y_base + active_offset + static_cast<std::size_t>(y) * model_row_stride;
            std::uint8_t* dst_row =
                y_plane.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
#if defined(__aarch64__)
            write_y_row_neon(src_row, dst_row, width, pixel_range);
#else
            for (int x = 0; x < width; ++x)
                dst_row[x] = quantize_byte(half_to_float(src_row[x]) / pixel_range);
#endif
        }

        const int uv_width = width / 2;
        const int uv_height = height / 2;
        auto convert_uv_plane = [&](const Float16Storage* src_base, std::uint8_t* dst_base) {
            for (int uy = 0; uy < uv_height; ++uy) {
                const Float16Storage* r0 =
                    src_base + active_offset +
                    static_cast<std::size_t>(uy * 2) * model_row_stride;
                const Float16Storage* r1 = r0 + model_row_stride;
                std::uint8_t* dst_row =
                    dst_base + static_cast<std::size_t>(uy) * static_cast<std::size_t>(uv_width);
#if defined(__aarch64__)
                write_uv_row_neon(r0, r1, dst_row, uv_width, pixel_range);
#else
                for (int ux = 0; ux < uv_width; ++ux) {
                    const int col = ux * 2;
                    const float sum =
                        half_to_float(r0[col]) / pixel_range +
                        half_to_float(r0[col + 1]) / pixel_range +
                        half_to_float(r1[col]) / pixel_range +
                        half_to_float(r1[col + 1]) / pixel_range;
                    dst_row[ux] = quantize_byte(sum * 0.25F);
                }
#endif
            }
        };
        convert_uv_plane(u_base, u_plane.data());
        convert_uv_plane(v_base, v_plane.data());
        return;
    }

    auto sample = [&](int channel, int y, int x) {
        const int model_y = x + padding.top;
        const int model_x = y + padding.left;
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
            y_plane[static_cast<std::size_t>(y) *
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
            u_plane[index] = quantize_byte(u * 0.25F);
            v_plane[index] = quantize_byte(v * 0.25F);
        }
    }
}

Yuv420Frame prepare_model_output(const Tensor& tensor, int width, int height,
                                 float pixel_range,
                                 const FramePadding& padding)
{
    validate_dimensions(width, height);
    Yuv420Frame result{width, height,
        std::vector<std::uint8_t>(static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height)),
        std::vector<std::uint8_t>(static_cast<std::size_t>(width / 2) *
                                  static_cast<std::size_t>(height / 2)),
        std::vector<std::uint8_t>(static_cast<std::size_t>(width / 2) *
                                  static_cast<std::size_t>(height / 2))};
    prepare_model_output_into(
        tensor, mutable_yuv420_view(result), pixel_range, padding);
    return result;
}

}  // namespace mlvc
