#include "execution_plan.hpp"

#include "core/half.hpp"
#include "core/yuv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mlvc::cpu_backend {
namespace {

void write_yuv420_to_nhwc_fp16(
    const std::uint8_t* yuv_bytes, const CodecIoConfig& io,
    const std::array<Float16Storage, 256>& lut,
    Float16Storage* dst_nhwc) noexcept
{
    const int width = io.width;
    const int height = io.height;
    const int model_w = io.model_width;
    const int model_h = io.model_height;
    const int uv_width = width / 2;
    const std::size_t y_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t uv_size = y_size / 4U;
    const std::uint8_t* src_y = yuv_bytes;
    const std::uint8_t* src_u = yuv_bytes + y_size;
    const std::uint8_t* src_v = src_u + uv_size;

    if (!io.padding.rotated) {
        const std::size_t row_elems = static_cast<std::size_t>(model_w) * 3U;
        for (int y = 0; y < height; ++y) {
            const std::uint8_t* sy =
                src_y + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            const std::size_t uv_row =
                static_cast<std::size_t>(y / 2) * static_cast<std::size_t>(uv_width);
            const std::uint8_t* su = src_u + uv_row;
            const std::uint8_t* sv = src_v + uv_row;
            Float16Storage* drow = dst_nhwc + static_cast<std::size_t>(y) * row_elems;

            int x = 0;
#if defined(__aarch64__)
            for (; x + 8 <= width; x += 8) {
                const int ux = x / 2;
                const std::uint16_t y8[8] = {
                    lut[sy[x]],     lut[sy[x + 1]], lut[sy[x + 2]], lut[sy[x + 3]],
                    lut[sy[x + 4]], lut[sy[x + 5]], lut[sy[x + 6]], lut[sy[x + 7]]};
                const std::uint16_t u4[4] = {
                    lut[su[ux]], lut[su[ux + 1]], lut[su[ux + 2]], lut[su[ux + 3]]};
                const std::uint16_t v4[4] = {
                    lut[sv[ux]], lut[sv[ux + 1]], lut[sv[ux + 2]], lut[sv[ux + 3]]};
                const uint16x4_t u_half = vld1_u16(u4);
                const uint16x4_t v_half = vld1_u16(v4);
                const uint16x8_t u_comb = vcombine_u16(u_half, u_half);
                const uint16x8_t v_comb = vcombine_u16(v_half, v_half);
                uint16x8x3_t yuv3;
                yuv3.val[0] = vld1q_u16(y8);
                yuv3.val[1] = vzip1q_u16(u_comb, u_comb);
                yuv3.val[2] = vzip1q_u16(v_comb, v_comb);
                vst3q_u16(reinterpret_cast<std::uint16_t*>(
                              drow + static_cast<std::size_t>(x) * 3U),
                          yuv3);
            }
#endif
            for (; x < width; ++x) {
                const std::size_t off = static_cast<std::size_t>(x) * 3U;
                const int ux = x / 2;
                drow[off] = lut[sy[x]];
                drow[off + 1U] = lut[su[ux]];
                drow[off + 2U] = lut[sv[ux]];
            }
            if (width < model_w) {
                const std::size_t last_off = static_cast<std::size_t>(width - 1) * 3U;
                const Float16Storage py = drow[last_off];
                const Float16Storage pu = drow[last_off + 1U];
                const Float16Storage pv = drow[last_off + 2U];
                for (int px = width; px < model_w; ++px) {
                    const std::size_t off = static_cast<std::size_t>(px) * 3U;
                    drow[off] = py;
                    drow[off + 1U] = pu;
                    drow[off + 2U] = pv;
                }
            }
        }
        if (height < model_h) {
            const Float16Storage* last_row =
                dst_nhwc + static_cast<std::size_t>(height - 1) * row_elems;
            const std::size_t row_bytes = row_elems * sizeof(Float16Storage);
            for (int y = height; y < model_h; ++y) {
                std::memcpy(dst_nhwc + static_cast<std::size_t>(y) * row_elems,
                            last_row, row_bytes);
            }
        }
        return;
    }

    for (int y = 0; y < model_h; ++y) {
        const int source_x = std::min(y, width - 1);
        Float16Storage* drow =
            dst_nhwc + static_cast<std::size_t>(y) * static_cast<std::size_t>(model_w) * 3U;
        for (int x = 0; x < model_w; ++x) {
            const int source_y = std::min(x, height - 1);
            const std::size_t y_off =
                static_cast<std::size_t>(source_y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(source_x);
            const std::size_t uv_off =
                static_cast<std::size_t>(source_y / 2) * static_cast<std::size_t>(uv_width) +
                static_cast<std::size_t>(source_x / 2);
            const std::size_t dst_off = static_cast<std::size_t>(x) * 3U;
            drow[dst_off] = lut[src_y[y_off]];
            drow[dst_off + 1U] = lut[src_u[uv_off]];
            drow[dst_off + 2U] = lut[src_v[uv_off]];
        }
    }
}

#if defined(__aarch64__)
inline void quantize_y8_neon(uint16x8_t bits, std::uint8_t* dst,
                             float pixel_range) noexcept
{
    const float32x4_t zero = vdupq_n_f32(0.0F);
    const float32x4_t one = vdupq_n_f32(1.0F);
    const float32x4_t v_range = vdupq_n_f32(pixel_range);
    const float16x8_t h = vreinterpretq_f16_u16(bits);
    float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h));
    float32x4_t f1 = vcvt_high_f32_f16(h);
    if (pixel_range != 1.0F) {
        f0 = vdivq_f32(f0, v_range);
        f1 = vdivq_f32(f1, v_range);
    }
    f0 = vminnmq_f32(vmaxnmq_f32(f0, zero), one);
    f1 = vminnmq_f32(vmaxnmq_f32(f1, zero), one);
    const uint32x4_t q0 = vcvtq_u32_f32(vrndiq_f32(vmulq_n_f32(f0, 255.0F)));
    const uint32x4_t q1 = vcvtq_u32_f32(vrndiq_f32(vmulq_n_f32(f1, 255.0F)));
    vst1_u8(dst, vqmovn_u16(vcombine_u16(vqmovn_u32(q0), vqmovn_u32(q1))));
}

inline void quantize_uv4_neon(uint16x8_t b0, uint16x8_t b1, std::uint8_t* dst,
                              float pixel_range) noexcept
{
    const float32x4_t zero = vdupq_n_f32(0.0F);
    const float32x4_t one = vdupq_n_f32(1.0F);
    const float32x4_t v_range = vdupq_n_f32(pixel_range);
    const float16x8_t h0 = vreinterpretq_f16_u16(b0);
    const float16x8_t h1 = vreinterpretq_f16_u16(b1);
    float32x4_t a0 = vcvt_f32_f16(vget_low_f16(h0));
    float32x4_t a1 = vcvt_high_f32_f16(h0);
    float32x4_t c0 = vcvt_f32_f16(vget_low_f16(h1));
    float32x4_t c1 = vcvt_high_f32_f16(h1);
    if (pixel_range != 1.0F) {
        a0 = vdivq_f32(a0, v_range);
        a1 = vdivq_f32(a1, v_range);
        c0 = vdivq_f32(c0, v_range);
        c1 = vdivq_f32(c1, v_range);
    }
    const float32x4_t row0_pair = vpaddq_f32(a0, a1);
    const float32x4_t row1_even = vuzp1q_f32(c0, c1);
    const float32x4_t row1_odd = vuzp2q_f32(c0, c1);
    const float32x4_t sum = vaddq_f32(vaddq_f32(row0_pair, row1_even), row1_odd);
    const float32x4_t clipped =
        vminnmq_f32(vmaxnmq_f32(vmulq_n_f32(sum, 0.25F), zero), one);
    const uint32x4_t q =
        vcvtq_u32_f32(vrndiq_f32(vmulq_n_f32(clipped, 255.0F)));
    const uint16x4_t q16 = vqmovn_u32(q);
    const uint8x8_t q8 = vqmovn_u16(vcombine_u16(q16, q16));
    const std::uint32_t packed4 = vget_lane_u32(vreinterpret_u32_u8(q8), 0);
    std::memcpy(dst, &packed4, sizeof(packed4));
}
#endif

void read_nhwc_fp16_to_yuv420(
    const Float16Storage* src_nhwc, const CodecIoConfig& io,
    std::uint8_t* yuv_dst) noexcept
{
    const int width = io.width;
    const int height = io.height;
    const int model_w = io.model_width;
    const int uv_width = width / 2;
    const int uv_height = height / 2;
    const float pixel_range = io.pixel_range;
    const std::size_t y_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t uv_size = y_size / 4U;
    std::uint8_t* y_dst = yuv_dst;
    std::uint8_t* u_dst = y_dst + y_size;
    std::uint8_t* v_dst = u_dst + uv_size;

    if (!io.padding.rotated) {
        const std::size_t model_row_elems = static_cast<std::size_t>(model_w) * 3U;
        const Float16Storage* active =
            src_nhwc +
            (static_cast<std::size_t>(io.padding.top) * static_cast<std::size_t>(model_w) +
             static_cast<std::size_t>(io.padding.left)) *
                3U;

        for (int uy = 0; uy < uv_height; ++uy) {
            const Float16Storage* r0 =
                active + static_cast<std::size_t>(uy * 2) * model_row_elems;
            const Float16Storage* r1 = r0 + model_row_elems;
            std::uint8_t* y0_out =
                y_dst + static_cast<std::size_t>(uy * 2) * static_cast<std::size_t>(width);
            std::uint8_t* y1_out = y0_out + width;
            std::uint8_t* u_out =
                u_dst + static_cast<std::size_t>(uy) * static_cast<std::size_t>(uv_width);
            std::uint8_t* v_out =
                v_dst + static_cast<std::size_t>(uy) * static_cast<std::size_t>(uv_width);

            int ux = 0;
#if defined(__aarch64__)
            for (; ux + 4 <= uv_width; ux += 4) {
                const int x = ux * 2;
                const std::size_t src_off = static_cast<std::size_t>(x) * 3U;
                const uint16x8x3_t p0 =
                    vld3q_u16(reinterpret_cast<const std::uint16_t*>(r0 + src_off));
                const uint16x8x3_t p1 =
                    vld3q_u16(reinterpret_cast<const std::uint16_t*>(r1 + src_off));
                quantize_y8_neon(p0.val[0], y0_out + x, pixel_range);
                quantize_y8_neon(p1.val[0], y1_out + x, pixel_range);
                quantize_uv4_neon(p0.val[1], p1.val[1], u_out + ux, pixel_range);
                quantize_uv4_neon(p0.val[2], p1.val[2], v_out + ux, pixel_range);
            }
#endif
            for (; ux < uv_width; ++ux) {
                const int x = ux * 2;
                const std::size_t o0 = static_cast<std::size_t>(x) * 3U;
                const std::size_t o1 = o0 + 3U;
                y0_out[x] = quantize_unit_float(half_to_float(r0[o0]) / pixel_range);
                y0_out[x + 1] = quantize_unit_float(half_to_float(r0[o1]) / pixel_range);
                y1_out[x] = quantize_unit_float(half_to_float(r1[o0]) / pixel_range);
                y1_out[x + 1] = quantize_unit_float(half_to_float(r1[o1]) / pixel_range);
                const float u_sum =
                    half_to_float(r0[o0 + 1U]) / pixel_range +
                    half_to_float(r0[o1 + 1U]) / pixel_range +
                    half_to_float(r1[o0 + 1U]) / pixel_range +
                    half_to_float(r1[o1 + 1U]) / pixel_range;
                const float v_sum =
                    half_to_float(r0[o0 + 2U]) / pixel_range +
                    half_to_float(r0[o1 + 2U]) / pixel_range +
                    half_to_float(r1[o0 + 2U]) / pixel_range +
                    half_to_float(r1[o1 + 2U]) / pixel_range;
                u_out[ux] = quantize_unit_float(u_sum * 0.25F);
                v_out[ux] = quantize_unit_float(v_sum * 0.25F);
            }
        }
        return;
    }

    auto sample = [&](int channel, int y, int x) noexcept -> float {
        const int model_y = x + io.padding.top;
        const int model_x = y + io.padding.left;
        const std::size_t idx =
            (static_cast<std::size_t>(model_y) * static_cast<std::size_t>(model_w) +
             static_cast<std::size_t>(model_x)) *
                3U +
            static_cast<std::size_t>(channel);
        return half_to_float(src_nhwc[idx]) / pixel_range;
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            y_dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(x)] =
                quantize_unit_float(sample(0, y, x));
        }
    }
    for (int y = 0; y < uv_height; ++y) {
        for (int x = 0; x < uv_width; ++x) {
            float u = 0.0F;
            float v = 0.0F;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    u += sample(1, y * 2 + dy, x * 2 + dx);
                    v += sample(2, y * 2 + dy, x * 2 + dx);
                }
            }
            const std::size_t idx =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(uv_width) +
                static_cast<std::size_t>(x);
            u_dst[idx] = quantize_unit_float(u * 0.25F);
            v_dst[idx] = quantize_unit_float(v * 0.25F);
        }
    }
}

template <typename ViewType, typename ByteVec>
ViewType make_packed_yuv_view(ByteVec& bytes, int width, int height)
{
    using Elem = std::conditional_t<std::is_const_v<ByteVec>,
                                    const std::uint8_t, std::uint8_t>;
    const std::size_t y_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t uv_size = y_size / 4U;
    const std::span<Elem> storage(bytes.data(), bytes.size());
    return ViewType(width, height, storage.first(y_size),
                    storage.subspan(y_size, uv_size),
                    storage.subspan(y_size + uv_size, uv_size));
}

}  // namespace

void CpuExecutionPlan::configure_codec_io(const CodecIoConfig& config)
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    if (codec_io_configured_)
        throw std::runtime_error("cpu: codec I/O is already configured");
    if (model_name_ != "MLVCEncoder" && model_name_ != "MLVCDecoder")
        throw std::runtime_error("cpu: unsupported codec graph");
    if (config.slots == 0U)
        throw std::runtime_error("cpu: codec slot count must be positive");
    if (!std::isfinite(config.pixel_range) || config.pixel_range <= 0.0F)
        throw std::runtime_error("cpu: pixel range must be finite and positive");
    if (config.padding.left < 0 || config.padding.right < 0 ||
        config.padding.top < 0 || config.padding.bottom < 0) {
        throw std::runtime_error("cpu: codec frame padding must be non-negative");
    }
    const std::size_t yuv_bytes = yuv420_frame_bytes(config.width, config.height);
    const int active_w = config.padding.rotated ? config.height : config.width;
    const int active_h = config.padding.rotated ? config.width : config.height;
    if (active_w + config.padding.left + config.padding.right != config.model_width ||
        active_h + config.padding.top + config.padding.bottom != config.model_height) {
        throw std::runtime_error(
            "cpu: codec I/O padded dimensions do not match model dimensions");
    }

    const std::vector<std::int64_t> image_shape{
        1, 3, config.model_height, config.model_width};
    bool found_q = false;
    bool found_img = false;
    for (std::size_t i = 0; i < input_ids_.size(); ++i) {
        const TensorMeta& m = require_meta(input_ids_[i]);
        if (m.state_input_slot >= 0)
            continue;
        if (m.name == "q_index_shifted") {
            if (m.dtype != aot::DataType::INT32 ||
                m.shape != std::vector<std::int64_t>{1}) {
                throw std::runtime_error("cpu: invalid q_index_shifted input");
            }
            q_input_graph_idx_ = i;
            found_q = true;
        } else if (model_name_ == "MLVCEncoder" && m.name == "x") {
            if (m.dtype != aot::DataType::FP16 || m.shape != image_shape) {
                throw std::runtime_error(
                    "cpu: codec I/O model dimensions do not match graph");
            }
            x_input_graph_idx_ = i;
            found_img = true;
        }
    }
    if (model_name_ == "MLVCDecoder") {
        for (const aot::TensorId out_id : output_ids_) {
            const TensorMeta& m = require_meta(out_id);
            if (m.state_output_slot >= 0 || m.name != "x_hat")
                continue;
            if (m.dtype != aot::DataType::FP16 || m.shape != image_shape) {
                throw std::runtime_error(
                    "cpu: codec I/O model dimensions do not match graph");
            }
            x_hat_output_id_ = out_id;
            found_img = true;
            break;
        }
    }
    if (!found_q || !found_img)
        throw std::runtime_error("cpu: missing required codec I/O tensors");

    if (model_name_ == "MLVCEncoder") {
        for (int v = 0; v < 256; ++v) {
            const float normalized =
                (static_cast<float>(v) / 255.0F) * config.pixel_range;
            codec_byte_lut_[static_cast<std::size_t>(v)] = float_to_half(normalized);
        }
    }

    codec_slots_.clear();
    codec_slots_.resize(config.slots);
    for (CodecSlot& slot : codec_slots_) {
        slot.yuv.assign(yuv_bytes, 0U);
        if (model_name_ == "MLVCEncoder") {
            slot.outputs.resize(output_ids_.size());
            for (std::size_t i = 0; i < output_ids_.size(); ++i) {
                const TensorMeta& out_m = require_meta(output_ids_[i]);
                if (out_m.state_output_slot >= 0)
                    continue;
                slot.outputs[i].resize(out_m.byte_size);
            }
        } else {
            slot.inputs.resize(input_ids_.size());
            for (std::size_t i = 0; i < input_ids_.size(); ++i) {
                const TensorMeta& in_m = require_meta(input_ids_[i]);
                if (in_m.state_input_slot >= 0 || i == q_input_graph_idx_)
                    continue;
                slot.inputs[i].resize(in_m.byte_size);
            }
        }
    }

    codec_io_ = config;
    codec_io_configured_ = true;
}

std::size_t CpuExecutionPlan::codec_slot_count() const noexcept
{
    return codec_io_configured_ ? codec_slots_.size() : 0U;
}

MutableYuv420FrameView CpuExecutionPlan::encoder_input_yuv(std::size_t slot_index)
{
    if (!codec_io_configured_ || model_name_ != "MLVCEncoder")
        throw std::runtime_error("cpu: encoder codec I/O is unavailable");
    if (slot_index >= codec_slots_.size())
        throw std::runtime_error("cpu: codec slot is out of range");
    return make_packed_yuv_view<MutableYuv420FrameView>(
        codec_slots_[slot_index].yuv, codec_io_.width, codec_io_.height);
}

void CpuExecutionPlan::submit_encoder(std::size_t slot_index, int shifted_q)
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    if (!codec_io_configured_ || model_name_ != "MLVCEncoder")
        throw std::runtime_error("cpu: encoder codec I/O is unavailable");
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error("cpu: reset_state() must be called before submit_encoder()");
    if (slot_index >= codec_slots_.size())
        throw std::runtime_error("cpu: codec slot is out of range");

    CodecSlot& slot = codec_slots_[slot_index];
    write_yuv420_to_nhwc_fp16(
        slot.yuv.data(), codec_io_, codec_byte_lut_,
        reinterpret_cast<Float16Storage*>(input_buffers_[x_input_graph_idx_].data()));
    const std::int32_t q32 = static_cast<std::int32_t>(shifted_q);
    std::memcpy(input_buffers_[q_input_graph_idx_].data(), &q32, sizeof(q32));

    execute_all_steps_and_swap_states();

    for (std::size_t i = 0; i < output_ids_.size(); ++i) {
        const aot::TensorId out_id = output_ids_[i];
        const TensorMeta& out_m = require_meta(out_id);
        if (out_m.state_output_slot >= 0)
            continue;
        write_graph_output_bytes(out_m, resolve_const_ptr(out_id), slot.outputs[i].data());
    }
}

std::vector<TensorView> CpuExecutionPlan::encoder_outputs(std::size_t slot_index) const
{
    if (!codec_io_configured_ || model_name_ != "MLVCEncoder")
        throw std::runtime_error("cpu: encoder codec I/O is unavailable");
    if (slot_index >= codec_slots_.size())
        throw std::runtime_error("cpu: codec slot is out of range");

    const CodecSlot& slot = codec_slots_[slot_index];
    std::vector<TensorView> outputs;
    outputs.reserve(output_ids_.size() - state_bindings_.size());
    for (std::size_t i = 0; i < output_ids_.size(); ++i) {
        const TensorMeta& out_m = require_meta(output_ids_[i]);
        if (out_m.state_output_slot >= 0)
            continue;
        outputs.push_back(TensorView::from_raw(
            out_m.name, std::span<const std::int64_t>(out_m.shape),
            aot::to_public_dtype(out_m.dtype), slot.outputs[i].data(),
            out_m.byte_size));
    }
    return outputs;
}

std::vector<MutableTensorView> CpuExecutionPlan::decoder_inputs(std::size_t slot_index)
{
    if (!codec_io_configured_ || model_name_ != "MLVCDecoder")
        throw std::runtime_error("cpu: decoder codec I/O is unavailable");
    if (slot_index >= codec_slots_.size())
        throw std::runtime_error("cpu: codec slot is out of range");

    CodecSlot& slot = codec_slots_[slot_index];
    std::vector<MutableTensorView> inputs;
    inputs.reserve(input_ids_.size() - state_bindings_.size() - 1U);
    for (std::size_t i = 0; i < input_ids_.size(); ++i) {
        const TensorMeta& in_m = require_meta(input_ids_[i]);
        if (in_m.state_input_slot >= 0 || i == q_input_graph_idx_)
            continue;
        inputs.push_back(MutableTensorView::from_raw(
            in_m.name, std::span<const std::int64_t>(in_m.shape),
            aot::to_public_dtype(in_m.dtype), slot.inputs[i].data(),
            in_m.byte_size));
    }
    return inputs;
}

void CpuExecutionPlan::submit_decoder(std::size_t slot_index, int shifted_q)
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    if (!codec_io_configured_ || model_name_ != "MLVCDecoder")
        throw std::runtime_error("cpu: decoder codec I/O is unavailable");
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error("cpu: reset_state() must be called before submit_decoder()");
    if (slot_index >= codec_slots_.size())
        throw std::runtime_error("cpu: codec slot is out of range");

    CodecSlot& slot = codec_slots_[slot_index];
    for (std::size_t i = 0; i < input_ids_.size(); ++i) {
        const TensorMeta& in_m = require_meta(input_ids_[i]);
        if (in_m.state_input_slot >= 0 || i == q_input_graph_idx_)
            continue;
        write_graph_input_bytes(in_m, slot.inputs[i].data(), input_buffers_[i].data());
    }
    const std::int32_t q32 = static_cast<std::int32_t>(shifted_q);
    std::memcpy(input_buffers_[q_input_graph_idx_].data(), &q32, sizeof(q32));

    execute_all_steps_and_swap_states();

    read_nhwc_fp16_to_yuv420(
        fp16_const(x_hat_output_id_), codec_io_, slot.yuv.data());
}

Yuv420FrameView CpuExecutionPlan::decoder_output_yuv(std::size_t slot_index) const
{
    if (!codec_io_configured_ || model_name_ != "MLVCDecoder")
        throw std::runtime_error("cpu: decoder codec I/O is unavailable");
    if (slot_index >= codec_slots_.size())
        throw std::runtime_error("cpu: codec slot is out of range");
    return make_packed_yuv_view<Yuv420FrameView>(
        codec_slots_[slot_index].yuv, codec_io_.width, codec_io_.height);
}

void CpuExecutionPlan::wait_codec_slot(std::size_t slot_index)
{
    if (!codec_io_configured_)
        throw std::runtime_error("cpu: codec I/O is unavailable");
    if (slot_index >= codec_slots_.size())
        throw std::runtime_error("cpu: codec slot is out of range");
}

}  // namespace mlvc::cpu_backend
