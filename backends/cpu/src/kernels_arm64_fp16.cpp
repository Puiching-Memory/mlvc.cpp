#include "kernels_common.hpp"

#include "mlvc/core/half.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mlvc::cpu_backend {
namespace {

inline float16x8_t vld_f16q(const Float16Storage* ptr) noexcept
{
    return vreinterpretq_f16_u16(vld1q_u16(ptr));
}

inline void vst_f16q(Float16Storage* ptr, float16x8_t val) noexcept
{
    vst1q_u16(ptr, vreinterpretq_u16_f16(val));
}

inline float32x4x2_t f16x8_to_f32x8(float16x8_t v) noexcept
{
    float32x4x2_t res;
    res.val[0] = vcvt_f32_f16(vget_low_f16(v));
    res.val[1] = vcvt_high_f32_f16(v);
    return res;
}

inline float16x8_t f32x8_to_f16x8(float32x4_t lo, float32x4_t hi) noexcept
{
    return vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
}

inline float16x8_t fast_sigmoid_f16x8(float16x8_t x) noexcept
{
    // Exact scalar expf rounded via NEON float32x4 matches CUDA float->half sigmoid boundary.
    alignas(16) float lo_arr[4];
    alignas(16) float hi_arr[4];
    vst1q_f32(lo_arr, vcvt_f32_f16(vget_low_f16(x)));
    vst1q_f32(hi_arr, vcvt_high_f32_f16(x));
    for (int i = 0; i < 4; ++i) {
        lo_arr[i] = 1.0F / (1.0F + std::exp(-lo_arr[i]));
        hi_arr[i] = 1.0F / (1.0F + std::exp(-hi_arr[i]));
    }
    return f32x8_to_f16x8(vld1q_f32(lo_arr), vld1q_f32(hi_arr));
}

void arm64_unary(
    const Float16Storage* input, Float16Storage* output, std::size_t count,
    UnaryKind op, float alpha, float clamp_min, float clamp_max,
    CpuThreadPool& pool)
{
    pool.parallel_for(count, 4096U, [&](std::size_t begin, std::size_t end) {
        std::size_t i = begin;
        if (op == UnaryKind::kLeakyRelu) {
            const float32x4_t va = vdupq_n_f32(alpha);
            const float16x8_t vz = vdupq_n_f16(static_cast<float16_t>(0.0F));
            for (; i + 8U <= end; i += 8U) {
                const float16x8_t h = vld_f16q(input + i);
                const uint16x8_t neg = vcltq_f16(h, vz);
                const float32x4x2_t f = f16x8_to_f32x8(h);
                const float16x8_t sc = f32x8_to_f16x8(
                    vmulq_f32(f.val[0], va), vmulq_f32(f.val[1], va));
                vst1q_u16(output + i, vbslq_u16(
                    neg, vreinterpretq_u16_f16(sc), vreinterpretq_u16_f16(h)));
            }
            for (; i < end; ++i) {
                const float fv = half_to_float(input[i]);
                output[i] = fv < 0.0F ? float_to_half(fv * alpha) : input[i];
            }
        } else if (op == UnaryKind::kClip) {
            const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(clamp_min));
            const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(clamp_max));
            for (; i + 8U <= end; i += 8U)
                vst_f16q(output + i, vmaxq_f16(vminq_f16(vld_f16q(input + i), vmax), vmin));
            for (; i < end; ++i)
                output[i] = float_to_half(std::min(std::max(half_to_float(input[i]), clamp_min), clamp_max));
        } else if (op == UnaryKind::kRound) {
            for (; i + 8U <= end; i += 8U)
                vst_f16q(output + i, vrndnq_f16(vld_f16q(input + i)));
            for (; i < end; ++i)
                output[i] = float_to_half(std::nearbyintf(half_to_float(input[i])));
        } else if (op == UnaryKind::kReciprocal) {
            const float32x4_t v_one = vdupq_n_f32(1.0F);
            for (; i + 8U <= end; i += 8U) {
                const float32x4x2_t f = f16x8_to_f32x8(vld_f16q(input + i));
                vst_f16q(output + i, f32x8_to_f16x8(
                    vdivq_f32(v_one, f.val[0]), vdivq_f32(v_one, f.val[1])));
            }
            for (; i < end; ++i)
                output[i] = float_to_half(1.0F / half_to_float(input[i]));
        } else {
            for (; i + 8U <= end; i += 8U)
                vst_f16q(output + i, fast_sigmoid_f16x8(vld_f16q(input + i)));
            for (; i < end; ++i)
                output[i] = float_to_half(1.0F / (1.0F + std::exp(-half_to_float(input[i]))));
        }
    });
}

void arm64_binary_contiguous(
    const Float16Storage* lhs, const Float16Storage* rhs,
    Float16Storage* output, std::size_t count, BinaryKind op,
    CpuThreadPool& pool)
{
    pool.parallel_for(count, 4096U, [&](std::size_t begin, std::size_t end) {
        std::size_t i = begin;
        if (op == BinaryKind::kAdd) {
            for (; i + 8U <= end; i += 8U)
                vst_f16q(output + i, vaddq_f16(vld_f16q(lhs + i), vld_f16q(rhs + i)));
            for (; i < end; ++i)
                output[i] = float_to_half(half_to_float(lhs[i]) + half_to_float(rhs[i]));
        } else if (op == BinaryKind::kMul) {
            for (; i + 8U <= end; i += 8U)
                vst_f16q(output + i, vmulq_f16(vld_f16q(lhs + i), vld_f16q(rhs + i)));
            for (; i < end; ++i)
                output[i] = float_to_half(half_to_float(lhs[i]) * half_to_float(rhs[i]));
        } else {
            for (; i + 8U <= end; i += 8U)
                vst_f16q(output + i, vsubq_f16(vld_f16q(lhs + i), vld_f16q(rhs + i)));
            for (; i < end; ++i)
                output[i] = float_to_half(half_to_float(lhs[i]) - half_to_float(rhs[i]));
        }
    });
}

void arm64_binary_channel_bcast(
    const Float16Storage* vec, const Float16Storage* chan,
    Float16Storage* output, std::size_t spatial, std::size_t channels,
    BinaryKind op, bool chan_is_lhs, CpuThreadPool& pool)
{
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* src = vec + p * channels;
            Float16Storage* dst = output + p * channels;
            std::size_t c = 0;
            for (; c + 8U <= channels; c += 8U) {
                const float16x8_t vv = vld_f16q(src + c);
                const float16x8_t cv = vld_f16q(chan + c);
                const float16x8_t a = chan_is_lhs ? cv : vv;
                const float16x8_t b = chan_is_lhs ? vv : cv;
                const float16x8_t out = op == BinaryKind::kAdd ? vaddq_f16(a, b)
                    : op == BinaryKind::kMul ? vmulq_f16(a, b) : vsubq_f16(a, b);
                vst_f16q(dst + c, out);
            }
            for (; c < channels; ++c) {
                const float a = half_to_float(chan_is_lhs ? chan[c] : src[c]);
                const float b = half_to_float(chan_is_lhs ? src[c] : chan[c]);
                dst[c] = float_to_half(
                    op == BinaryKind::kAdd ? (a + b) : op == BinaryKind::kMul ? (a * b) : (a - b));
            }
        }
    });
}

void arm64_binary_scalar_bcast(
    const Float16Storage* vec, Float16Storage scalar,
    Float16Storage* output, std::size_t count, BinaryKind op,
    bool scalar_is_lhs, CpuThreadPool& pool)
{
    const float sf = half_to_float(scalar);
    const float16x8_t sv = vdupq_n_f16(static_cast<float16_t>(sf));
    pool.parallel_for(count, 4096U, [&](std::size_t begin, std::size_t end) {
        std::size_t i = begin;
        for (; i + 8U <= end; i += 8U) {
            const float16x8_t vv = vld_f16q(vec + i);
            const float16x8_t a = scalar_is_lhs ? sv : vv;
            const float16x8_t b = scalar_is_lhs ? vv : sv;
            const float16x8_t out = op == BinaryKind::kAdd ? vaddq_f16(a, b)
                : op == BinaryKind::kMul ? vmulq_f16(a, b) : vsubq_f16(a, b);
            vst_f16q(output + i, out);
        }
        for (; i < end; ++i) {
            const float vf = half_to_float(vec[i]);
            const float a = scalar_is_lhs ? sf : vf;
            const float b = scalar_is_lhs ? vf : sf;
            output[i] = float_to_half(
                op == BinaryKind::kAdd ? (a + b) : op == BinaryKind::kMul ? (a * b) : (a - b));
        }
    });
}

void arm64_reglu_nhwc(
    const Float16Storage* input, Float16Storage* output,
    std::size_t spatial, std::size_t gate_channels,
    float clamp_min, float clamp_max, CpuThreadPool& pool)
{
    const std::size_t in_c = gate_channels * 2U;
    const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(clamp_max));
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* gptr = input + p * in_c;
            const Float16Storage* lptr = gptr + gate_channels;
            Float16Storage* out_px = output + p * gate_channels;
            std::size_t c = 0;
            for (; c + 8U <= gate_channels; c += 8U) {
                const float16x8_t g = vmaxq_f16(vminq_f16(vld_f16q(gptr + c), vmax), vmin);
                vst_f16q(out_px + c, vmulq_f16(g, vld_f16q(lptr + c)));
            }
            for (; c < gate_channels; ++c) {
                const float clp = std::min(
                    std::max(half_to_float(gptr[c]), clamp_min), clamp_max);
                const Float16Storage cg = float_to_half(clp);
                out_px[c] = float_to_half(half_to_float(cg) * half_to_float(lptr[c]));
            }
        }
    });
}

void arm64_feature_update_nhwc(
    const Float16Storage* input, const Float16Storage* history,
    const Float16Storage* scale, Float16Storage* output,
    std::size_t spatial, std::size_t channels, CpuThreadPool& pool)
{
    const std::size_t in_stride = channels * 3U;
    const std::size_t out_stride = channels * 2U;
    const float16x8_t v_one = vdupq_n_f16(static_cast<float16_t>(1.0F));
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* in_px = input + p * in_stride;
            const Float16Storage* hist_px = history + p * channels;
            Float16Storage* out_px = output + p * out_stride;
            std::size_t c = 0;
            for (; c + 8U <= channels; c += 8U) {
                const float16x8_t a = vld_f16q(in_px + c);
                const float16x8_t sig_b = fast_sigmoid_f16x8(vld_f16q(in_px + channels + c));
                const float16x8_t sig_c = fast_sigmoid_f16x8(vld_f16q(in_px + channels * 2U + c));
                const float16x8_t prev = vld_f16q(hist_px + c);
                const float16x8_t inv_gate = vsubq_f16(v_one, sig_b);
                const float16x8_t blended = vaddq_f16(
                    vmulq_f16(sig_b, prev), vmulq_f16(inv_gate, a));
                const float16x8_t gated = vmulq_f16(sig_c, blended);
                const float16x8_t scaled = vmulq_f16(gated, vld_f16q(scale + c));
                vst_f16q(out_px + c, scaled);
                vst_f16q(out_px + channels + c, blended);
            }
            for (; c < channels; ++c) {
                const Float16Storage a = in_px[c];
                const Float16Storage sig_b =
                    float_to_half(1.0F / (1.0F + std::exp(-half_to_float(in_px[c + channels]))));
                const Float16Storage sig_c =
                    float_to_half(1.0F / (1.0F + std::exp(-half_to_float(in_px[c + channels * 2U]))));
                const Float16Storage w_b =
                    float_to_half(half_to_float(sig_b) * half_to_float(hist_px[c]));
                const Float16Storage inv_g = float_to_half(1.0F - half_to_float(sig_b));
                const Float16Storage w_a = float_to_half(half_to_float(inv_g) * half_to_float(a));
                const Float16Storage blended =
                    float_to_half(half_to_float(w_b) + half_to_float(w_a));
                const Float16Storage gated =
                    float_to_half(half_to_float(sig_c) * half_to_float(blended));
                out_px[c] = float_to_half(half_to_float(gated) * half_to_float(scale[c]));
                out_px[c + channels] = blended;
            }
        }
    });
}

void arm64_feature_update_outputs_nhwc(
    const Float16Storage* input, const Float16Storage* history,
    const Float16Storage* scale, Float16Storage* scaled_output,
    Float16Storage* blended_output, Float16Storage* feature_output,
    std::size_t spatial, std::size_t channels, CpuThreadPool& pool)
{
    const std::size_t in_stride = channels * 3U;
    const std::size_t feat_stride = channels * 2U;
    const float16x8_t v_one = vdupq_n_f16(static_cast<float16_t>(1.0F));
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* in_px = input + p * in_stride;
            const Float16Storage* hist_px = history + p * channels;
            Float16Storage* scaled_px = scaled_output + p * channels;
            Float16Storage* blended_px = blended_output + p * channels;
            Float16Storage* feat_px = feature_output + p * feat_stride;
            std::size_t c = 0;
            for (; c + 8U <= channels; c += 8U) {
                const float16x8_t a = vld_f16q(in_px + c);
                const float16x8_t sig_b = fast_sigmoid_f16x8(vld_f16q(in_px + channels + c));
                const float16x8_t sig_c = fast_sigmoid_f16x8(vld_f16q(in_px + channels * 2U + c));
                const float16x8_t prev = vld_f16q(hist_px + c);
                const float16x8_t inv_gate = vsubq_f16(v_one, sig_b);
                const float16x8_t blended = vaddq_f16(
                    vmulq_f16(sig_b, prev), vmulq_f16(inv_gate, a));
                const float16x8_t gated = vmulq_f16(sig_c, blended);
                const float16x8_t scaled = vmulq_f16(gated, vld_f16q(scale + c));
                vst_f16q(scaled_px + c, scaled);
                vst_f16q(blended_px + c, blended);
                vst_f16q(feat_px + c, scaled);
                vst_f16q(feat_px + channels + c, blended);
            }
            for (; c < channels; ++c) {
                const Float16Storage a = in_px[c];
                const Float16Storage sig_b =
                    float_to_half(1.0F / (1.0F + std::exp(-half_to_float(in_px[c + channels]))));
                const Float16Storage sig_c =
                    float_to_half(1.0F / (1.0F + std::exp(-half_to_float(in_px[c + channels * 2U]))));
                const Float16Storage w_b =
                    float_to_half(half_to_float(sig_b) * half_to_float(hist_px[c]));
                const Float16Storage inv_g = float_to_half(1.0F - half_to_float(sig_b));
                const Float16Storage w_a = float_to_half(half_to_float(inv_g) * half_to_float(a));
                const Float16Storage blended =
                    float_to_half(half_to_float(w_b) + half_to_float(w_a));
                const Float16Storage gated =
                    float_to_half(half_to_float(sig_c) * half_to_float(blended));
                const Float16Storage scaled =
                    float_to_half(half_to_float(gated) * half_to_float(scale[c]));
                scaled_px[c] = scaled;
                blended_px[c] = blended;
                feat_px[c] = scaled;
                feat_px[c + channels] = blended;
            }
        }
    });
}

void arm64_y0_tail_nhwc(
    const Float16Storage* latent, const Float16Storage* prior,
    const Float16Storage* quant_scale, const Float16Storage* update_scale0,
    const Float16Storage* update_scale1, Float16Storage* normalized,
    Float16Storage* clipped, Float16Storage* y_raw, Float16Storage* concat2,
    Float16Storage* concat3, float clip_min,
    std::size_t spatial, std::size_t quant_channels, CpuThreadPool& pool)
{
    const std::size_t norm_c = quant_channels * 2U;
    const std::size_t prior_c = quant_channels * 4U;
    const std::size_t c3_c = quant_channels * 6U;
    const float16x8_t v_cmin = vdupq_n_f16(static_cast<float16_t>(clip_min));
    const float32x4_t v_one = vdupq_n_f32(1.0F);

    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* lat_px = latent + p * norm_c;
            const Float16Storage* pri_px = prior + p * prior_c;
            const Float16Storage* qsc_px = quant_scale + p * norm_c;
            const Float16Storage* us0_px = update_scale0 + p * quant_channels;
            const Float16Storage* us1_px = update_scale1 + p * quant_channels;
            Float16Storage* nrm_px = normalized + p * norm_c;
            Float16Storage* clp_px = clipped + p * norm_c;
            Float16Storage* raw_px = y_raw + p * quant_channels;
            Float16Storage* c2_px = concat2 + p * norm_c;
            Float16Storage* c3_px = concat3 + p * c3_c;

            std::size_t c = 0;
            for (; c + 8U <= quant_channels; c += 8U) {
                const float16x8_t clip0 = vmaxq_f16(vld_f16q(pri_px + c), v_cmin);
                const float16x8_t clip1 = vmaxq_f16(vld_f16q(pri_px + c + quant_channels), v_cmin);
                const float32x4x2_t c0f = f16x8_to_f32x8(clip0);
                const float32x4x2_t c1f = f16x8_to_f32x8(clip1);
                const float16x8_t inv0 = f32x8_to_f16x8(
                    vdivq_f32(v_one, c0f.val[0]), vdivq_f32(v_one, c0f.val[1]));
                const float16x8_t inv1 = f32x8_to_f16x8(
                    vdivq_f32(v_one, c1f.val[0]), vdivq_f32(v_one, c1f.val[1]));
                const float16x8_t nrm0 = vmulq_f16(vld_f16q(lat_px + c), inv0);
                const float16x8_t nrm1 = vmulq_f16(vld_f16q(lat_px + c + quant_channels), inv1);
                vst_f16q(nrm_px + c, nrm0);
                vst_f16q(nrm_px + c + quant_channels, nrm1);
                vst_f16q(clp_px + c, clip0);
                vst_f16q(clp_px + c + quant_channels, clip1);

                const float16x8_t pt0 = vld_f16q(pri_px + norm_c + c);
                const float16x8_t pt1 = vld_f16q(pri_px + norm_c + c + quant_channels);
                const float16x8_t s0 = vmulq_f16(vsubq_f16(nrm0, pt0), vld_f16q(qsc_px + c));
                const float16x8_t s1 = vmulq_f16(vsubq_f16(nrm1, pt1), vld_f16q(qsc_px + c + quant_channels));
                const float16x8_t rounded = vrndnq_f16(vaddq_f16(s0, s1));
                vst_f16q(raw_px + c, rounded);

                const float16x8_t upd0 = vmulq_f16(vaddq_f16(rounded, pt0), vld_f16q(us0_px + c));
                const float16x8_t upd1 = vmulq_f16(vaddq_f16(rounded, pt1), vld_f16q(us1_px + c));
                vst_f16q(c2_px + c, upd0);
                vst_f16q(c2_px + c + quant_channels, upd1);
                vst_f16q(c3_px + c, upd0);
                vst_f16q(c3_px + c + quant_channels, upd1);
            }
            for (; c < quant_channels; ++c) {
                const std::size_t cp = c + quant_channels;
                const Float16Storage clip0 = float_to_half(std::max(half_to_float(pri_px[c]), clip_min));
                const Float16Storage clip1 = float_to_half(std::max(half_to_float(pri_px[cp]), clip_min));
                const Float16Storage inv0 = float_to_half(1.0F / half_to_float(clip0));
                const Float16Storage inv1 = float_to_half(1.0F / half_to_float(clip1));
                const Float16Storage nrm0 = float_to_half(half_to_float(lat_px[c]) * half_to_float(inv0));
                const Float16Storage nrm1 = float_to_half(half_to_float(lat_px[cp]) * half_to_float(inv1));
                nrm_px[c] = nrm0;
                nrm_px[cp] = nrm1;
                clp_px[c] = clip0;
                clp_px[cp] = clip1;

                const Float16Storage pt0 = pri_px[norm_c + c];
                const Float16Storage pt1 = pri_px[norm_c + cp];
                const Float16Storage s0 = float_to_half(
                    half_to_float(float_to_half(half_to_float(nrm0) - half_to_float(pt0))) *
                    half_to_float(qsc_px[c]));
                const Float16Storage s1 = float_to_half(
                    half_to_float(float_to_half(half_to_float(nrm1) - half_to_float(pt1))) *
                    half_to_float(qsc_px[cp]));
                const Float16Storage rounded =
                    float_to_half(std::nearbyintf(half_to_float(s0) + half_to_float(s1)));
                raw_px[c] = rounded;
                const Float16Storage upd0 = float_to_half(
                    half_to_float(float_to_half(half_to_float(rounded) + half_to_float(pt0))) *
                    half_to_float(us0_px[c]));
                const Float16Storage upd1 = float_to_half(
                    half_to_float(float_to_half(half_to_float(rounded) + half_to_float(pt1))) *
                    half_to_float(us1_px[c]));
                c2_px[c] = upd0;
                c2_px[cp] = upd1;
                c3_px[c] = upd0;
                c3_px[cp] = upd1;
            }
            std::memcpy(c3_px + norm_c, pri_px, prior_c * sizeof(Float16Storage));
        }
    });
}

void arm64_y1_tail_nhwc(
    const Float16Storage* normalized, const Float16Storage* prior,
    const Float16Storage* quant_scale, const Float16Storage* update_scale0,
    const Float16Storage* update_scale1, const Float16Storage* concat2,
    const Float16Storage* clipped, Float16Storage* y_raw,
    Float16Storage* output, std::size_t spatial,
    std::size_t quant_channels, CpuThreadPool& pool)
{
    const std::size_t norm_c = quant_channels * 2U;
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* nrm_px = normalized + p * norm_c;
            const Float16Storage* pri_px = prior + p * norm_c;
            const Float16Storage* qsc_px = quant_scale + p * norm_c;
            const Float16Storage* us0_px = update_scale0 + p * quant_channels;
            const Float16Storage* us1_px = update_scale1 + p * quant_channels;
            const Float16Storage* c2_px = concat2 + p * norm_c;
            const Float16Storage* clp_px = clipped + p * norm_c;
            Float16Storage* raw_px = y_raw + p * quant_channels;
            Float16Storage* out_px = output + p * norm_c;

            std::size_t c = 0;
            for (; c + 8U <= quant_channels; c += 8U) {
                const float16x8_t p0v = vld_f16q(pri_px + c);
                const float16x8_t p1v = vld_f16q(pri_px + c + quant_channels);
                const float16x8_t s0 = vmulq_f16(
                    vsubq_f16(vld_f16q(nrm_px + c), p0v), vld_f16q(qsc_px + c));
                const float16x8_t s1 = vmulq_f16(
                    vsubq_f16(vld_f16q(nrm_px + c + quant_channels), p1v),
                    vld_f16q(qsc_px + c + quant_channels));
                const float16x8_t rounded = vrndnq_f16(vaddq_f16(s0, s1));
                vst_f16q(raw_px + c, rounded);

                const float16x8_t upd0 = vmulq_f16(vaddq_f16(rounded, p0v), vld_f16q(us0_px + c));
                const float16x8_t upd1 = vmulq_f16(vaddq_f16(rounded, p1v), vld_f16q(us1_px + c));
                const float16x8_t add0 = vaddq_f16(vld_f16q(c2_px + c), upd0);
                const float16x8_t add1 = vaddq_f16(vld_f16q(c2_px + c + quant_channels), upd1);
                vst_f16q(out_px + c, vmulq_f16(add0, vld_f16q(clp_px + c)));
                vst_f16q(out_px + c + quant_channels,
                         vmulq_f16(add1, vld_f16q(clp_px + c + quant_channels)));
            }
            for (; c < quant_channels; ++c) {
                const std::size_t cp = c + quant_channels;
                const Float16Storage d0 =
                    float_to_half(half_to_float(nrm_px[c]) - half_to_float(pri_px[c]));
                const Float16Storage d1 =
                    float_to_half(half_to_float(nrm_px[cp]) - half_to_float(pri_px[cp]));
                const Float16Storage s0 = float_to_half(half_to_float(d0) * half_to_float(qsc_px[c]));
                const Float16Storage s1 = float_to_half(half_to_float(d1) * half_to_float(qsc_px[cp]));
                const Float16Storage rounded =
                    float_to_half(std::nearbyintf(half_to_float(s0) + half_to_float(s1)));
                raw_px[c] = rounded;
                const Float16Storage uin0 = float_to_half(half_to_float(rounded) + half_to_float(pri_px[c]));
                const Float16Storage uin1 = float_to_half(half_to_float(rounded) + half_to_float(pri_px[cp]));
                const Float16Storage upd0 = float_to_half(half_to_float(uin0) * half_to_float(us0_px[c]));
                const Float16Storage upd1 = float_to_half(half_to_float(uin1) * half_to_float(us1_px[c]));
                const Float16Storage add0 = float_to_half(half_to_float(c2_px[c]) + half_to_float(upd0));
                const Float16Storage add1 = float_to_half(half_to_float(c2_px[cp]) + half_to_float(upd1));
                out_px[c] = float_to_half(half_to_float(add0) * half_to_float(clp_px[c]));
                out_px[cp] = float_to_half(half_to_float(add1) * half_to_float(clp_px[cp]));
            }
        }
    });
}

}  // namespace

void arm64_fp16_reglu_post(
    const Float16Storage* reglu_buf, Float16Storage* dst_block,
    std::size_t m_rows, std::size_t n, std::size_t dst_stride,
    float clamp_min, float clamp_max) noexcept
{
    const std::size_t gate_c = n / 2U;
    const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(clamp_max));
    for (std::size_t mi = 0; mi < m_rows; ++mi) {
        const Float16Storage* gptr = reglu_buf + mi * n;
        const Float16Storage* lptr = gptr + gate_c;
        Float16Storage* out_row = dst_block + mi * dst_stride;
        std::size_t c = 0;
        for (; c + 8U <= gate_c; c += 8U) {
            const float16x8_t g = vmaxq_f16(vminq_f16(vld_f16q(gptr + c), vmax), vmin);
            vst_f16q(out_row + c, vmulq_f16(g, vld_f16q(lptr + c)));
        }
        for (; c < gate_c; ++c) {
            const Float16Storage cg = float_to_half(
                std::min(std::max(half_to_float(gptr[c]), clamp_min), clamp_max));
            out_row[c] = float_to_half(half_to_float(cg) * half_to_float(lptr[c]));
        }
    }
}

void apply_arm64_fp16_non_gemm_ops(KernelSet& k) noexcept
{
    k.unary = &arm64_unary;
    k.binary_contiguous = &arm64_binary_contiguous;
    k.binary_channel_bcast = &arm64_binary_channel_bcast;
    k.binary_scalar_bcast = &arm64_binary_scalar_bcast;
    k.reglu_nhwc = &arm64_reglu_nhwc;
    k.feature_update_nhwc = &arm64_feature_update_nhwc;
    k.feature_update_outputs_nhwc = &arm64_feature_update_outputs_nhwc;
    k.y0_tail_nhwc = &arm64_y0_tail_nhwc;
    k.y1_tail_nhwc = &arm64_y1_tail_nhwc;
}

}  // namespace mlvc::cpu_backend
