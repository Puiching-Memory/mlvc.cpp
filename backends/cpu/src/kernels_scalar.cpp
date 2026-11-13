#include "kernels_common.hpp"

#include "mlvc/core/half.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mlvc::cpu_backend {

std::vector<Float16Storage> pack_conv_weights_f16p16x1bias(
    const Float16Storage* weight_oihw, const Float16Storage* bias,
    std::size_t oc, std::size_t ic, std::size_t kh, std::size_t kw)
{
    const std::size_t k_dim = kh * kw * ic;
    const std::size_t n_blocks = (oc + 15U) / 16U;
    const std::size_t block_elems = 16U + k_dim * 16U;
    std::vector<Float16Storage> packed(n_blocks * block_elems, 0);

    for (std::size_t nb = 0; nb < oc; nb += 16U) {
        Float16Storage* dst_block = packed.data() + (nb / 16U) * block_elems;
        const std::size_t valid_n = std::min<std::size_t>(16U, oc - nb);
        for (std::size_t ni = 0; ni < valid_n; ++ni)
            dst_block[ni] = bias ? bias[nb + ni] : 0;

        Float16Storage* dst_weights = dst_block + 16U;
        if (kh == 1U && kw == 1U) {
            for (std::size_t c = 0; c < ic; ++c) {
                Float16Storage* dst_row = dst_weights + c * 16U;
                for (std::size_t ni = 0; ni < valid_n; ++ni)
                    dst_row[ni] = weight_oihw[(nb + ni) * ic + c];
            }
        } else {
            const std::size_t kernel_plane = kh * kw;
            for (std::size_t ky = 0; ky < kh; ++ky) {
                for (std::size_t kx = 0; kx < kw; ++kx) {
                    const std::size_t k_spatial = (ky * kw + kx) * ic;
                    const std::size_t k_off = ky * kw + kx;
                    for (std::size_t c = 0; c < ic; ++c) {
                        Float16Storage* dst_row =
                            dst_weights + (k_spatial + c) * 16U;
                        for (std::size_t ni = 0; ni < valid_n; ++ni) {
                            const std::size_t src_idx =
                                ((nb + ni) * ic + c) * kernel_plane + k_off;
                            dst_row[ni] = weight_oihw[src_idx];
                        }
                    }
                }
            }
        }
    }
    return packed;
}

std::vector<Float16Storage> pack_depthwise_weights(
    const Float16Storage* weight_c1hw, const Float16Storage* bias,
    std::size_t channels, std::size_t kh, std::size_t kw)
{
    const std::size_t kernel_plane = kh * kw;
    const std::size_t pad_channels = ((channels + 15U) / 16U) * 16U;
    std::vector<Float16Storage> packed((kernel_plane + 1U) * pad_channels, 0);

    Float16Storage* dst_bias = packed.data();
    for (std::size_t c = 0; c < channels; ++c)
        dst_bias[c] = bias ? bias[c] : 0;

    Float16Storage* dst_w = packed.data() + pad_channels;
    for (std::size_t k = 0; k < kernel_plane; ++k) {
        Float16Storage* row = dst_w + k * pad_channels;
        for (std::size_t c = 0; c < channels; ++c)
            row[c] = weight_c1hw[c * kernel_plane + k];
    }
    return packed;
}

namespace {

inline float apply_unary_scalar(float x, UnaryKind op, float alpha,
                                float clamp_min, float clamp_max) noexcept
{
    switch (op) {
    case UnaryKind::kLeakyRelu:
        return x < 0.0F ? x * alpha : x;
    case UnaryKind::kClip:
        return std::min(std::max(x, clamp_min), clamp_max);
    case UnaryKind::kSigmoid:
        return 1.0F / (1.0F + std::exp(-x));
    case UnaryKind::kReciprocal:
        return 1.0F / x;
    case UnaryKind::kRound:
        return std::nearbyintf(x);
    }
    return x;
}

inline float apply_binary_scalar(float a, float b, BinaryKind op) noexcept
{
    switch (op) {
    case BinaryKind::kAdd:
        return a + b;
    case BinaryKind::kMul:
        return a * b;
    case BinaryKind::kSub:
        return a - b;
    }
    return a;
}

void scalar_apply_gemm_epilogue(
    std::size_t m_rows, std::size_t n, std::size_t k,
    const Float16Storage* lhs_block, std::size_t lhs_stride,
    const Float16Storage* rhs_packed,
    Float16Storage* dst_block, std::size_t dst_stride,
    const GemmEpilogue& epilogue, std::size_t row_offset,
    Float16Storage* scratch, std::size_t scratch_elems)
{
    const std::size_t block_elems = 16U + k * 16U;
    if (epilogue.kind == GemmEpilogueKind::kReglu) {
        const std::size_t gate_c = n / 2U;
        const std::size_t req = m_rows * n;
        std::vector<Float16Storage> local_buf;
        Float16Storage* temp = scratch;
        if (!temp || scratch_elems < req) {
            local_buf.resize(req);
            temp = local_buf.data();
        }
        std::vector<float> accum(n, 0.0F);
        for (std::size_t mi = 0; mi < m_rows; ++mi) {
            const Float16Storage* a_row = lhs_block + mi * lhs_stride;
            for (std::size_t nb = 0; nb < n; nb += 16U) {
                const Float16Storage* wblk = rhs_packed + (nb / 16U) * block_elems;
                const std::size_t valid_n = std::min<std::size_t>(16U, n - nb);
                for (std::size_t ni = 0; ni < valid_n; ++ni)
                    accum[nb + ni] = half_to_float(wblk[ni]);
                const Float16Storage* wptr = wblk + 16U;
                for (std::size_t kk = 0; kk < k; ++kk) {
                    const float av = half_to_float(a_row[kk]);
                    const Float16Storage* wrow = wptr + kk * 16U;
                    for (std::size_t ni = 0; ni < valid_n; ++ni)
                        accum[nb + ni] += av * half_to_float(wrow[ni]);
                }
            }
            Float16Storage* out_row = dst_block + mi * dst_stride;
            for (std::size_t c = 0; c < gate_c; ++c) {
                const Float16Storage g = float_to_half(accum[c]);
                const Float16Storage lin = float_to_half(accum[c + gate_c]);
                const float clamped = std::min(
                    std::max(half_to_float(g), epilogue.clamp_min), epilogue.clamp_max);
                const Float16Storage cg = float_to_half(clamped);
                out_row[c] = float_to_half(half_to_float(cg) * half_to_float(lin));
            }
        }
        (void)temp;
        return;
    }

    for (std::size_t mi = 0; mi < m_rows; ++mi) {
        const Float16Storage* a_row = lhs_block + mi * lhs_stride;
        Float16Storage* out_row = dst_block + mi * dst_stride;
        const Float16Storage* res_row =
            epilogue.residual ? (epilogue.residual + (row_offset + mi) * dst_stride) : nullptr;
        for (std::size_t nb = 0; nb < n; nb += 16U) {
            const Float16Storage* wblk = rhs_packed + (nb / 16U) * block_elems;
            const std::size_t valid_n = std::min<std::size_t>(16U, n - nb);
            std::array<float, 16> acc{};
            for (std::size_t ni = 0; ni < valid_n; ++ni)
                acc[ni] = half_to_float(wblk[ni]);
            const Float16Storage* wptr = wblk + 16U;
            for (std::size_t kk = 0; kk < k; ++kk) {
                const float av = half_to_float(a_row[kk]);
                const Float16Storage* wrow = wptr + kk * 16U;
                for (std::size_t ni = 0; ni < valid_n; ++ni)
                    acc[ni] += av * half_to_float(wrow[ni]);
            }
            for (std::size_t ni = 0; ni < valid_n; ++ni) {
                Float16Storage hv = float_to_half(acc[ni]);
                if (epilogue.kind == GemmEpilogueKind::kClamp) {
                    const float fv = std::min(
                        std::max(half_to_float(hv), epilogue.clamp_min), epilogue.clamp_max);
                    hv = float_to_half(fv);
                } else if (epilogue.kind == GemmEpilogueKind::kLeakyRelu) {
                    const float fv = half_to_float(hv);
                    if (fv < 0.0F)
                        hv = float_to_half(fv * epilogue.alpha);
                } else if (epilogue.kind == GemmEpilogueKind::kResidualAdd && res_row) {
                    hv = float_to_half(
                        half_to_float(hv) + half_to_float(res_row[nb + ni]));
                }
                out_row[nb + ni] = hv;
            }
        }
    }
}

#if defined(__aarch64__)
// Baseline ARMv8.0 NEON FP32-accumulating 4x16 GEMM kernel (mandatory on every
// AArch64 SoC even when FEAT_FP16 is absent).
void neon_fp32_run_block(
    std::size_t m_rows, std::size_t n, std::size_t k,
    const Float16Storage* lhs_block, std::size_t lhs_stride,
    const Float16Storage* rhs_packed,
    Float16Storage* dst_block, std::size_t dst_stride,
    const GemmEpilogue& epilogue, std::size_t row_offset,
    Float16Storage* scratch, std::size_t scratch_elems)
{
    if (n % 16U != 0U) {
        scalar_apply_gemm_epilogue(
            m_rows, n, k, lhs_block, lhs_stride, rhs_packed,
            dst_block, dst_stride, epilogue, row_offset, scratch, scratch_elems);
        return;
    }

    const std::size_t block_elems = 16U + k * 16U;
    const bool is_reglu = (epilogue.kind == GemmEpilogueKind::kReglu);
    const std::size_t gate_c = is_reglu ? (n / 2U) : n;
    std::vector<Float16Storage> local_reglu;
    Float16Storage* reglu_buf = nullptr;
    if (is_reglu) {
        const std::size_t req = m_rows * n;
        if (scratch && scratch_elems >= req) {
            reglu_buf = scratch;
        } else {
            local_reglu.resize(req);
            reglu_buf = local_reglu.data();
        }
    }

    for (std::size_t mi = 0; mi < m_rows; mi += 4U) {
        const std::size_t mr = std::min<std::size_t>(4U, m_rows - mi);
        const Float16Storage* a0 = lhs_block + (mi + 0U) * lhs_stride;
        const Float16Storage* a1 = mr > 1U ? lhs_block + (mi + 1U) * lhs_stride : a0;
        const Float16Storage* a2 = mr > 2U ? lhs_block + (mi + 2U) * lhs_stride : a0;
        const Float16Storage* a3 = mr > 3U ? lhs_block + (mi + 3U) * lhs_stride : a0;

        for (std::size_t nb = 0; nb < n; nb += 16U) {
            const Float16Storage* wblk = rhs_packed + (nb / 16U) * block_elems;
            const float32x4_t b0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wblk + 0U)));
            const float32x4_t b1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wblk + 4U)));
            const float32x4_t b2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wblk + 8U)));
            const float32x4_t b3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wblk + 12U)));

            float32x4_t c00 = b0, c01 = b1, c02 = b2, c03 = b3;
            float32x4_t c10 = b0, c11 = b1, c12 = b2, c13 = b3;
            float32x4_t c20 = b0, c21 = b1, c22 = b2, c23 = b3;
            float32x4_t c30 = b0, c31 = b1, c32 = b2, c33 = b3;

            const Float16Storage* wptr = wblk + 16U;
            std::size_t kk = 0;
            for (; kk + 4U <= k; kk += 4U) {
                const float32x4_t av0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a0 + kk)));
                const float32x4_t av1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a1 + kk)));
                const float32x4_t av2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a2 + kk)));
                const float32x4_t av3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a3 + kk)));
                auto mac4 = [&](int lane) {
                    const float32x4_t w0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wptr + 0U)));
                    const float32x4_t w1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wptr + 4U)));
                    const float32x4_t w2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wptr + 8U)));
                    const float32x4_t w3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wptr + 12U)));
                    wptr += 16U;
                    switch (lane) {
                    case 0:
                        c00 = vfmaq_laneq_f32(c00, w0, av0, 0); c01 = vfmaq_laneq_f32(c01, w1, av0, 0);
                        c02 = vfmaq_laneq_f32(c02, w2, av0, 0); c03 = vfmaq_laneq_f32(c03, w3, av0, 0);
                        c10 = vfmaq_laneq_f32(c10, w0, av1, 0); c11 = vfmaq_laneq_f32(c11, w1, av1, 0);
                        c12 = vfmaq_laneq_f32(c12, w2, av1, 0); c13 = vfmaq_laneq_f32(c13, w3, av1, 0);
                        c20 = vfmaq_laneq_f32(c20, w0, av2, 0); c21 = vfmaq_laneq_f32(c21, w1, av2, 0);
                        c22 = vfmaq_laneq_f32(c22, w2, av2, 0); c23 = vfmaq_laneq_f32(c23, w3, av2, 0);
                        c30 = vfmaq_laneq_f32(c30, w0, av3, 0); c31 = vfmaq_laneq_f32(c31, w1, av3, 0);
                        c32 = vfmaq_laneq_f32(c32, w2, av3, 0); c33 = vfmaq_laneq_f32(c33, w3, av3, 0);
                        break;
                    case 1:
                        c00 = vfmaq_laneq_f32(c00, w0, av0, 1); c01 = vfmaq_laneq_f32(c01, w1, av0, 1);
                        c02 = vfmaq_laneq_f32(c02, w2, av0, 1); c03 = vfmaq_laneq_f32(c03, w3, av0, 1);
                        c10 = vfmaq_laneq_f32(c10, w0, av1, 1); c11 = vfmaq_laneq_f32(c11, w1, av1, 1);
                        c12 = vfmaq_laneq_f32(c12, w2, av1, 1); c13 = vfmaq_laneq_f32(c13, w3, av1, 1);
                        c20 = vfmaq_laneq_f32(c20, w0, av2, 1); c21 = vfmaq_laneq_f32(c21, w1, av2, 1);
                        c22 = vfmaq_laneq_f32(c22, w2, av2, 1); c23 = vfmaq_laneq_f32(c23, w3, av2, 1);
                        c30 = vfmaq_laneq_f32(c30, w0, av3, 1); c31 = vfmaq_laneq_f32(c31, w1, av3, 1);
                        c32 = vfmaq_laneq_f32(c32, w2, av3, 1); c33 = vfmaq_laneq_f32(c33, w3, av3, 1);
                        break;
                    case 2:
                        c00 = vfmaq_laneq_f32(c00, w0, av0, 2); c01 = vfmaq_laneq_f32(c01, w1, av0, 2);
                        c02 = vfmaq_laneq_f32(c02, w2, av0, 2); c03 = vfmaq_laneq_f32(c03, w3, av0, 2);
                        c10 = vfmaq_laneq_f32(c10, w0, av1, 2); c11 = vfmaq_laneq_f32(c11, w1, av1, 2);
                        c12 = vfmaq_laneq_f32(c12, w2, av1, 2); c13 = vfmaq_laneq_f32(c13, w3, av1, 2);
                        c20 = vfmaq_laneq_f32(c20, w0, av2, 2); c21 = vfmaq_laneq_f32(c21, w1, av2, 2);
                        c22 = vfmaq_laneq_f32(c22, w2, av2, 2); c23 = vfmaq_laneq_f32(c23, w3, av2, 2);
                        c30 = vfmaq_laneq_f32(c30, w0, av3, 2); c31 = vfmaq_laneq_f32(c31, w1, av3, 2);
                        c32 = vfmaq_laneq_f32(c32, w2, av3, 2); c33 = vfmaq_laneq_f32(c33, w3, av3, 2);
                        break;
                    case 3:
                        c00 = vfmaq_laneq_f32(c00, w0, av0, 3); c01 = vfmaq_laneq_f32(c01, w1, av0, 3);
                        c02 = vfmaq_laneq_f32(c02, w2, av0, 3); c03 = vfmaq_laneq_f32(c03, w3, av0, 3);
                        c10 = vfmaq_laneq_f32(c10, w0, av1, 3); c11 = vfmaq_laneq_f32(c11, w1, av1, 3);
                        c12 = vfmaq_laneq_f32(c12, w2, av1, 3); c13 = vfmaq_laneq_f32(c13, w3, av1, 3);
                        c20 = vfmaq_laneq_f32(c20, w0, av2, 3); c21 = vfmaq_laneq_f32(c21, w1, av2, 3);
                        c22 = vfmaq_laneq_f32(c22, w2, av2, 3); c23 = vfmaq_laneq_f32(c23, w3, av2, 3);
                        c30 = vfmaq_laneq_f32(c30, w0, av3, 3); c31 = vfmaq_laneq_f32(c31, w1, av3, 3);
                        c32 = vfmaq_laneq_f32(c32, w2, av3, 3); c33 = vfmaq_laneq_f32(c33, w3, av3, 3);
                        break;
                    default:
                        break;
                    }
                };
                mac4(0);
                mac4(1);
                mac4(2);
                mac4(3);
            }
            for (; kk < k; ++kk, wptr += 16U) {
                const float32x4_t w0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wptr + 0U)));
                const float32x4_t w1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wptr + 4U)));
                const float32x4_t w2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wptr + 8U)));
                const float32x4_t w3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wptr + 12U)));
                const float32x4_t s0 = vdupq_n_f32(half_to_float(a0[kk]));
                const float32x4_t s1 = vdupq_n_f32(half_to_float(a1[kk]));
                const float32x4_t s2 = vdupq_n_f32(half_to_float(a2[kk]));
                const float32x4_t s3 = vdupq_n_f32(half_to_float(a3[kk]));
                c00 = vfmaq_f32(c00, w0, s0); c01 = vfmaq_f32(c01, w1, s0);
                c02 = vfmaq_f32(c02, w2, s0); c03 = vfmaq_f32(c03, w3, s0);
                c10 = vfmaq_f32(c10, w0, s1); c11 = vfmaq_f32(c11, w1, s1);
                c12 = vfmaq_f32(c12, w2, s1); c13 = vfmaq_f32(c13, w3, s1);
                c20 = vfmaq_f32(c20, w0, s2); c21 = vfmaq_f32(c21, w1, s2);
                c22 = vfmaq_f32(c22, w2, s2); c23 = vfmaq_f32(c23, w3, s2);
                c30 = vfmaq_f32(c30, w0, s3); c31 = vfmaq_f32(c31, w1, s3);
                c32 = vfmaq_f32(c32, w2, s3); c33 = vfmaq_f32(c33, w3, s3);
            }

            float32x4_t rows[4][4] = {
                {c00, c01, c02, c03},
                {c10, c11, c12, c13},
                {c20, c21, c22, c23},
                {c30, c31, c32, c33},
            };
            const float32x4_t v_min = vdupq_n_f32(epilogue.clamp_min);
            const float32x4_t v_max = vdupq_n_f32(epilogue.clamp_max);
            const float32x4_t v_alpha = vdupq_n_f32(epilogue.alpha);
            const float32x4_t v_zero = vdupq_n_f32(0.0F);
            for (std::size_t r = 0; r < mr; ++r) {
                Float16Storage* out_row = is_reglu
                    ? (reglu_buf + (mi + r) * n + nb)
                    : (dst_block + (mi + r) * dst_stride + nb);
                if (epilogue.kind == GemmEpilogueKind::kClamp) {
                    for (int g = 0; g < 4; ++g) {
                        const float16x4_t h = vcvt_f16_f32(rows[r][g]);
                        const float32x4_t f = vmaxq_f32(vminq_f32(vcvt_f32_f16(h), v_max), v_min);
                        vst1_u16(out_row + g * 4, vreinterpret_u16_f16(vcvt_f16_f32(f)));
                    }
                } else if (epilogue.kind == GemmEpilogueKind::kLeakyRelu) {
                    for (int g = 0; g < 4; ++g) {
                        const float32x4_t h_as_f = vcvt_f32_f16(vcvt_f16_f32(rows[r][g]));
                        const uint32x4_t neg = vcltq_f32(h_as_f, v_zero);
                        const float32x4_t f = vbslq_f32(neg, vmulq_f32(h_as_f, v_alpha), h_as_f);
                        vst1_u16(out_row + g * 4, vreinterpret_u16_f16(vcvt_f16_f32(f)));
                    }
                } else if (epilogue.kind == GemmEpilogueKind::kResidualAdd && epilogue.residual) {
                    const Float16Storage* res_ptr =
                        epilogue.residual + (row_offset + mi + r) * dst_stride + nb;
                    for (int g = 0; g < 4; ++g) {
                        const float32x4_t h_as_f = vcvt_f32_f16(vcvt_f16_f32(rows[r][g]));
                        const float32x4_t rv = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(res_ptr + g * 4)));
                        vst1_u16(out_row + g * 4,
                                 vreinterpret_u16_f16(vcvt_f16_f32(vaddq_f32(h_as_f, rv))));
                    }
                } else {
                    for (int g = 0; g < 4; ++g)
                        vst1_u16(out_row + g * 4, vreinterpret_u16_f16(vcvt_f16_f32(rows[r][g])));
                }
            }
        }
    }

    if (is_reglu) {
        const float32x4_t v_min = vdupq_n_f32(epilogue.clamp_min);
        const float32x4_t v_max = vdupq_n_f32(epilogue.clamp_max);
        for (std::size_t mi = 0; mi < m_rows; ++mi) {
            const Float16Storage* gptr = reglu_buf + mi * n;
            const Float16Storage* lptr = gptr + gate_c;
            Float16Storage* out_row = dst_block + mi * dst_stride;
            std::size_t c = 0;
            for (; c + 4U <= gate_c; c += 4U) {
                const float32x4_t gv = vmaxq_f32(vminq_f32(
                    vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(gptr + c))), v_max), v_min);
                const float32x4_t cg = vcvt_f32_f16(vcvt_f16_f32(gv));
                const float32x4_t lv = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(lptr + c)));
                vst1_u16(out_row + c, vreinterpret_u16_f16(vcvt_f16_f32(vmulq_f32(cg, lv))));
            }
            for (; c < gate_c; ++c) {
                const float cg = half_to_float(float_to_half(
                    std::min(std::max(half_to_float(gptr[c]), epilogue.clamp_min), epilogue.clamp_max)));
                out_row[c] = float_to_half(cg * half_to_float(lptr[c]));
            }
        }
    }
}
#endif

void scalar_depthwise3x3_s1p1(
    const Float16Storage* input, const Float16Storage* packed_weight,
    Float16Storage* output, int n, int h, int w, int channels,
    CpuThreadPool& pool)
{
    const std::size_t pad_c = ((static_cast<std::size_t>(channels) + 15U) / 16U) * 16U;
    const Float16Storage* bias = packed_weight;
    const Float16Storage* weights = packed_weight + pad_c;
    const int hw = h * w;
    const std::size_t total_rows = static_cast<std::size_t>(n) * h;

    pool.parallel_for(total_rows, 2U, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t row = r0; row < r1; ++row) {
            const int batch = static_cast<int>(row / static_cast<std::size_t>(h));
            const int oy = static_cast<int>(row % static_cast<std::size_t>(h));
            const Float16Storage* in_batch =
                input + static_cast<std::size_t>(batch) * hw * channels;
            Float16Storage* out_row =
                output + (static_cast<std::size_t>(batch) * h + oy) * w * channels;

            for (int ox = 0; ox < w; ++ox) {
                Float16Storage* out_px = out_row + static_cast<std::size_t>(ox) * channels;
                for (int c = 0; c < channels; ++c) {
                    float acc = half_to_float(bias[c]);
                    for (int ky = 0; ky < 3; ++ky) {
                        const int iy = oy + ky - 1;
                        if (iy < 0 || iy >= h)
                            continue;
                        for (int kx = 0; kx < 3; ++kx) {
                            const int ix = ox + kx - 1;
                            if (ix < 0 || ix >= w)
                                continue;
                            const Float16Storage* in_px =
                                in_batch + (static_cast<std::size_t>(iy) * w + ix) * channels;
                            const Float16Storage* w_row =
                                weights + static_cast<std::size_t>(ky * 3 + kx) * pad_c;
                            acc += half_to_float(in_px[c]) * half_to_float(w_row[c]);
                        }
                    }
                    out_px[c] = float_to_half(acc);
                }
            }
        }
    });
}

void scalar_unary(
    const Float16Storage* input, Float16Storage* output, std::size_t count,
    UnaryKind op, float alpha, float clamp_min, float clamp_max,
    CpuThreadPool& pool)
{
    pool.parallel_for(count, 4096U, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const float x = half_to_float(input[i]);
            output[i] = float_to_half(apply_unary_scalar(x, op, alpha, clamp_min, clamp_max));
        }
    });
}

void scalar_binary_contiguous(
    const Float16Storage* lhs, const Float16Storage* rhs,
    Float16Storage* output, std::size_t count, BinaryKind op,
    CpuThreadPool& pool)
{
    pool.parallel_for(count, 4096U, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            output[i] = float_to_half(
                apply_binary_scalar(half_to_float(lhs[i]), half_to_float(rhs[i]), op));
        }
    });
}

void scalar_binary_channel_bcast(
    const Float16Storage* vec, const Float16Storage* chan,
    Float16Storage* output, std::size_t spatial, std::size_t channels,
    BinaryKind op, bool chan_is_lhs, CpuThreadPool& pool)
{
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* src = vec + p * channels;
            Float16Storage* dst = output + p * channels;
            for (std::size_t c = 0; c < channels; ++c) {
                const float vf = half_to_float(src[c]);
                const float cf = half_to_float(chan[c]);
                dst[c] = float_to_half(
                    apply_binary_scalar(chan_is_lhs ? cf : vf, chan_is_lhs ? vf : cf, op));
            }
        }
    });
}

void scalar_binary_scalar_bcast(
    const Float16Storage* vec, Float16Storage scalar,
    Float16Storage* output, std::size_t count, BinaryKind op,
    bool scalar_is_lhs, CpuThreadPool& pool)
{
    const float sf = half_to_float(scalar);
    pool.parallel_for(count, 4096U, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const float vf = half_to_float(vec[i]);
            output[i] = float_to_half(
                apply_binary_scalar(scalar_is_lhs ? sf : vf, scalar_is_lhs ? vf : sf, op));
        }
    });
}

void scalar_reglu_nhwc(
    const Float16Storage* input, Float16Storage* output,
    std::size_t spatial, std::size_t gate_channels,
    float clamp_min, float clamp_max, CpuThreadPool& pool)
{
    const std::size_t in_c = gate_channels * 2U;
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* in_px = input + p * in_c;
            Float16Storage* out_px = output + p * gate_channels;
            for (std::size_t c = 0; c < gate_channels; ++c) {
                const float g = half_to_float(in_px[c]);
                const Float16Storage cg =
                    float_to_half(std::min(std::max(g, clamp_min), clamp_max));
                out_px[c] = float_to_half(half_to_float(cg) * half_to_float(in_px[c + gate_channels]));
            }
        }
    });
}

void scalar_feature_update_nhwc(
    const Float16Storage* input, const Float16Storage* history,
    const Float16Storage* scale, Float16Storage* output,
    std::size_t spatial, std::size_t channels, CpuThreadPool& pool)
{
    const std::size_t in_stride = channels * 3U;
    const std::size_t out_stride = channels * 2U;
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* in_px = input + p * in_stride;
            const Float16Storage* hist_px = history + p * channels;
            Float16Storage* out_px = output + p * out_stride;
            for (std::size_t c = 0; c < channels; ++c) {
                const Float16Storage a = in_px[c];
                const Float16Storage b = in_px[c + channels];
                const Float16Storage cg = in_px[c + channels * 2U];
                const Float16Storage prev = hist_px[c];
                const Float16Storage sig_b =
                    float_to_half(1.0F / (1.0F + std::exp(-half_to_float(b))));
                const Float16Storage sig_c =
                    float_to_half(1.0F / (1.0F + std::exp(-half_to_float(cg))));
                const Float16Storage w_b =
                    float_to_half(half_to_float(sig_b) * half_to_float(prev));
                const Float16Storage inv_gate =
                    float_to_half(1.0F - half_to_float(sig_b));
                const Float16Storage w_a =
                    float_to_half(half_to_float(inv_gate) * half_to_float(a));
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

void scalar_feature_update_outputs_nhwc(
    const Float16Storage* input, const Float16Storage* history,
    const Float16Storage* scale, Float16Storage* scaled_output,
    Float16Storage* blended_output, Float16Storage* feature_output,
    std::size_t spatial, std::size_t channels, CpuThreadPool& pool)
{
    const std::size_t in_stride = channels * 3U;
    const std::size_t feat_stride = channels * 2U;
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* in_px = input + p * in_stride;
            const Float16Storage* hist_px = history + p * channels;
            Float16Storage* scaled_px = scaled_output + p * channels;
            Float16Storage* blended_px = blended_output + p * channels;
            Float16Storage* feat_px = feature_output + p * feat_stride;
            for (std::size_t c = 0; c < channels; ++c) {
                const Float16Storage a = in_px[c];
                const Float16Storage b = in_px[c + channels];
                const Float16Storage cg = in_px[c + channels * 2U];
                const Float16Storage prev = hist_px[c];
                const Float16Storage sig_b =
                    float_to_half(1.0F / (1.0F + std::exp(-half_to_float(b))));
                const Float16Storage sig_c =
                    float_to_half(1.0F / (1.0F + std::exp(-half_to_float(cg))));
                const Float16Storage w_b =
                    float_to_half(half_to_float(sig_b) * half_to_float(prev));
                const Float16Storage inv_gate =
                    float_to_half(1.0F - half_to_float(sig_b));
                const Float16Storage w_a =
                    float_to_half(half_to_float(inv_gate) * half_to_float(a));
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

void scalar_y0_tail_nhwc(
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

            for (std::size_t c = 0; c < quant_channels; ++c) {
                const std::size_t c_pair = c + quant_channels;
                const Float16Storage clip0 =
                    float_to_half(std::max(half_to_float(pri_px[c]), clip_min));
                const Float16Storage clip1 =
                    float_to_half(std::max(half_to_float(pri_px[c_pair]), clip_min));
                const Float16Storage inv0 = float_to_half(1.0F / half_to_float(clip0));
                const Float16Storage inv1 = float_to_half(1.0F / half_to_float(clip1));
                const Float16Storage nrm0 =
                    float_to_half(half_to_float(lat_px[c]) * half_to_float(inv0));
                const Float16Storage nrm1 =
                    float_to_half(half_to_float(lat_px[c_pair]) * half_to_float(inv1));
                nrm_px[c] = nrm0;
                nrm_px[c_pair] = nrm1;
                clp_px[c] = clip0;
                clp_px[c_pair] = clip1;

                const Float16Storage ptail0 = pri_px[c + norm_c];
                const Float16Storage ptail1 = pri_px[c_pair + norm_c];
                const Float16Storage d0 =
                    float_to_half(half_to_float(nrm0) - half_to_float(ptail0));
                const Float16Storage d1 =
                    float_to_half(half_to_float(nrm1) - half_to_float(ptail1));
                const Float16Storage s0 =
                    float_to_half(half_to_float(d0) * half_to_float(qsc_px[c]));
                const Float16Storage s1 =
                    float_to_half(half_to_float(d1) * half_to_float(qsc_px[c_pair]));
                const Float16Storage sum_s =
                    float_to_half(half_to_float(s0) + half_to_float(s1));
                const Float16Storage rounded =
                    float_to_half(std::nearbyintf(half_to_float(sum_s)));
                raw_px[c] = rounded;

                const Float16Storage uin0 =
                    float_to_half(half_to_float(rounded) + half_to_float(ptail0));
                const Float16Storage uin1 =
                    float_to_half(half_to_float(rounded) + half_to_float(ptail1));
                const Float16Storage upd0 =
                    float_to_half(half_to_float(uin0) * half_to_float(us0_px[c]));
                const Float16Storage upd1 =
                    float_to_half(half_to_float(uin1) * half_to_float(us1_px[c]));
                c2_px[c] = upd0;
                c2_px[c_pair] = upd1;
                c3_px[c] = upd0;
                c3_px[c_pair] = upd1;
            }
            std::memcpy(c3_px + norm_c, pri_px, prior_c * sizeof(Float16Storage));
        }
    });
}

void scalar_y1_tail_nhwc(
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

            for (std::size_t c = 0; c < quant_channels; ++c) {
                const std::size_t c_pair = c + quant_channels;
                const Float16Storage d0 =
                    float_to_half(half_to_float(nrm_px[c]) - half_to_float(pri_px[c]));
                const Float16Storage d1 =
                    float_to_half(half_to_float(nrm_px[c_pair]) - half_to_float(pri_px[c_pair]));
                const Float16Storage s0 =
                    float_to_half(half_to_float(d0) * half_to_float(qsc_px[c]));
                const Float16Storage s1 =
                    float_to_half(half_to_float(d1) * half_to_float(qsc_px[c_pair]));
                const Float16Storage sum_s =
                    float_to_half(half_to_float(s0) + half_to_float(s1));
                const Float16Storage rounded =
                    float_to_half(std::nearbyintf(half_to_float(sum_s)));
                raw_px[c] = rounded;

                const Float16Storage uin0 =
                    float_to_half(half_to_float(rounded) + half_to_float(pri_px[c]));
                const Float16Storage uin1 =
                    float_to_half(half_to_float(rounded) + half_to_float(pri_px[c_pair]));
                const Float16Storage upd0 =
                    float_to_half(half_to_float(uin0) * half_to_float(us0_px[c]));
                const Float16Storage upd1 =
                    float_to_half(half_to_float(uin1) * half_to_float(us1_px[c]));
                const Float16Storage add0 =
                    float_to_half(half_to_float(c2_px[c]) + half_to_float(upd0));
                const Float16Storage add1 =
                    float_to_half(half_to_float(c2_px[c_pair]) + half_to_float(upd1));
                out_px[c] = float_to_half(half_to_float(add0) * half_to_float(clp_px[c]));
                out_px[c_pair] = float_to_half(half_to_float(add1) * half_to_float(clp_px[c_pair]));
            }
        }
    });
}

void scalar_nchw_to_nhwc(const Float16Storage* nchw, Float16Storage* nhwc,
                         std::size_t n, std::size_t c, std::size_t hw,
                         CpuThreadPool& pool)
{
    if (c == 1U || hw == 1U) {
        std::memcpy(nhwc, nchw, n * c * hw * sizeof(Float16Storage));
        return;
    }
    const std::size_t total_px = n * hw;
    pool.parallel_for(total_px, 256U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const std::size_t batch = p / hw;
            const std::size_t s = p % hw;
            const Float16Storage* batch_in = nchw + batch * c * hw + s;
            Float16Storage* out_px = nhwc + p * c;
            for (std::size_t ch = 0; ch < c; ++ch)
                out_px[ch] = batch_in[ch * hw];
        }
    });
}

void scalar_nhwc_to_nchw(const Float16Storage* nhwc, Float16Storage* nchw,
                         std::size_t n, std::size_t hw, std::size_t c,
                         CpuThreadPool& pool)
{
    if (c == 1U || hw == 1U) {
        std::memcpy(nchw, nhwc, n * c * hw * sizeof(Float16Storage));
        return;
    }
    for (std::size_t batch = 0; batch < n; ++batch) {
        const Float16Storage* batch_in = nhwc + batch * hw * c;
        Float16Storage* batch_out = nchw + batch * c * hw;
        pool.parallel_for(c, 8U, [&](std::size_t c0, std::size_t c1) {
            for (std::size_t ch = c0; ch < c1; ++ch) {
                Float16Storage* dst = batch_out + ch * hw;
                for (std::size_t s = 0; s < hw; ++s)
                    dst[s] = batch_in[s * c + ch];
            }
        });
    }
}

}  // namespace

const KernelSet& get_scalar_kernels() noexcept
{
    static const KernelSet scalar_set = []() {
        KernelSet k;
        k.mode = GemmMode::kScalarFp32;
        k.gemm.run_block = &scalar_apply_gemm_epilogue;
        k.depthwise3x3_s1p1 = &scalar_depthwise3x3_s1p1;
        k.unary = &scalar_unary;
        k.binary_contiguous = &scalar_binary_contiguous;
        k.binary_channel_bcast = &scalar_binary_channel_bcast;
        k.binary_scalar_bcast = &scalar_binary_scalar_bcast;
        k.reglu_nhwc = &scalar_reglu_nhwc;
        k.feature_update_nhwc = &scalar_feature_update_nhwc;
        k.feature_update_outputs_nhwc = &scalar_feature_update_outputs_nhwc;
        k.y0_tail_nhwc = &scalar_y0_tail_nhwc;
        k.y1_tail_nhwc = &scalar_y1_tail_nhwc;
        k.nchw_to_nhwc = &scalar_nchw_to_nhwc;
        k.nhwc_to_nchw = &scalar_nhwc_to_nchw;
        return k;
    }();
    return scalar_set;
}

const KernelSet& select_kernels(const CpuFeatures& features) noexcept
{
    switch (features.preferred_gemm) {
#if defined(MLVC_CPU_ENABLE_ARM64)
    case GemmMode::kArm64KleidiFp16:
        return get_arm64_fp16_kernels();
    case GemmMode::kArm64Fp16Fml:
        return get_arm64_fp16fml_kernels();
    case GemmMode::kArm64NeonFp32: {
        if (features.arm64_fp16) {
            static const KernelSet neon_v82_fp32 = []() {
                KernelSet k = get_arm64_fp16_kernels();
                k.mode = GemmMode::kArm64NeonFp32;
                k.gemm.run_block = &neon_fp32_run_block;
                return k;
            }();
            return neon_v82_fp32;
        }
        static const KernelSet neon_fp32 = []() {
            KernelSet k = get_scalar_kernels();
            k.mode = GemmMode::kArm64NeonFp32;
            k.gemm.run_block = &neon_fp32_run_block;
            return k;
        }();
        return neon_fp32;
    }
#endif
#if defined(MLVC_CPU_ENABLE_X86_AVX2)
    case GemmMode::kX86Avx2Fp32:
        return get_x86_avx2_kernels();
#endif
    case GemmMode::kScalarFp32:
    default:
        return get_scalar_kernels();
    }
}

void execute_gemm_nhwc(
    const KernelSet& kernels, std::size_t m, std::size_t n, std::size_t k,
    const Float16Storage* lhs, const Float16Storage* rhs_packed,
    Float16Storage* dst, const GemmEpilogue& epilogue, CpuThreadPool& pool)
{
    if (m == 0 || n == 0)
        return;

    const std::size_t out_cols =
        epilogue.kind == GemmEpilogueKind::kReglu ? (n / 2U) : n;
    constexpr std::size_t kRowBlock = 24U;
    const std::size_t m_tiles = (m + kRowBlock - 1U) / kRowBlock;
    const bool split_n = (epilogue.kind != GemmEpilogueKind::kReglu) &&
                         (m_tiles < pool.thread_count() * 2U) && (n >= 64U) &&
                         (n % 16U == 0U);
    if (split_n) {
        constexpr std::size_t kColBlock = 64U;
        const std::size_t n_tiles = (n + kColBlock - 1U) / kColBlock;
        const std::size_t block_elems = 16U + k * 16U;
        const std::size_t total_tasks = m_tiles * n_tiles;
        pool.parallel_for(total_tasks, 1U, [&](std::size_t t0, std::size_t t1) {
            for (std::size_t t = t0; t < t1; ++t) {
                const std::size_t mt = t / n_tiles;
                const std::size_t nt = t % n_tiles;
                const std::size_t m0 = mt * kRowBlock;
                const std::size_t mc = std::min(kRowBlock, m - m0);
                const std::size_t n0 = nt * kColBlock;
                const std::size_t nc = std::min(kColBlock, n - n0);
                GemmEpilogue sub_epi = epilogue;
                if (sub_epi.residual)
                    sub_epi.residual += n0;
                kernels.gemm.run_block(
                    mc, nc, k,
                    lhs + m0 * k, k,
                    rhs_packed + (n0 / 16U) * block_elems,
                    dst + m0 * out_cols + n0, out_cols,
                    sub_epi, m0, nullptr, 0);
            }
        });
        return;
    }

    pool.parallel_for(m_tiles, 1U, [&](std::size_t mt0, std::size_t mt1) {
        std::vector<Float16Storage> scratch;
        if (epilogue.kind == GemmEpilogueKind::kReglu)
            scratch.resize(kRowBlock * n);
        for (std::size_t mt = mt0; mt < mt1; ++mt) {
            const std::size_t m0 = mt * kRowBlock;
            const std::size_t mc = std::min(kRowBlock, m - m0);
            kernels.gemm.run_block(
                mc, n, k,
                lhs + m0 * k, k,
                rhs_packed,
                dst + m0 * out_cols, out_cols,
                epilogue, m0, scratch.data(), scratch.size());
        }
    });
}

void execute_spatial_conv_nhwc(
    const KernelSet& kernels, const Float16Storage* input,
    const Float16Storage* rhs_packed, Float16Storage* output,
    int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int ph, int pw,
    const GemmEpilogue& epilogue, CpuThreadPool& pool)
{
    const std::size_t out_cols =
        epilogue.kind == GemmEpilogueKind::kReglu ? static_cast<std::size_t>(oc / 2)
                                                 : static_cast<std::size_t>(oc);
    const std::size_t total_spatial = static_cast<std::size_t>(n) * oh * ow;
    const std::size_t k_dim = static_cast<std::size_t>(kh) * kw * ic;
    const std::size_t row_bytes = static_cast<std::size_t>(ic) * sizeof(Float16Storage);
    const int in_plane = ih * iw;
    const int out_plane = oh * ow;
    constexpr std::size_t kStripRows = 12U;
    const std::size_t strips = (total_spatial + kStripRows - 1U) / kStripRows;

    pool.parallel_for(strips, 1U, [&](std::size_t s0, std::size_t s1) {
        std::vector<Float16Storage> im2col_buf(kStripRows * k_dim);
        std::vector<Float16Storage> scratch;
        if (epilogue.kind == GemmEpilogueKind::kReglu)
            scratch.resize(kStripRows * static_cast<std::size_t>(oc));

        for (std::size_t st = s0; st < s1; ++st) {
            const std::size_t m0 = st * kStripRows;
            const std::size_t mc = std::min(kStripRows, total_spatial - m0);
            for (std::size_t mr = 0; mr < mc; ++mr) {
                const std::size_t linear = m0 + mr;
                const int batch = static_cast<int>(linear / static_cast<std::size_t>(out_plane));
                const int plane = static_cast<int>(linear % static_cast<std::size_t>(out_plane));
                const int oy = plane / ow;
                const int ox = plane % ow;
                const Float16Storage* in_batch =
                    input + static_cast<std::size_t>(batch) * in_plane * ic;
                Float16Storage* col_dst = im2col_buf.data() + mr * k_dim;
                for (int ky = 0; ky < kh; ++ky) {
                    const int iy = oy * sh - ph + ky;
                    if (iy < 0 || iy >= ih) {
                        std::memset(col_dst, 0, static_cast<std::size_t>(kw) * row_bytes);
                        col_dst += static_cast<std::size_t>(kw) * ic;
                        continue;
                    }
                    const Float16Storage* in_row =
                        in_batch + static_cast<std::size_t>(iy) * iw * ic;
                    for (int kx = 0; kx < kw; ++kx) {
                        const int ix = ox * sw - pw + kx;
                        if (ix >= 0 && ix < iw) {
                            std::memcpy(col_dst, in_row + static_cast<std::size_t>(ix) * ic,
                                        row_bytes);
                        } else {
                            std::memset(col_dst, 0, row_bytes);
                        }
                        col_dst += ic;
                    }
                }
            }
            kernels.gemm.run_block(
                mc, static_cast<std::size_t>(oc), k_dim,
                im2col_buf.data(), k_dim, rhs_packed,
                output + m0 * out_cols, out_cols,
                epilogue, m0, scratch.data(), scratch.size());
        }
    });
}

void execute_depthwise_generic_nhwc(
    const Float16Storage* input, const Float16Storage* packed_weight,
    Float16Storage* output, int n, int ic, int ih, int iw,
    int oh, int ow, int kh, int kw, int sh, int sw, int ph, int pw,
    CpuThreadPool& pool)
{
    const std::size_t pad_c = ((static_cast<std::size_t>(ic) + 15U) / 16U) * 16U;
    const Float16Storage* bias = packed_weight;
    const Float16Storage* weights = packed_weight + pad_c;
    const std::size_t total_rows = static_cast<std::size_t>(n) * oh;
    const int in_plane = ih * iw;

    pool.parallel_for(total_rows, 2U, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t row = r0; row < r1; ++row) {
            const int batch = static_cast<int>(row / static_cast<std::size_t>(oh));
            const int oy = static_cast<int>(row % static_cast<std::size_t>(oh));
            const Float16Storage* in_batch =
                input + static_cast<std::size_t>(batch) * in_plane * ic;
            Float16Storage* out_row =
                output + (static_cast<std::size_t>(batch) * oh + oy) * ow * ic;

            for (int ox = 0; ox < ow; ++ox) {
                Float16Storage* out_px = out_row + static_cast<std::size_t>(ox) * ic;
                for (int c = 0; c < ic; ++c) {
                    float acc = half_to_float(bias[c]);
                    for (int ky = 0; ky < kh; ++ky) {
                        const int iy = oy * sh - ph + ky;
                        if (iy < 0 || iy >= ih)
                            continue;
                        for (int kx = 0; kx < kw; ++kx) {
                            const int ix = ox * sw - pw + kx;
                            if (ix < 0 || ix >= iw)
                                continue;
                            const Float16Storage* in_px =
                                in_batch + (static_cast<std::size_t>(iy) * iw + ix) * ic;
                            const Float16Storage* w_row =
                                weights + static_cast<std::size_t>(ky * kw + kx) * pad_c;
                            acc += half_to_float(in_px[c]) * half_to_float(w_row[c]);
                        }
                    }
                    out_px[c] = float_to_half(acc);
                }
            }
        }
    });
}

void execute_slice_nhwc(
    const Float16Storage* input, Float16Storage* output,
    const int* id, const int* od, const int* starts, const int* steps,
    CpuThreadPool& pool)
{
    // Note: id, od, starts, steps remain in logical NCHW order [N, C, H, W]
    // while underlying buffers store NHWC [N, H, W, C].
    const int on = od[0], oc = od[1], oh = od[2], ow = od[3];
    const int in_c = id[1], in_h = id[2], in_w = id[3];

    const bool unit_steps = (steps[0] == 1 && steps[1] == 1 &&
                             steps[2] == 1 && steps[3] == 1);
    if (unit_steps && starts[0] == 0 && starts[1] == 0 &&
        starts[3] == 0 && on == 1 && oc == in_c && ow == in_w) {
        // Pure H-slice in NHWC is a single contiguous block copy!
        const std::size_t elems =
            static_cast<std::size_t>(oh) * ow * oc;
        const std::size_t offset =
            static_cast<std::size_t>(starts[2]) * in_w * in_c;
        std::memcpy(output, input + offset, elems * sizeof(Float16Storage));
        return;
    }
    if (unit_steps && starts[0] == 0 && starts[2] == 0 &&
        starts[3] == 0 && on == id[0] && oh == in_h && ow == in_w) {
        // Pure C-slice in NHWC copies contiguous channel segments per pixel.
        const std::size_t spatial = static_cast<std::size_t>(on) * oh * ow;
        const std::size_t copy_bytes = static_cast<std::size_t>(oc) * sizeof(Float16Storage);
        const int c_off = starts[1];
        pool.parallel_for(spatial, 128U, [&](std::size_t p0, std::size_t p1) {
            for (std::size_t p = p0; p < p1; ++p) {
                std::memcpy(output + p * oc,
                            input + p * in_c + c_off, copy_bytes);
            }
        });
        return;
    }

    const std::size_t total_spatial = static_cast<std::size_t>(on) * oh * ow;
    const std::size_t oc_bytes = static_cast<std::size_t>(oc) * sizeof(Float16Storage);
    const bool contiguous_c = (steps[1] == 1);
    pool.parallel_for(total_spatial, 128U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const int b = static_cast<int>(p / static_cast<std::size_t>(oh * ow));
            const int hw_idx = static_cast<int>(p % static_cast<std::size_t>(oh * ow));
            const int y = hw_idx / ow;
            const int x = hw_idx % ow;
            const int ib = starts[0] + b * steps[0];
            const int iy = starts[2] + y * steps[2];
            const int ix = starts[3] + x * steps[3];
            const Float16Storage* in_px =
                input + (((static_cast<std::size_t>(ib) * in_h + iy) * in_w + ix) * in_c);
            Float16Storage* out_px = output + p * oc;
            if (contiguous_c) {
                std::memcpy(out_px, in_px + starts[1], oc_bytes);
            } else {
                for (int c = 0; c < oc; ++c)
                    out_px[c] = in_px[starts[1] + c * steps[1]];
            }
        }
    });
}

void execute_concat_nhwc(
    const Float16Storage* const* inputs, const int* const* input_dims_nchw,
    std::size_t input_count, Float16Storage* output, const int* od_nchw,
    int axis_nchw, CpuThreadPool& pool)
{
    const int on = od_nchw[0], oc = od_nchw[1], oh = od_nchw[2], ow = od_nchw[3];
    if (axis_nchw == 2 && on == 1) {
        // H-axis concatenation in NHWC is literally sequential memcpy chunks!
        std::size_t out_offset = 0;
        for (std::size_t idx = 0; idx < input_count; ++idx) {
            const int* id = input_dims_nchw[idx];
            const std::size_t elems =
                static_cast<std::size_t>(id[0]) * id[2] * id[3] * id[1];
            std::memcpy(output + out_offset, inputs[idx],
                        elems * sizeof(Float16Storage));
            out_offset += elems;
        }
        return;
    }

    if (axis_nchw == 1) {
        const std::size_t spatial = static_cast<std::size_t>(on) * oh * ow;
        pool.parallel_for(spatial, 128U, [&](std::size_t p0, std::size_t p1) {
            for (std::size_t p = p0; p < p1; ++p) {
                Float16Storage* dst_px = output + p * oc;
                std::size_t c_offset = 0;
                for (std::size_t idx = 0; idx < input_count; ++idx) {
                    const int ic = input_dims_nchw[idx][1];
                    std::memcpy(dst_px + c_offset,
                                inputs[idx] + p * ic,
                                static_cast<std::size_t>(ic) * sizeof(Float16Storage));
                    c_offset += static_cast<std::size_t>(ic);
                }
            }
        });
        return;
    }

    std::array<int, 4> axis_offset{0, 0, 0, 0};
    for (std::size_t idx = 0; idx < input_count; ++idx) {
        const int* id = input_dims_nchw[idx];
        const Float16Storage* in = inputs[idx];
        for (int b = 0; b < id[0]; ++b) {
            for (int y = 0; y < id[2]; ++y) {
                for (int x = 0; x < id[3]; ++x) {
                    const Float16Storage* src_px =
                        in + ((static_cast<std::size_t>(b) * id[2] + y) * id[3] + x) * id[1];
                    Float16Storage* dst_px =
                        output +
                        ((static_cast<std::size_t>(b + axis_offset[0]) * oh +
                          (y + axis_offset[2])) * ow + (x + axis_offset[3])) * oc +
                        axis_offset[1];
                    std::memcpy(dst_px, src_px,
                                static_cast<std::size_t>(id[1]) * sizeof(Float16Storage));
                }
            }
        }
        axis_offset[static_cast<std::size_t>(axis_nchw)] +=
            id[static_cast<std::size_t>(axis_nchw)];
    }
}

void execute_depth_to_space_nhwc(
    const Float16Storage* input, Float16Storage* output,
    int n, int in_c, int in_h, int in_w, int block, CpuThreadPool& pool)
{
    // CRD layout matching PyTorch PixelShuffle: in_c = out_c * block * block.
    const int b2 = block * block;
    const int out_c = in_c / b2;
    const int out_w = in_w * block;
    const std::size_t total_rows = static_cast<std::size_t>(n) * in_h;

    pool.parallel_for(total_rows, 2U, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t row = r0; row < r1; ++row) {
            const int batch = static_cast<int>(row / static_cast<std::size_t>(in_h));
            const int iy = static_cast<int>(row % static_cast<std::size_t>(in_h));
            const Float16Storage* in_row =
                input + (static_cast<std::size_t>(batch) * in_h + iy) * in_w * in_c;
            Float16Storage* out_batch =
                output + static_cast<std::size_t>(batch) * (in_h * block) * out_w * out_c;

            for (int ix = 0; ix < in_w; ++ix) {
                const Float16Storage* in_px = in_row + static_cast<std::size_t>(ix) * in_c;
                for (int dy = 0; dy < block; ++dy) {
                    const int oy = iy * block + dy;
                    Float16Storage* out_y_row =
                        out_batch + (static_cast<std::size_t>(oy) * out_w + ix * block) * out_c;
                    for (int dx = 0; dx < block; ++dx) {
                        Float16Storage* out_px = out_y_row + static_cast<std::size_t>(dx) * out_c;
                        const int off = dy * block + dx;
                        for (int oc = 0; oc < out_c; ++oc)
                            out_px[oc] = in_px[oc * b2 + off];
                    }
                }
            }
        }
    });
}

void execute_space_to_depth_nhwc(
    const Float16Storage* input, Float16Storage* output,
    int n, int in_c, int in_h, int in_w, int block, CpuThreadPool& pool)
{
    const int b2 = block * block;
    const int out_h = in_h / block;
    const int out_w = in_w / block;
    const int out_c = in_c * b2;
    const std::size_t total_rows = static_cast<std::size_t>(n) * out_h;

    pool.parallel_for(total_rows, 2U, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t row = r0; row < r1; ++row) {
            const int batch = static_cast<int>(row / static_cast<std::size_t>(out_h));
            const int oy = static_cast<int>(row % static_cast<std::size_t>(out_h));
            const Float16Storage* in_batch =
                input + static_cast<std::size_t>(batch) * in_h * in_w * in_c;
            Float16Storage* out_row =
                output + (static_cast<std::size_t>(batch) * out_h + oy) * out_w * out_c;

            for (int ox = 0; ox < out_w; ++ox) {
                Float16Storage* out_px = out_row + static_cast<std::size_t>(ox) * out_c;
                for (int dy = 0; dy < block; ++dy) {
                    const int iy = oy * block + dy;
                    const Float16Storage* in_y_row =
                        in_batch + (static_cast<std::size_t>(iy) * in_w + ox * block) * in_c;
                    for (int dx = 0; dx < block; ++dx) {
                        const Float16Storage* in_px = in_y_row + static_cast<std::size_t>(dx) * in_c;
                        const int off = dy * block + dx;
                        for (int c = 0; c < in_c; ++c)
                            out_px[c * b2 + off] = in_px[c];
                    }
                }
            }
        }
    });
}

}  // namespace mlvc::cpu_backend
