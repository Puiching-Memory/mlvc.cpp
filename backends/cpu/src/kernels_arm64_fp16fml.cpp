#include "kernels_common.hpp"

#include "mlvc/core/half.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <type_traits>
#include <vector>

namespace mlvc::cpu_backend {
namespace {

// 4x16 FP16->FP32 fused multiply-accumulate GEMM kernel using ARMv8.2+fp16fml
// (vfmlalq_low_f16 / vfmlalq_high_f16). Accumulates pairs of FP16 reduction
// elements directly into 4x4 = 16 FP32 vector accumulators with full FP32 range.
void fp16fml_run_block(
    std::size_t m_rows, std::size_t n, std::size_t k,
    const Float16Storage* lhs_block, std::size_t lhs_stride,
    const Float16Storage* rhs_packed,
    Float16Storage* dst_block, std::size_t dst_stride,
    const GemmEpilogue& epilogue, std::size_t row_offset,
    Float16Storage* scratch, std::size_t scratch_elems)
{
    if (n % 16U != 0U) {
        get_scalar_kernels().gemm.run_block(
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
            auto mac_step = [](const float16x8_t w0, const float16x8_t w1,
                               const float16x8_t av,
                               float32x4_t& r0, float32x4_t& r1,
                               float32x4_t& r2, float32x4_t& r3) {
                r0 = vfmlalq_low_f16(r0, w0, av);
                r1 = vfmlalq_high_f16(r1, w0, av);
                r2 = vfmlalq_low_f16(r2, w1, av);
                r3 = vfmlalq_high_f16(r3, w1, av);
            };
            auto mac_lane = []<int Lane>(const float16x8_t w0, const float16x8_t w1,
                                          const float16x4_t av,
                                          float32x4_t& r0, float32x4_t& r1,
                                          float32x4_t& r2, float32x4_t& r3) {
                r0 = vfmlalq_lane_low_f16(r0, w0, av, Lane);
                r1 = vfmlalq_lane_high_f16(r1, w0, av, Lane);
                r2 = vfmlalq_lane_low_f16(r2, w1, av, Lane);
                r3 = vfmlalq_lane_high_f16(r3, w1, av, Lane);
            };

            std::size_t kk = 0;
            for (; kk + 4U <= k; kk += 4U, wptr += 64U) {
                const float16x4_t av0 = vreinterpret_f16_u16(vld1_u16(a0 + kk));
                const float16x4_t av1 = vreinterpret_f16_u16(vld1_u16(a1 + kk));
                const float16x4_t av2 = vreinterpret_f16_u16(vld1_u16(a2 + kk));
                const float16x4_t av3 = vreinterpret_f16_u16(vld1_u16(a3 + kk));

                auto step4 = [&](auto lane_tag, std::size_t woff) {
                    const float16x8_t w0 = vreinterpretq_f16_u16(vld1q_u16(wptr + woff));
                    const float16x8_t w1 = vreinterpretq_f16_u16(vld1q_u16(wptr + woff + 8U));
                    constexpr int L = decltype(lane_tag)::value;
                    mac_lane.template operator()<L>(w0, w1, av0, c00, c01, c02, c03);
                    mac_lane.template operator()<L>(w0, w1, av1, c10, c11, c12, c13);
                    mac_lane.template operator()<L>(w0, w1, av2, c20, c21, c22, c23);
                    mac_lane.template operator()<L>(w0, w1, av3, c30, c31, c32, c33);
                };
                step4(std::integral_constant<int, 0>{}, 0U);
                step4(std::integral_constant<int, 1>{}, 16U);
                step4(std::integral_constant<int, 2>{}, 32U);
                step4(std::integral_constant<int, 3>{}, 48U);
            }
            for (; kk < k; ++kk, wptr += 16U) {
                const float16x8_t w0 = vreinterpretq_f16_u16(vld1q_u16(wptr + 0U));
                const float16x8_t w1 = vreinterpretq_f16_u16(vld1q_u16(wptr + 8U));
                mac_step(w0, w1, vreinterpretq_f16_u16(vdupq_n_u16(a0[kk])), c00, c01, c02, c03);
                mac_step(w0, w1, vreinterpretq_f16_u16(vdupq_n_u16(a1[kk])), c10, c11, c12, c13);
                mac_step(w0, w1, vreinterpretq_f16_u16(vdupq_n_u16(a2[kk])), c20, c21, c22, c23);
                mac_step(w0, w1, vreinterpretq_f16_u16(vdupq_n_u16(a3[kk])), c30, c31, c32, c33);
            }

            float32x4_t rows[4][4] = {
                {c00, c01, c02, c03},
                {c10, c11, c12, c13},
                {c20, c21, c22, c23},
                {c30, c31, c32, c33},
            };
            const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(epilogue.clamp_min));
            const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(epilogue.clamp_max));
            const float32x4_t valpha = vdupq_n_f32(epilogue.alpha);
            const float16x8_t vzero = vdupq_n_f16(static_cast<float16_t>(0.0F));

            for (std::size_t r = 0; r < mr; ++r) {
                Float16Storage* out_row = is_reglu
                    ? (reglu_buf + (mi + r) * n + nb)
                    : (dst_block + (mi + r) * dst_stride + nb);
                float16x8_t h0 = vcombine_f16(vcvt_f16_f32(rows[r][0]), vcvt_f16_f32(rows[r][1]));
                float16x8_t h1 = vcombine_f16(vcvt_f16_f32(rows[r][2]), vcvt_f16_f32(rows[r][3]));
                if (epilogue.kind == GemmEpilogueKind::kClamp) {
                    h0 = vmaxq_f16(vminq_f16(h0, vmax), vmin);
                    h1 = vmaxq_f16(vminq_f16(h1, vmax), vmin);
                } else if (epilogue.kind == GemmEpilogueKind::kLeakyRelu) {
                    const uint16x8_t n0 = vcltq_f16(h0, vzero);
                    const uint16x8_t n1 = vcltq_f16(h1, vzero);
                    const float16x8_t s0 = vcombine_f16(
                        vcvt_f16_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(h0)), valpha)),
                        vcvt_f16_f32(vmulq_f32(vcvt_high_f32_f16(h0), valpha)));
                    const float16x8_t s1 = vcombine_f16(
                        vcvt_f16_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(h1)), valpha)),
                        vcvt_f16_f32(vmulq_f32(vcvt_high_f32_f16(h1), valpha)));
                    h0 = vreinterpretq_f16_u16(vbslq_u16(n0, vreinterpretq_u16_f16(s0), vreinterpretq_u16_f16(h0)));
                    h1 = vreinterpretq_f16_u16(vbslq_u16(n1, vreinterpretq_u16_f16(s1), vreinterpretq_u16_f16(h1)));
                } else if (epilogue.kind == GemmEpilogueKind::kResidualAdd && epilogue.residual) {
                    const Float16Storage* res_ptr =
                        epilogue.residual + (row_offset + mi + r) * dst_stride + nb;
                    h0 = vaddq_f16(h0, vreinterpretq_f16_u16(vld1q_u16(res_ptr + 0U)));
                    h1 = vaddq_f16(h1, vreinterpretq_f16_u16(vld1q_u16(res_ptr + 8U)));
                }
                vst1q_u16(out_row + 0U, vreinterpretq_u16_f16(h0));
                vst1q_u16(out_row + 8U, vreinterpretq_u16_f16(h1));
            }
        }
    }

    if (is_reglu) {
        const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(epilogue.clamp_min));
        const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(epilogue.clamp_max));
        for (std::size_t mi = 0; mi < m_rows; ++mi) {
            const Float16Storage* gptr = reglu_buf + mi * n;
            const Float16Storage* lptr = gptr + gate_c;
            Float16Storage* out_row = dst_block + mi * dst_stride;
            for (std::size_t c = 0; c < gate_c; c += 8U) {
                const float16x8_t gv = vmaxq_f16(vminq_f16(
                    vreinterpretq_f16_u16(vld1q_u16(gptr + c)), vmax), vmin);
                const float16x8_t lv = vreinterpretq_f16_u16(vld1q_u16(lptr + c));
                vst1q_u16(out_row + c, vreinterpretq_u16_f16(vmulq_f16(gv, lv)));
            }
        }
    }
}

}  // namespace

const KernelSet& get_arm64_fp16fml_kernels() noexcept
{
    static const KernelSet fml_set = []() {
        KernelSet k = get_arm64_fp16_kernels();
        k.mode = GemmMode::kArm64Fp16Fml;
        k.gemm.run_block = &fp16fml_run_block;
        return k;
    }();
    return fml_set;
}

}  // namespace mlvc::cpu_backend
