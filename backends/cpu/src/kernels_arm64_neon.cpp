#include "kernels_common.hpp"

#include "mlvc/core/half.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace mlvc::cpu_backend {

static_assert(offsetof(Arm64AsmGemmArgs, m_rows) == 0U, "ASM offset mismatch: m_rows");
static_assert(offsetof(Arm64AsmGemmArgs, n_cols) == 8U, "ASM offset mismatch: n_cols");
static_assert(offsetof(Arm64AsmGemmArgs, k_dim) == 16U, "ASM offset mismatch: k_dim");
static_assert(offsetof(Arm64AsmGemmArgs, lhs) == 24U, "ASM offset mismatch: lhs");
static_assert(offsetof(Arm64AsmGemmArgs, lhs_stride) == 32U, "ASM offset mismatch: lhs_stride");
static_assert(offsetof(Arm64AsmGemmArgs, rhs) == 40U, "ASM offset mismatch: rhs");
static_assert(offsetof(Arm64AsmGemmArgs, dst) == 48U, "ASM offset mismatch: dst");
static_assert(offsetof(Arm64AsmGemmArgs, dst_stride) == 56U, "ASM offset mismatch: dst_stride");
static_assert(offsetof(Arm64AsmGemmArgs, res) == 64U, "ASM offset mismatch: res");
static_assert(offsetof(Arm64AsmGemmArgs, epi_kind) == 72U, "ASM offset mismatch: epi_kind");
static_assert(offsetof(Arm64AsmGemmArgs, alpha) == 80U, "ASM offset mismatch: alpha");
static_assert(offsetof(Arm64AsmGemmArgs, clamp_min) == 84U, "ASM offset mismatch: clamp_min");
static_assert(offsetof(Arm64AsmGemmArgs, clamp_max) == 88U, "ASM offset mismatch: clamp_max");

namespace {

inline float32x4x2_t neon_f16x8_to_f32x8(uint16x8_t bits) noexcept
{
    const float16x8_t h = vreinterpretq_f16_u16(bits);
    float32x4x2_t out;
    out.val[0] = vcvt_f32_f16(vget_low_f16(h));
    out.val[1] = vcvt_high_f32_f16(h);
    return out;
}

inline uint16x8_t neon_f32x8_to_u16x8(float32x4_t lo, float32x4_t hi) noexcept
{
    return vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
}

void arm64_neon_reglu_post(
    const Float16Storage* reglu_buf, Float16Storage* dst_block,
    std::size_t m_rows, std::size_t n, std::size_t dst_stride,
    float clamp_min, float clamp_max) noexcept
{
    const std::size_t gate_c = n / 2U;
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    for (std::size_t mi = 0; mi < m_rows; ++mi) {
        const Float16Storage* gptr = reglu_buf + mi * n;
        const Float16Storage* lptr = gptr + gate_c;
        Float16Storage* out_row = dst_block + mi * dst_stride;
        std::size_t c = 0;
        for (; c + 4U <= gate_c; c += 4U) {
            const float32x4_t gv = vmaxq_f32(vminq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(gptr + c))), vmax), vmin);
            const float32x4_t cg = vcvt_f32_f16(vcvt_f16_f32(gv));
            const float32x4_t lv = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(lptr + c)));
            vst1_u16(out_row + c, vreinterpret_u16_f16(vcvt_f16_f32(vmulq_f32(cg, lv))));
        }
        for (; c < gate_c; ++c) {
            const Float16Storage cg = float_to_half(std::min(
                std::max(half_to_float(gptr[c]), clamp_min), clamp_max));
            out_row[c] = float_to_half(half_to_float(cg) * half_to_float(lptr[c]));
        }
    }
}

std::size_t map_asm_epilogue_kind(const GemmEpilogue& epilogue) noexcept
{
    switch (epilogue.kind) {
    case GemmEpilogueKind::kClamp:
        return 1U;
    case GemmEpilogueKind::kLeakyRelu:
        return 2U;
    case GemmEpilogueKind::kResidualAdd:
        return epilogue.residual ? 3U : 0U;
    case GemmEpilogueKind::kNone:
    case GemmEpilogueKind::kReglu:
    default:
        return 0U;
    }
}

void arm64_neon_fp32_run_block(
    std::size_t m_rows, std::size_t n, std::size_t k,
    const Float16Storage* lhs_block, std::size_t lhs_stride,
    const Float16Storage* rhs_packed,
    Float16Storage* dst_block, std::size_t dst_stride,
    const GemmEpilogue& epilogue, std::size_t row_offset,
    Float16Storage* scratch, std::size_t scratch_elems)
{
    arm64_run_asm_gemm_block(
        &mlvc_arm64_gemm_neon_fp32_4x16, false,
        m_rows, n, k, lhs_block, lhs_stride, rhs_packed,
        dst_block, dst_stride, epilogue, row_offset, scratch, scratch_elems);
}

void arm64_neon_depthwise3x3_s1p1(
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
                const Float16Storage* neighbors[9] = {};
                const Float16Storage* wrows[9] = {};
                int kcount = 0;
                for (int ky = 0; ky < 3; ++ky) {
                    const int iy = oy + ky - 1;
                    if (iy < 0 || iy >= h)
                        continue;
                    const Float16Storage* in_r =
                        in_batch + static_cast<std::size_t>(iy) * w * channels;
                    for (int kx = 0; kx < 3; ++kx) {
                        const int ix = ox + kx - 1;
                        if (ix < 0 || ix >= w)
                            continue;
                        neighbors[kcount] = in_r + static_cast<std::size_t>(ix) * channels;
                        wrows[kcount] = weights + static_cast<std::size_t>(ky * 3 + kx) * pad_c;
                        ++kcount;
                    }
                }

                int c = 0;
                for (; c + 8 <= channels; c += 8) {
                    const float32x4x2_t b = neon_f16x8_to_f32x8(vld1q_u16(bias + c));
                    float32x4_t acc0 = b.val[0];
                    float32x4_t acc1 = b.val[1];
                    for (int idx = 0; idx < kcount; ++idx) {
                        const float32x4x2_t iv = neon_f16x8_to_f32x8(vld1q_u16(neighbors[idx] + c));
                        const float32x4x2_t wv = neon_f16x8_to_f32x8(vld1q_u16(wrows[idx] + c));
                        acc0 = vfmaq_f32(acc0, iv.val[0], wv.val[0]);
                        acc1 = vfmaq_f32(acc1, iv.val[1], wv.val[1]);
                    }
                    vst1q_u16(out_px + c, neon_f32x8_to_u16x8(acc0, acc1));
                }
                for (; c < channels; ++c) {
                    float acc = half_to_float(bias[c]);
                    for (int idx = 0; idx < kcount; ++idx)
                        acc += half_to_float(neighbors[idx][c]) * half_to_float(wrows[idx][c]);
                    out_px[c] = float_to_half(acc);
                }
            }
        }
    });
}

inline void transpose8x8_u16(
    const Float16Storage* src, std::size_t src_stride,
    Float16Storage* dst, std::size_t dst_stride) noexcept
{
    uint16x8_t r0 = vld1q_u16(src + 0U * src_stride);
    uint16x8_t r1 = vld1q_u16(src + 1U * src_stride);
    uint16x8_t r2 = vld1q_u16(src + 2U * src_stride);
    uint16x8_t r3 = vld1q_u16(src + 3U * src_stride);
    uint16x8_t r4 = vld1q_u16(src + 4U * src_stride);
    uint16x8_t r5 = vld1q_u16(src + 5U * src_stride);
    uint16x8_t r6 = vld1q_u16(src + 6U * src_stride);
    uint16x8_t r7 = vld1q_u16(src + 7U * src_stride);

    const uint16x8_t a0 = vtrn1q_u16(r0, r1);
    const uint16x8_t a1 = vtrn2q_u16(r0, r1);
    const uint16x8_t a2 = vtrn1q_u16(r2, r3);
    const uint16x8_t a3 = vtrn2q_u16(r2, r3);
    const uint16x8_t a4 = vtrn1q_u16(r4, r5);
    const uint16x8_t a5 = vtrn2q_u16(r4, r5);
    const uint16x8_t a6 = vtrn1q_u16(r6, r7);
    const uint16x8_t a7 = vtrn2q_u16(r6, r7);

    const uint32x4_t b0 = vtrn1q_u32(vreinterpretq_u32_u16(a0), vreinterpretq_u32_u16(a2));
    const uint32x4_t b1 = vtrn1q_u32(vreinterpretq_u32_u16(a1), vreinterpretq_u32_u16(a3));
    const uint32x4_t b2 = vtrn2q_u32(vreinterpretq_u32_u16(a0), vreinterpretq_u32_u16(a2));
    const uint32x4_t b3 = vtrn2q_u32(vreinterpretq_u32_u16(a1), vreinterpretq_u32_u16(a3));
    const uint32x4_t b4 = vtrn1q_u32(vreinterpretq_u32_u16(a4), vreinterpretq_u32_u16(a6));
    const uint32x4_t b5 = vtrn1q_u32(vreinterpretq_u32_u16(a5), vreinterpretq_u32_u16(a7));
    const uint32x4_t b6 = vtrn2q_u32(vreinterpretq_u32_u16(a4), vreinterpretq_u32_u16(a6));
    const uint32x4_t b7 = vtrn2q_u32(vreinterpretq_u32_u16(a5), vreinterpretq_u32_u16(a7));

    vst1q_u16(dst + 0U * dst_stride, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(b0), vreinterpretq_u64_u32(b4))));
    vst1q_u16(dst + 1U * dst_stride, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(b1), vreinterpretq_u64_u32(b5))));
    vst1q_u16(dst + 2U * dst_stride, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(b2), vreinterpretq_u64_u32(b6))));
    vst1q_u16(dst + 3U * dst_stride, vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(b3), vreinterpretq_u64_u32(b7))));
    vst1q_u16(dst + 4U * dst_stride, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(b0), vreinterpretq_u64_u32(b4))));
    vst1q_u16(dst + 5U * dst_stride, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(b1), vreinterpretq_u64_u32(b5))));
    vst1q_u16(dst + 6U * dst_stride, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(b2), vreinterpretq_u64_u32(b6))));
    vst1q_u16(dst + 7U * dst_stride, vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(b3), vreinterpretq_u64_u32(b7))));
}

void arm64_transpose_2d_u16(
    const Float16Storage* src, Float16Storage* dst,
    std::size_t batches, std::size_t rows, std::size_t cols,
    CpuThreadPool& pool)
{
    if (rows == 0U || cols == 0U || batches == 0U)
        return;
    if (rows == 1U || cols == 1U) {
        std::memcpy(dst, src, batches * rows * cols * sizeof(Float16Storage));
        return;
    }

    // Fast path for planar 3-channel images (NCHW [3, HW] -> NHWC [HW, 3]).
    if (rows == 3U) {
        for (std::size_t b = 0; b < batches; ++b) {
            const Float16Storage* p0 = src + b * 3U * cols;
            const Float16Storage* p1 = p0 + cols;
            const Float16Storage* p2 = p1 + cols;
            Float16Storage* db = dst + b * cols * 3U;
            pool.parallel_for(cols, 1024U, [&](std::size_t s0, std::size_t s1) {
                std::size_t s = s0;
                for (; s + 8U <= s1; s += 8U) {
                    uint16x8x3_t rgb;
                    rgb.val[0] = vld1q_u16(p0 + s);
                    rgb.val[1] = vld1q_u16(p1 + s);
                    rgb.val[2] = vld1q_u16(p2 + s);
                    vst3q_u16(db + s * 3U, rgb);
                }
                for (; s < s1; ++s) {
                    Float16Storage* px = db + s * 3U;
                    px[0] = p0[s];
                    px[1] = p1[s];
                    px[2] = p2[s];
                }
            });
        }
        return;
    }

    // Fast path for packed 3-channel images (NHWC [HW, 3] -> NCHW [3, HW]).
    if (cols == 3U) {
        for (std::size_t b = 0; b < batches; ++b) {
            const Float16Storage* sb = src + b * rows * 3U;
            Float16Storage* p0 = dst + b * 3U * rows;
            Float16Storage* p1 = p0 + rows;
            Float16Storage* p2 = p1 + rows;
            pool.parallel_for(rows, 1024U, [&](std::size_t s0, std::size_t s1) {
                std::size_t s = s0;
                for (; s + 8U <= s1; s += 8U) {
                    const uint16x8x3_t rgb = vld3q_u16(sb + s * 3U);
                    vst1q_u16(p0 + s, rgb.val[0]);
                    vst1q_u16(p1 + s, rgb.val[1]);
                    vst1q_u16(p2 + s, rgb.val[2]);
                }
                for (; s < s1; ++s) {
                    const Float16Storage* px = sb + s * 3U;
                    p0[s] = px[0];
                    p1[s] = px[1];
                    p2[s] = px[2];
                }
            });
        }
        return;
    }

    const std::size_t row_tiles = (rows + 7U) / 8U;
    for (std::size_t b = 0; b < batches; ++b) {
        const Float16Storage* sb = src + b * rows * cols;
        Float16Storage* db = dst + b * rows * cols;
        if (rows < 8U && cols >= 64U) {
            pool.parallel_for(cols, 1024U, [&](std::size_t c0, std::size_t c1) {
                for (std::size_t r = 0; r < rows; ++r) {
                    const Float16Storage* sr = sb + r * cols;
                    for (std::size_t c = c0; c < c1; ++c)
                        db[c * rows + r] = sr[c];
                }
            });
            continue;
        }
        pool.parallel_for(row_tiles, 32U, [&](std::size_t rt0, std::size_t rt1) {
            for (std::size_t rt = rt0; rt < rt1; ++rt) {
                const std::size_t r0 = rt * 8U;
                const std::size_t rem_r = std::min<std::size_t>(8U, rows - r0);
                std::size_t c = 0;
                if (rem_r == 8U) {
                    for (; c + 8U <= cols; c += 8U)
                        transpose8x8_u16(sb + r0 * cols + c, cols, db + c * rows + r0, rows);
                }
                for (std::size_t rr = r0; rr < r0 + rem_r; ++rr) {
                    const Float16Storage* sr = sb + rr * cols;
                    for (std::size_t cc = c; cc < cols; ++cc)
                        db[cc * rows + rr] = sr[cc];
                }
            }
        });
    }
}

void arm64_nchw_to_nhwc(const Float16Storage* nchw, Float16Storage* nhwc,
                        std::size_t n, std::size_t c, std::size_t hw,
                        CpuThreadPool& pool)
{
    arm64_transpose_2d_u16(nchw, nhwc, n, c, hw, pool);
}

void arm64_nhwc_to_nchw(const Float16Storage* nhwc, Float16Storage* nchw,
                        std::size_t n, std::size_t hw, std::size_t c,
                        CpuThreadPool& pool)
{
    arm64_transpose_2d_u16(nhwc, nchw, n, hw, c, pool);
}

}  // namespace

void arm64_run_asm_gemm_block(
    Arm64AsmGemmFn gemm_fn, bool use_fp16_reglu,
    std::size_t m_rows, std::size_t n, std::size_t k,
    const Float16Storage* lhs_block, std::size_t lhs_stride,
    const Float16Storage* rhs_packed,
    Float16Storage* dst_block, std::size_t dst_stride,
    const GemmEpilogue& epilogue, std::size_t row_offset,
    Float16Storage* scratch, std::size_t scratch_elems)
{
    const bool is_reglu = (epilogue.kind == GemmEpilogueKind::kReglu);
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

    Arm64AsmGemmArgs args{};
    args.m_rows = m_rows;
    args.n_cols = n;
    args.k_dim = k;
    args.lhs = lhs_block;
    args.lhs_stride = lhs_stride * sizeof(Float16Storage);
    args.rhs = rhs_packed;
    args.dst = is_reglu ? reglu_buf : dst_block;
    args.dst_stride = (is_reglu ? n : dst_stride) * sizeof(Float16Storage);
    args.epi_kind = map_asm_epilogue_kind(epilogue);
    args.res = (args.epi_kind == 3U)
        ? (epilogue.residual + row_offset * dst_stride)
        : nullptr;
    args.alpha = epilogue.alpha;
    args.clamp_min = epilogue.clamp_min;
    args.clamp_max = epilogue.clamp_max;

    gemm_fn(&args);

    if (is_reglu) {
        if (use_fp16_reglu) {
            arm64_fp16_reglu_post(
                reglu_buf, dst_block, m_rows, n, dst_stride,
                epilogue.clamp_min, epilogue.clamp_max);
        } else {
            arm64_neon_reglu_post(
                reglu_buf, dst_block, m_rows, n, dst_stride,
                epilogue.clamp_min, epilogue.clamp_max);
        }
    }
}

const KernelSet& get_arm64_neon_fp32_kernels(bool has_fp16) noexcept
{
    auto build_set = [](bool enable_fp16_ops) {
        KernelSet k = get_scalar_kernels();
        k.mode = GemmMode::kArm64NeonFp32;
        k.gemm.run_block = &arm64_neon_fp32_run_block;
        k.depthwise3x3_s1p1 = &arm64_neon_depthwise3x3_s1p1;
        k.nchw_to_nhwc = &arm64_nchw_to_nhwc;
        k.nhwc_to_nchw = &arm64_nhwc_to_nchw;
        if (enable_fp16_ops)
            apply_arm64_fp16_non_gemm_ops(k);
        return k;
    };

    if (has_fp16) {
        static const KernelSet fp16_ops_set = build_set(true);
        return fp16_ops_set;
    }
    static const KernelSet baseline_set = build_set(false);
    return baseline_set;
}

}  // namespace mlvc::cpu_backend
