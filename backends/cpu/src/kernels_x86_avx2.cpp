#include "kernels_common.hpp"

#include "mlvc/core/half.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mlvc::cpu_backend {
namespace {

constexpr int kRoundPsFlags = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
constexpr int kCvtPhFlags = _MM_FROUND_TO_NEAREST_INT;

inline __m256 load_f16x8_as_f32(const Float16Storage* ptr) noexcept
{
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
}

inline void store_f32x8_as_f16(Float16Storage* ptr, __m256 v) noexcept
{
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), _mm256_cvtps_ph(v, kCvtPhFlags));
}

inline __m256 fast_exp_ps_avx2(__m256 x) noexcept
{
    const __m256 clamp_lo = _mm256_set1_ps(-87.33654475F);
    const __m256 clamp_hi = _mm256_set1_ps(88.72283905F);
    x = _mm256_min_ps(_mm256_max_ps(x, clamp_lo), clamp_hi);

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634F);
    const __m256 ln2_hi = _mm256_set1_ps(0.693145751953125F);
    const __m256 ln2_lo = _mm256_set1_ps(1.428606765330187E-6F);

    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), kRoundPsFlags);
    const __m256 f0 = _mm256_fnmadd_ps(fx, ln2_hi, x);
    const __m256 r = _mm256_fnmadd_ps(fx, ln2_lo, f0);

    __m256 poly = _mm256_set1_ps(0.0013888889F);
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(0.0083333338F));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(0.0416666679F));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(0.1666666716F));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(0.5F));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(1.0F));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(1.0F));

    const __m256i exp_bits = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(poly, _mm256_castsi256_ps(exp_bits));
}

inline __m256 fast_sigmoid_ps_avx2(__m256 x) noexcept
{
    const __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    const __m256 one = _mm256_set1_ps(1.0F);
    return _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_ps_avx2(neg_x)));
}

inline __m256 round_to_f16_precision(__m256 v) noexcept
{
    return _mm256_cvtph_ps(_mm256_cvtps_ph(v, kCvtPhFlags));
}

void avx2_run_block(
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

    const __m256 vmin = _mm256_set1_ps(epilogue.clamp_min);
    const __m256 vmax = _mm256_set1_ps(epilogue.clamp_max);
    const __m256 valpha = _mm256_set1_ps(epilogue.alpha);
    const __m256 vzero = _mm256_setzero_ps();

    for (std::size_t mi = 0; mi < m_rows; mi += 6U) {
        const std::size_t mr = std::min<std::size_t>(6U, m_rows - mi);
        const Float16Storage* a0 = lhs_block + (mi + 0U) * lhs_stride;
        const Float16Storage* a1 = mr > 1U ? lhs_block + (mi + 1U) * lhs_stride : a0;
        const Float16Storage* a2 = mr > 2U ? lhs_block + (mi + 2U) * lhs_stride : a0;
        const Float16Storage* a3 = mr > 3U ? lhs_block + (mi + 3U) * lhs_stride : a0;
        const Float16Storage* a4 = mr > 4U ? lhs_block + (mi + 4U) * lhs_stride : a0;
        const Float16Storage* a5 = mr > 5U ? lhs_block + (mi + 5U) * lhs_stride : a0;

        for (std::size_t nb = 0; nb < n; nb += 16U) {
            const Float16Storage* wblk = rhs_packed + (nb / 16U) * block_elems;
            const __m256 b0 = load_f16x8_as_f32(wblk + 0U);
            const __m256 b1 = load_f16x8_as_f32(wblk + 8U);

            __m256 c00 = b0, c01 = b1;
            __m256 c10 = b0, c11 = b1;
            __m256 c20 = b0, c21 = b1;
            __m256 c30 = b0, c31 = b1;
            __m256 c40 = b0, c41 = b1;
            __m256 c50 = b0, c51 = b1;

            const Float16Storage* wptr = wblk + 16U;
            if (mr == 6U) {
                for (std::size_t kk = 0; kk < k; ++kk, wptr += 16U) {
                    const __m256 w0 = load_f16x8_as_f32(wptr + 0U);
                    const __m256 w1 = load_f16x8_as_f32(wptr + 8U);

                    const __m256 s0 = _mm256_set1_ps(half_to_float(a0[kk]));
                    c00 = _mm256_fmadd_ps(w0, s0, c00);
                    c01 = _mm256_fmadd_ps(w1, s0, c01);

                    const __m256 s1 = _mm256_set1_ps(half_to_float(a1[kk]));
                    c10 = _mm256_fmadd_ps(w0, s1, c10);
                    c11 = _mm256_fmadd_ps(w1, s1, c11);

                    const __m256 s2 = _mm256_set1_ps(half_to_float(a2[kk]));
                    c20 = _mm256_fmadd_ps(w0, s2, c20);
                    c21 = _mm256_fmadd_ps(w1, s2, c21);

                    const __m256 s3 = _mm256_set1_ps(half_to_float(a3[kk]));
                    c30 = _mm256_fmadd_ps(w0, s3, c30);
                    c31 = _mm256_fmadd_ps(w1, s3, c31);

                    const __m256 s4 = _mm256_set1_ps(half_to_float(a4[kk]));
                    c40 = _mm256_fmadd_ps(w0, s4, c40);
                    c41 = _mm256_fmadd_ps(w1, s4, c41);

                    const __m256 s5 = _mm256_set1_ps(half_to_float(a5[kk]));
                    c50 = _mm256_fmadd_ps(w0, s5, c50);
                    c51 = _mm256_fmadd_ps(w1, s5, c51);
                }
            } else {
                for (std::size_t kk = 0; kk < k; ++kk, wptr += 16U) {
                    const __m256 w0 = load_f16x8_as_f32(wptr + 0U);
                    const __m256 w1 = load_f16x8_as_f32(wptr + 8U);

                    const __m256 s0 = _mm256_set1_ps(half_to_float(a0[kk]));
                    c00 = _mm256_fmadd_ps(w0, s0, c00);
                    c01 = _mm256_fmadd_ps(w1, s0, c01);
                    if (mr > 1U) {
                        const __m256 s1 = _mm256_set1_ps(half_to_float(a1[kk]));
                        c10 = _mm256_fmadd_ps(w0, s1, c10);
                        c11 = _mm256_fmadd_ps(w1, s1, c11);
                    }
                    if (mr > 2U) {
                        const __m256 s2 = _mm256_set1_ps(half_to_float(a2[kk]));
                        c20 = _mm256_fmadd_ps(w0, s2, c20);
                        c21 = _mm256_fmadd_ps(w1, s2, c21);
                    }
                    if (mr > 3U) {
                        const __m256 s3 = _mm256_set1_ps(half_to_float(a3[kk]));
                        c30 = _mm256_fmadd_ps(w0, s3, c30);
                        c31 = _mm256_fmadd_ps(w1, s3, c31);
                    }
                    if (mr > 4U) {
                        const __m256 s4 = _mm256_set1_ps(half_to_float(a4[kk]));
                        c40 = _mm256_fmadd_ps(w0, s4, c40);
                        c41 = _mm256_fmadd_ps(w1, s4, c41);
                    }
                }
            }

            const __m256 rows[6][2] = {
                {c00, c01}, {c10, c11}, {c20, c21},
                {c30, c31}, {c40, c41}, {c50, c51},
            };
            for (std::size_t r = 0; r < mr; ++r) {
                Float16Storage* out_row = is_reglu
                    ? (reglu_buf + (mi + r) * n + nb)
                    : (dst_block + (mi + r) * dst_stride + nb);
                __m256 h0 = round_to_f16_precision(rows[r][0]);
                __m256 h1 = round_to_f16_precision(rows[r][1]);
                if (epilogue.kind == GemmEpilogueKind::kClamp) {
                    h0 = _mm256_max_ps(_mm256_min_ps(h0, vmax), vmin);
                    h1 = _mm256_max_ps(_mm256_min_ps(h1, vmax), vmin);
                } else if (epilogue.kind == GemmEpilogueKind::kLeakyRelu) {
                    const __m256 m0 = _mm256_cmp_ps(h0, vzero, _CMP_LT_OQ);
                    const __m256 m1 = _mm256_cmp_ps(h1, vzero, _CMP_LT_OQ);
                    h0 = _mm256_blendv_ps(h0, _mm256_mul_ps(h0, valpha), m0);
                    h1 = _mm256_blendv_ps(h1, _mm256_mul_ps(h1, valpha), m1);
                } else if (epilogue.kind == GemmEpilogueKind::kResidualAdd && epilogue.residual) {
                    const Float16Storage* res_ptr =
                        epilogue.residual + (row_offset + mi + r) * dst_stride + nb;
                    h0 = _mm256_add_ps(h0, load_f16x8_as_f32(res_ptr + 0U));
                    h1 = _mm256_add_ps(h1, load_f16x8_as_f32(res_ptr + 8U));
                }
                store_f32x8_as_f16(out_row + 0U, h0);
                store_f32x8_as_f16(out_row + 8U, h1);
            }
        }
    }

    if (is_reglu) {
        for (std::size_t mi = 0; mi < m_rows; ++mi) {
            const Float16Storage* gptr = reglu_buf + mi * n;
            const Float16Storage* lptr = gptr + gate_c;
            Float16Storage* out_row = dst_block + mi * dst_stride;
            std::size_t c = 0;
            for (; c + 8U <= gate_c; c += 8U) {
                const __m256 gv = _mm256_max_ps(_mm256_min_ps(load_f16x8_as_f32(gptr + c), vmax), vmin);
                const __m256 cg = round_to_f16_precision(gv);
                const __m256 lv = load_f16x8_as_f32(lptr + c);
                store_f32x8_as_f16(out_row + c, _mm256_mul_ps(cg, lv));
            }
            for (; c < gate_c; ++c) {
                const float cg = half_to_float(float_to_half(
                    std::min(std::max(half_to_float(gptr[c]), epilogue.clamp_min), epilogue.clamp_max)));
                out_row[c] = float_to_half(cg * half_to_float(lptr[c]));
            }
        }
    }
}

void avx2_depthwise3x3_s1p1(
    const Float16Storage* input, const Float16Storage* packed_weight,
    Float16Storage* output, int n, int h, int w, int channels,
    CpuThreadPool& pool)
{
    if (channels % 8 != 0) {
        get_scalar_kernels().depthwise3x3_s1p1(
            input, packed_weight, output, n, h, w, channels, pool);
        return;
    }

    const std::size_t c_sz = static_cast<std::size_t>(channels);
    const std::size_t pad_c = ((c_sz + 15U) / 16U) * 16U;
    const Float16Storage* bias = packed_weight;
    const Float16Storage* wbase = packed_weight + pad_c;
    const std::size_t row_stride = static_cast<std::size_t>(w) * c_sz;

    for (int batch = 0; batch < n; ++batch) {
        const Float16Storage* in_b =
            input + static_cast<std::size_t>(batch) * static_cast<std::size_t>(h) * row_stride;
        Float16Storage* out_b =
            output + static_cast<std::size_t>(batch) * static_cast<std::size_t>(h) * row_stride;

        pool.parallel_for(static_cast<std::size_t>(h), 4U, [&](std::size_t y0, std::size_t y1) {
            for (std::size_t y_idx = y0; y_idx < y1; ++y_idx) {
                const int y = static_cast<int>(y_idx);
                const bool has_top = (y > 0);
                const bool has_bot = (y + 1 < h);
                const Float16Storage* r_top = has_top ? in_b + static_cast<std::size_t>(y - 1) * row_stride : nullptr;
                const Float16Storage* r_mid = in_b + static_cast<std::size_t>(y) * row_stride;
                const Float16Storage* r_bot = has_bot ? in_b + static_cast<std::size_t>(y + 1) * row_stride : nullptr;
                Float16Storage* out_row = out_b + static_cast<std::size_t>(y) * row_stride;

                for (int x = 0; x < w; ++x) {
                    const std::size_t xoff = static_cast<std::size_t>(x) * c_sz;
                    const bool has_left = (x > 0);
                    const bool has_right = (x + 1 < w);

                    for (std::size_t c = 0; c < c_sz; c += 8U) {
                        __m256 acc = load_f16x8_as_f32(bias + c);
                        auto tap = [&](const Float16Storage* row, std::ptrdiff_t dx, std::size_t k_idx) {
                            const __m256 wt = load_f16x8_as_f32(wbase + k_idx * pad_c + c);
                            const __m256 iv = load_f16x8_as_f32(
                                row + static_cast<std::ptrdiff_t>(xoff) + dx * static_cast<std::ptrdiff_t>(c_sz) + c);
                            acc = _mm256_fmadd_ps(iv, wt, acc);
                        };
                        if (has_top) {
                            if (has_left) tap(r_top, -1, 0U);
                            tap(r_top, 0, 1U);
                            if (has_right) tap(r_top, 1, 2U);
                        }
                        if (has_left) tap(r_mid, -1, 3U);
                        tap(r_mid, 0, 4U);
                        if (has_right) tap(r_mid, 1, 5U);
                        if (has_bot) {
                            if (has_left) tap(r_bot, -1, 6U);
                            tap(r_bot, 0, 7U);
                            if (has_right) tap(r_bot, 1, 8U);
                        }
                        store_f32x8_as_f16(out_row + xoff + c, acc);
                    }
                }
            }
        });
    }
}

void avx2_unary(
    const Float16Storage* input, Float16Storage* output,
    std::size_t count, UnaryKind op, float alpha,
    float clamp_min, float clamp_max, CpuThreadPool& pool)
{
    pool.parallel_for(count, 4096U, [&](std::size_t begin, std::size_t end) {
        std::size_t i = begin;
        if (op == UnaryKind::kLeakyRelu) {
            const __m256 vzero = _mm256_setzero_ps();
            const __m256 valpha = _mm256_set1_ps(alpha);
            for (; i + 8U <= end; i += 8U) {
                const __m256 v = load_f16x8_as_f32(input + i);
                const __m256 m = _mm256_cmp_ps(v, vzero, _CMP_LT_OQ);
                store_f32x8_as_f16(output + i, _mm256_blendv_ps(v, _mm256_mul_ps(v, valpha), m));
            }
            for (; i < end; ++i) {
                const float x = half_to_float(input[i]);
                output[i] = float_to_half(x < 0.0F ? x * alpha : x);
            }
        } else if (op == UnaryKind::kClip) {
            const __m256 vmin = _mm256_set1_ps(clamp_min);
            const __m256 vmax = _mm256_set1_ps(clamp_max);
            for (; i + 8U <= end; i += 8U) {
                const __m256 v = load_f16x8_as_f32(input + i);
                store_f32x8_as_f16(output + i, _mm256_max_ps(_mm256_min_ps(v, vmax), vmin));
            }
            for (; i < end; ++i) {
                const float x = half_to_float(input[i]);
                output[i] = float_to_half(std::min(std::max(x, clamp_min), clamp_max));
            }
        } else if (op == UnaryKind::kSigmoid) {
            for (; i + 8U <= end; i += 8U)
                store_f32x8_as_f16(output + i, fast_sigmoid_ps_avx2(load_f16x8_as_f32(input + i)));
            for (; i < end; ++i) {
                const float x = half_to_float(input[i]);
                output[i] = float_to_half(1.0F / (1.0F + std::exp(-x)));
            }
        } else if (op == UnaryKind::kReciprocal) {
            const __m256 vone = _mm256_set1_ps(1.0F);
            for (; i + 8U <= end; i += 8U)
                store_f32x8_as_f16(output + i, _mm256_div_ps(vone, load_f16x8_as_f32(input + i)));
            for (; i < end; ++i)
                output[i] = float_to_half(1.0F / half_to_float(input[i]));
        } else {
            for (; i + 8U <= end; i += 8U)
                store_f32x8_as_f16(output + i, _mm256_round_ps(load_f16x8_as_f32(input + i), kRoundPsFlags));
            for (; i < end; ++i)
                output[i] = float_to_half(std::nearbyintf(half_to_float(input[i])));
        }
    });
}

inline __m256 eval_binary_ps(__m256 a, __m256 b, BinaryKind op) noexcept
{
    switch (op) {
    case BinaryKind::kAdd:
        return _mm256_add_ps(a, b);
    case BinaryKind::kMul:
        return _mm256_mul_ps(a, b);
    case BinaryKind::kSub:
        return _mm256_sub_ps(a, b);
    }
    return a;
}

inline float eval_binary_scalar(float a, float b, BinaryKind op) noexcept
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

void avx2_binary_contiguous(
    const Float16Storage* lhs, const Float16Storage* rhs, Float16Storage* output,
    std::size_t count, BinaryKind op, CpuThreadPool& pool)
{
    pool.parallel_for(count, 4096U, [&](std::size_t begin, std::size_t end) {
        std::size_t i = begin;
        for (; i + 8U <= end; i += 8U) {
            const __m256 a = load_f16x8_as_f32(lhs + i);
            const __m256 b = load_f16x8_as_f32(rhs + i);
            store_f32x8_as_f16(output + i, eval_binary_ps(a, b, op));
        }
        for (; i < end; ++i) {
            output[i] = float_to_half(
                eval_binary_scalar(half_to_float(lhs[i]), half_to_float(rhs[i]), op));
        }
    });
}

void avx2_binary_channel_bcast(
    const Float16Storage* vec, const Float16Storage* chan, Float16Storage* output,
    std::size_t spatial, std::size_t channels, BinaryKind op,
    bool chan_is_lhs, CpuThreadPool& pool)
{
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* vpx = vec + p * channels;
            Float16Storage* opx = output + p * channels;
            std::size_t c = 0;
            for (; c + 8U <= channels; c += 8U) {
                const __m256 vv = load_f16x8_as_f32(vpx + c);
                const __m256 cv = load_f16x8_as_f32(chan + c);
                const __m256 res = chan_is_lhs ? eval_binary_ps(cv, vv, op) : eval_binary_ps(vv, cv, op);
                store_f32x8_as_f16(opx + c, res);
            }
            for (; c < channels; ++c) {
                const float vf = half_to_float(vpx[c]);
                const float cf = half_to_float(chan[c]);
                opx[c] = float_to_half(eval_binary_scalar(chan_is_lhs ? cf : vf, chan_is_lhs ? vf : cf, op));
            }
        }
    });
}

void avx2_binary_scalar_bcast(
    const Float16Storage* vec, Float16Storage scalar, Float16Storage* output,
    std::size_t count, BinaryKind op, bool scalar_is_lhs, CpuThreadPool& pool)
{
    const float sf = half_to_float(scalar);
    const __m256 vs = _mm256_set1_ps(sf);
    pool.parallel_for(count, 4096U, [&](std::size_t begin, std::size_t end) {
        std::size_t i = begin;
        for (; i + 8U <= end; i += 8U) {
            const __m256 vv = load_f16x8_as_f32(vec + i);
            store_f32x8_as_f16(output + i,
                               scalar_is_lhs ? eval_binary_ps(vs, vv, op) : eval_binary_ps(vv, vs, op));
        }
        for (; i < end; ++i) {
            const float vf = half_to_float(vec[i]);
            output[i] = float_to_half(eval_binary_scalar(scalar_is_lhs ? sf : vf, scalar_is_lhs ? vf : sf, op));
        }
    });
}

void avx2_reglu_nhwc(
    const Float16Storage* input, Float16Storage* output,
    std::size_t spatial, std::size_t gate_channels,
    float clamp_min, float clamp_max, CpuThreadPool& pool)
{
    const std::size_t in_c = gate_channels * 2U;
    const __m256 vmin = _mm256_set1_ps(clamp_min);
    const __m256 vmax = _mm256_set1_ps(clamp_max);
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* in_px = input + p * in_c;
            Float16Storage* out_px = output + p * gate_channels;
            std::size_t c = 0;
            for (; c + 8U <= gate_channels; c += 8U) {
                const __m256 cg = round_to_f16_precision(
                    _mm256_max_ps(_mm256_min_ps(load_f16x8_as_f32(in_px + c), vmax), vmin));
                const __m256 lv = load_f16x8_as_f32(in_px + c + gate_channels);
                store_f32x8_as_f16(out_px + c, _mm256_mul_ps(cg, lv));
            }
            for (; c < gate_channels; ++c) {
                const float cg = half_to_float(float_to_half(
                    std::min(std::max(half_to_float(in_px[c]), clamp_min), clamp_max)));
                out_px[c] = float_to_half(cg * half_to_float(in_px[c + gate_channels]));
            }
        }
    });
}

void avx2_feature_update_nhwc(
    const Float16Storage* input, const Float16Storage* history,
    const Float16Storage* scale, Float16Storage* output,
    std::size_t spatial, std::size_t channels, CpuThreadPool& pool)
{
    const std::size_t in_stride = channels * 3U;
    const std::size_t out_stride = channels * 2U;
    const __m256 vone = _mm256_set1_ps(1.0F);
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* in_px = input + p * in_stride;
            const Float16Storage* hist_px = history + p * channels;
            Float16Storage* out_px = output + p * out_stride;
            std::size_t c = 0;
            for (; c + 8U <= channels; c += 8U) {
                const __m256 a = load_f16x8_as_f32(in_px + c);
                const __m256 b = load_f16x8_as_f32(in_px + c + channels);
                const __m256 cg = load_f16x8_as_f32(in_px + c + channels * 2U);
                const __m256 prev = load_f16x8_as_f32(hist_px + c);
                const __m256 sc = load_f16x8_as_f32(scale + c);

                const __m256 sig_b = round_to_f16_precision(fast_sigmoid_ps_avx2(b));
                const __m256 sig_c = round_to_f16_precision(fast_sigmoid_ps_avx2(cg));
                const __m256 w_b = round_to_f16_precision(_mm256_mul_ps(sig_b, prev));
                const __m256 inv_gate = round_to_f16_precision(_mm256_sub_ps(vone, sig_b));
                const __m256 w_a = round_to_f16_precision(_mm256_mul_ps(inv_gate, a));
                const __m256 blended = round_to_f16_precision(_mm256_add_ps(w_b, w_a));
                const __m256 gated = round_to_f16_precision(_mm256_mul_ps(sig_c, blended));
                const __m256 scaled = _mm256_mul_ps(gated, sc);

                store_f32x8_as_f16(out_px + c, scaled);
                store_f32x8_as_f16(out_px + c + channels, blended);
            }
            for (; c < channels; ++c) {
                const float a = half_to_float(in_px[c]);
                const float b = half_to_float(in_px[c + channels]);
                const float cg = half_to_float(in_px[c + channels * 2U]);
                const float prev = half_to_float(hist_px[c]);
                const float sig_b = half_to_float(float_to_half(1.0F / (1.0F + std::exp(-b))));
                const float sig_c = half_to_float(float_to_half(1.0F / (1.0F + std::exp(-cg))));
                const float w_b = half_to_float(float_to_half(sig_b * prev));
                const float inv_gate = half_to_float(float_to_half(1.0F - sig_b));
                const float w_a = half_to_float(float_to_half(inv_gate * a));
                const Float16Storage blended_h = float_to_half(w_b + w_a);
                const float blended = half_to_float(blended_h);
                const float gated = half_to_float(float_to_half(sig_c * blended));
                out_px[c] = float_to_half(gated * half_to_float(scale[c]));
                out_px[c + channels] = blended_h;
            }
        }
    });
}

void avx2_feature_update_outputs_nhwc(
    const Float16Storage* input, const Float16Storage* history,
    const Float16Storage* scale, Float16Storage* scaled_output,
    Float16Storage* blended_output, Float16Storage* feature_output,
    std::size_t spatial, std::size_t channels, CpuThreadPool& pool)
{
    const std::size_t in_stride = channels * 3U;
    const std::size_t feat_stride = channels * 2U;
    const __m256 vone = _mm256_set1_ps(1.0F);
    pool.parallel_for(spatial, 64U, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const Float16Storage* in_px = input + p * in_stride;
            const Float16Storage* hist_px = history + p * channels;
            Float16Storage* scaled_px = scaled_output + p * channels;
            Float16Storage* blended_px = blended_output + p * channels;
            Float16Storage* feat_px = feature_output + p * feat_stride;
            std::size_t c = 0;
            for (; c + 8U <= channels; c += 8U) {
                const __m256 a = load_f16x8_as_f32(in_px + c);
                const __m256 b = load_f16x8_as_f32(in_px + c + channels);
                const __m256 cg = load_f16x8_as_f32(in_px + c + channels * 2U);
                const __m256 prev = load_f16x8_as_f32(hist_px + c);
                const __m256 sc = load_f16x8_as_f32(scale + c);

                const __m256 sig_b = round_to_f16_precision(fast_sigmoid_ps_avx2(b));
                const __m256 sig_c = round_to_f16_precision(fast_sigmoid_ps_avx2(cg));
                const __m256 w_b = round_to_f16_precision(_mm256_mul_ps(sig_b, prev));
                const __m256 inv_gate = round_to_f16_precision(_mm256_sub_ps(vone, sig_b));
                const __m256 w_a = round_to_f16_precision(_mm256_mul_ps(inv_gate, a));
                const __m256 blended = round_to_f16_precision(_mm256_add_ps(w_b, w_a));
                const __m256 gated = round_to_f16_precision(_mm256_mul_ps(sig_c, blended));
                const __m256 scaled = round_to_f16_precision(_mm256_mul_ps(gated, sc));

                store_f32x8_as_f16(scaled_px + c, scaled);
                store_f32x8_as_f16(blended_px + c, blended);
                store_f32x8_as_f16(feat_px + c, scaled);
                store_f32x8_as_f16(feat_px + c + channels, blended);
            }
            for (; c < channels; ++c) {
                const float a = half_to_float(in_px[c]);
                const float b = half_to_float(in_px[c + channels]);
                const float cg = half_to_float(in_px[c + channels * 2U]);
                const float prev = half_to_float(hist_px[c]);
                const float sig_b = half_to_float(float_to_half(1.0F / (1.0F + std::exp(-b))));
                const float sig_c = half_to_float(float_to_half(1.0F / (1.0F + std::exp(-cg))));
                const float w_b = half_to_float(float_to_half(sig_b * prev));
                const float inv_gate = half_to_float(float_to_half(1.0F - sig_b));
                const float w_a = half_to_float(float_to_half(inv_gate * a));
                const Float16Storage blended_h = float_to_half(w_b + w_a);
                const float blended = half_to_float(blended_h);
                const float gated = half_to_float(float_to_half(sig_c * blended));
                const Float16Storage scaled_h = float_to_half(gated * half_to_float(scale[c]));
                scaled_px[c] = scaled_h;
                blended_px[c] = blended_h;
                feat_px[c] = scaled_h;
                feat_px[c + channels] = blended_h;
            }
        }
    });
}

void avx2_y0_tail_nhwc(
    const Float16Storage* latent, const Float16Storage* prior,
    const Float16Storage* quant_scale, const Float16Storage* update_scale0,
    const Float16Storage* update_scale1, Float16Storage* normalized,
    Float16Storage* clipped, Float16Storage* y_raw, Float16Storage* concat2,
    Float16Storage* concat3, float clip_min,
    std::size_t spatial, std::size_t quant_channels, CpuThreadPool& pool)
{
    if (quant_channels % 8U != 0U) {
        get_scalar_kernels().y0_tail_nhwc(
            latent, prior, quant_scale, update_scale0, update_scale1,
            normalized, clipped, y_raw, concat2, concat3, clip_min,
            spatial, quant_channels, pool);
        return;
    }

    const std::size_t norm_c = quant_channels * 2U;
    const std::size_t prior_c = quant_channels * 4U;
    const std::size_t c3_c = quant_channels * 6U;
    const __m256 vmin = _mm256_set1_ps(clip_min);
    const __m256 vone = _mm256_set1_ps(1.0F);

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

            for (std::size_t c = 0; c < quant_channels; c += 8U) {
                const std::size_t cp = c + quant_channels;
                const __m256 clip0 = round_to_f16_precision(_mm256_max_ps(load_f16x8_as_f32(pri_px + c), vmin));
                const __m256 clip1 = round_to_f16_precision(_mm256_max_ps(load_f16x8_as_f32(pri_px + cp), vmin));
                const __m256 inv0 = round_to_f16_precision(_mm256_div_ps(vone, clip0));
                const __m256 inv1 = round_to_f16_precision(_mm256_div_ps(vone, clip1));
                const __m256 nrm0 = round_to_f16_precision(_mm256_mul_ps(load_f16x8_as_f32(lat_px + c), inv0));
                const __m256 nrm1 = round_to_f16_precision(_mm256_mul_ps(load_f16x8_as_f32(lat_px + cp), inv1));

                store_f32x8_as_f16(nrm_px + c, nrm0);
                store_f32x8_as_f16(nrm_px + cp, nrm1);
                store_f32x8_as_f16(clp_px + c, clip0);
                store_f32x8_as_f16(clp_px + cp, clip1);

                const __m256 pt0 = load_f16x8_as_f32(pri_px + c + norm_c);
                const __m256 pt1 = load_f16x8_as_f32(pri_px + cp + norm_c);
                const __m256 d0 = round_to_f16_precision(_mm256_sub_ps(nrm0, pt0));
                const __m256 d1 = round_to_f16_precision(_mm256_sub_ps(nrm1, pt1));
                const __m256 s0 = round_to_f16_precision(_mm256_mul_ps(d0, load_f16x8_as_f32(qsc_px + c)));
                const __m256 s1 = round_to_f16_precision(_mm256_mul_ps(d1, load_f16x8_as_f32(qsc_px + cp)));
                const __m256 sum_s = round_to_f16_precision(_mm256_add_ps(s0, s1));
                const __m256 rounded = round_to_f16_precision(_mm256_round_ps(sum_s, kRoundPsFlags));
                store_f32x8_as_f16(raw_px + c, rounded);

                const __m256 uin0 = round_to_f16_precision(_mm256_add_ps(rounded, pt0));
                const __m256 uin1 = round_to_f16_precision(_mm256_add_ps(rounded, pt1));
                const __m256 upd0 = round_to_f16_precision(_mm256_mul_ps(uin0, load_f16x8_as_f32(us0_px + c)));
                const __m256 upd1 = round_to_f16_precision(_mm256_mul_ps(uin1, load_f16x8_as_f32(us1_px + c)));
                store_f32x8_as_f16(c2_px + c, upd0);
                store_f32x8_as_f16(c2_px + cp, upd1);
                store_f32x8_as_f16(c3_px + c, upd0);
                store_f32x8_as_f16(c3_px + cp, upd1);
            }
            std::memcpy(c3_px + norm_c, pri_px, prior_c * sizeof(Float16Storage));
        }
    });
}

void avx2_y1_tail_nhwc(
    const Float16Storage* normalized, const Float16Storage* prior,
    const Float16Storage* quant_scale, const Float16Storage* update_scale0,
    const Float16Storage* update_scale1, const Float16Storage* concat2,
    const Float16Storage* clipped, Float16Storage* y_raw,
    Float16Storage* output, std::size_t spatial,
    std::size_t quant_channels, CpuThreadPool& pool)
{
    if (quant_channels % 8U != 0U) {
        get_scalar_kernels().y1_tail_nhwc(
            normalized, prior, quant_scale, update_scale0, update_scale1,
            concat2, clipped, y_raw, output, spatial, quant_channels, pool);
        return;
    }

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

            for (std::size_t c = 0; c < quant_channels; c += 8U) {
                const std::size_t cp = c + quant_channels;
                const __m256 pt0 = load_f16x8_as_f32(pri_px + c);
                const __m256 pt1 = load_f16x8_as_f32(pri_px + cp);
                const __m256 d0 = round_to_f16_precision(_mm256_sub_ps(load_f16x8_as_f32(nrm_px + c), pt0));
                const __m256 d1 = round_to_f16_precision(_mm256_sub_ps(load_f16x8_as_f32(nrm_px + cp), pt1));
                const __m256 s0 = round_to_f16_precision(_mm256_mul_ps(d0, load_f16x8_as_f32(qsc_px + c)));
                const __m256 s1 = round_to_f16_precision(_mm256_mul_ps(d1, load_f16x8_as_f32(qsc_px + cp)));
                const __m256 sum_s = round_to_f16_precision(_mm256_add_ps(s0, s1));
                const __m256 rounded = round_to_f16_precision(_mm256_round_ps(sum_s, kRoundPsFlags));
                store_f32x8_as_f16(raw_px + c, rounded);

                const __m256 uin0 = round_to_f16_precision(_mm256_add_ps(rounded, pt0));
                const __m256 uin1 = round_to_f16_precision(_mm256_add_ps(rounded, pt1));
                const __m256 upd0 = round_to_f16_precision(_mm256_mul_ps(uin0, load_f16x8_as_f32(us0_px + c)));
                const __m256 upd1 = round_to_f16_precision(_mm256_mul_ps(uin1, load_f16x8_as_f32(us1_px + c)));
                const __m256 add0 = round_to_f16_precision(_mm256_add_ps(load_f16x8_as_f32(c2_px + c), upd0));
                const __m256 add1 = round_to_f16_precision(_mm256_add_ps(load_f16x8_as_f32(c2_px + cp), upd1));

                store_f32x8_as_f16(out_px + c, _mm256_mul_ps(add0, load_f16x8_as_f32(clp_px + c)));
                store_f32x8_as_f16(out_px + cp, _mm256_mul_ps(add1, load_f16x8_as_f32(clp_px + cp)));
            }
        }
    });
}

inline void transpose_u16_8x8(const Float16Storage* src, std::size_t src_stride,
                              Float16Storage* dst, std::size_t dst_stride) noexcept
{
    __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0U * src_stride));
    __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 1U * src_stride));
    __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 2U * src_stride));
    __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 3U * src_stride));
    __m128i r4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4U * src_stride));
    __m128i r5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 5U * src_stride));
    __m128i r6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 6U * src_stride));
    __m128i r7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 7U * src_stride));

    const __m128i t0 = _mm_unpacklo_epi16(r0, r1);
    const __m128i t1 = _mm_unpackhi_epi16(r0, r1);
    const __m128i t2 = _mm_unpacklo_epi16(r2, r3);
    const __m128i t3 = _mm_unpackhi_epi16(r2, r3);
    const __m128i t4 = _mm_unpacklo_epi16(r4, r5);
    const __m128i t5 = _mm_unpackhi_epi16(r4, r5);
    const __m128i t6 = _mm_unpacklo_epi16(r6, r7);
    const __m128i t7 = _mm_unpackhi_epi16(r6, r7);

    const __m128i u0 = _mm_unpacklo_epi32(t0, t2);
    const __m128i u1 = _mm_unpackhi_epi32(t0, t2);
    const __m128i u2 = _mm_unpacklo_epi32(t1, t3);
    const __m128i u3 = _mm_unpackhi_epi32(t1, t3);
    const __m128i u4 = _mm_unpacklo_epi32(t4, t6);
    const __m128i u5 = _mm_unpackhi_epi32(t4, t6);
    const __m128i u6 = _mm_unpacklo_epi32(t5, t7);
    const __m128i u7 = _mm_unpackhi_epi32(t5, t7);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 0U * dst_stride), _mm_unpacklo_epi64(u0, u4));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 1U * dst_stride), _mm_unpackhi_epi64(u0, u4));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 2U * dst_stride), _mm_unpacklo_epi64(u1, u5));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 3U * dst_stride), _mm_unpackhi_epi64(u1, u5));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 4U * dst_stride), _mm_unpacklo_epi64(u2, u6));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 5U * dst_stride), _mm_unpackhi_epi64(u2, u6));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 6U * dst_stride), _mm_unpacklo_epi64(u3, u7));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 7U * dst_stride), _mm_unpackhi_epi64(u3, u7));
}

void avx2_nchw_to_nhwc(const Float16Storage* nchw, Float16Storage* nhwc,
                       std::size_t n, std::size_t c, std::size_t hw,
                       CpuThreadPool& pool)
{
    if (c == 1U || hw == 1U) {
        std::memcpy(nhwc, nchw, n * c * hw * sizeof(Float16Storage));
        return;
    }
    const std::size_t hw_tiles = (hw + 7U) / 8U;
    for (std::size_t batch = 0; batch < n; ++batch) {
        const Float16Storage* bin = nchw + batch * c * hw;
        Float16Storage* bout = nhwc + batch * hw * c;
        pool.parallel_for(hw_tiles, 32U, [&](std::size_t t0, std::size_t t1) {
            for (std::size_t t = t0; t < t1; ++t) {
                const std::size_t s0 = t * 8U;
                const std::size_t rem_s = std::min<std::size_t>(8U, hw - s0);
                std::size_t ch = 0;
                if (rem_s == 8U) {
                    for (; ch + 8U <= c; ch += 8U) {
                        transpose_u16_8x8(bin + ch * hw + s0, hw,
                                          bout + s0 * c + ch, c);
                    }
                }
                for (std::size_t ds = 0; ds < rem_s; ++ds) {
                    Float16Storage* dst_px = bout + (s0 + ds) * c;
                    for (std::size_t cc = ch; cc < c; ++cc)
                        dst_px[cc] = bin[cc * hw + s0 + ds];
                }
            }
        });
    }
}

void avx2_nhwc_to_nchw(const Float16Storage* nhwc, Float16Storage* nchw,
                       std::size_t n, std::size_t hw, std::size_t c,
                       CpuThreadPool& pool)
{
    if (c == 1U || hw == 1U) {
        std::memcpy(nchw, nhwc, n * c * hw * sizeof(Float16Storage));
        return;
    }
    const std::size_t hw_tiles = (hw + 7U) / 8U;
    for (std::size_t batch = 0; batch < n; ++batch) {
        const Float16Storage* bin = nhwc + batch * hw * c;
        Float16Storage* bout = nchw + batch * c * hw;
        pool.parallel_for(hw_tiles, 32U, [&](std::size_t t0, std::size_t t1) {
            for (std::size_t t = t0; t < t1; ++t) {
                const std::size_t s0 = t * 8U;
                const std::size_t rem_s = std::min<std::size_t>(8U, hw - s0);
                std::size_t ch = 0;
                if (rem_s == 8U) {
                    for (; ch + 8U <= c; ch += 8U) {
                        transpose_u16_8x8(bin + s0 * c + ch, c,
                                          bout + ch * hw + s0, hw);
                    }
                }
                for (std::size_t ds = 0; ds < rem_s; ++ds) {
                    const Float16Storage* src_px = bin + (s0 + ds) * c;
                    for (std::size_t cc = (rem_s == 8U ? ch : 0U); cc < c; ++cc)
                        bout[cc * hw + s0 + ds] = src_px[cc];
                }
            }
        });
    }
}

}  // namespace

const KernelSet& get_x86_avx2_kernels() noexcept
{
    static const KernelSet avx2_set = []() {
        KernelSet k = get_scalar_kernels();
        k.mode = GemmMode::kX86Avx2Fp32;
        k.gemm.run_block = &avx2_run_block;
        k.depthwise3x3_s1p1 = &avx2_depthwise3x3_s1p1;
        k.unary = &avx2_unary;
        k.binary_contiguous = &avx2_binary_contiguous;
        k.binary_channel_bcast = &avx2_binary_channel_bcast;
        k.binary_scalar_bcast = &avx2_binary_scalar_bcast;
        k.reglu_nhwc = &avx2_reglu_nhwc;
        k.feature_update_nhwc = &avx2_feature_update_nhwc;
        k.feature_update_outputs_nhwc = &avx2_feature_update_outputs_nhwc;
        k.y0_tail_nhwc = &avx2_y0_tail_nhwc;
        k.y1_tail_nhwc = &avx2_y1_tail_nhwc;
        k.nchw_to_nhwc = &avx2_nchw_to_nhwc;
        k.nhwc_to_nchw = &avx2_nhwc_to_nchw;
        return k;
    }();
    return avx2_set;
}

}  // namespace mlvc::cpu_backend
