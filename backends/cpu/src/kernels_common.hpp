#pragma once

#include "cpu_features.hpp"
#include "thread_pool.hpp"

#include "mlvc/core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mlvc::cpu_backend {

enum class GemmEpilogueKind {
    kNone,
    kClamp,
    kLeakyRelu,
    kResidualAdd,
    kReglu,
};

struct GemmEpilogue {
    GemmEpilogueKind kind = GemmEpilogueKind::kNone;
    float alpha = 0.0F;
    float clamp_min = -std::numeric_limits<float>::infinity();
    float clamp_max = std::numeric_limits<float>::infinity();
    const Float16Storage* residual = nullptr;
};

enum class UnaryKind {
    kLeakyRelu,
    kClip,
    kSigmoid,
    kReciprocal,
    kRound,
};

enum class BinaryKind {
    kAdd,
    kMul,
    kSub,
};

// Packs OIHW weights [oc, ic, kh, kw] and optional bias [oc] into unified
// f16p16x1biasf16 layout consumed by KleidiAI NEON, FP16FML, ARMv8 NEON FP32,
// AVX2+F16C, and scalar GEMM kernels. For spatial convs, K = kh * kw * ic with
// layout order (ky, kx, ic) matching NHWC contiguous input channel strips.
std::vector<Float16Storage> pack_conv_weights_f16p16x1bias(
    const Float16Storage* weight_oihw, const Float16Storage* bias,
    std::size_t oc, std::size_t ic, std::size_t kh, std::size_t kw);

// Packs depthwise conv weights [c, 1, kh, kw] and optional bias [c] into
// NHWC-friendly [kh * kw, pad16(c)] + bias[pad16(c)].
std::vector<Float16Storage> pack_depthwise_weights(
    const Float16Storage* weight_c1hw, const Float16Storage* bias,
    std::size_t channels, std::size_t kh, std::size_t kw);

struct GemmTileOps {
    // Executes GEMM on a contiguous M-block slice of LHS [m_rows, k]
    // producing [m_rows, n] in dst_base with optional epilogue.
    // Note: residual inside epilogue has row stride dst_stride, pre-offset to the
    // tile's starting column n0, so row (row_offset + mi) * dst_stride aligns with dst_block.
    void (*run_block)(std::size_t m_rows, std::size_t n, std::size_t k,
                      const Float16Storage* lhs_block, std::size_t lhs_stride,
                      const Float16Storage* rhs_packed,
                      Float16Storage* dst_block, std::size_t dst_stride,
                      const GemmEpilogue& epilogue, std::size_t row_offset,
                      Float16Storage* scratch, std::size_t scratch_elems) = nullptr;
};

struct KernelSet {
    GemmMode mode = GemmMode::kScalarFp32;
    GemmTileOps gemm;

    void (*depthwise3x3_s1p1)(const Float16Storage* input,
                              const Float16Storage* packed_weight,
                              Float16Storage* output,
                              int n, int h, int w, int channels,
                              CpuThreadPool& pool) = nullptr;

    void (*unary)(const Float16Storage* input, Float16Storage* output,
                  std::size_t count, UnaryKind op, float alpha,
                  float clamp_min, float clamp_max,
                  CpuThreadPool& pool) = nullptr;

    void (*binary_contiguous)(const Float16Storage* lhs, const Float16Storage* rhs,
                              Float16Storage* output, std::size_t count,
                              BinaryKind op, CpuThreadPool& pool) = nullptr;

    // Broadcast [N*H*W, C] op [C] (or reversed operand order when rhs is 4D).
    void (*binary_channel_bcast)(const Float16Storage* vec, const Float16Storage* chan,
                                 Float16Storage* output, std::size_t spatial,
                                 std::size_t channels, BinaryKind op,
                                 bool chan_is_lhs, CpuThreadPool& pool) = nullptr;

    void (*binary_scalar_bcast)(const Float16Storage* vec, Float16Storage scalar,
                                Float16Storage* output, std::size_t count,
                                BinaryKind op, bool scalar_is_lhs,
                                CpuThreadPool& pool) = nullptr;

    void (*reglu_nhwc)(const Float16Storage* input, Float16Storage* output,
                       std::size_t spatial, std::size_t gate_channels,
                       float clamp_min, float clamp_max,
                       CpuThreadPool& pool) = nullptr;

    void (*feature_update_nhwc)(const Float16Storage* input,
                                const Float16Storage* history,
                                const Float16Storage* scale,
                                Float16Storage* output,
                                std::size_t spatial, std::size_t channels,
                                CpuThreadPool& pool) = nullptr;

    void (*feature_update_outputs_nhwc)(const Float16Storage* input,
                                        const Float16Storage* history,
                                        const Float16Storage* scale,
                                        Float16Storage* scaled_output,
                                        Float16Storage* blended_output,
                                        Float16Storage* feature_output,
                                        std::size_t spatial, std::size_t channels,
                                        CpuThreadPool& pool) = nullptr;

    void (*y0_tail_nhwc)(const Float16Storage* latent, const Float16Storage* prior,
                         const Float16Storage* quant_scale,
                         const Float16Storage* update_scale0,
                         const Float16Storage* update_scale1,
                         Float16Storage* normalized, Float16Storage* clipped,
                         Float16Storage* y_raw, Float16Storage* concat2,
                         Float16Storage* concat3, float clip_min,
                         std::size_t spatial, std::size_t quant_channels,
                         CpuThreadPool& pool) = nullptr;

    void (*y1_tail_nhwc)(const Float16Storage* normalized, const Float16Storage* prior,
                         const Float16Storage* quant_scale,
                         const Float16Storage* update_scale0,
                         const Float16Storage* update_scale1,
                         const Float16Storage* concat2,
                         const Float16Storage* clipped,
                         Float16Storage* y_raw, Float16Storage* output,
                         std::size_t spatial, std::size_t quant_channels,
                         CpuThreadPool& pool) = nullptr;

    void (*nchw_to_nhwc)(const Float16Storage* nchw, Float16Storage* nhwc,
                         std::size_t n, std::size_t c, std::size_t hw,
                         CpuThreadPool& pool) = nullptr;

    void (*nhwc_to_nchw)(const Float16Storage* nhwc, Float16Storage* nchw,
                         std::size_t n, std::size_t hw, std::size_t c,
                         CpuThreadPool& pool) = nullptr;
};

const KernelSet& get_scalar_kernels() noexcept;
#if defined(MLVC_CPU_ENABLE_ARM64)
const KernelSet& get_arm64_fp16_kernels() noexcept;
const KernelSet& get_arm64_fp16fml_kernels() noexcept;
#endif
#if defined(MLVC_CPU_ENABLE_X86_AVX2)
const KernelSet& get_x86_avx2_kernels() noexcept;
#endif

// Shared execution helpers used across all ISAs.
const KernelSet& select_kernels(const CpuFeatures& features) noexcept;

void execute_gemm_nhwc(const KernelSet& kernels,
                       std::size_t m, std::size_t n, std::size_t k,
                       const Float16Storage* lhs, const Float16Storage* rhs_packed,
                       Float16Storage* dst, const GemmEpilogue& epilogue,
                       CpuThreadPool& pool);

void execute_spatial_conv_nhwc(const KernelSet& kernels,
                               const Float16Storage* input,
                               const Float16Storage* rhs_packed,
                               Float16Storage* output,
                               int n, int ic, int ih, int iw,
                               int oc, int oh, int ow,
                               int kh, int kw, int sh, int sw, int ph, int pw,
                               const GemmEpilogue& epilogue,
                               CpuThreadPool& pool);

void execute_depthwise_generic_nhwc(const Float16Storage* input,
                                    const Float16Storage* packed_weight,
                                    Float16Storage* output,
                                    int n, int ic, int ih, int iw,
                                    int oh, int ow, int kh, int kw,
                                    int sh, int sw, int ph, int pw,
                                    CpuThreadPool& pool);

void execute_slice_nhwc(const Float16Storage* input, Float16Storage* output,
                        const int* id, const int* od,
                        const int* starts_nchw, const int* steps_nchw,
                        CpuThreadPool& pool);

void execute_concat_nhwc(const Float16Storage* const* inputs,
                         const int* const* input_dims_nchw,
                         std::size_t input_count,
                         Float16Storage* output,
                         const int* od_nchw,
                         int axis_nchw,
                         CpuThreadPool& pool);

void execute_depth_to_space_nhwc(const Float16Storage* input, Float16Storage* output,
                                 int n, int c, int h, int w, int block_size,
                                 CpuThreadPool& pool);

void execute_space_to_depth_nhwc(const Float16Storage* input, Float16Storage* output,
                                 int n, int c, int h, int w, int block_size,
                                 CpuThreadPool& pool);

}  // namespace mlvc::cpu_backend
