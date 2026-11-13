#include "kernels_common.hpp"

namespace mlvc::cpu_backend {
namespace {

void fp16fml_run_block(
    std::size_t m_rows, std::size_t n, std::size_t k,
    const Float16Storage* lhs_block, std::size_t lhs_stride,
    const Float16Storage* rhs_packed,
    Float16Storage* dst_block, std::size_t dst_stride,
    const GemmEpilogue& epilogue, std::size_t row_offset,
    Float16Storage* scratch, std::size_t scratch_elems)
{
    arm64_run_asm_gemm_block(
        &mlvc_arm64_gemm_fp16fml_6x16, true,
        m_rows, n, k, lhs_block, lhs_stride, rhs_packed,
        dst_block, dst_stride, epilogue, row_offset, scratch, scratch_elems);
}

}  // namespace

const KernelSet& get_arm64_fp16fml_kernels() noexcept
{
    static const KernelSet fml_set = []() {
        KernelSet k = get_arm64_neon_fp32_kernels(true);
        k.mode = GemmMode::kArm64Fp16Fml;
        k.gemm.run_block = &fp16fml_run_block;
        return k;
    }();
    return fml_set;
}

}  // namespace mlvc::cpu_backend
