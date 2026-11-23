#pragma once

#include <string_view>

namespace mlvc::cpu_backend {

enum class GemmMode {
    kScalarFp32,
    kArm64NeonFp32,
    kArm64Fp16Fml,
    kX86Avx2Fp32,
};

struct CpuFeatures {
    bool arm64_fp16 = false;
    bool arm64_fp16fml = false;
    bool x86_avx2_fma_f16c = false;
    GemmMode preferred_gemm = GemmMode::kScalarFp32;
};

const CpuFeatures& detect_cpu_features() noexcept;
std::string_view gemm_mode_name(GemmMode mode) noexcept;

}  // namespace mlvc::cpu_backend
