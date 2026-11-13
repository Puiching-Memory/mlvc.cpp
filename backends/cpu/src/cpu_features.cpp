#include "cpu_features.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#if defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__) || defined(__OHOS__))
#include <sys/auxv.h>
#ifndef HWCAP_FPHP
#define HWCAP_FPHP (1UL << 9)
#endif
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1UL << 10)
#endif
#ifndef HWCAP_ASIMDFHM
#define HWCAP_ASIMDFHM (1UL << 23)
#endif
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
#include <intrin.h>
#endif

namespace mlvc::cpu_backend {
namespace {

std::string lowercase_env(const char* value)
{
    if (!value)
        return {};
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return result;
}

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
bool detect_msvc_x86_avx2_fma_f16c() noexcept
{
    int cpu_info[4] = {};
    __cpuid(cpu_info, 0);
    if (cpu_info[0] < 7)
        return false;

    int leaf1[4] = {};
    __cpuid(leaf1, 1);
    const bool has_osxsave = (leaf1[2] & (1 << 27)) != 0;
    const bool has_fma = (leaf1[2] & (1 << 12)) != 0;
    const bool has_f16c = (leaf1[2] & (1 << 29)) != 0;
    const bool has_avx = (leaf1[2] & (1 << 28)) != 0;
    if (!has_osxsave || !has_fma || !has_f16c || !has_avx)
        return false;

    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6ULL) != 0x6ULL)
        return false;

    int leaf7[4] = {};
    __cpuidex(leaf7, 7, 0);
    return (leaf7[1] & (1 << 5)) != 0;  // AVX2
}
#endif

CpuFeatures probe_features() noexcept
{
    CpuFeatures features;

#if defined(__aarch64__)
#if defined(__linux__) || defined(__ANDROID__) || defined(__OHOS__)
    const unsigned long hwcap = ::getauxval(AT_HWCAP);
    features.arm64_fp16 =
        (hwcap & HWCAP_FPHP) != 0 && (hwcap & HWCAP_ASIMDHP) != 0;
    features.arm64_fp16fml =
        features.arm64_fp16 && ((hwcap & HWCAP_ASIMDFHM) != 0);
#elif defined(__APPLE__)
    features.arm64_fp16 = true;
    features.arm64_fp16fml = true;
#endif
    // Pure-FP16 accumulation (KleidiAI neon_mla) can exceed MLVC accuracy limits
    // on deep K=864..2304 conv chains, so AUTO selects FP16->FP32 accumulation
    // (+fp16fml when available, else NEON FP32). Explicit MLVC_CPU_ISA=fp16|kleidi
    // remains available for targeted pure-FP16 testing.
    if (features.arm64_fp16fml) {
        features.preferred_gemm = GemmMode::kArm64Fp16Fml;
    } else {
        features.preferred_gemm = GemmMode::kArm64NeonFp32;
    }
#elif defined(MLVC_CPU_ENABLE_X86_AVX2)
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
    features.x86_avx2_fma_f16c = detect_msvc_x86_avx2_fma_f16c();
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    __builtin_cpu_init();
    features.x86_avx2_fma_f16c = __builtin_cpu_supports("avx2") &&
                                 __builtin_cpu_supports("fma") &&
                                 __builtin_cpu_supports("f16c");
#endif
    features.preferred_gemm = features.x86_avx2_fma_f16c
        ? GemmMode::kX86Avx2Fp32
        : GemmMode::kScalarFp32;
#endif

    // Optional runtime ISA selection override for A/B conformance & benchmarking.
    const std::string override_mode = lowercase_env(std::getenv("MLVC_CPU_ISA"));
    if (override_mode == "scalar") {
        features.preferred_gemm = GemmMode::kScalarFp32;
    } else if (override_mode == "fp32" || override_mode == "neon_fp32") {
#if defined(__aarch64__)
        features.preferred_gemm = GemmMode::kArm64NeonFp32;
#else
        features.preferred_gemm = GemmMode::kScalarFp32;
#endif
    } else if (override_mode == "fp16fml" && features.arm64_fp16fml) {
        features.preferred_gemm = GemmMode::kArm64Fp16Fml;
    } else if ((override_mode == "fp16" || override_mode == "kleidi") &&
               features.arm64_fp16) {
        features.preferred_gemm = GemmMode::kArm64KleidiFp16;
    } else if (override_mode == "avx2" && features.x86_avx2_fma_f16c) {
        features.preferred_gemm = GemmMode::kX86Avx2Fp32;
    }

    return features;
}

}  // namespace

const CpuFeatures& detect_cpu_features() noexcept
{
    static const CpuFeatures features = probe_features();
    return features;
}

std::string_view gemm_mode_name(GemmMode mode) noexcept
{
    switch (mode) {
    case GemmMode::kScalarFp32:
        return "scalar-fp32";
    case GemmMode::kArm64NeonFp32:
        return "arm64-neon-fp32";
    case GemmMode::kArm64Fp16Fml:
        return "arm64-fp16fml";
    case GemmMode::kArm64KleidiFp16:
        return "arm64-kleidi-fp16";
    case GemmMode::kX86Avx2Fp32:
        return "x86-avx2-f16c";
    }
    return "unknown";
}

}  // namespace mlvc::cpu_backend
