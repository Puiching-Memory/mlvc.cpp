#include "core/half.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mlvc::detail {

bool integer_values_x86_avx2(const Float16Storage* source, std::int32_t* result,
                             std::size_t count) noexcept
{
    const __m256 lower = _mm256_set1_ps(-32768.0F);
    const __m256 upper = _mm256_set1_ps(32767.0F);
    std::size_t i = 0;
    for (; i + 8U <= count; i += 8U) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(source + i));
        const __m256 values = _mm256_cvtph_ps(packed);
        const __m256 ordered = _mm256_cmp_ps(values, values, _CMP_ORD_Q);
        const __m256 above = _mm256_cmp_ps(values, lower, _CMP_GE_OQ);
        const __m256 below = _mm256_cmp_ps(values, upper, _CMP_LE_OQ);
        if (_mm256_movemask_ps(_mm256_and_ps(
                ordered, _mm256_and_ps(above, below))) != 0xff) {
            return false;
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i),
                            _mm256_cvttps_epi32(values));
    }
    for (; i < count; ++i) {
        const float value = half_to_float(source[i]);
        if (!std::isfinite(value) || value < -32768.0F || value > 32767.0F)
            return false;
        result[i] = static_cast<std::int32_t>(value);
    }
    return true;
}

void fp16_values_x86_avx2(const std::int32_t* values, Float16Storage* storage,
                          std::size_t count) noexcept
{
    std::size_t i = 0;
    for (; i + 8U <= count; i += 8U) {
        const __m256i integers = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(values + i));
        const __m128i packed = _mm256_cvtps_ph(
            _mm256_cvtepi32_ps(integers),
            _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(storage + i), packed);
    }
    for (; i < count; ++i)
        storage[i] = float_to_half(static_cast<float>(values[i]));
}

}  // namespace mlvc::detail
