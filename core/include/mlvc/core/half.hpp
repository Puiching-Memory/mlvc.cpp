#pragma once

#include "mlvc/core/tensor.hpp"

#include <bit>
#include <cstdint>

namespace mlvc {

inline float half_to_float(Float16Storage value) noexcept
{
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fU;
    std::uint32_t mantissa = value & 0x03ffU;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int unbiased = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --unbiased;
            }
            mantissa &= 0x03ffU;
            bits = sign |
                (static_cast<std::uint32_t>(unbiased + 127) << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13);
        if (mantissa != 0)
            bits |= 0x00400000U;
    } else {
        exponent += 127U - 15U;
        bits = sign | (exponent << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(bits);
}

inline Float16Storage float_to_half(float value) noexcept
{
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
    const std::uint32_t exponent = (bits >> 23) & 0xffU;
    std::uint32_t mantissa = bits & 0x007fffffU;

    if (exponent == 0xffU) {
        if (mantissa == 0)
            return static_cast<Float16Storage>(sign | 0x7c00U);
        const std::uint16_t payload = static_cast<std::uint16_t>(mantissa >> 13);
        return static_cast<Float16Storage>(sign | 0x7c00U | payload | 0x0200U);
    }

    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31)
        return static_cast<Float16Storage>(sign | 0x7c00U);

    if (half_exponent <= 0) {
        if (half_exponent < -10)
            return sign;
        mantissa |= 0x00800000U;
        const unsigned shift = static_cast<unsigned>(14 - half_exponent);
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const std::uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U)))
            ++rounded;
        return static_cast<Float16Storage>(sign | rounded);
    }

    std::uint32_t rounded = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U))) {
        ++rounded;
        if (rounded == 0x0400U) {
            rounded = 0;
            ++half_exponent;
            if (half_exponent >= 31)
                return static_cast<Float16Storage>(sign | 0x7c00U);
        }
    }
    return static_cast<Float16Storage>(
        sign | (static_cast<std::uint16_t>(half_exponent) << 10) |
        static_cast<std::uint16_t>(rounded));
}

}  // namespace mlvc
