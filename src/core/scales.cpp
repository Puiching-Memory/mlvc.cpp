#include "core/scales.hpp"

#include "core/half.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mlvc {

ScalePair extract_scales(const Tensor& z_raw, const ModelConfig& config,
                         int scale_max_index)
{
    ScalePair result;
    extract_scales(tensor_view(z_raw), config, scale_max_index,
                   result.first, result.second);
    return result;
}

void extract_scales(TensorView z_raw, const ModelConfig& config,
                    int scale_max_index, std::vector<std::int32_t>& scales_0,
                    std::vector<std::int32_t>& scales_1)
{
    const std::vector<int64_t> expected{
        1, config.hyperprior_channels,
        config.hyperprior_height(), config.hyperprior_width()};
    const std::span<const std::int64_t> shape = z_raw.shape();
    if (z_raw.data_type() != TensorDataType::kFloat16 ||
        shape.size() != expected.size() ||
        !std::equal(shape.begin(), shape.end(), expected.begin()) ||
        z_raw.element_count() != checked_tensor_element_count(expected)) {
        throw std::runtime_error("z_raw shape or dtype does not match model metadata");
    }

    const int channels = config.latent_channels;
    const int half_channels = channels / 2;
    const int base_channels = channels / config.y_scale_repeat;
    const int y_height = config.latent_height();
    const int y_width = config.latent_width();
    const int z_height = config.hyperprior_height();
    const int z_width = config.hyperprior_width();
    const int spatial_repeat = config.downsample_hyperprior;
    const std::span<const Float16Storage> z = z_raw.fp16_data();
    const std::size_t output_count =
        static_cast<std::size_t>(half_channels) * y_height * y_width;
    scales_0.resize(output_count);
    scales_1.resize(output_count);

    if (base_channels > config.hyperprior_channels)
        throw std::runtime_error("scale decoder channel mapping is invalid");
    std::vector<std::int32_t> base_scales(
        static_cast<std::size_t>(base_channels) * z_height * z_width);
    std::size_t base_idx = 0;
#if defined(__aarch64__)
    const std::size_t count = base_scales.size();
    const int32x4_t zero = vdupq_n_s32(0);
    const int32x4_t max_idx = vdupq_n_s32(scale_max_index);
    for (; base_idx + 8 <= count; base_idx += 8) {
        std::uint16_t bits[8];
        std::memcpy(bits, z.data() + base_idx, sizeof(bits));
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(bits));
        const float32x4_t a0 = vabsq_f32(vcvt_f32_f16(vget_low_f16(h)));
        const float32x4_t a1 = vabsq_f32(vcvt_high_f32_f16(h));
        const int32x4_t s0 = vminq_s32(vmaxq_s32(vcvtq_s32_f32(a0), zero), max_idx);
        const int32x4_t s1 = vminq_s32(vmaxq_s32(vcvtq_s32_f32(a1), zero), max_idx);
        vst1q_s32(base_scales.data() + base_idx, s0);
        vst1q_s32(base_scales.data() + base_idx + 4, s1);
    }
#endif
    for (; base_idx < base_scales.size(); ++base_idx) {
        const float value = std::abs(half_to_float(z[base_idx]));
        base_scales[base_idx] =
            std::clamp(static_cast<std::int32_t>(value), 0, scale_max_index);
    }

    for (int channel = 0; channel < half_channels; ++channel) {
        const int first_channel = channel / config.y_scale_repeat;
        const int second_channel =
            (channel + half_channels) / config.y_scale_repeat;
        for (int y = 0; y < y_height; ++y) {
            const int zy = std::min(y / spatial_repeat, z_height - 1);
            const auto* first_row = base_scales.data() +
                (static_cast<std::size_t>(first_channel) * z_height + zy) *
                    z_width;
            const auto* second_row = base_scales.data() +
                (static_cast<std::size_t>(second_channel) * z_height + zy) *
                    z_width;
            for (int x = 0; x < y_width; ++x) {
                const bool checkerboard_0 = ((x + y) & 1) == 0;
                const int zx = std::min(x / spatial_repeat, z_width - 1);
                const int first = first_row[zx];
                const int second = second_row[zx];
                const std::size_t index =
                    (static_cast<std::size_t>(channel) * y_height + y) * y_width + x;
                scales_0[index] = checkerboard_0 ? first : second;
                scales_1[index] = checkerboard_0 ? second : first;
            }
        }
    }
}

}  // namespace mlvc
