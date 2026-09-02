#include "mlvc/scales.hpp"

#include "mlvc/half.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mlvc {

ScalePair extract_scales(const Tensor& z_raw, const ModelConfig& config,
                         int scale_max_index)
{
    const std::vector<int64_t> expected{
        1, config.hyperprior_channels,
        config.hyperprior_height(), config.hyperprior_width()};
    if (z_raw.data_type() != TensorDataType::kFloat16 || z_raw.shape != expected)
        throw std::runtime_error("z_raw shape or dtype does not match model metadata");

    const int channels = config.latent_channels;
    const int half_channels = channels / 2;
    const int base_channels = channels / config.y_scale_repeat;
    const int y_height = config.latent_height();
    const int y_width = config.latent_width();
    const int z_height = config.hyperprior_height();
    const int z_width = config.hyperprior_width();
    const int spatial_repeat = config.downsample_hyperprior;
    const auto& z = std::get<std::vector<Float16Storage>>(z_raw.data);
    const std::size_t output_count =
        static_cast<std::size_t>(half_channels) * y_height * y_width;
    std::vector<std::int32_t> scales_0(output_count);
    std::vector<std::int32_t> scales_1(output_count);

    auto scale_value = [&](int channel, int y, int x) {
        const int base_channel = channel / config.y_scale_repeat;
        if (base_channel >= base_channels || base_channel >= config.hyperprior_channels)
            throw std::runtime_error("scale decoder channel mapping is invalid");
        const int zy = std::min(y / spatial_repeat, z_height - 1);
        const int zx = std::min(x / spatial_repeat, z_width - 1);
        const std::size_t index =
            (static_cast<std::size_t>(base_channel) * z_height + zy) * z_width + zx;
        const float value = std::abs(half_to_float(z[index]));
        return std::clamp(static_cast<std::int32_t>(value), 0, scale_max_index);
    };

    for (int channel = 0; channel < half_channels; ++channel) {
        for (int y = 0; y < y_height; ++y) {
            for (int x = 0; x < y_width; ++x) {
                const bool checkerboard_0 = ((x + y) & 1) == 0;
                const int first = scale_value(channel, y, x);
                const int second = scale_value(channel + half_channels, y, x);
                const std::size_t index =
                    (static_cast<std::size_t>(channel) * y_height + y) * y_width + x;
                scales_0[index] = checkerboard_0 ? first : second;
                scales_1[index] = checkerboard_0 ? second : first;
            }
        }
    }
    return {std::move(scales_0), std::move(scales_1)};
}

}  // namespace mlvc
