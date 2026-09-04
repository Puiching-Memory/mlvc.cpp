#include "mlvc/core/scales.hpp"

#include "mlvc/core/half.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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
    if (z_raw.data_type != TensorDataType::kFloat16 ||
        !std::equal(z_raw.shape.begin(), z_raw.shape.end(),
                    expected.begin(), expected.end()) ||
        z_raw.shape.size() != expected.size() ||
        z_raw.bytes != static_cast<std::size_t>(config.hyperprior_channels) *
            config.hyperprior_height() * config.hyperprior_width() *
            sizeof(Float16Storage)) {
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
    const auto* z = static_cast<const Float16Storage*>(z_raw.data);
    const std::size_t output_count =
        static_cast<std::size_t>(half_channels) * y_height * y_width;
    scales_0.resize(output_count);
    scales_1.resize(output_count);

    if (base_channels > config.hyperprior_channels)
        throw std::runtime_error("scale decoder channel mapping is invalid");
    std::vector<std::int32_t> base_scales(
        static_cast<std::size_t>(base_channels) * z_height * z_width);
    for (std::size_t index = 0; index < base_scales.size(); ++index) {
        const float value = std::abs(half_to_float(z[index]));
        base_scales[index] =
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
