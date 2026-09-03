#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace mlvc {

struct ModelConfig {
    std::string name;
    std::string model_version;
    int model_width = 0;
    int model_height = 0;
    float pixel_range = 1.0F;
    int qp_num = 0;
    int total_qp_num = 0;
    std::vector<int> frame_index_map;
    std::vector<int> qp_shift;
    int feature_channels = 0;
    int latent_channels = 0;
    int hyperprior_channels = 0;
    int downsample_feature = 0;
    int downsample_latent = 0;
    int downsample_hyperprior = 0;
    int y_scale_repeat = 0;
    std::optional<int> iframe_period;
    std::optional<int> reset_period;
    bool disable_feature_reset = false;

    int feature_height() const noexcept;
    int feature_width() const noexcept;
    int latent_height() const noexcept;
    int latent_width() const noexcept;
    int hyperprior_height() const noexcept;
    int hyperprior_width() const noexcept;
    int frame_in_gop(int absolute_frame) const;
    int q_index_shift(int frame_index) const;
};

ModelConfig load_model_config(const std::filesystem::path& model_dir);
std::vector<std::string> embedded_model_profiles();

}  // namespace mlvc
