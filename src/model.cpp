#include "mlvc/model.hpp"

#include "model_assets.hpp"

#include <nlohmann/json.hpp>

#include <stdexcept>

namespace mlvc {
namespace {

int divide_up(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

std::optional<int> optional_int(const nlohmann::json& value)
{
    if (value.is_null())
        return std::nullopt;
    return value.get<int>();
}

void validate_bundle_manifest(const std::filesystem::path& model_dir,
                              const nlohmann::json& metadata)
{
    const auto manifest_path = model_dir / "model_bundle.json";
    if (!detail::model_asset_exists(model_dir, "model_bundle.json"))
        return;
    const nlohmann::json manifest = nlohmann::json::parse(
        detail::read_model_text(model_dir, "model_bundle.json"));
    const auto& full = metadata.at("params").at("full_model_params");
    const auto& split = metadata.at("params").at("split_model_params");
    if (manifest.at("schema_version").get<int>() != 1 ||
        manifest.at("profile").get<std::string>() !=
            metadata.at("name").get<std::string>() ||
        manifest.at("model_version").get<std::string>() !=
            full.at("model_version").get<std::string>() ||
        manifest.at("model_size").at("width").get<int>() !=
            split.at("model_width").get<int>() ||
        manifest.at("model_size").at("height").get<int>() !=
            split.at("model_height").get<int>() ||
        manifest.at("compatibility").at("container").get<std::string>() !=
            "mlvc-frame-le-v1" ||
        manifest.at("compatibility").at("entropy").get<std::string>() !=
            "canonical-pmf-v1") {
        throw std::runtime_error("model bundle manifest does not match metadata: " +
                                 manifest_path.string());
    }
}

}  // namespace

int ModelConfig::feature_height() const noexcept
{
    return divide_up(model_height, downsample_feature);
}

int ModelConfig::feature_width() const noexcept
{
    return divide_up(model_width, downsample_feature);
}

int ModelConfig::latent_height() const noexcept
{
    return divide_up(model_height, downsample_latent);
}

int ModelConfig::latent_width() const noexcept
{
    return divide_up(model_width, downsample_latent);
}

int ModelConfig::hyperprior_height() const noexcept
{
    return divide_up(model_height,
                     downsample_latent * downsample_hyperprior);
}

int ModelConfig::hyperprior_width() const noexcept
{
    return divide_up(model_width,
                     downsample_latent * downsample_hyperprior);
}

int ModelConfig::frame_in_gop(int absolute_frame) const
{
    if (absolute_frame < 0)
        throw std::runtime_error("frame index must be non-negative");
    if (iframe_period && *iframe_period > 0)
        return absolute_frame % *iframe_period;
    return absolute_frame;
}

int ModelConfig::q_index_shift(int frame_index) const
{
    if (frame_index_map.empty() || qp_shift.empty())
        throw std::runtime_error("model metadata has no QP shift mapping");
    const int map_index = frame_index_map.at(
        static_cast<std::size_t>((frame_index + 1) % frame_index_map.size()));
    return qp_shift.at(static_cast<std::size_t>(map_index));
}

ModelConfig load_model_config(const std::filesystem::path& model_dir)
{
    const auto path = model_dir / "metadata.json";
    const nlohmann::json root = nlohmann::json::parse(
        detail::read_model_text(model_dir, "metadata.json"));
    validate_bundle_manifest(model_dir, root);
    const auto& full = root.at("params").at("full_model_params");
    const auto& split = root.at("params").at("split_model_params");

    ModelConfig result;
    result.name = root.at("name").get<std::string>();
    result.model_version = full.at("model_version").get<std::string>();
    result.model_width = split.at("model_width").get<int>();
    result.model_height = split.at("model_height").get<int>();
    result.pixel_range = full.at("pixel_range").get<float>();
    result.qp_num = full.at("qp_num").get<int>();
    result.total_qp_num = full.at("total_qp_num").get<int>();
    result.frame_index_map = full.at("frame_index_map").get<std::vector<int>>();
    result.qp_shift = full.at("qp_shift").get<std::vector<int>>();
    result.feature_channels = full.at("feature_channels").get<int>();
    result.latent_channels = full.at("latent_channels").get<int>();
    result.hyperprior_channels = full.at("hyperprior_channels").get<int>();
    result.downsample_feature = full.at("downsample_feature").get<int>();
    result.downsample_latent = full.at("downsample_latent").get<int>();
    result.downsample_hyperprior =
        full.at("downsample_hyperprior").get<int>();
    result.y_scale_repeat = full.at("y_scale_repeat").get<int>();
    result.iframe_period = optional_int(full.at("iframe_period"));
    result.reset_period = optional_int(full.at("reset_period"));
    result.disable_feature_reset =
        full.at("disable_feature_reset").get<bool>();

    if (result.model_width <= 0 || result.model_height <= 0 ||
        result.feature_channels <= 0 || result.latent_channels <= 0 ||
        result.hyperprior_channels <= 0 || result.y_scale_repeat <= 0 ||
        result.latent_channels % 2 != 0 ||
        result.latent_channels % result.y_scale_repeat != 0) {
        throw std::runtime_error("invalid fixed-shape model metadata: " +
                                 path.string());
    }
    if (!result.disable_feature_reset || result.reset_period.has_value()) {
        throw std::runtime_error(
            "this runtime currently requires the dmc61sbr_e1d1 no-reset profile");
    }
    return result;
}

std::vector<std::string> embedded_model_profiles()
{
    return detail::list_embedded_model_profiles();
}

}  // namespace mlvc
