#include "mlvc/core/model.hpp"

#include "model_assets.hpp"

#include "model_proto.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mlvc {
namespace {

int divide_up(int value, std::int64_t divisor)
{
    if (value <= 0 || divisor <= 0 ||
        divisor > std::numeric_limits<int>::max()) {
        throw std::runtime_error("invalid model downsample dimensions");
    }
    return static_cast<int>(
        static_cast<std::int64_t>(value) / divisor +
        (static_cast<std::int64_t>(value) % divisor != 0 ? 1 : 0));
}

int checked_int(std::uint32_t value)
{
    if (value > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("model integer exceeds runtime range");
    return static_cast<int>(value);
}

void validate_bundle_manifest(const std::filesystem::path& model_dir,
                              const wire::ModelMetadata& metadata)
{
    const auto bundle = detail::read_model_proto<wire::ModelBundle>(model_dir, "model_bundle.pb");
    if (metadata.schema_version() != 1 || bundle.schema_version() != 2 ||
        bundle.profile() != metadata.profile() || bundle.model_version() != metadata.model_version() ||
        bundle.width() != metadata.width() || bundle.height() != metadata.height() ||
        metadata.checkpoint_sha256().size() != 32 || bundle.entropy_model_sha256().size() != 32 ||
        bundle.checkpoint_sha256() != metadata.checkpoint_sha256() ||
        bundle.contract().container() != "mlvc-frame-le-v1" ||
        bundle.contract().entropy() != "canonical-pmf-v1")
        throw std::runtime_error("model bundle does not match Protobuf metadata");
}

}  // namespace

void ModelConfig::validate() const
{
    const bool invalid_dimensions =
        model_width <= 0 || model_height <= 0 || feature_channels <= 0 ||
        latent_channels <= 0 || hyperprior_channels <= 0 ||
        downsample_feature <= 0 || downsample_latent <= 0 ||
        downsample_hyperprior <= 0 || y_scale_repeat <= 0;
    if (name.empty() || model_version.empty() || invalid_dimensions ||
        !std::isfinite(pixel_range) || pixel_range <= 0.0F || qp_num <= 0 ||
        total_qp_num < qp_num || latent_channels % 2 != 0 ||
        latent_channels % y_scale_repeat != 0 ||
        latent_channels / y_scale_repeat > hyperprior_channels ||
        frame_index_map.empty() || qp_shift.empty()) {
        throw std::runtime_error("invalid fixed-shape model metadata");
    }

    const auto invalid_map_index = [&](int index) {
        return index < 0 ||
            static_cast<std::size_t>(index) >= qp_shift.size();
    };
    if (std::any_of(frame_index_map.begin(), frame_index_map.end(),
                    invalid_map_index)) {
        throw std::runtime_error("model QP shift mapping is out of range");
    }
    if (std::any_of(qp_shift.begin(), qp_shift.end(), [&](int shift) {
            return shift < 0 || shift > total_qp_num - qp_num;
        })) {
        throw std::runtime_error("model QP shift is outside the model range");
    }
    if (iframe_period && *iframe_period <= 0)
        throw std::runtime_error("model I-frame period must be positive");
    if (reset_period && *reset_period <= 0)
        throw std::runtime_error("model reset period must be positive");
    if (static_cast<std::int64_t>(downsample_latent) *
            downsample_hyperprior >
        std::numeric_limits<int>::max()) {
        throw std::runtime_error("model downsample factor is too large");
    }
    if (!disable_feature_reset || reset_period.has_value()) {
        throw std::runtime_error(
            "this runtime currently requires the dmc61sbr_e1d1 no-reset profile");
    }
}

int ModelConfig::feature_height() const
{
    return divide_up(model_height, downsample_feature);
}

int ModelConfig::feature_width() const
{
    return divide_up(model_width, downsample_feature);
}

int ModelConfig::latent_height() const
{
    return divide_up(model_height, downsample_latent);
}

int ModelConfig::latent_width() const
{
    return divide_up(model_width, downsample_latent);
}

int ModelConfig::hyperprior_height() const
{
    return divide_up(model_height,
        static_cast<std::int64_t>(downsample_latent) *
            downsample_hyperprior);
}

int ModelConfig::hyperprior_width() const
{
    return divide_up(model_width,
        static_cast<std::int64_t>(downsample_latent) *
            downsample_hyperprior);
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
    if (frame_index < 0)
        throw std::runtime_error("frame index must be non-negative");
    if (frame_index_map.empty() || qp_shift.empty())
        throw std::runtime_error("model metadata has no QP shift mapping");
    const std::size_t frame = static_cast<std::size_t>(frame_index);
    const std::size_t frame_map_index =
        (frame % frame_index_map.size() + 1U) % frame_index_map.size();
    const int map_index = frame_index_map.at(frame_map_index);
    return qp_shift.at(static_cast<std::size_t>(map_index));
}

ModelConfig load_model_config(const std::filesystem::path& model_dir)
{
    const auto path = model_dir / "metadata.pb";
    const auto metadata = detail::read_model_proto<wire::ModelMetadata>(model_dir, "metadata.pb");
    validate_bundle_manifest(model_dir, metadata);
    ModelConfig result;
    result.name = metadata.profile();
    result.model_version = metadata.model_version();
    result.model_width = checked_int(metadata.width());
    result.model_height = checked_int(metadata.height());
    result.pixel_range = metadata.pixel_range();
    result.qp_num = checked_int(metadata.qp_num());
    result.total_qp_num = checked_int(metadata.total_qp_num());
    result.feature_channels = checked_int(metadata.feature_channels());
    result.latent_channels = checked_int(metadata.latent_channels());
    result.hyperprior_channels = checked_int(metadata.hyperprior_channels());
    result.downsample_feature = checked_int(metadata.downsample_feature());
    result.downsample_latent = checked_int(metadata.downsample_latent());
    result.downsample_hyperprior = checked_int(metadata.downsample_hyperprior());
    result.y_scale_repeat = checked_int(metadata.y_scale_repeat());
    for (const auto value : metadata.frame_index_map())
        result.frame_index_map.push_back(checked_int(value));
    result.qp_shift.assign(metadata.qp_shift().begin(), metadata.qp_shift().end());
    if (metadata.has_iframe_period()) result.iframe_period = checked_int(metadata.iframe_period());
    if (metadata.has_reset_period()) result.reset_period = checked_int(metadata.reset_period());
    result.disable_feature_reset = metadata.disable_feature_reset();

    try {
        result.validate();
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(error.what()) + ": " + path.string());
    }
    return result;
}

std::vector<std::string> embedded_model_profiles()
{
    return detail::list_embedded_model_profiles();
}

}  // namespace mlvc
