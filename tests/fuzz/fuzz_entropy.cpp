#include "core/assets.hpp"
#include "core/entropy.hpp"
#include "core/model.hpp"
#include "core/sha256.hpp"
#include "mlvc_model.pb.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::byte> serialize(const google::protobuf::MessageLite& msg)
{
    const std::string encoded = msg.SerializeAsString();
    const auto bytes = std::as_bytes(std::span<const char>{encoded.data(), encoded.size()});
    return {bytes.begin(), bytes.end()};
}

class StaticAssetReader final : public mlvc::ModelAssetReader {
public:
    void put(const std::string& key, std::vector<std::byte> bytes)
    {
        files_[key] = std::move(bytes);
    }

    std::string location_label() const override { return "fuzz-memory"; }

    bool exists(const std::filesystem::path& rel) const override
    {
        return files_.contains(rel.generic_string());
    }

    std::vector<std::byte> read_binary(const std::filesystem::path& rel) const override
    {
        const auto it = files_.find(rel.generic_string());
        if (it == files_.end())
            throw std::runtime_error("missing fuzz asset");
        return it->second;
    }

private:
    std::map<std::string, std::vector<std::byte>> files_;
};

const mlvc::EntropyCodec& fuzz_codec()
{
    static const mlvc::EntropyCodec codec = []() {
        mlvc::wire::GaussianPmf gaussian;
        gaussian.set_schema_version(1);
        gaussian.set_scale_levels(4);
        gaussian.set_index_space(true);
        for (int row = 0; row < 4; ++row) {
            gaussian.mutable_table()->add_lengths(3);
            gaussian.mutable_table()->add_offsets(-1);
            gaussian.mutable_table()->add_values(16384);
            gaussian.mutable_table()->add_values(32768);
            gaussian.mutable_table()->add_values(16384);
        }

        mlvc::wire::BitEstimatorPmf bit_estimator;
        bit_estimator.set_schema_version(1);
        bit_estimator.set_qp_num(8);
        bit_estimator.set_channels(2);
        for (int row = 0; row < 16; ++row) {
            bit_estimator.mutable_table()->add_lengths(3);
            bit_estimator.mutable_table()->add_offsets(-1);
            bit_estimator.mutable_table()->add_values(16384);
            bit_estimator.mutable_table()->add_values(32768);
            bit_estimator.mutable_table()->add_values(16384);
        }

        auto reader = std::make_shared<StaticAssetReader>();
        reader->put("gaussian_pmf.pb", serialize(gaussian));
        reader->put("bit_estimator_pmf.pb", serialize(bit_estimator));

        mlvc::ModelConfig config;
        config.name = "fuzz-v1";
        config.model_version = "v1";
        config.model_width = 64;
        config.model_height = 64;
        config.pixel_range = 1.0F;
        config.qp_num = 4;
        config.total_qp_num = 8;
        config.frame_index_map = {0};
        config.qp_shift = {0};
        config.feature_channels = 2;
        config.latent_channels = 4;
        config.hyperprior_channels = 2;
        config.downsample_feature = 16;
        config.downsample_latent = 16;
        config.downsample_hyperprior = 2;
        config.y_scale_repeat = 2;
        config.disable_feature_reset = true;
        return mlvc::EntropyCodec(reader, config);
    }();
    return codec;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    if (!data || size < 2U || size > (1U << 18))
        return 0;

    const int q_index = static_cast<int>(data[0] % 8U);
    const std::span<const std::byte> payload(
        reinterpret_cast<const std::byte*>(data + 1), size - 1U);
    try {
        static_cast<void>(fuzz_codec().decode(payload, q_index));
    } catch (const std::exception&) {
    }
    return 0;
}
