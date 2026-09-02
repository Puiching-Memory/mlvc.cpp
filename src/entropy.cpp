#include "mlvc/entropy.hpp"

#include "mlvc/half.hpp"
#include "mlvc/scales.hpp"

#include <msrtc_rans/EntropyCoder.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace mlvc {
namespace {

struct PmfTable {
    std::vector<std::int32_t> lengths;
    std::vector<std::int32_t> offsets;
    std::vector<std::int32_t> table;
};

struct GaussianPmf : PmfTable {
    int scale_levels = 0;
    bool index_space = false;
};

struct BitEstimatorPmf : PmfTable {
    int qp_num = 0;
    int channels = 0;
};

nlohmann::json load_json(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open PMF file: " + path.string());
    nlohmann::json result;
    input >> result;
    return result;
}

PmfTable parse_table(const nlohmann::json& value)
{
    return PmfTable{
        value.at("pmf_lengths").get<std::vector<std::int32_t>>(),
        value.at("pmf_offsets").get<std::vector<std::int32_t>>(),
        value.at("pmf_table").get<std::vector<std::int32_t>>()};
}

GaussianPmf load_gaussian(const std::filesystem::path& model_dir)
{
    const auto value = load_json(model_dir / "gaussian_pmf.json");
    PmfTable table = parse_table(value);
    GaussianPmf result;
    static_cast<PmfTable&>(result) = std::move(table);
    result.scale_levels = value.at("scale_levels").get<int>();
    result.index_space = value.at("index_space").get<bool>();
    if (!result.index_space)
        throw std::runtime_error("MLVC AOT runtime requires index-space Gaussian PMFs");
    return result;
}

BitEstimatorPmf load_bit_estimator(const std::filesystem::path& model_dir)
{
    const auto value = load_json(model_dir / "bit_estimator_pmf.json");
    PmfTable table = parse_table(value);
    BitEstimatorPmf result;
    static_cast<PmfTable&>(result) = std::move(table);
    result.qp_num = value.at("qp_num").get<int>();
    result.channels = value.at("channels").get<int>();
    return result;
}

void check_error(const std::error_code& error, const char* operation)
{
    if (error)
        throw std::runtime_error(std::string("rANS ") + operation +
                                 " failed: " + error.message());
}

template <typename T>
msrtc_rans::span<const T> const_span(const std::vector<T>& values)
{
    return {values.data(), values.size()};
}

msrtc_rans::span<std::int32_t> mutable_span(std::vector<std::int32_t>& values)
{
    return {values.data(), values.size()};
}

std::vector<std::int32_t> integer_values(const Tensor& tensor)
{
    if (tensor.data_type() != TensorDataType::kFloat16)
        throw std::runtime_error("entropy input must be FP16");
    const auto& source = std::get<std::vector<Float16Storage>>(tensor.data);
    std::vector<std::int32_t> result(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        const float value = half_to_float(source[i]);
        if (!std::isfinite(value) || value < -32768.0F || value > 32767.0F)
            throw std::runtime_error("entropy symbol is outside signed 16-bit range");
        result[i] = static_cast<std::int32_t>(value);
    }
    return result;
}

Tensor fp16_tensor(std::string name, std::vector<int64_t> shape,
                   const std::vector<std::int32_t>& values)
{
    std::vector<Float16Storage> storage(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        storage[i] = float_to_half(static_cast<float>(values[i]));
    return Tensor{std::move(name), std::move(shape), std::move(storage)};
}

class CoderPair {
public:
    explicit CoderPair(const PmfTable& pmf)
    {
        check_error(encoder.Initialize(
            msrtc_rans::RansVariant::RansByte,
            const_span(pmf.lengths), const_span(pmf.offsets),
            const_span(pmf.table), 16, 2), "encoder initialization");
        check_error(decoder.Initialize(
            msrtc_rans::RansVariant::RansByte,
            const_span(pmf.lengths), const_span(pmf.offsets),
            const_span(pmf.table), 16, 2), "decoder initialization");
    }

    msrtc_rans::EntropyEncoder encoder;
    msrtc_rans::EntropyDecoder decoder;
};

}  // namespace

class EntropyCodec::Impl final {
public:
    Impl(const std::filesystem::path& model_dir, ModelConfig model_config)
        : config(std::move(model_config)),
          gaussian_pmf(load_gaussian(model_dir)),
          bit_estimator_pmf(load_bit_estimator(model_dir)),
          gaussian(gaussian_pmf), bit_estimator(bit_estimator_pmf)
    {
        if (gaussian_pmf.scale_levels <= 0 ||
            static_cast<int>(gaussian_pmf.lengths.size()) != gaussian_pmf.scale_levels)
            throw std::runtime_error("invalid Gaussian PMF dimensions");
        if (bit_estimator_pmf.qp_num != config.total_qp_num ||
            bit_estimator_pmf.channels != config.hyperprior_channels ||
            static_cast<int>(bit_estimator_pmf.lengths.size()) !=
                bit_estimator_pmf.qp_num * bit_estimator_pmf.channels) {
            throw std::runtime_error("bit-estimator PMF does not match model metadata");
        }
    }

    std::vector<std::int32_t> z_indices(int q_index) const
    {
        if (q_index < 0 || q_index >= bit_estimator_pmf.qp_num)
            throw std::runtime_error("q-index is outside the PMF range");
        const int height = config.hyperprior_height();
        const int width = config.hyperprior_width();
        std::vector<std::int32_t> result(
            static_cast<std::size_t>(config.hyperprior_channels) * height * width);
        for (int channel = 0; channel < config.hyperprior_channels; ++channel) {
            const std::int32_t index =
                q_index * bit_estimator_pmf.channels + channel;
            for (int i = 0; i < height * width; ++i)
                result[static_cast<std::size_t>(channel) * height * width + i] = index;
        }
        return result;
    }

    ModelConfig config;
    GaussianPmf gaussian_pmf;
    BitEstimatorPmf bit_estimator_pmf;
    CoderPair gaussian;
    CoderPair bit_estimator;
};

EntropyCodec::EntropyCodec(const std::filesystem::path& model_dir,
                           const ModelConfig& config)
    : impl_(std::make_unique<Impl>(model_dir, config))
{
}

EntropyCodec::~EntropyCodec() = default;
EntropyCodec::EntropyCodec(EntropyCodec&&) noexcept = default;
EntropyCodec& EntropyCodec::operator=(EntropyCodec&&) noexcept = default;

std::vector<std::byte> EntropyCodec::encode(
    const Tensor& y_raw_1, const Tensor& y_raw_0,
    const Tensor& z_raw, int q_index) const
{
    const auto scales = extract_scales(
        z_raw, impl_->config, impl_->gaussian_pmf.scale_levels - 1);
    const std::vector<std::int32_t> y1 = integer_values(y_raw_1);
    const std::vector<std::int32_t> y0 = integer_values(y_raw_0);
    const std::vector<std::int32_t> z = integer_values(z_raw);
    const std::vector<std::int32_t> z_indices = impl_->z_indices(q_index);
    if (y1.size() != scales.second.size() || y0.size() != scales.first.size() ||
        z.size() != z_indices.size())
        throw std::runtime_error("entropy tensor shape mismatch");

    msrtc_rans::HeapResizableBuffer buffer;
    msrtc_rans::RansEncoderStream stream;
    check_error(stream.Initialize(msrtc_rans::RansVariant::RansByte, buffer),
                "stream initialization");
    check_error(impl_->gaussian.encoder.Encode(
        stream, const_span(scales.second), const_span(y1)),
        "y_raw_1 encode");
    check_error(impl_->gaussian.encoder.Encode(
        stream, const_span(scales.first), const_span(y0)),
        "y_raw_0 encode");
    check_error(impl_->bit_estimator.encoder.Encode(
        stream, const_span(z_indices), const_span(z)), "z_raw encode");
    const auto encoded = stream.Flush();
    return {encoded.data(), encoded.data() + encoded.size()};
}

EntropyCodec::DecodedLatents EntropyCodec::decode(
    const std::vector<std::byte>& payload, int q_index) const
{
    msrtc_rans::RansDecoderStream stream;
    check_error(stream.Initialize(msrtc_rans::RansVariant::RansByte),
                "decoder stream initialization");
    check_error(stream.Open({payload.data(), payload.size()}), "stream open");

    const std::vector<std::int32_t> z_indices = impl_->z_indices(q_index);
    std::vector<std::int32_t> z_values(z_indices.size());
    check_error(impl_->bit_estimator.decoder.Decode(
        mutable_span(z_values), const_span(z_indices), stream), "z_raw decode");
    Tensor z = fp16_tensor("z_raw",
        {1, impl_->config.hyperprior_channels,
         impl_->config.hyperprior_height(), impl_->config.hyperprior_width()},
        z_values);

    const auto scales = extract_scales(
        z, impl_->config, impl_->gaussian_pmf.scale_levels - 1);
    std::vector<std::int32_t> y0_values(scales.first.size());
    std::vector<std::int32_t> y1_values(scales.second.size());
    check_error(impl_->gaussian.decoder.Decode(
        mutable_span(y0_values), const_span(scales.first), stream),
        "y_raw_0 decode");
    check_error(impl_->gaussian.decoder.Decode(
        mutable_span(y1_values), const_span(scales.second), stream),
        "y_raw_1 decode");
    if (!stream.CheckEOF())
        throw std::runtime_error("rANS payload has trailing or invalid data");
    stream.Close();

    const std::vector<int64_t> y_shape{
        1, impl_->config.latent_channels / 2,
        impl_->config.latent_height(), impl_->config.latent_width()};
    return DecodedLatents{
        std::move(z),
        fp16_tensor("y_raw_0", y_shape, y0_values),
        fp16_tensor("y_raw_1", y_shape, y1_values)};
}

}  // namespace mlvc
