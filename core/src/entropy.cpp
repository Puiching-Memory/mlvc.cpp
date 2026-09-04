#include "mlvc/core/entropy.hpp"

#include "mlvc/core/half.hpp"
#include "mlvc/core/scales.hpp"

#include "model_assets.hpp"

#include <msrtc_rans/EntropyCoder.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
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

nlohmann::json load_json(const std::filesystem::path& model_dir,
                         const std::filesystem::path& relative_path)
{
    return nlohmann::json::parse(
        detail::read_model_text(model_dir, relative_path));
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
    const auto value = load_json(model_dir, "gaussian_pmf.json");
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
    const auto value = load_json(model_dir, "bit_estimator_pmf.json");
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

bool shape_equals(std::span<const std::int64_t> actual,
                  const std::vector<std::int64_t>& expected)
{
    return actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin());
}

__attribute__((target("f16c,avx2")))
void integer_values_f16c(const Float16Storage* source, std::int32_t* result,
                         std::size_t count)
{
    const __m256 lower = _mm256_set1_ps(-32768.0F);
    const __m256 upper = _mm256_set1_ps(32767.0F);
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(source + i));
        const __m256 values = _mm256_cvtph_ps(packed);
        const __m256 ordered = _mm256_cmp_ps(values, values, _CMP_ORD_Q);
        const __m256 above = _mm256_cmp_ps(values, lower, _CMP_GE_OQ);
        const __m256 below = _mm256_cmp_ps(values, upper, _CMP_LE_OQ);
        if (_mm256_movemask_ps(_mm256_and_ps(
                ordered, _mm256_and_ps(above, below))) != 0xff) {
            throw std::runtime_error(
                "entropy symbol is outside signed 16-bit range");
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i),
                            _mm256_cvttps_epi32(values));
    }
    for (; i < count; ++i) {
        const float value = half_to_float(source[i]);
        if (!std::isfinite(value) || value < -32768.0F || value > 32767.0F)
            throw std::runtime_error(
                "entropy symbol is outside signed 16-bit range");
        result[i] = static_cast<std::int32_t>(value);
    }
}

void integer_values(TensorView tensor, std::vector<std::int32_t>& result)
{
    if (tensor.data_type() != TensorDataType::kFloat16)
        throw std::runtime_error("entropy input must be FP16");
    const std::span<const Float16Storage> source = tensor.fp16_data();
    result.resize(source.size());
    if (__builtin_cpu_supports("f16c") && __builtin_cpu_supports("avx2")) {
        integer_values_f16c(source.data(), result.data(), source.size());
        return;
    }
    for (std::size_t i = 0; i < source.size(); ++i) {
        const float value = half_to_float(source[i]);
        if (!std::isfinite(value) || value < -32768.0F || value > 32767.0F)
            throw std::runtime_error(
                "entropy symbol is outside signed 16-bit range");
        result[i] = static_cast<std::int32_t>(value);
    }
}

__attribute__((target("f16c,avx2")))
void fp16_values_f16c(const std::int32_t* values, Float16Storage* storage,
                      std::size_t count)
{
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256i integers = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(values + i));
        const __m128i packed = _mm256_cvtps_ph(
            _mm256_cvtepi32_ps(integers),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(storage + i), packed);
    }
    for (; i < count; ++i)
        storage[i] = float_to_half(static_cast<float>(values[i]));
}

void fp16_values(const std::vector<std::int32_t>& values,
                 MutableTensorView destination)
{
    if (destination.data_type() != TensorDataType::kFloat16 ||
        destination.element_count() != values.size()) {
        throw std::runtime_error("entropy output must be matching FP16 storage");
    }
    const std::span<Float16Storage> storage = destination.fp16_data();
    if (__builtin_cpu_supports("f16c") && __builtin_cpu_supports("avx2")) {
        fp16_values_f16c(values.data(), storage.data(), values.size());
        return;
    }
    for (std::size_t i = 0; i < values.size(); ++i)
        storage[i] = float_to_half(static_cast<float>(values[i]));
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
          gaussian(gaussian_pmf), bit_estimator(bit_estimator_pmf),
          encode_buffer(1U << 20)
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
        z_indices_by_q.reserve(static_cast<std::size_t>(bit_estimator_pmf.qp_num));
        for (int q_index = 0; q_index < bit_estimator_pmf.qp_num; ++q_index)
            z_indices_by_q.push_back(make_z_indices(q_index));
        check_error(encode_stream.Initialize(
            msrtc_rans::RansVariant::RansByte, encode_buffer),
            "stream initialization");
        check_error(decode_stream.Initialize(msrtc_rans::RansVariant::RansByte),
                    "decoder stream initialization");
    }

    std::vector<std::int32_t> make_z_indices(int q_index) const
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

    const std::vector<std::int32_t>& z_indices(int q_index) const
    {
        if (q_index < 0 || q_index >= bit_estimator_pmf.qp_num)
            throw std::runtime_error("q-index is outside the PMF range");
        return z_indices_by_q[static_cast<std::size_t>(q_index)];
    }

    ModelConfig config;
    GaussianPmf gaussian_pmf;
    BitEstimatorPmf bit_estimator_pmf;
    CoderPair gaussian;
    CoderPair bit_estimator;
    std::vector<std::vector<std::int32_t>> z_indices_by_q;
    mutable std::vector<std::int32_t> scales_0;
    mutable std::vector<std::int32_t> scales_1;
    mutable std::vector<std::int32_t> y0_values;
    mutable std::vector<std::int32_t> y1_values;
    mutable std::vector<std::int32_t> z_values;
    mutable msrtc_rans::HeapResizableBuffer encode_buffer;
    mutable msrtc_rans::RansEncoderStream encode_stream;
    mutable msrtc_rans::RansDecoderStream decode_stream;
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
    return encode(tensor_view(y_raw_1), tensor_view(y_raw_0),
                  tensor_view(z_raw), q_index);
}

std::vector<std::byte> EntropyCodec::encode(
    TensorView y_raw_1, TensorView y_raw_0, TensorView z_raw,
    int q_index) const
{
    extract_scales(z_raw, impl_->config,
                   impl_->gaussian_pmf.scale_levels - 1,
                   impl_->scales_0, impl_->scales_1);
    integer_values(y_raw_1, impl_->y1_values);
    integer_values(y_raw_0, impl_->y0_values);
    integer_values(z_raw, impl_->z_values);
    const auto& z_indices = impl_->z_indices(q_index);
    if (impl_->y1_values.size() != impl_->scales_1.size() ||
        impl_->y0_values.size() != impl_->scales_0.size() ||
        impl_->z_values.size() != z_indices.size()) {
        throw std::runtime_error("entropy tensor shape mismatch");
    }

    check_error(impl_->gaussian.encoder.Encode(
        impl_->encode_stream, const_span(impl_->scales_1),
        const_span(impl_->y1_values)),
        "y_raw_1 encode");
    check_error(impl_->gaussian.encoder.Encode(
        impl_->encode_stream, const_span(impl_->scales_0),
        const_span(impl_->y0_values)),
        "y_raw_0 encode");
    check_error(impl_->bit_estimator.encoder.Encode(
        impl_->encode_stream, const_span(z_indices),
        const_span(impl_->z_values)), "z_raw encode");
    const auto encoded = impl_->encode_stream.Flush();
    return {encoded.data(), encoded.data() + encoded.size()};
}

EntropyCodec::DecodedLatents EntropyCodec::decode(
    const std::vector<std::byte>& payload, int q_index) const
{
    Tensor z{"z_raw",
        {1, impl_->config.hyperprior_channels,
         impl_->config.hyperprior_height(), impl_->config.hyperprior_width()},
        std::vector<Float16Storage>(
            static_cast<std::size_t>(impl_->config.hyperprior_channels) *
            impl_->config.hyperprior_height() *
            impl_->config.hyperprior_width())};
    const std::vector<int64_t> y_shape{
        1, impl_->config.latent_channels / 2,
        impl_->config.latent_height(), impl_->config.latent_width()};
    Tensor y0{"y_raw_0", y_shape, std::vector<Float16Storage>(
        static_cast<std::size_t>(impl_->config.latent_channels / 2) *
        impl_->config.latent_height() * impl_->config.latent_width())};
    Tensor y1{"y_raw_1", y_shape, std::vector<Float16Storage>(
        static_cast<std::size_t>(impl_->config.latent_channels / 2) *
        impl_->config.latent_height() * impl_->config.latent_width())};
    decode_into(payload, q_index, mutable_tensor_view(z),
                mutable_tensor_view(y0), mutable_tensor_view(y1));
    return DecodedLatents{std::move(z), std::move(y0), std::move(y1)};
}

void EntropyCodec::decode_into(
    const std::vector<std::byte>& payload, int q_index,
    MutableTensorView z_raw, MutableTensorView y_raw_0,
    MutableTensorView y_raw_1) const
{
    const std::vector<std::int64_t> expected_z{
        1, impl_->config.hyperprior_channels,
        impl_->config.hyperprior_height(), impl_->config.hyperprior_width()};
    const std::vector<std::int64_t> expected_y{
        1, impl_->config.latent_channels / 2,
        impl_->config.latent_height(), impl_->config.latent_width()};
    if (!shape_equals(z_raw.shape(), expected_z) ||
        !shape_equals(y_raw_0.shape(), expected_y) ||
        !shape_equals(y_raw_1.shape(), expected_y)) {
        throw std::runtime_error("entropy output tensor shape mismatch");
    }

    check_error(impl_->decode_stream.Open({payload.data(), payload.size()}),
                "stream open");
    const auto& z_indices = impl_->z_indices(q_index);
    impl_->z_values.resize(z_indices.size());
    check_error(impl_->bit_estimator.decoder.Decode(
        mutable_span(impl_->z_values), const_span(z_indices),
        impl_->decode_stream), "z_raw decode");
    fp16_values(impl_->z_values, z_raw);

    extract_scales(static_cast<TensorView>(z_raw), impl_->config,
                   impl_->gaussian_pmf.scale_levels - 1,
                   impl_->scales_0, impl_->scales_1);
    impl_->y0_values.resize(impl_->scales_0.size());
    impl_->y1_values.resize(impl_->scales_1.size());
    check_error(impl_->gaussian.decoder.Decode(
        mutable_span(impl_->y0_values), const_span(impl_->scales_0),
        impl_->decode_stream), "y_raw_0 decode");
    check_error(impl_->gaussian.decoder.Decode(
        mutable_span(impl_->y1_values), const_span(impl_->scales_1),
        impl_->decode_stream), "y_raw_1 decode");
    if (!impl_->decode_stream.CheckEOF())
        throw std::runtime_error("rANS payload has trailing or invalid data");
    impl_->decode_stream.Close();
    fp16_values(impl_->y0_values, y_raw_0);
    fp16_values(impl_->y1_values, y_raw_1);
}

}  // namespace mlvc
