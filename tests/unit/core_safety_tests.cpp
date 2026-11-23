#include "core/bitstream.hpp"
#include "core/model.hpp"
#include "core/tensor.hpp"
#include "core/yuv.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Operation>
void require_throws(Operation&& operation, const char* message)
{
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

mlvc::ModelConfig valid_model_config()
{
    mlvc::ModelConfig config;
    config.name = "test-model";
    config.model_version = "test-v1";
    config.model_width = 8;
    config.model_height = 8;
    config.pixel_range = 1.0F;
    config.qp_num = 64;
    config.total_qp_num = 72;
    config.frame_index_map = {0, 1};
    config.qp_shift = {0, 8};
    config.feature_channels = 8;
    config.latent_channels = 8;
    config.hyperprior_channels = 8;
    config.downsample_feature = 2;
    config.downsample_latent = 2;
    config.downsample_hyperprior = 2;
    config.y_scale_repeat = 2;
    config.iframe_period = 64;
    config.disable_feature_reset = true;
    return config;
}

void test_model_validation()
{
    mlvc::ModelConfig config = valid_model_config();
    config.validate();
    require(config.feature_height() == 4 && config.hyperprior_width() == 2,
            "valid model dimensions were computed incorrectly");

    config.downsample_feature = 0;
    require_throws([&] { config.validate(); },
                   "zero model downsample factor was accepted");
    require_throws([&] { config.feature_height(); },
                   "zero model downsample factor reached integer division");

    config = valid_model_config();
    config.frame_index_map = {2};
    require_throws([&] { config.validate(); },
                   "out-of-range QP mapping was accepted");

    config = valid_model_config();
    config.qp_shift = {0, 9};
    require_throws([&] { config.validate(); },
                   "QP shift outside the total range was accepted");
}

void test_tensor_views()
{
    const std::array<std::int64_t, 2> shape{2, 2};
    std::array<mlvc::Float16Storage, 4> values{};
    const mlvc::TensorView view("tensor", shape, values);
    require(view.element_count() == values.size() &&
                view.byte_size() == sizeof(values),
            "typed tensor view has incorrect storage metadata");

    require_throws(
        [&] {
            static_cast<void>(mlvc::TensorView::from_raw(
                "bad", shape, mlvc::TensorDataType::kFloat16,
                values.data(), sizeof(values) - 1));
        },
        "tensor view accepted mismatched byte storage");
    require_throws(
        [&] {
            static_cast<void>(mlvc::TensorView::from_raw(
                "bad-type", shape,
                static_cast<mlvc::TensorDataType>(99), values.data(),
                sizeof(values)));
        },
        "tensor view accepted an unsupported data type");

    mlvc::Tensor invalid{"bad", {2, 2},
                         std::vector<mlvc::Float16Storage>(3)};
    require_throws([&] { invalid.validate(); },
                   "owning tensor accepted mismatched shape and storage");
}

void test_yuv_views()
{
    constexpr int width = 4;
    constexpr int height = 2;
    std::array<std::uint8_t, 12> expected{};
    for (std::size_t index = 0; index < expected.size(); ++index)
        expected[index] = static_cast<std::uint8_t>(index * 17U);

    std::array<std::uint8_t, 12> storage{};
    mlvc::MutableYuv420FrameView frame(
        width, height, std::span(storage).first(8),
        std::span(storage).subspan(8, 2), std::span(storage).subspan(10, 2));
    const std::string bytes(
        reinterpret_cast<const char*>(expected.data()), expected.size());
    std::istringstream input(bytes, std::ios::binary);
    require(mlvc::read_yuv420_frame(input, frame),
            "cannot read into checked YUV view");
    require(storage == expected, "checked YUV view changed input bytes");

    require_throws(
        [&] {
            static_cast<void>(mlvc::MutableYuv420FrameView(
                width, height, std::span(storage).first(8),
                std::span(storage).subspan(8, 1),
                std::span(storage).subspan(10, 2)));
        },
        "YUV view accepted an undersized chroma plane");
}

void test_rotated_output_validation()
{
    mlvc::Yuv420Frame source{
        4, 6, std::vector<std::uint8_t>(24, 64),
        std::vector<std::uint8_t>(6, 128),
        std::vector<std::uint8_t>(6, 192)};
    mlvc::FramePadding padding;
    const mlvc::Tensor model_input =
        mlvc::prepare_model_input(source, 6, 4, 1.0F, padding);
    require(padding.rotated, "portrait frame did not select rotated layout");
    const mlvc::Yuv420Frame output =
        mlvc::prepare_model_output(model_input, 4, 6, 1.0F, padding);
    require(output.width == 4 && output.height == 6,
            "rotated output dimensions were not restored");

    mlvc::Tensor invalid{"x_hat", {1, 3, 2, 6},
                         std::vector<mlvc::Float16Storage>(36)};
    require_throws(
        [&] {
            static_cast<void>(
                mlvc::prepare_model_output(invalid, 4, 6, 1.0F, padding));
        },
        "rotated output accepted an invalid spatial shape");

    mlvc::FramePadding invalid_padding = padding;
    invalid_padding.left = -1;
    require_throws(
        [&] {
            static_cast<void>(mlvc::prepare_model_output(
                model_input, 4, 6, 1.0F, invalid_padding));
        },
        "model output accepted negative frame padding");
}

void test_bitstream_limit()
{
    mlvc::BitstreamEncoder encoder;
    mlvc::FrameData frame;
    frame.model_width = 64;
    frame.model_height = 64;
    frame.frame_type = mlvc::FrameType::kIFrame;
    frame.qp = 21;
    const std::vector<std::byte> payload(5, std::byte{0x42});
    frame.payload = payload;

    const auto access_unit = encoder.encode(frame);
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    mlvc::write_access_unit(stream, access_unit);
    stream.seekg(0);

    mlvc::AccessUnitReader reader(stream, 4);
    std::vector<std::byte> decoded{std::byte{0x11}, std::byte{0x22}};
    bool hit_limit = false;
    try {
        static_cast<void>(reader.read(decoded));
    } catch (const mlvc::BitstreamError& error) {
        hit_limit = (error.code() == mlvc::BitstreamErrorCode::kSizeLimitExceeded);
    }
    require(hit_limit, "oversized NALU access unit was accepted");
    require(decoded == (std::vector<std::byte>{std::byte{0x11}, std::byte{0x22}}),
            "failed access unit read modified the destination");

    mlvc::FrameData invalid_qp = frame;
    invalid_qp.qp = 52;
    require_throws([&] { static_cast<void>(encoder.encode(invalid_qp)); },
                   "QP above 51 was accepted");
    invalid_qp.qp = -1;
    require_throws([&] { static_cast<void>(encoder.encode(invalid_qp)); },
                   "negative QP was accepted");
}

}  // namespace

int main()
{
    try {
        test_model_validation();
        test_tensor_views();
        test_yuv_views();
        test_rotated_output_validation();
        test_bitstream_limit();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core safety test failed: " << error.what() << '\n';
        return 1;
    }
}
