#include "test_support.hpp"

#include "core/assets.hpp"
#include "core/bitstream.hpp"
#include "core/env.hpp"
#include "core/half.hpp"
#include "core/logging.hpp"
#include "core/model.hpp"
#include "core/scales.hpp"
#include "core/sha256.hpp"
#include "core/tensor.hpp"
#include "core/yuv.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
    MLVC_REQUIRE(config.feature_height() == 4 && config.hyperprior_width() == 2,
                 "valid model dimensions were computed incorrectly");

    config.downsample_feature = 0;
    MLVC_REQUIRE_THROWS(config.validate(),
                        "zero model downsample factor was accepted");
    MLVC_REQUIRE_THROWS(static_cast<void>(config.feature_height()),
                        "zero model downsample factor reached integer division");

    config = valid_model_config();
    config.frame_index_map = {2};
    MLVC_REQUIRE_THROWS(config.validate(),
                        "out-of-range QP mapping was accepted");

    config = valid_model_config();
    config.qp_shift = {0, 9};
    MLVC_REQUIRE_THROWS(config.validate(),
                        "QP shift outside the total range was accepted");
}

void test_tensor_views()
{
    const std::array<std::int64_t, 2> shape{2, 2};
    std::array<mlvc::Float16Storage, 4> values{};
    const mlvc::TensorView view("tensor", shape, values);
    MLVC_REQUIRE(view.element_count() == values.size() &&
                     view.byte_size() == sizeof(values),
                 "typed tensor view has incorrect storage metadata");

    MLVC_REQUIRE_THROWS(
        static_cast<void>(mlvc::TensorView::from_raw(
            "bad", shape, mlvc::TensorDataType::kFloat16,
            values.data(), sizeof(values) - 1)),
        "tensor view accepted mismatched byte storage");
    MLVC_REQUIRE_THROWS(
        static_cast<void>(mlvc::TensorView::from_raw(
            "bad-type", shape,
            static_cast<mlvc::TensorDataType>(99), values.data(),
            sizeof(values))),
        "tensor view accepted an unsupported data type");

    mlvc::Tensor invalid{"bad", {2, 2},
                         std::vector<mlvc::Float16Storage>(3)};
    MLVC_REQUIRE_THROWS(invalid.validate(),
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
    MLVC_REQUIRE(mlvc::read_yuv420_frame(input, frame),
                 "cannot read into checked YUV view");
    MLVC_REQUIRE(storage == expected, "checked YUV view changed input bytes");

    MLVC_REQUIRE_THROWS(
        static_cast<void>(mlvc::MutableYuv420FrameView(
            width, height, std::span(storage).first(8),
            std::span(storage).subspan(8, 1),
            std::span(storage).subspan(10, 2))),
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
    MLVC_REQUIRE(padding.rotated, "portrait frame did not select rotated layout");
    const mlvc::Yuv420Frame output =
        mlvc::prepare_model_output(model_input, 4, 6, 1.0F, padding);
    MLVC_REQUIRE(output.width == 4 && output.height == 6,
                 "rotated output dimensions were not restored");

    mlvc::Tensor invalid{"x_hat", {1, 3, 2, 6},
                         std::vector<mlvc::Float16Storage>(36)};
    MLVC_REQUIRE_THROWS(
        static_cast<void>(
            mlvc::prepare_model_output(invalid, 4, 6, 1.0F, padding)),
        "rotated output accepted an invalid spatial shape");

    mlvc::FramePadding invalid_padding = padding;
    invalid_padding.left = -1;
    MLVC_REQUIRE_THROWS(
        static_cast<void>(mlvc::prepare_model_output(
            model_input, 4, 6, 1.0F, invalid_padding)),
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
    MLVC_REQUIRE(hit_limit, "oversized NALU access unit was accepted");
    MLVC_REQUIRE(decoded == (std::vector<std::byte>{std::byte{0x11}, std::byte{0x22}}),
                 "failed access unit read modified the destination");

    mlvc::FrameData invalid_qp = frame;
    invalid_qp.qp = 52;
    MLVC_REQUIRE_THROWS(static_cast<void>(encoder.encode(invalid_qp)),
                        "QP above 51 was accepted");
    invalid_qp.qp = -1;
    MLVC_REQUIRE_THROWS(static_cast<void>(encoder.encode(invalid_qp)),
                        "negative QP was accepted");
}

void test_strided_yuv_and_nan_safety()
{
    constexpr int width = 4;
    constexpr int height = 4;
    constexpr std::size_t y_stride = 8;
    constexpr std::size_t uv_stride = 4;

    std::vector<std::uint8_t> y_buf(height * y_stride, 0xCC);
    std::vector<std::uint8_t> u_buf((height / 2) * uv_stride, 0xDD);
    std::vector<std::uint8_t> v_buf((height / 2) * uv_stride, 0xEE);

    mlvc::Yuv420Frame packed{
        width, height,
        std::vector<std::uint8_t>(width * height),
        std::vector<std::uint8_t>((width / 2) * (height / 2)),
        std::vector<std::uint8_t>((width / 2) * (height / 2))};
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const auto val = static_cast<std::uint8_t>(row * 40 + col * 10);
            packed.y[static_cast<std::size_t>(row) * width + col] = val;
            y_buf[static_cast<std::size_t>(row) * y_stride + col] = val;
        }
    }
    for (int row = 0; row < height / 2; ++row) {
        for (int col = 0; col < width / 2; ++col) {
            const auto uval = static_cast<std::uint8_t>(80 + row * 10 + col);
            const auto vval = static_cast<std::uint8_t>(160 + row * 10 + col);
            packed.u[static_cast<std::size_t>(row) * (width / 2) + col] = uval;
            packed.v[static_cast<std::size_t>(row) * (width / 2) + col] = vval;
            u_buf[static_cast<std::size_t>(row) * uv_stride + col] = uval;
            v_buf[static_cast<std::size_t>(row) * uv_stride + col] = vval;
        }
    }

    const mlvc::Yuv420FrameView strided_view(
        width, height, y_buf, y_stride, u_buf, uv_stride, v_buf, uv_stride);
    MLVC_REQUIRE(!strided_view.is_tightly_packed(),
                 "strided view must not report tightly packed");

    mlvc::FramePadding pad_packed{};
    mlvc::FramePadding pad_strided{};
    const mlvc::Tensor t_packed = mlvc::prepare_model_input(packed, 6, 6, 1.0F, pad_packed);
    const mlvc::Tensor t_strided = mlvc::prepare_model_input(strided_view, 6, 6, 1.0F, pad_strided);
    MLVC_REQUIRE(std::get<std::vector<mlvc::Float16Storage>>(t_packed.data) ==
                     std::get<std::vector<mlvc::Float16Storage>>(t_strided.data),
                 "strided prepare_model_input must match tightly packed input");

    // Verify strided output preserves padding bytes untouched and matches packed output.
    std::vector<std::uint8_t> out_y(height * y_stride, 0xAA);
    std::vector<std::uint8_t> out_u((height / 2) * uv_stride, 0xBB);
    std::vector<std::uint8_t> out_v((height / 2) * uv_stride, 0xCC);
    mlvc::MutableYuv420FrameView out_strided(
        width, height, out_y, y_stride, out_u, uv_stride, out_v, uv_stride);
    mlvc::prepare_model_output_into(t_strided, out_strided, 1.0F, pad_strided);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            MLVC_REQUIRE(out_y[static_cast<std::size_t>(row) * y_stride + col] ==
                             packed.y[static_cast<std::size_t>(row) * width + col],
                         "strided output Y mismatch");
        }
        for (std::size_t col = width; col < y_stride; ++col) {
            MLVC_REQUIRE(out_y[static_cast<std::size_t>(row) * y_stride + col] == 0xAA,
                         "strided output clobbered padding bytes");
        }
    }

    // Invalid stride must be rejected.
    MLVC_REQUIRE_THROWS(
        static_cast<void>(mlvc::Yuv420FrameView(
            width, height, y_buf, width - 1, u_buf, uv_stride, v_buf, uv_stride)),
        "stride smaller than width was accepted");

    // NaN and +/-Inf values in x_hat must not cause UB and must clamp safely.
    std::vector<mlvc::Float16Storage> extreme(3U * 4U * 4U, mlvc::float_to_half(0.5F));
    extreme[0] = 0x7e00U;  // qNaN
    extreme[1] = 0x7c00U;  // +Inf
    extreme[2] = 0xfc00U;  // -Inf
    mlvc::Tensor nan_tensor{"x_hat", {1, 3, 4, 4}, std::move(extreme)};
    const mlvc::Yuv420Frame clamped =
        mlvc::prepare_model_output(nan_tensor, 4, 4, 1.0F, mlvc::FramePadding{});
    MLVC_REQUIRE(clamped.y[0] == 0 && clamped.y[1] == 255 && clamped.y[2] == 0,
                 "NaN/+Inf/-Inf model output did not quantize safely");
}

void test_scales_and_env_parsing()
{
    const mlvc::ModelConfig config = valid_model_config();
    const std::size_t z_elems =
        static_cast<std::size_t>(config.hyperprior_channels) *
        static_cast<std::size_t>(config.hyperprior_height()) *
        static_cast<std::size_t>(config.hyperprior_width());
    std::vector<mlvc::Float16Storage> z_vals(z_elems, mlvc::float_to_half(3.5F));
    z_vals[0] = 0x7e00U;  // qNaN -> 0
    z_vals[1] = 0x7c00U;  // +Inf -> scale_max_index
    z_vals[2] = 0xfc00U;  // -Inf -> abs -> +Inf -> scale_max_index
    z_vals[3] = mlvc::float_to_half(-5.0F);  // abs -> 5
    const mlvc::Tensor z_raw{
        "z_raw",
        {1, config.hyperprior_channels,
         config.hyperprior_height(), config.hyperprior_width()},
        std::move(z_vals)};

    constexpr int kMaxScale = 63;
    const auto [s0, s1] = mlvc::extract_scales(z_raw, config, kMaxScale);
    MLVC_REQUIRE(!s0.empty() && s0.size() == s1.size(),
                 "extract_scales produced empty outputs");
    // Channel 0 maps to base channel 0 (z_vals[0..3]): (y=0,x=0)->z[0]=NaN->0,
    // (y=0,x=2)->z[1]=+Inf->63, (y=2,x=0)->z[2]=-Inf->63, (y=2,x=2)->z[3]=-5->5.
    const int yw = config.latent_width();
    MLVC_REQUIRE(s0[0] == 0, "NaN scale should quantize to 0");
    MLVC_REQUIRE(s0[2] == kMaxScale, "+Inf scale should saturate to scale_max_index");
    MLVC_REQUIRE(s0[static_cast<std::size_t>(2 * yw)] == kMaxScale,
                 "-Inf scale should saturate to scale_max_index");
    MLVC_REQUIRE(s0[static_cast<std::size_t>(2 * yw + 2)] == 5,
                 "negative finite scale should take abs and truncate");

    MLVC_REQUIRE_THROWS(
        static_cast<void>(mlvc::extract_scales(z_raw, config, -1)),
        "negative scale_max_index must be rejected");

    // Safe environment variable integer parsing.
    {
        mlvc::test::ScopedEnvVar guard("MLVC_TEST_PARSE_INT", "42");
        MLVC_REQUIRE(mlvc::detail::parse_env_int("MLVC_TEST_PARSE_INT", 7, 0, 100) == 42,
                     "valid env int should parse");
    }
    {
        mlvc::test::ScopedEnvVar guard("MLVC_TEST_PARSE_INT", "999999999999999999999");
        MLVC_REQUIRE(mlvc::detail::parse_env_int("MLVC_TEST_PARSE_INT", 7, 0, 100) == 7,
                     "overflowing env int must fall back to default");
    }
    {
        mlvc::test::ScopedEnvVar guard("MLVC_TEST_PARSE_INT", "12abc");
        MLVC_REQUIRE(mlvc::detail::parse_env_int("MLVC_TEST_PARSE_INT", 7, 0, 100) == 7,
                     "garbage suffix in env int must fall back to default");
    }
    {
        mlvc::test::ScopedEnvVar guard("MLVC_TEST_PARSE_INT", "-5");
        MLVC_REQUIRE(mlvc::detail::parse_env_int("MLVC_TEST_PARSE_INT", 7, 0, 100) == 0,
                     "below-min env int must clamp to min");
    }
}

void test_sha256_assets_and_logging()
{
    // NIST SHA-256 test vector for "abc":
    // ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    const std::string abc = "abc";
    const auto abc_span = std::as_bytes(std::span<const char>{abc.data(), abc.size()});
    const mlvc::Sha256Digest digest = mlvc::sha256_digest(abc_span);
    const std::array<std::uint8_t, 32> expected_abc = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    const std::string expected_str(
        reinterpret_cast<const char*>(expected_abc.data()), expected_abc.size());
    MLVC_REQUIRE(mlvc::sha256_equals(digest, expected_str), "SHA-256('abc') mismatch");

    // DirectoryAssetReader path traversal rejection and zero-copy mmap read_view.
    {
        mlvc::test::ScopedTempDir temp("mlvc-assets-test");
        const auto large_path = temp.path() / "large.bin";
        std::vector<std::byte> payload(128U * 1024U, std::byte{0x5A});
        payload.front() = std::byte{0x12};
        payload.back() = std::byte{0x34};
        {
            std::ofstream out(large_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(payload.data()),
                      static_cast<std::streamsize>(payload.size()));
            MLVC_REQUIRE(static_cast<bool>(out), "failed to write temp asset");
        }
        const auto reader = mlvc::make_directory_asset_reader(temp.path());
        MLVC_REQUIRE(reader->exists("large.bin"), "existing asset not found");
        const mlvc::AssetView view = reader->read_view("large.bin");
        MLVC_REQUIRE(view.size() == payload.size() &&
                         view.data()[0] == std::byte{0x12} &&
                         view.data()[payload.size() - 1U] == std::byte{0x34},
                     "read_view() content mismatch");
        MLVC_REQUIRE_THROWS(static_cast<void>(reader->read_view("../escape.bin")),
                            "path traversal was accepted by DirectoryAssetReader");
        MLVC_REQUIRE_THROWS(static_cast<void>(reader->exists("/etc/passwd")),
                            "absolute path was accepted by DirectoryAssetReader");
    }

    // Logging callback delivery and filtering.
    struct LogCapture {
        std::atomic<int> count{0};
        std::string last;
    } capture;
    auto cb = [](mlvc::LogLevel, const char*, const char* msg, void* ud) noexcept {
        auto* c = static_cast<LogCapture*>(ud);
        c->count.fetch_add(1, std::memory_order_relaxed);
        c->last = msg ? msg : "";
    };
    mlvc::set_log_callback(cb, &capture, mlvc::LogLevel::kWarn);
    mlvc::log_message(mlvc::LogLevel::kInfo, "test", "filtered-out");
    MLVC_REQUIRE(capture.count.load() == 0, "info log should be filtered at kWarn");
    mlvc::log_message(mlvc::LogLevel::kWarn, "test", "delivered-warning");
    MLVC_REQUIRE(capture.count.load() == 1 && capture.last == "delivered-warning",
                 "warning log should be delivered");
    mlvc::set_log_callback(nullptr);
    mlvc::log_message(mlvc::LogLevel::kError, "test", "disabled");
    MLVC_REQUIRE(capture.count.load() == 1, "disabled logger should not invoke callback");
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
        test_strided_yuv_and_nan_safety();
        test_scales_and_env_parsing();
        test_sha256_assets_and_logging();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core safety test failed: " << error.what() << '\n';
        return 1;
    }
}
