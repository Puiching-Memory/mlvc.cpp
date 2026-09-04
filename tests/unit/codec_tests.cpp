#include "mlvc/core/bitstream.hpp"
#include "mlvc/core/entropy.hpp"
#include "mlvc/core/half.hpp"
#include "mlvc/core/model.hpp"
#include "mlvc/core/yuv.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<mlvc::Float16Storage> read_fp16(const std::filesystem::path& path)
{
    const auto bytes = std::filesystem::file_size(path);
    if (bytes % sizeof(mlvc::Float16Storage) != 0)
        throw std::runtime_error("invalid FP16 fixture size: " + path.string());
    std::vector<mlvc::Float16Storage> result(bytes / sizeof(mlvc::Float16Storage));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(bytes));
    if (!input)
        throw std::runtime_error("cannot read fixture: " + path.string());
    return result;
}

std::vector<std::byte> read_bytes(const std::filesystem::path& path)
{
    const auto size = std::filesystem::file_size(path);
    std::vector<std::byte> result(size);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(size));
    if (!input)
        throw std::runtime_error("cannot read fixture: " + path.string());
    return result;
}

std::vector<std::byte> from_hex(const std::string& hex)
{
    if (hex.size() % 2 != 0)
        throw std::runtime_error("invalid hex fixture");
    std::vector<std::byte> result(hex.size() / 2);
    auto nibble = [](char value) -> unsigned {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        throw std::runtime_error("invalid hex fixture");
    };
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = static_cast<std::byte>((nibble(hex[i * 2]) << 4) |
                                           nibble(hex[i * 2 + 1]));
    return result;
}

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

bool fp16_numerically_equal(const mlvc::Tensor& lhs, const mlvc::Tensor& rhs)
{
    const auto& a = std::get<std::vector<mlvc::Float16Storage>>(lhs.data);
    const auto& b = std::get<std::vector<mlvc::Float16Storage>>(rhs.data);
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (mlvc::half_to_float(a[i]) != mlvc::half_to_float(b[i]))
            return false;
    }
    return true;
}

void require_yuv_matches(const mlvc::Yuv420Frame& frame,
                         const std::filesystem::path& reference_path,
                         int maximum_delta = 1)
{
    const std::vector<std::byte> expected = read_bytes(reference_path);
    const std::size_t frame_bytes = mlvc::yuv420_frame_bytes(frame.width, frame.height);
    require(expected.size() >= frame_bytes, "reference YUV is truncated");
    std::size_t offset = 0;
    auto compare_plane = [&](const std::vector<std::uint8_t>& plane) {
        for (std::uint8_t value : plane) {
            const int reference = static_cast<std::uint8_t>(expected[offset]);
            require(std::abs(reference - static_cast<int>(value)) <= maximum_delta,
                    "YUV postprocessing differs from Python reference");
            ++offset;
        }
    };
    compare_plane(frame.y);
    compare_plane(frame.u);
    compare_plane(frame.v);
}

void test_half_conversion()
{
    for (std::uint32_t bits = 0; bits <= 0xffffU; ++bits) {
        const auto half = static_cast<mlvc::Float16Storage>(bits);
        const bool nan = (half & 0x7c00U) == 0x7c00U && (half & 0x03ffU) != 0;
        if (!nan)
            require(mlvc::float_to_half(mlvc::half_to_float(half)) == half,
                    "FP16 round trip mismatch");
    }
}

void test_yuv_views()
{
    constexpr int width = 4;
    constexpr int height = 2;
    std::array<std::uint8_t, 12> expected{};
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = static_cast<std::uint8_t>(i * 17U);

    const std::string input_bytes(
        reinterpret_cast<const char*>(expected.data()), expected.size());
    std::istringstream input(input_bytes, std::ios::binary);
    std::array<std::uint8_t, 12> storage{};
    mlvc::MutableYuv420FrameView frame{
        width, height, storage.data(), storage.data() + 8,
        storage.data() + 10};
    require(mlvc::read_yuv420_frame(input, frame),
            "cannot read into non-owning YUV view");
    require(storage == expected, "non-owning YUV read differs from source");

    std::ostringstream output(std::ios::binary);
    mlvc::write_yuv420_frame(output, static_cast<mlvc::Yuv420FrameView>(frame));
    require(output.str() == input_bytes,
            "non-owning YUV write differs from source");
}

void test_mlvc_codec(const std::filesystem::path& assets)
{
    auto model_dir = assets.parent_path() / "canonical/mlvc-psnr-v1/640x368";
    if (!std::filesystem::exists(model_dir / "metadata.json"))
        model_dir = assets.parent_path() / "canonical/mlvc-psnr-v1/640x368";
    const auto encoder_case = assets / "benchmarks/gray-q21/encoder";
    const auto decoder_case = assets / "benchmarks/gray-q21/decoder";
    const mlvc::ModelConfig config = mlvc::load_model_config(model_dir);
    require(config.model_version == "dmc61sbr_reglu", "unexpected MLVC profile");
    require(config.feature_channels == 256 && config.latent_channels == 128 &&
            config.hyperprior_channels == 128, "unexpected MLVC dimensions");
    require(config.frame_in_gop(0) == 0 && config.frame_in_gop(1) == 1 &&
            config.frame_in_gop(63) == 63 && config.frame_in_gop(64) == 0 &&
            config.frame_in_gop(65) == 1,
            "MLVC GOP/I-frame reset schedule mismatch");
    require(config.q_index_shift(config.frame_in_gop(0)) == 8 &&
            config.q_index_shift(config.frame_in_gop(1)) == 0 &&
            config.q_index_shift(config.frame_in_gop(64)) == 8,
            "MLVC GOP QP-shift schedule mismatch");

    std::ifstream yuv(assets / "input/gray_640x360_1f.yuv",
                      std::ios::binary);
    mlvc::Yuv420Frame frame;
    require(mlvc::read_yuv420_frame(yuv, 640, 360, frame),
            "cannot read YUV fixture");
    mlvc::FramePadding padding;
    const mlvc::Tensor input = mlvc::prepare_model_input(
        frame, config.model_width, config.model_height, config.pixel_range, padding);
    require(std::get<std::vector<mlvc::Float16Storage>>(input.data) ==
                read_fp16(encoder_case / "input-00-x.fp16.raw"),
            "YUV preprocessing differs from Python reference");

    mlvc::Tensor z{"z_raw", {1, 128, 3, 5},
        read_fp16(encoder_case / "expected-01-z_raw.fp16.raw")};
    mlvc::Tensor y0{"y_raw_0", {1, 64, 23, 40},
        read_fp16(encoder_case / "expected-02-y_raw_0.fp16.raw")};
    mlvc::Tensor y1{"y_raw_1", {1, 64, 23, 40},
        read_fp16(encoder_case / "expected-03-y_raw_1.fp16.raw")};
    mlvc::EntropyCodec entropy(model_dir, config);
    const std::vector<std::byte> payload = entropy.encode(y1, y0, z, 21);
    const std::vector<std::byte> expected = from_hex(
        "b029a10685248f46194353d1af7ad64f1c8c6aa504a0156148068bcd8ad73f05"
        "aa7959fe1bbe9d13b9af863171d4b0385fcd9279a6971d41cf07db7e41cc219e"
        "b0385fcc394c419e11d09d8fbc10a8cdef1918225e1917b54acd2c80f60d98f"
        "0c8f6b1a52bb08336ea6118");
    require(payload == expected, "rANS payload differs from Python reference");

    auto decoded = entropy.decode(payload, 21);
    require(fp16_numerically_equal(decoded.z_raw, z),
            "z rANS round trip mismatch");
    require(fp16_numerically_equal(decoded.y_raw_0, y0),
            "y0 rANS round trip mismatch");
    require(fp16_numerically_equal(decoded.y_raw_1, y1),
            "y1 rANS round trip mismatch");

    const std::array<std::int64_t, 4> z_shape{1, 128, 3, 5};
    const std::array<std::int64_t, 4> y_shape{1, 64, 23, 40};
    std::vector<mlvc::Float16Storage> direct_z(z.element_count());
    std::vector<mlvc::Float16Storage> direct_y0(y0.element_count());
    std::vector<mlvc::Float16Storage> direct_y1(y1.element_count());
    entropy.decode_into(
        payload, 21,
        mlvc::MutableTensorView{"z_raw", z_shape,
            mlvc::TensorDataType::kFloat16, direct_z.data(),
            direct_z.size() * sizeof(direct_z.front())},
        mlvc::MutableTensorView{"y_raw_0", y_shape,
            mlvc::TensorDataType::kFloat16, direct_y0.data(),
            direct_y0.size() * sizeof(direct_y0.front())},
        mlvc::MutableTensorView{"y_raw_1", y_shape,
            mlvc::TensorDataType::kFloat16, direct_y1.data(),
            direct_y1.size() * sizeof(direct_y1.front())});
    require(direct_z == std::get<std::vector<mlvc::Float16Storage>>(
                            decoded.z_raw.data) &&
            direct_y0 == std::get<std::vector<mlvc::Float16Storage>>(
                             decoded.y_raw_0.data) &&
            direct_y1 == std::get<std::vector<mlvc::Float16Storage>>(
                             decoded.y_raw_1.data),
            "decode_into differs from owning entropy decode");

    mlvc::Tensor x_hat{"x_hat", {1, 3, 368, 640},
        read_fp16(decoder_case / "expected-00-x_hat.fp16.raw")};
    const auto reconstructed = mlvc::prepare_model_output(
        x_hat, 640, 360, config.pixel_range, padding);
    require(reconstructed.y.size() == 640U * 360U &&
            reconstructed.u.size() == 320U * 180U &&
            reconstructed.v.size() == 320U * 180U,
            "YUV postprocessing produced invalid planes");
    const auto reference_yuv = assets /
        "references/mlvc-psnr-v1/gray-q21-2f/reference.yuv";
    if (std::filesystem::exists(reference_yuv)) {
        mlvc::Tensor reference_x_hat{"x_hat", {1, 3, 368, 640},
            read_fp16(assets / "references/mlvc-psnr-v1/gray-q21-2f/"
                             "decoder/frame-000000/output-00-x_hat.fp16.raw")};
        require_yuv_matches(mlvc::prepare_model_output(
            reference_x_hat, 640, 360, config.pixel_range, padding), reference_yuv);
    }

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    mlvc::write_encoded_frame(stream, mlvc::EncodedFrame{21, payload});
    stream.seekg(0);
    mlvc::EncodedFrame restored;
    require(mlvc::read_encoded_frame(stream, restored), "cannot read frame container");
    require(restored.q_index == 21 && restored.payload == payload,
            "frame container round trip mismatch");
}

void test_mlvc_s_codec(const std::filesystem::path& assets)
{
    auto model_dir = assets.parent_path() / "canonical/mlvc-s-psnr-v1/640x368";
    if (!std::filesystem::exists(model_dir / "metadata.json"))
        model_dir = assets.parent_path() / "canonical/mlvc-s-psnr-v1/640x368";
    if (!std::filesystem::exists(model_dir / "metadata.json"))
        return;
    const mlvc::ModelConfig config = mlvc::load_model_config(model_dir);
    require(config.model_version == "dmc61sbr_mini_reglu", "unexpected MLVC-S profile");
    require(config.feature_channels == 96 && config.latent_channels == 48 &&
            config.hyperprior_channels == 48 && config.y_scale_repeat == 4,
            "unexpected MLVC-S dimensions");

    const auto encoder_case = assets / "benchmarks/mlvc-s-gray-q21/encoder";
    std::ifstream yuv(assets / "input/gray_640x360_1f.yuv",
                      std::ios::binary);
    mlvc::Yuv420Frame frame;
    require(mlvc::read_yuv420_frame(yuv, 640, 360, frame),
            "cannot read MLVC-S YUV fixture");
    mlvc::FramePadding padding;
    const mlvc::Tensor input = mlvc::prepare_model_input(
        frame, config.model_width, config.model_height, config.pixel_range, padding);
    require(std::get<std::vector<mlvc::Float16Storage>>(input.data) ==
                read_fp16(encoder_case / "input-00-x.fp16.raw"),
            "MLVC-S preprocessing differs from Python reference");

    mlvc::Tensor z{"z_raw", {1, 48, 6, 10},
        read_fp16(encoder_case / "expected-01-z_raw.fp16.raw")};
    mlvc::Tensor y0{"y_raw_0", {1, 24, 23, 40},
        read_fp16(encoder_case / "expected-02-y_raw_0.fp16.raw")};
    mlvc::Tensor y1{"y_raw_1", {1, 24, 23, 40},
        read_fp16(encoder_case / "expected-03-y_raw_1.fp16.raw")};
    mlvc::EntropyCodec entropy(model_dir, config);
    const std::vector<std::byte> payload = entropy.encode(y1, y0, z, 21);
    const std::vector<std::byte> expected = from_hex(
        "299e46143dcc973fa03e4933ce87a468ec5d09fc904e61c66e82528c356c84b"
        "abf02ddf788bbd45dee7badc281cd4c5b463ce103a6d23e39badfda3192bc8"
        "8c98989349eca2904c2f68842c49495dade197248229eca2904c2f5e06eda319"
        "2bc87ccce98a4790e97a1e49dfdd83d73cc230e470095229eca2904c2f68842"
        "c49502d55c197248229ec92a448a8bdfdfda3192bc87ccce98a4790ad7c5e49d"
        "fdd83d73cc230e47a674229eca2904c2f68842c48535045c197248229ec92a44"
        "8a7d94dfda3192bc87ccce99a3396fd2c5e49dfdd93c38068a54f017d38fcd9"
        "c8c4b00a0077b758b386ba04c3cd38fcd9c8d4e0cb24512a3c140ee9d7020b"
        "0be691fe6881068ce4d080439508d00f392de7f5fe31b63795c206df5db947c"
        "992ad2d2c5e49cfe1844b163bdd7");
    require(payload == expected, "MLVC-S rANS payload differs from Python reference");
    const auto decoded = entropy.decode(payload, 21);
    require(fp16_numerically_equal(decoded.z_raw, z),
            "MLVC-S z rANS round trip mismatch");
    require(fp16_numerically_equal(decoded.y_raw_0, y0),
            "MLVC-S y0 rANS round trip mismatch");
    require(fp16_numerically_equal(decoded.y_raw_1, y1),
            "MLVC-S y1 rANS round trip mismatch");

    const auto reference_root = assets /
        "references/mlvc-s-psnr-v1/gray-q21-2f";
    if (std::filesystem::exists(reference_root / "reference.yuv")) {
        mlvc::Tensor x_hat{"x_hat", {1, 3, 368, 640}, read_fp16(
            reference_root / "decoder/frame-000000/output-00-x_hat.fp16.raw")};
        require_yuv_matches(mlvc::prepare_model_output(
            x_hat, 640, 360, config.pixel_range, padding),
            reference_root / "reference.yuv");
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 2)
            throw std::runtime_error("usage: mlvc_codec_tests ASSETS_ROOT");
        test_half_conversion();
        test_yuv_views();
        test_mlvc_codec(std::filesystem::path(argv[1]));
        test_mlvc_s_codec(std::filesystem::path(argv[1]));
        std::cout << "codec tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "codec test failure: " << error.what() << '\n';
        return 1;
    }
}
