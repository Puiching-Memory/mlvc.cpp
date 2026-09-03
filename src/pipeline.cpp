#include "mlvc/pipeline.hpp"

#include "mlvc/bitstream.hpp"
#include "mlvc/entropy.hpp"
#include "mlvc/model.hpp"
#include "mlvc/yuv.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mlvc {
namespace {

BackendOptions make_backend_options(const CodecOptions& options)
{
    BackendOptions result;
    result.model_dir = options.model_dir;
    result.device_id = options.device_id;
    result.workspace_size = options.workspace_size;
    result.engine_cache_dir = options.engine_cache_dir;
    return result;
}

void validate_options(const CodecOptions& options, const ModelConfig& config,
                      bool encoding)
{
    if (options.input_path.empty() || options.output_path.empty() ||
        options.model_dir.empty())
        throw std::runtime_error("--input, --output and --model-dir are required");
    if (options.width <= 0 || options.height <= 0 ||
        options.width % 2 != 0 || options.height % 2 != 0)
        throw std::runtime_error("--width and --height must be positive even integers");
    frame_padding(options.width, options.height,
                  config.model_width, config.model_height);
    if (options.frames < 0)
        throw std::runtime_error("--frames must be non-negative");
    if (encoding && (options.q_index < 0 || options.q_index >= config.qp_num))
        throw std::runtime_error("--q-index is outside the model range");
}

std::vector<Float16Storage> zero_reference(const ModelConfig& config)
{
    return std::vector<Float16Storage>(
        static_cast<std::size_t>(config.feature_channels) *
        config.feature_height() * config.feature_width(), 0);
}

Tensor make_reference(const ModelConfig& config,
                      const std::vector<Float16Storage>& values)
{
    return Tensor{"ref_feature",
        {1, config.feature_channels,
         config.feature_height(), config.feature_width()}, values};
}

Tensor make_q_index(int q_index)
{
    return Tensor{"q_index_shifted", {1},
                  std::vector<std::int32_t>{q_index}};
}

void validate_fp16_tensor(const Tensor& tensor, const std::vector<int64_t>& shape,
                          const char* name)
{
    if (tensor.data_type() != TensorDataType::kFloat16 || tensor.shape != shape)
        throw std::runtime_error(std::string("backend output ") + name +
                                 " has unexpected shape or dtype");
}

std::vector<Float16Storage> copy_fp16(const Tensor& tensor)
{
    return std::get<std::vector<Float16Storage>>(tensor.data);
}

void create_parent_directory(const std::string& path)
{
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
}

std::size_t regular_file_size(const std::string& path)
{
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0 : static_cast<std::size_t>(size);
}

bool streaming_path(const std::string& path)
{
    if (path == "-")
        return true;
    std::error_code error;
    return std::filesystem::is_fifo(path, error);
}

const char* tensor_dtype_name(TensorDataType type)
{
    return type == TensorDataType::kFloat16 ? "fp16" : "int32";
}

void write_debug_tensors(const CodecOptions& options, const char* stage,
                         int frame_index, const char* direction,
                         const std::vector<Tensor>& tensors)
{
    if (options.debug_dir.empty())
        return;
    std::ostringstream frame_name;
    frame_name << "frame-" << std::setw(6) << std::setfill('0') << frame_index;
    const std::filesystem::path directory =
        std::filesystem::path(options.debug_dir) / stage / frame_name.str();
    std::filesystem::create_directories(directory);

    nlohmann::json manifest = nlohmann::json::array();
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const Tensor& tensor = tensors[index];
        std::ostringstream filename;
        filename << direction << '-' << std::setw(2) << std::setfill('0') << index
                 << '-' << (tensor.name.empty() ? "unnamed" : tensor.name) << '.'
                 << tensor_dtype_name(tensor.data_type()) << ".raw";
        const std::filesystem::path path = directory / filename.str();
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char*>(tensor.raw_data()),
                     static_cast<std::streamsize>(tensor.byte_size()));
        if (!output)
            throw std::runtime_error("failed to write debug tensor: " + path.string());
        manifest.push_back({
            {"direction", direction},
            {"index", index},
            {"name", tensor.name},
            {"dtype", tensor_dtype_name(tensor.data_type())},
            {"shape", tensor.shape},
            {"bytes", tensor.byte_size()},
            {"path", filename.str()},
        });
    }
    const std::filesystem::path manifest_path =
        directory / (std::string(direction) + "s.json");
    std::ofstream manifest_output(manifest_path, std::ios::trunc);
    manifest_output << manifest.dump(2) << '\n';
    if (!manifest_output)
        throw std::runtime_error("failed to write debug manifest: " +
                                 manifest_path.string());
}

}  // namespace

CodecStats encode_video(const CodecOptions& options)
{
    const ModelConfig config = load_model_config(options.model_dir);
    validate_options(options, config, true);
    EntropyCodec entropy(options.model_dir, config);
    std::unique_ptr<InferenceBackend> backend = create_backend(make_backend_options(options));
    backend->load("MLVCEncoder");

    std::ifstream input_file;
    std::istream* input = &std::cin;
    if (options.input_path != "-") {
        input_file.open(options.input_path, std::ios::binary);
        input = &input_file;
    }
    if (!*input)
        throw std::runtime_error("cannot open input YUV: " + options.input_path);
    create_parent_directory(options.output_path);
    std::ofstream output_file;
    std::ostream* output = &std::cout;
    const bool stream_output = streaming_path(options.output_path);
    if (options.output_path != "-") {
        output_file.open(options.output_path, std::ios::binary | std::ios::trunc);
        output = &output_file;
    }
    if (!*output)
        throw std::runtime_error("cannot create output bitstream: " + options.output_path);

    const auto started = std::chrono::steady_clock::now();
    CodecStats stats;
    std::vector<Float16Storage> reference = zero_reference(config);
    Yuv420Frame frame;
    while ((options.frames == 0 || stats.frames < options.frames) &&
           read_yuv420_frame(*input, options.width, options.height, frame)) {
        const int gop_frame = config.frame_in_gop(stats.frames);
        if (gop_frame == 0)
            std::fill(reference.begin(), reference.end(), 0);
        const int shifted_q = options.q_index + config.q_index_shift(gop_frame);
        if (shifted_q < 0 || shifted_q >= config.total_qp_num)
            throw std::runtime_error("shifted q-index is outside the model range");

        FramePadding padding;
        Tensor x = prepare_model_input(frame, config.model_width,
                                       config.model_height, config.pixel_range,
                                       padding);
        std::vector<Tensor> inputs{
            std::move(x), make_reference(config, reference), make_q_index(shifted_q)};
        write_debug_tensors(options, "encoder", stats.frames, "input", inputs);
        std::vector<Tensor> outputs = backend->run(inputs);
        if (outputs.size() != 4)
            throw std::runtime_error("MLVCEncoder returned an unexpected output count");
        outputs[0].name = "feature";
        outputs[1].name = "z_raw";
        outputs[2].name = "y_raw_0";
        outputs[3].name = "y_raw_1";
        validate_fp16_tensor(outputs[0],
            {1, config.feature_channels, config.feature_height(), config.feature_width()},
            "feature");
        validate_fp16_tensor(outputs[1],
            {1, config.hyperprior_channels, config.hyperprior_height(), config.hyperprior_width()},
            "z_raw");
        const std::vector<int64_t> y_shape{
            1, config.latent_channels / 2,
            config.latent_height(), config.latent_width()};
        validate_fp16_tensor(outputs[2], y_shape, "y_raw_0");
        validate_fp16_tensor(outputs[3], y_shape, "y_raw_1");
        write_debug_tensors(options, "encoder", stats.frames, "output", outputs);

        EncodedFrame encoded;
        encoded.q_index = options.q_index;
        encoded.payload = entropy.encode(outputs[3], outputs[2], outputs[1],
                                         options.q_index);
        write_encoded_frame(*output, encoded);
        if (stream_output)
            output->flush();
        if (!*output)
            throw std::runtime_error("failed to flush output bitstream");
        reference = copy_fp16(outputs[0]);
        ++stats.frames;
        stats.input_bytes += yuv420_frame_bytes(options.width, options.height);
        stats.output_bytes += encoded.payload.size() + 8;
    }
    output->flush();
    if (!*output)
        throw std::runtime_error("failed to finalize output bitstream");
    if (!stream_output)
        stats.output_bytes = regular_file_size(options.output_path);
    stats.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return stats;
}

CodecStats decode_video(const CodecOptions& options)
{
    const ModelConfig config = load_model_config(options.model_dir);
    validate_options(options, config, false);
    EntropyCodec entropy(options.model_dir, config);
    std::unique_ptr<InferenceBackend> backend = create_backend(make_backend_options(options));
    backend->load("MLVCDecoder");

    std::ifstream input_file;
    std::istream* input = &std::cin;
    if (options.input_path != "-") {
        input_file.open(options.input_path, std::ios::binary);
        input = &input_file;
    }
    if (!*input)
        throw std::runtime_error("cannot open MLVC bitstream: " + options.input_path);
    create_parent_directory(options.output_path);
    std::ofstream output_file;
    std::ostream* output = &std::cout;
    const bool stream_output = streaming_path(options.output_path);
    if (options.output_path != "-") {
        output_file.open(options.output_path, std::ios::binary | std::ios::trunc);
        output = &output_file;
    }
    if (!*output)
        throw std::runtime_error("cannot create output YUV: " + options.output_path);

    const auto started = std::chrono::steady_clock::now();
    CodecStats stats;
    std::vector<Float16Storage> reference = zero_reference(config);
    EncodedFrame encoded;
    const FramePadding padding = frame_padding(
        options.width, options.height, config.model_width, config.model_height);
    while ((options.frames == 0 || stats.frames < options.frames) &&
           read_encoded_frame(*input, encoded)) {
        if (encoded.q_index < 0 || encoded.q_index >= config.qp_num)
            throw std::runtime_error("bitstream q-index is outside the model range");
        const int gop_frame = config.frame_in_gop(stats.frames);
        if (gop_frame == 0)
            std::fill(reference.begin(), reference.end(), 0);
        const int shifted_q = encoded.q_index + config.q_index_shift(gop_frame);
        if (shifted_q < 0 || shifted_q >= config.total_qp_num)
            throw std::runtime_error("shifted q-index is outside the model range");

        EntropyCodec::DecodedLatents latents =
            entropy.decode(encoded.payload, encoded.q_index);
        std::vector<Tensor> inputs{
            std::move(latents.z_raw), std::move(latents.y_raw_0),
            std::move(latents.y_raw_1), make_reference(config, reference),
            make_q_index(shifted_q)};
        write_debug_tensors(options, "decoder", stats.frames, "input", inputs);
        std::vector<Tensor> outputs = backend->run(inputs);
        if (outputs.size() != 2)
            throw std::runtime_error("MLVCDecoder returned an unexpected output count");
        outputs[0].name = "x_hat";
        outputs[1].name = "feature";
        validate_fp16_tensor(outputs[0],
            {1, 3, config.model_height, config.model_width}, "x_hat");
        validate_fp16_tensor(outputs[1],
            {1, config.feature_channels, config.feature_height(), config.feature_width()},
            "feature");
        write_debug_tensors(options, "decoder", stats.frames, "output", outputs);
        write_yuv420_frame(*output, prepare_model_output(
            outputs[0], options.width, options.height, config.pixel_range, padding));
        if (stream_output)
            output->flush();
        if (!*output)
            throw std::runtime_error("failed to flush output YUV");
        reference = copy_fp16(outputs[1]);
        ++stats.frames;
        stats.input_bytes += encoded.payload.size() + 8;
        stats.output_bytes += yuv420_frame_bytes(options.width, options.height);
    }
    output->flush();
    if (!*output)
        throw std::runtime_error("failed to finalize output YUV");
    stats.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return stats;
}

}  // namespace mlvc
