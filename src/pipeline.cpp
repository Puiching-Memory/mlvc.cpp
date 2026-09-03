#include "mlvc/pipeline.hpp"

#include "mlvc/bitstream.hpp"
#include "mlvc/entropy.hpp"
#include "mlvc/model.hpp"
#include "mlvc/yuv.hpp"

#include "model_assets.hpp"

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

std::filesystem::path resolve_model_location(const CodecOptions& options)
{
    if (!options.model_dir.empty())
        return options.model_dir;
    const std::filesystem::path embedded = detail::default_model_location();
    if (!embedded.empty())
        return embedded;
    throw std::runtime_error("--model-dir is required for this backend");
}

BackendOptions make_backend_options(const CodecOptions& options,
                                    const std::filesystem::path& model_dir)
{
    BackendOptions result;
    result.model_dir = model_dir.string();
    result.device_id = options.device_id;
    result.workspace_size = options.workspace_size;
    result.engine_cache_dir = options.engine_cache_dir;
    return result;
}

void validate_options(const CodecOptions& options, const ModelConfig& config,
                      bool encoding)
{
    if (options.input_path.empty() || options.output_path.empty())
        throw std::runtime_error("--input and --output are required");
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
         config.feature_height(), config.feature_width()},
        values};
}

ModelExecutionConfig make_execution_config(const ModelConfig& config,
                                            bool encoding,
                                            bool debug_tensors)
{
    ModelExecutionConfig result;
    if (!debug_tensors) {
        result.state_bindings.push_back({
            encoding ? 1U : 3U,
            encoding ? 0U : 1U,
            TensorDataType::kFloat16,
            {1, config.feature_channels,
             config.feature_height(), config.feature_width()},
        });
    }
    return result;
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
    const std::filesystem::path model_dir = resolve_model_location(options);
    const ModelConfig config = load_model_config(model_dir);
    validate_options(options, config, true);
    EntropyCodec entropy(model_dir, config);
    std::unique_ptr<InferenceBackend> backend = create_backend(
        make_backend_options(options, model_dir));
    const ModelExecutionConfig execution = make_execution_config(
        config, true, !options.debug_dir.empty());
    backend->load("MLVCEncoder", execution);
    const bool device_state = !execution.state_bindings.empty();

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
    std::vector<Float16Storage> reference =
        device_state ? std::vector<Float16Storage>{} : zero_reference(config);
    Yuv420Frame frame;
    while ((options.frames == 0 || stats.frames < options.frames) &&
           read_yuv420_frame(*input, options.width, options.height, frame)) {
        const int gop_frame = config.frame_in_gop(stats.frames);
        const bool reset_reference = gop_frame == 0;
        if (reset_reference) {
            if (device_state)
                backend->reset_state();
            else
                std::fill(reference.begin(), reference.end(), 0);
        }
        const int shifted_q = options.q_index + config.q_index_shift(gop_frame);
        if (shifted_q < 0 || shifted_q >= config.total_qp_num)
            throw std::runtime_error("shifted q-index is outside the model range");

        FramePadding padding;
        Tensor x = prepare_model_input(frame, config.model_width,
                                       config.model_height, config.pixel_range,
                                       padding);
        std::vector<Tensor> inputs;
        inputs.push_back(std::move(x));
        if (!device_state)
            inputs.push_back(make_reference(config, reference));
        inputs.push_back(make_q_index(shifted_q));
        write_debug_tensors(options, "encoder", stats.frames, "input", inputs);
        std::vector<Tensor> outputs = backend->run(inputs);
        if (outputs.size() != (device_state ? 3U : 4U))
            throw std::runtime_error("MLVCEncoder returned an unexpected output count");
        const std::size_t latent_offset = device_state ? 0U : 1U;
        if (!device_state) {
            outputs[0].name = "feature";
            validate_fp16_tensor(outputs[0],
                {1, config.feature_channels, config.feature_height(),
                 config.feature_width()},
                "feature");
        }
        outputs[latent_offset].name = "z_raw";
        outputs[latent_offset + 1].name = "y_raw_0";
        outputs[latent_offset + 2].name = "y_raw_1";
        validate_fp16_tensor(outputs[latent_offset],
            {1, config.hyperprior_channels, config.hyperprior_height(), config.hyperprior_width()},
            "z_raw");
        const std::vector<int64_t> y_shape{
            1, config.latent_channels / 2,
            config.latent_height(), config.latent_width()};
        validate_fp16_tensor(outputs[latent_offset + 1], y_shape, "y_raw_0");
        validate_fp16_tensor(outputs[latent_offset + 2], y_shape, "y_raw_1");
        write_debug_tensors(options, "encoder", stats.frames, "output", outputs);

        EncodedFrame encoded;
        encoded.q_index = options.q_index;
        encoded.payload = entropy.encode(
            outputs[latent_offset + 2], outputs[latent_offset + 1],
            outputs[latent_offset], options.q_index);
        write_encoded_frame(*output, encoded);
        if (stream_output)
            output->flush();
        if (!*output)
            throw std::runtime_error("failed to flush output bitstream");
        if (!device_state)
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
    const std::filesystem::path model_dir = resolve_model_location(options);
    const ModelConfig config = load_model_config(model_dir);
    validate_options(options, config, false);
    EntropyCodec entropy(model_dir, config);
    std::unique_ptr<InferenceBackend> backend = create_backend(
        make_backend_options(options, model_dir));
    const ModelExecutionConfig execution = make_execution_config(
        config, false, !options.debug_dir.empty());
    backend->load("MLVCDecoder", execution);
    const bool device_state = !execution.state_bindings.empty();

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
    std::vector<Float16Storage> reference =
        device_state ? std::vector<Float16Storage>{} : zero_reference(config);
    EncodedFrame encoded;
    const FramePadding padding = frame_padding(
        options.width, options.height, config.model_width, config.model_height);
    while ((options.frames == 0 || stats.frames < options.frames) &&
           read_encoded_frame(*input, encoded)) {
        if (encoded.q_index < 0 || encoded.q_index >= config.qp_num)
            throw std::runtime_error("bitstream q-index is outside the model range");
        const int gop_frame = config.frame_in_gop(stats.frames);
        const bool reset_reference = gop_frame == 0;
        if (reset_reference) {
            if (device_state)
                backend->reset_state();
            else
                std::fill(reference.begin(), reference.end(), 0);
        }
        const int shifted_q = encoded.q_index + config.q_index_shift(gop_frame);
        if (shifted_q < 0 || shifted_q >= config.total_qp_num)
            throw std::runtime_error("shifted q-index is outside the model range");

        EntropyCodec::DecodedLatents latents =
            entropy.decode(encoded.payload, encoded.q_index);
        std::vector<Tensor> inputs{
            std::move(latents.z_raw), std::move(latents.y_raw_0),
            std::move(latents.y_raw_1)};
        if (!device_state)
            inputs.push_back(make_reference(config, reference));
        inputs.push_back(make_q_index(shifted_q));
        write_debug_tensors(options, "decoder", stats.frames, "input", inputs);
        std::vector<Tensor> outputs = backend->run(inputs);
        if (outputs.size() != (device_state ? 1U : 2U))
            throw std::runtime_error("MLVCDecoder returned an unexpected output count");
        outputs[0].name = "x_hat";
        validate_fp16_tensor(outputs[0],
            {1, 3, config.model_height, config.model_width}, "x_hat");
        if (!device_state) {
            outputs[1].name = "feature";
            validate_fp16_tensor(outputs[1],
                {1, config.feature_channels, config.feature_height(),
                 config.feature_width()},
                "feature");
        }
        write_debug_tensors(options, "decoder", stats.frames, "output", outputs);
        write_yuv420_frame(*output, prepare_model_output(
            outputs[0], options.width, options.height, config.pixel_range, padding));
        if (stream_output)
            output->flush();
        if (!*output)
            throw std::runtime_error("failed to flush output YUV");
        if (!device_state)
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
