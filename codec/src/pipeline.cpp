#include "mlvc/codec/pipeline.hpp"

#include "mlvc/core/bitstream.hpp"
#include "mlvc/core/entropy.hpp"
#include "mlvc/core/model.hpp"
#include "mlvc/core/yuv.hpp"
#include "mlvc/runtime/backend.hpp"

#include "model_assets.hpp"

#include <nlohmann/json.hpp>

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mlvc {
namespace {

constexpr std::string_view kPackagedModelPrefix = "packaged:";
constexpr std::string_view kDefaultModelProfile = "mlvc-psnr-v1";
char kLibraryLocationAnchor = 0;

bool valid_profile_name(std::string_view profile)
{
    return !profile.empty() && profile.find('/') == std::string_view::npos &&
        profile.find('\\') == std::string_view::npos;
}

std::filesystem::path packaged_models_root()
{
    Dl_info information{};
    if (dladdr(static_cast<const void*>(&kLibraryLocationAnchor), &information) == 0 ||
        information.dli_fname == nullptr) {
        throw std::runtime_error(
            "cannot locate libmlvc_codec.so for packaged model discovery");
    }
    std::error_code error;
    const std::filesystem::path library =
        std::filesystem::weakly_canonical(information.dli_fname, error);
    if (error || library.parent_path().empty()) {
        throw std::runtime_error(
            "cannot resolve libmlvc_codec.so path for packaged models");
    }
    return library.parent_path().parent_path() / "share/mlvc/models";
}

std::vector<std::filesystem::path> packaged_profile_bundles(
    const std::filesystem::path& profile_dir)
{
    std::vector<std::filesystem::path> result;
    std::error_code error;
    if (!std::filesystem::is_directory(profile_dir, error) || error)
        return result;
    for (std::filesystem::directory_iterator iterator(profile_dir, error), end;
         !error && iterator != end; iterator.increment(error)) {
        const std::filesystem::path candidate = iterator->path();
        std::error_code candidate_error;
        if (iterator->is_directory(candidate_error) && !candidate_error &&
            std::filesystem::is_regular_file(
                candidate / "model_bundle.json", candidate_error) &&
            !candidate_error) {
            result.push_back(candidate);
        }
    }
    if (error) {
        throw std::runtime_error(
            "cannot enumerate packaged model directory: " +
            profile_dir.string());
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::filesystem::path packaged_model_location(std::string_view profile)
{
    if (!valid_profile_name(profile))
        throw std::runtime_error("invalid packaged model profile");
    const std::filesystem::path profile_dir =
        packaged_models_root() / std::string(profile);
    const std::vector<std::filesystem::path> bundles =
        packaged_profile_bundles(profile_dir);
    if (bundles.size() != 1) {
        throw std::runtime_error(
            "expected exactly one packaged model bundle for " +
            std::string(profile) + " below " + profile_dir.string());
    }
    return bundles.front();
}

std::vector<std::string> packaged_model_profiles()
{
    std::vector<std::string> result;
    const std::filesystem::path root = packaged_models_root();
    std::error_code error;
    if (!std::filesystem::is_directory(root, error) || error)
        return result;
    for (std::filesystem::directory_iterator iterator(root, error), end;
         !error && iterator != end; iterator.increment(error)) {
        std::error_code candidate_error;
        if (!iterator->is_directory(candidate_error) || candidate_error)
            continue;
        if (packaged_profile_bundles(iterator->path()).size() == 1)
            result.push_back(iterator->path().filename().string());
    }
    if (error) {
        throw std::runtime_error(
            "cannot enumerate packaged model root: " + root.string());
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::filesystem::path resolve_model_location(const CodecOptions& options)
{
    if (!options.model_dir.empty()) {
        const std::string_view location = options.model_dir;
        if (location.starts_with(kPackagedModelPrefix)) {
            return packaged_model_location(
                location.substr(kPackagedModelPrefix.size()));
        }
        return options.model_dir;
    }
    const std::filesystem::path embedded = detail::default_model_location();
    if (!embedded.empty())
        return embedded;
    return packaged_model_location(kDefaultModelProfile);
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
    tensor.validate();
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
        tensor.validate();
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

bool shape_matches(std::span<const int64_t> actual,
                   const std::vector<int64_t>& expected)
{
    return actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin());
}

const TensorView& find_tensor_view(const std::vector<TensorView>& tensors,
                                   std::string_view name)
{
    const auto found = std::find_if(
        tensors.begin(), tensors.end(),
        [&](const TensorView& tensor) { return tensor.name() == name; });
    if (found == tensors.end())
        throw std::runtime_error("buffered backend output is missing " +
                                 std::string(name));
    return *found;
}

MutableTensorView find_tensor_view(
    const std::vector<MutableTensorView>& tensors, std::string_view name)
{
    const auto found = std::find_if(
        tensors.begin(), tensors.end(),
        [&](const MutableTensorView& tensor) { return tensor.name() == name; });
    if (found == tensors.end())
        throw std::runtime_error("buffered backend input is missing " +
                                 std::string(name));
    return *found;
}

void validate_fp16_tensor(TensorView tensor,
                          const std::vector<int64_t>& shape,
                          const char* name)
{
    if (tensor.data_type() != TensorDataType::kFloat16 ||
        !shape_matches(tensor.shape(), shape) ||
        tensor.element_count() != checked_tensor_element_count(shape)) {
        throw std::runtime_error(std::string("backend output ") + name +
                                 " has unexpected shape or dtype");
    }
}

CodecIoConfig make_codec_io_config(const CodecOptions& options,
                                   const ModelConfig& config)
{
    CodecIoConfig result;
    result.width = options.width;
    result.height = options.height;
    result.model_width = config.model_width;
    result.model_height = config.model_height;
    result.pixel_range = config.pixel_range;
    result.padding = frame_padding(options.width, options.height,
                                   config.model_width, config.model_height);
    result.slots = 2;
    return result;
}

CodecStats encode_buffered(const CodecOptions& options,
                           const ModelConfig& config,
                           EntropyCodec& entropy,
                           InferenceBackend& backend,
                           BufferedCodecBackend& buffered,
                           std::istream& input, std::ostream& output,
                           bool stream_output)
{
    buffered.configure_codec_io(make_codec_io_config(options, config));
    const std::size_t slots = buffered.codec_slot_count();
    if (slots < 2)
        throw std::runtime_error("buffered backend requires at least two slots");

    const auto started = std::chrono::steady_clock::now();
    CodecStats stats;
    int submitted = 0;
    auto consume = [&](std::size_t slot) {
        buffered.wait_codec_slot(slot);
        const std::vector<TensorView> outputs = buffered.encoder_outputs(slot);
        const TensorView& z = find_tensor_view(outputs, "z_raw");
        const TensorView& y0 = find_tensor_view(outputs, "y_raw_0");
        const TensorView& y1 = find_tensor_view(outputs, "y_raw_1");
        validate_fp16_tensor(z,
            {1, config.hyperprior_channels, config.hyperprior_height(),
             config.hyperprior_width()}, "z_raw");
        const std::vector<int64_t> y_shape{
            1, config.latent_channels / 2,
            config.latent_height(), config.latent_width()};
        validate_fp16_tensor(y0, y_shape, "y_raw_0");
        validate_fp16_tensor(y1, y_shape, "y_raw_1");

        EncodedFrame encoded;
        encoded.q_index = options.q_index;
        encoded.payload = entropy.encode(y1, y0, z, options.q_index);
        write_encoded_frame(output, encoded);
        if (stream_output)
            output.flush();
        if (!output)
            throw std::runtime_error("failed to flush output bitstream");
        ++stats.frames;
        stats.input_bytes += yuv420_frame_bytes(options.width, options.height);
        stats.output_bytes += encoded.payload.size() + 8;
    };

    while (options.frames == 0 || submitted < options.frames) {
        if (submitted - stats.frames == static_cast<int>(slots))
            consume(static_cast<std::size_t>(stats.frames) % slots);
        const std::size_t slot = static_cast<std::size_t>(submitted) % slots;
        MutableYuv420FrameView frame = buffered.encoder_input_yuv(slot);
        if (!read_yuv420_frame(input, frame))
            break;

        const int gop_frame = config.frame_in_gop(submitted);
        if (gop_frame == 0)
            backend.reset_state();
        const int shifted_q = options.q_index + config.q_index_shift(gop_frame);
        if (shifted_q < 0 || shifted_q >= config.total_qp_num)
            throw std::runtime_error("shifted q-index is outside the model range");
        buffered.submit_encoder(slot, shifted_q);
        ++submitted;
    }
    while (stats.frames < submitted)
        consume(static_cast<std::size_t>(stats.frames) % slots);

    output.flush();
    if (!output)
        throw std::runtime_error("failed to finalize output bitstream");
    if (!stream_output)
        stats.output_bytes = regular_file_size(options.output_path);
    stats.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return stats;
}

CodecStats decode_buffered(const CodecOptions& options,
                           const ModelConfig& config,
                           EntropyCodec& entropy,
                           InferenceBackend& backend,
                           BufferedCodecBackend& buffered,
                           std::istream& input, std::ostream& output,
                           bool stream_output)
{
    buffered.configure_codec_io(make_codec_io_config(options, config));
    const std::size_t slots = buffered.codec_slot_count();
    if (slots < 2)
        throw std::runtime_error("buffered backend requires at least two slots");

    const auto started = std::chrono::steady_clock::now();
    CodecStats stats;
    int submitted = 0;
    auto consume = [&](std::size_t slot) {
        buffered.wait_codec_slot(slot);
        write_yuv420_frame(output, buffered.decoder_output_yuv(slot));
        if (stream_output)
            output.flush();
        if (!output)
            throw std::runtime_error("failed to flush output YUV");
        ++stats.frames;
        stats.output_bytes += yuv420_frame_bytes(options.width, options.height);
    };

    EncodedFrame encoded;
    while ((options.frames == 0 || submitted < options.frames) &&
           read_encoded_frame(input, encoded)) {
        if (submitted - stats.frames == static_cast<int>(slots))
            consume(static_cast<std::size_t>(stats.frames) % slots);
        if (encoded.q_index < 0 || encoded.q_index >= config.qp_num)
            throw std::runtime_error("bitstream q-index is outside the model range");
        const int gop_frame = config.frame_in_gop(submitted);
        if (gop_frame == 0)
            backend.reset_state();
        const int shifted_q = encoded.q_index + config.q_index_shift(gop_frame);
        if (shifted_q < 0 || shifted_q >= config.total_qp_num)
            throw std::runtime_error("shifted q-index is outside the model range");

        const std::size_t slot = static_cast<std::size_t>(submitted) % slots;
        const std::vector<MutableTensorView> inputs =
            buffered.decoder_inputs(slot);
        MutableTensorView z = find_tensor_view(inputs, "z_raw");
        MutableTensorView y0 = find_tensor_view(inputs, "y_raw_0");
        MutableTensorView y1 = find_tensor_view(inputs, "y_raw_1");
        entropy.decode_into(encoded.payload, encoded.q_index, z, y0, y1);
        buffered.submit_decoder(slot, shifted_q);
        stats.input_bytes += encoded.payload.size() + 8;
        ++submitted;
    }
    while (stats.frames < submitted)
        consume(static_cast<std::size_t>(stats.frames) % slots);

    output.flush();
    if (!output)
        throw std::runtime_error("failed to finalize output YUV");
    stats.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return stats;
}

}  // namespace

std::vector<std::string> available_model_profiles()
{
    std::vector<std::string> result = embedded_model_profiles();
    if (!result.empty())
        return result;
    return packaged_model_profiles();
}

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

    if (options.debug_dir.empty()) {
        if (auto* buffered = dynamic_cast<BufferedCodecBackend*>(backend.get())) {
            return encode_buffered(options, config, entropy, *backend, *buffered,
                                   *input, *output, stream_output);
        }
    }

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

    if (options.debug_dir.empty()) {
        if (auto* buffered = dynamic_cast<BufferedCodecBackend*>(backend.get())) {
            return decode_buffered(options, config, entropy, *backend, *buffered,
                                   *input, *output, stream_output);
        }
    }

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
