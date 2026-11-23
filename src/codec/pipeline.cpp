#include "codec/pipeline.hpp"

#include "core/assets.hpp"
#include "core/bitstream.hpp"
#include "core/entropy.hpp"
#include "core/model.hpp"
#include "core/yuv.hpp"
#include "runtime/backend.hpp"
#include "mlvc_model.pb.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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

#if defined(_WIN32)
#  include <fcntl.h>
#  include <io.h>
#endif

namespace mlvc {
namespace {

void configure_binary_stdio(bool use_stdin, bool use_stdout)
{
#if defined(_WIN32)
    if (use_stdin) {
        if (_setmode(_fileno(stdin), _O_BINARY) == -1)
            throw std::runtime_error("failed to switch stdin to binary mode");
    }
    if (use_stdout) {
        if (_setmode(_fileno(stdout), _O_BINARY) == -1)
            throw std::runtime_error("failed to switch stdout to binary mode");
    }
#else
    (void)use_stdin;
    (void)use_stdout;
#endif
}

SessionOptions make_session_options(const CodecOptions& options)
{
    SessionOptions session;
    session.width = options.width;
    session.height = options.height;
    session.q_index = options.q_index;
    session.device_id = options.device_id;
    session.num_threads = options.num_threads;
    session.execution_preference = options.execution_preference;
    session.allow_cpu_fallback = options.allow_cpu_fallback;
    session.share_thread_pool = options.share_thread_pool;
    session.model_dir = options.model_dir;
    session.debug_dir = options.debug_dir;
    session.backend_name = options.backend_name;
    session.asset_reader = options.asset_reader;
    return session;
}

BackendOptions make_backend_options(const SessionOptions& options,
                                    const std::shared_ptr<const ModelAssetReader>& assets)
{
    BackendOptions result;
    result.model_dir = options.model_dir.empty() ? assets->location_label() : options.model_dir;
    result.device_id = options.device_id;
    result.num_threads = options.num_threads;
    result.execution_preference = options.execution_preference;
    result.allow_cpu_fallback = options.allow_cpu_fallback;
    result.share_thread_pool = options.share_thread_pool;
    result.backend_name = options.backend_name;
    result.asset_reader = assets;
    return result;
}

void validate_qp(int q_index, const ModelConfig& config, const char* source)
{
    if (q_index < kMinBitstreamQp || q_index > kMaxBitstreamQp) {
        throw std::runtime_error(std::string(source) +
                                 " must be in [0, 51] for MLVC NALU bitstream");
    }
    if (q_index >= config.qp_num)
        throw std::runtime_error(std::string(source) + " is outside the model range");
}

void validate_dimensions(int width, int height, const ModelConfig& config,
                         FramePadding* padding_out = nullptr)
{
    if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0)
        throw std::runtime_error("--width and --height must be positive even integers");
    const FramePadding padding = frame_padding(
        width, height, config.model_width, config.model_height);
    if (padding_out)
        *padding_out = padding;
}

void validate_file_options(const CodecOptions& options, bool encoding)
{
    if (options.input_path.empty() || options.output_path.empty())
        throw std::runtime_error("--input and --output are required");
    if (options.frames < 0)
        throw std::runtime_error("--frames must be non-negative");
    if (encoding) {
        if (options.width <= 0 || options.height <= 0 ||
            options.width % 2 != 0 || options.height % 2 != 0) {
            throw std::runtime_error("--width and --height must be positive even integers");
        }
    } else if ((options.width == 0) != (options.height == 0)) {
        throw std::runtime_error(
            "--width and --height must both be positive even integers or both zero");
    }
}

FrameData make_frame_data(const ModelConfig& config, const FramePadding& padding,
                          int q_index, int gop_frame_index, bool is_idr,
                          std::span<const std::byte> entropy_payload)
{
    FrameData data{};
    data.mlvc_version = {kSupportedMlvcMajorVersion, kDefaultMlvcMinorVersion};
    data.model_width = config.model_width;
    data.model_height = config.model_height;
    data.crop_offsets = {padding.left, padding.right, padding.top, padding.bottom};
    data.transpose_flag = padding.rotated;
    data.frame_idx_bits = kDefaultFrameIdxBits;
    data.temporal_id = 0;
    data.max_temporal_layers = 1;
    data.frame_type = is_idr ? FrameType::kIFrame : FrameType::kPFrame;
    data.qp = q_index;
    data.feature_reset_flag = false;
    data.cur_frame_idx = is_idr ? 0 : gop_frame_index;
    data.ref_frame_idx = is_idr ? 0 : gop_frame_index - 1;
    data.payload = entropy_payload;
    return data;
}

void validate_decoded_frame(const FrameData& frame_data,
                            const ModelConfig& config,
                            const FramePadding& padding,
                            int expected_width, int expected_height)
{
    if (frame_data.model_width != config.model_width ||
        frame_data.model_height != config.model_height) {
        throw std::runtime_error(
            "bitstream model dimensions do not match loaded model");
    }
    if (frame_data.display_width() != expected_width ||
        frame_data.display_height() != expected_height) {
        throw std::runtime_error(
            "bitstream display dimensions do not match decoder configuration");
    }
    if (frame_data.transpose_flag != padding.rotated ||
        frame_data.crop_offsets.left != padding.left ||
        frame_data.crop_offsets.right != padding.right ||
        frame_data.crop_offsets.top != padding.top ||
        frame_data.crop_offsets.bottom != padding.bottom) {
        throw std::runtime_error(
            "bitstream crop or transpose settings do not match decoder padding");
    }
    validate_qp(frame_data.qp, config, "bitstream QP");
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

void write_debug_tensors(const std::string& debug_dir, const char* stage,
                         int frame_index, const char* direction,
                         const std::vector<Tensor>& tensors)
{
    if (debug_dir.empty())
        return;
    std::ostringstream frame_name;
    frame_name << "frame-" << std::setw(6) << std::setfill('0') << frame_index;
    const std::filesystem::path directory =
        std::filesystem::path(debug_dir) / stage / frame_name.str();
    std::filesystem::create_directories(directory);

    wire::DebugTensorSet manifest;
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
        auto* record = manifest.add_tensors();
        record->set_direction(direction);
        record->set_index(static_cast<std::uint32_t>(index));
        record->set_name(tensor.name);
        record->set_dtype(tensor.data_type() == TensorDataType::kFloat16 ? wire::FP16 : wire::INT32);
        for (const auto dimension : tensor.shape) record->add_shape(dimension);
        record->set_size_bytes(tensor.byte_size());
        record->set_path(filename.str());
    }
    const std::filesystem::path manifest_path =
        directory / (std::string(direction) + "s.pb");
    std::ofstream manifest_output(manifest_path, std::ios::binary | std::ios::trunc);
    if (!manifest.SerializeToOstream(&manifest_output))
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

CodecIoConfig make_codec_io_config(int width, int height,
                                   const ModelConfig& config,
                                   const FramePadding& padding,
                                   std::size_t slots)
{
    CodecIoConfig result;
    result.width = width;
    result.height = height;
    result.model_width = config.model_width;
    result.model_height = config.model_height;
    result.pixel_range = config.pixel_range;
    result.padding = padding;
    result.slots = slots;
    return result;
}

CodecIoConfig make_codec_io_config(const CodecOptions& options,
                                   const ModelConfig& config)
{
    return make_codec_io_config(
        options.width, options.height, config,
        frame_padding(options.width, options.height,
                      config.model_width, config.model_height),
        2);
}

void copy_yuv_plane(const std::uint8_t* src, std::size_t src_stride,
                    std::uint8_t* dst, std::size_t dst_stride,
                    std::size_t row_width, std::size_t rows) noexcept
{
    if (src_stride == row_width && dst_stride == row_width) {
        std::memcpy(dst, src, row_width * rows);
        return;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        std::memcpy(dst + row * dst_stride, src + row * src_stride, row_width);
    }
}

void copy_yuv420_frame(Yuv420FrameView src, MutableYuv420FrameView dst) noexcept
{
    const std::size_t w = static_cast<std::size_t>(src.width());
    const std::size_t h = static_cast<std::size_t>(src.height());
    copy_yuv_plane(src.y().data(), src.y_stride(),
                   dst.y().data(), dst.y_stride(), w, h);
    copy_yuv_plane(src.u().data(), src.u_stride(),
                   dst.u().data(), dst.u_stride(), w / 2U, h / 2U);
    copy_yuv_plane(src.v().data(), src.v_stride(),
                   dst.v().data(), dst.v_stride(), w / 2U, h / 2U);
}

struct GopFramePlan {
    int gop_frame_index = 0;
    bool is_idr = true;
    int shifted_q = 0;
};

GopFramePlan plan_encoder_frame(const ModelConfig& config, int q_index,
                                int frame_index, int current_gop_index,
                                bool force_idr)
{
    const bool periodic_idr =
        force_idr ||
        (frame_index == 0) ||
        (config.iframe_period && current_gop_index >= *config.iframe_period) ||
        (current_gop_index >= (1 << kDefaultFrameIdxBits));
    const int effective_gop = periodic_idr ? 0 : current_gop_index;
    const int shifted_q = q_index + config.q_index_shift(effective_gop);
    if (shifted_q < 0 || shifted_q >= config.total_qp_num)
        throw std::runtime_error("shifted q-index is outside the model range");
    return GopFramePlan{effective_gop, effective_gop == 0, shifted_q};
}

class DecoderReferenceTracker {
public:
    void reset() noexcept
    {
        has_active_reference_ = false;
        last_frame_idx_ = -1;
    }

    // Validates frame type & reference continuity. Returns true if backend state
    // must be reset before decoding this frame.
    bool validate_and_check_reset(const FrameData& frame_data) const
    {
        if (frame_data.frame_type == FrameType::kLtrRecovery) {
            throw std::runtime_error(
                "LTR recovery frames are not supported by this decoder runtime");
        }
        const bool resets_reference =
            frame_data.frame_type == FrameType::kIFrame ||
            frame_data.feature_reset_flag;
        if (!resets_reference) {
            if (!has_active_reference_) {
                throw std::runtime_error(
                    "cannot decode P-frame before an IDR or feature-reset frame");
            }
            if (frame_data.ref_frame_idx != last_frame_idx_) {
                throw std::runtime_error(
                    "missing reference frame: bitstream ref_frame_idx does not match "
                    "last decoded frame (packet loss requires IDR recovery)");
            }
        }
        return resets_reference;
    }

    void commit(const FrameData& frame_data) noexcept
    {
        has_active_reference_ = true;
        last_frame_idx_ = frame_data.cur_frame_idx;
    }

private:
    bool has_active_reference_ = false;
    int last_frame_idx_ = -1;
};

int compute_decoder_shifted_q(const ModelConfig& config,
                              const FrameData& frame_data)
{
    const int shifted_q =
        frame_data.qp + config.q_index_shift(frame_data.cur_frame_idx);
    if (shifted_q < 0 || shifted_q >= config.total_qp_num)
        throw std::runtime_error("shifted q-index is outside the model range");
    return shifted_q;
}

template <typename TensorLike>
void validate_encoder_latents(const TensorLike& z, const TensorLike& y0,
                              const TensorLike& y1, const ModelConfig& config)
{
    validate_fp16_tensor(z,
        {1, config.hyperprior_channels, config.hyperprior_height(),
         config.hyperprior_width()}, "z_raw");
    const std::vector<int64_t> y_shape{
        1, config.latent_channels / 2,
        config.latent_height(), config.latent_width()};
    validate_fp16_tensor(y0, y_shape, "y_raw_0");
    validate_fp16_tensor(y1, y_shape, "y_raw_1");
}

struct OpenedIoStreams {
    std::ifstream input_file;
    std::ofstream output_file;
    std::istream* input = nullptr;
    std::ostream* output = nullptr;
    bool stream_output = false;

    OpenedIoStreams(const std::string& input_path, const char* input_label,
                    const std::string& output_path, const char* output_label)
    {
        configure_binary_stdio(input_path == "-", output_path == "-");
        input = &std::cin;
        if (input_path != "-") {
            input_file.open(input_path, std::ios::binary);
            input = &input_file;
        }
        if (!*input)
            throw std::runtime_error(std::string("cannot open ") + input_label + ": " + input_path);
        create_parent_directory(output_path);
        output = &std::cout;
        stream_output = streaming_path(output_path);
        if (output_path != "-") {
            output_file.open(output_path, std::ios::binary | std::ios::trunc);
            output = &output_file;
        }
        if (!*output)
            throw std::runtime_error(std::string("cannot create ") + output_label + ": " + output_path);
    }
};

CodecStats encode_buffered(const CodecOptions& options,
                           const ModelConfig& config,
                           const FramePadding& padding,
                           EntropyCodec& entropy,
                           InferenceBackend& backend,
                           BufferedCodecBackend& buffered,
                           std::istream& input, std::ostream& output,
                           bool stream_output)
{
    validate_qp(options.q_index, config, "--q-index");
    buffered.configure_codec_io(make_codec_io_config(options, config));
    const std::size_t slots = buffered.codec_slot_count();
    if (slots < 2)
        throw std::runtime_error("buffered backend requires at least two slots");

    const auto started = std::chrono::steady_clock::now();
    CodecStats stats;
    BitstreamEncoder bitstream_encoder;
    std::vector<int> slot_gop_frames(slots, 0);
    std::vector<bool> slot_is_idr(slots, false);
    int gop_frame_index = 0;
    int submitted = 0;
    auto consume = [&](std::size_t slot) {
        buffered.wait_codec_slot(slot);
        const std::vector<TensorView> outputs = buffered.encoder_outputs(slot);
        const TensorView& z = find_tensor_view(outputs, "z_raw");
        const TensorView& y0 = find_tensor_view(outputs, "y_raw_0");
        const TensorView& y1 = find_tensor_view(outputs, "y_raw_1");
        validate_encoder_latents(z, y0, y1, config);

        const std::vector<std::byte> entropy_payload =
            entropy.encode(y1, y0, z, options.q_index);
        const FrameData frame_data = make_frame_data(
            config, padding, options.q_index, slot_gop_frames[slot],
            slot_is_idr[slot], entropy_payload);
        const std::span<const std::byte> access_unit =
            bitstream_encoder.encode(frame_data);
        write_access_unit(output, access_unit);
        if (stream_output)
            output.flush();
        if (!output)
            throw std::runtime_error("failed to flush output bitstream");
        ++stats.frames;
        stats.input_bytes += yuv420_frame_bytes(options.width, options.height);
        stats.output_bytes += access_unit.size();
    };

    while (options.frames == 0 || submitted < options.frames) {
        if (submitted - stats.frames == static_cast<int>(slots))
            consume(static_cast<std::size_t>(stats.frames) % slots);
        const std::size_t slot = static_cast<std::size_t>(submitted) % slots;
        MutableYuv420FrameView frame = buffered.encoder_input_yuv(slot);
        if (!read_yuv420_frame(input, frame))
            break;

        const GopFramePlan plan = plan_encoder_frame(
            config, options.q_index, submitted, gop_frame_index, false);
        if (plan.is_idr)
            backend.reset_state();
        slot_gop_frames[slot] = plan.gop_frame_index;
        slot_is_idr[slot] = plan.is_idr;
        buffered.submit_encoder(slot, plan.shifted_q);
        gop_frame_index = plan.gop_frame_index + 1;
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
                           const FramePadding& padding,
                           EntropyCodec& entropy,
                           InferenceBackend& backend,
                           BufferedCodecBackend& buffered,
                           AccessUnitReader& reader,
                           std::optional<std::vector<std::byte>> initial_au,
                           std::ostream& output,
                           bool stream_output)
{
    buffered.configure_codec_io(make_codec_io_config(options, config));
    const std::size_t slots = buffered.codec_slot_count();
    if (slots < 2)
        throw std::runtime_error("buffered backend requires at least two slots");

    const auto started = std::chrono::steady_clock::now();
    CodecStats stats;
    BitstreamDecoder bitstream_decoder;
    DecoderReferenceTracker ref_tracker;
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

    std::vector<std::byte> access_unit;
    auto next_access_unit = [&]() -> bool {
        if (initial_au.has_value()) {
            access_unit = std::move(*initial_au);
            initial_au.reset();
            return true;
        }
        return reader.read(access_unit);
    };

    while ((options.frames == 0 || submitted < options.frames) &&
           next_access_unit()) {
        if (submitted - stats.frames == static_cast<int>(slots))
            consume(static_cast<std::size_t>(stats.frames) % slots);

        const FrameData frame_data = bitstream_decoder.decode(access_unit);
        validate_decoded_frame(frame_data, config, padding,
                               options.width, options.height);
        const bool resets_reference =
            ref_tracker.validate_and_check_reset(frame_data);
        const int shifted_q = compute_decoder_shifted_q(config, frame_data);

        const std::size_t slot = static_cast<std::size_t>(submitted) % slots;
        const std::vector<MutableTensorView> inputs =
            buffered.decoder_inputs(slot);
        MutableTensorView z = find_tensor_view(inputs, "z_raw");
        MutableTensorView y0 = find_tensor_view(inputs, "y_raw_0");
        MutableTensorView y1 = find_tensor_view(inputs, "y_raw_1");
        entropy.decode_into(frame_data.payload, frame_data.qp, z, y0, y1);
        if (resets_reference)
            backend.reset_state();
        buffered.submit_decoder(slot, shifted_q);
        ref_tracker.commit(frame_data);
        stats.input_bytes += access_unit.size();
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

Yuv420Frame allocate_yuv420_frame(int width, int height)
{
    static_cast<void>(yuv420_frame_bytes(width, height));
    Yuv420Frame frame;
    frame.width = width;
    frame.height = height;
    frame.y.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    const std::size_t chroma =
        static_cast<std::size_t>(width / 2) * static_cast<std::size_t>(height / 2);
    frame.u.resize(chroma);
    frame.v.resize(chroma);
    return frame;
}

}  // namespace

struct EncoderSession::Impl {
    SessionOptions options;
    std::shared_ptr<const ModelAssetReader> assets;
    ModelConfig config;
    FramePadding padding;
    EntropyCodec entropy;
    BitstreamEncoder bitstream_encoder;
    std::unique_ptr<InferenceBackend> backend;
    BufferedCodecBackend* buffered_backend = nullptr;
    bool buffered_configured = false;
    bool device_state = false;
    std::vector<Float16Storage> reference;
    int gop_frame_index = 0;
    int frame_index = 0;

    explicit Impl(const SessionOptions& session_options)
        : options(session_options),
          assets(resolve_model_asset_reader(
              session_options.model_dir, session_options.asset_reader)),
          config(load_model_config(*assets)),
          entropy(assets, config)
    {
        validate_dimensions(options.width, options.height, config, &padding);
        backend = create_backend(make_backend_options(options, assets));
        const ModelExecutionConfig execution = make_execution_config(
            config, true, !options.debug_dir.empty());
        backend->load("MLVCEncoder", execution);
        device_state = !execution.state_bindings.empty();
        if (options.debug_dir.empty())
            buffered_backend = dynamic_cast<BufferedCodecBackend*>(backend.get());
        if (!device_state)
            reference = zero_reference(config);
    }

    void reset()
    {
        gop_frame_index = 0;
        frame_index = 0;
        bitstream_encoder.reset();
        if (device_state)
            backend->reset_state();
        else
            std::fill(reference.begin(), reference.end(), 0);
    }

    EncodedFrame encode_frame(Yuv420FrameView frame, int q_index, bool force_idr)
    {
        if (frame.width() != options.width || frame.height() != options.height)
            throw std::runtime_error("input YUV frame dimensions do not match session configuration");
        validate_qp(q_index, config, "--q-index");

        const GopFramePlan plan = plan_encoder_frame(
            config, q_index, frame_index, gop_frame_index, force_idr);

        if (buffered_backend) {
            if (!buffered_configured) {
                buffered_backend->configure_codec_io(
                    make_codec_io_config(options.width, options.height,
                                         config, padding, 1));
                buffered_configured = true;
            }
            copy_yuv420_frame(frame, buffered_backend->encoder_input_yuv(0));
            if (plan.is_idr && device_state)
                backend->reset_state();
            buffered_backend->submit_encoder(0, plan.shifted_q);
            buffered_backend->wait_codec_slot(0);
            const std::vector<TensorView> outputs =
                buffered_backend->encoder_outputs(0);
            const TensorView& z = find_tensor_view(outputs, "z_raw");
            const TensorView& y0 = find_tensor_view(outputs, "y_raw_0");
            const TensorView& y1 = find_tensor_view(outputs, "y_raw_1");
            validate_encoder_latents(z, y0, y1, config);

            const std::vector<std::byte> entropy_payload =
                entropy.encode(y1, y0, z, q_index);
            const FrameData frame_data = make_frame_data(
                config, padding, q_index, plan.gop_frame_index, plan.is_idr,
                entropy_payload);
            const std::span<const std::byte> access_unit =
                bitstream_encoder.encode(frame_data);

            EncodedFrame encoded;
            encoded.q_index = q_index;
            encoded.frame_type = frame_data.frame_type;
            encoded.bitstream.assign(access_unit.begin(), access_unit.end());
            gop_frame_index = plan.gop_frame_index + 1;
            ++frame_index;
            return encoded;
        }

        FramePadding prepared_padding;
        Tensor x = prepare_model_input(frame, config.model_width,
                                       config.model_height, config.pixel_range,
                                       prepared_padding);
        std::vector<Tensor> inputs;
        inputs.push_back(std::move(x));
        if (!device_state) {
            if (plan.is_idr)
                inputs.push_back(make_reference(config, zero_reference(config)));
            else
                inputs.push_back(make_reference(config, reference));
        }
        inputs.push_back(make_q_index(plan.shifted_q));
        write_debug_tensors(options.debug_dir, "encoder", frame_index, "input", inputs);

        if (plan.is_idr && device_state)
            backend->reset_state();

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
        validate_encoder_latents(
            outputs[latent_offset], outputs[latent_offset + 1],
            outputs[latent_offset + 2], config);
        write_debug_tensors(options.debug_dir, "encoder", frame_index, "output", outputs);

        const std::vector<std::byte> entropy_payload = entropy.encode(
            outputs[latent_offset + 2], outputs[latent_offset + 1],
            outputs[latent_offset], q_index);
        const FrameData frame_data = make_frame_data(
            config, padding, q_index, plan.gop_frame_index, plan.is_idr,
            entropy_payload);
        const std::span<const std::byte> access_unit =
            bitstream_encoder.encode(frame_data);

        EncodedFrame encoded;
        encoded.q_index = q_index;
        encoded.frame_type = frame_data.frame_type;
        encoded.bitstream.assign(access_unit.begin(), access_unit.end());
        if (!device_state)
            reference = copy_fp16(outputs[0]);
        gop_frame_index = plan.gop_frame_index + 1;
        ++frame_index;
        return encoded;
    }
};

struct DecoderSession::Impl {
    SessionOptions options;
    std::shared_ptr<const ModelAssetReader> assets;
    ModelConfig config;
    FramePadding padding;
    EntropyCodec entropy;
    BitstreamDecoder bitstream_decoder;
    std::unique_ptr<InferenceBackend> backend;
    BufferedCodecBackend* buffered_backend = nullptr;
    bool buffered_configured = false;
    bool device_state = false;
    std::vector<Float16Storage> reference;
    DecoderReferenceTracker ref_tracker;
    int frame_index = 0;

    explicit Impl(const SessionOptions& session_options)
        : options(session_options),
          assets(resolve_model_asset_reader(
              session_options.model_dir, session_options.asset_reader)),
          config(load_model_config(*assets)),
          entropy(assets, config)
    {
        validate_dimensions(options.width, options.height, config, &padding);
        backend = create_backend(make_backend_options(options, assets));
        const ModelExecutionConfig execution = make_execution_config(
            config, false, !options.debug_dir.empty());
        backend->load("MLVCDecoder", execution);
        device_state = !execution.state_bindings.empty();
        if (options.debug_dir.empty())
            buffered_backend = dynamic_cast<BufferedCodecBackend*>(backend.get());
        if (!device_state)
            reference = zero_reference(config);
    }

    void reset()
    {
        frame_index = 0;
        ref_tracker.reset();
        bitstream_decoder.reset();
        if (device_state)
            backend->reset_state();
        else
            std::fill(reference.begin(), reference.end(), 0);
    }

    void decode_frame_into(std::span<const std::byte> access_unit,
                           MutableYuv420FrameView output)
    {
        if (output.width() != options.width || output.height() != options.height)
            throw std::runtime_error("output YUV frame dimensions do not match session configuration");

        const FrameData frame_data = bitstream_decoder.decode(access_unit);
        validate_decoded_frame(frame_data, config, padding,
                               options.width, options.height);
        const bool resets_reference =
            ref_tracker.validate_and_check_reset(frame_data);
        const int shifted_q = compute_decoder_shifted_q(config, frame_data);

        if (buffered_backend) {
            if (!buffered_configured) {
                buffered_backend->configure_codec_io(
                    make_codec_io_config(options.width, options.height,
                                         config, padding, 1));
                buffered_configured = true;
            }
            const std::vector<MutableTensorView> inputs =
                buffered_backend->decoder_inputs(0);
            MutableTensorView z = find_tensor_view(inputs, "z_raw");
            MutableTensorView y0 = find_tensor_view(inputs, "y_raw_0");
            MutableTensorView y1 = find_tensor_view(inputs, "y_raw_1");
            entropy.decode_into(frame_data.payload, frame_data.qp, z, y0, y1);
            if (resets_reference && device_state)
                backend->reset_state();
            buffered_backend->submit_decoder(0, shifted_q);
            buffered_backend->wait_codec_slot(0);
            copy_yuv420_frame(buffered_backend->decoder_output_yuv(0), output);
            ref_tracker.commit(frame_data);
            ++frame_index;
            return;
        }

        EntropyCodec::DecodedLatents latents =
            entropy.decode(frame_data.payload, frame_data.qp);
        std::vector<Tensor> inputs{
            std::move(latents.z_raw), std::move(latents.y_raw_0),
            std::move(latents.y_raw_1)};
        if (!device_state) {
            if (resets_reference)
                inputs.push_back(make_reference(config, zero_reference(config)));
            else
                inputs.push_back(make_reference(config, reference));
        }
        inputs.push_back(make_q_index(shifted_q));
        write_debug_tensors(options.debug_dir, "decoder", frame_index, "input", inputs);

        if (resets_reference && device_state)
            backend->reset_state();

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
        write_debug_tensors(options.debug_dir, "decoder", frame_index, "output", outputs);
        prepare_model_output_into(outputs[0], output, config.pixel_range, padding);
        if (!device_state)
            reference = copy_fp16(outputs[1]);
        ref_tracker.commit(frame_data);
        ++frame_index;
    }
};

EncoderSession::EncoderSession(const SessionOptions& options)
    : impl_(std::make_unique<Impl>(options))
{
}

EncoderSession::~EncoderSession() = default;
EncoderSession::EncoderSession(EncoderSession&&) noexcept = default;
EncoderSession& EncoderSession::operator=(EncoderSession&&) noexcept = default;

void EncoderSession::reset()
{
    if (!impl_)
        throw std::runtime_error("EncoderSession has been moved from");
    impl_->reset();
}

int EncoderSession::frame_index() const noexcept
{
    return impl_ ? impl_->frame_index : 0;
}

const ModelConfig& EncoderSession::model_config() const
{
    if (!impl_)
        throw std::runtime_error("EncoderSession has been moved from");
    return impl_->config;
}

EncodedFrame EncoderSession::encode_frame(Yuv420FrameView frame)
{
    if (!impl_)
        throw std::runtime_error("EncoderSession has been moved from");
    return impl_->encode_frame(frame, impl_->options.q_index, false);
}

EncodedFrame EncoderSession::encode_frame(Yuv420FrameView frame, int q_index)
{
    if (!impl_)
        throw std::runtime_error("EncoderSession has been moved from");
    return impl_->encode_frame(frame, q_index, false);
}

EncodedFrame EncoderSession::encode_frame(Yuv420FrameView frame, int q_index,
                                          bool force_idr)
{
    if (!impl_)
        throw std::runtime_error("EncoderSession has been moved from");
    return impl_->encode_frame(frame, q_index, force_idr);
}

DecoderSession::DecoderSession(const SessionOptions& options)
    : impl_(std::make_unique<Impl>(options))
{
}

DecoderSession::~DecoderSession() = default;
DecoderSession::DecoderSession(DecoderSession&&) noexcept = default;
DecoderSession& DecoderSession::operator=(DecoderSession&&) noexcept = default;

void DecoderSession::reset()
{
    if (!impl_)
        throw std::runtime_error("DecoderSession has been moved from");
    impl_->reset();
}

int DecoderSession::frame_index() const noexcept
{
    return impl_ ? impl_->frame_index : 0;
}

const ModelConfig& DecoderSession::model_config() const
{
    if (!impl_)
        throw std::runtime_error("DecoderSession has been moved from");
    return impl_->config;
}

Yuv420Frame DecoderSession::decode_frame(const EncodedFrame& encoded)
{
    if (!impl_)
        throw std::runtime_error("DecoderSession has been moved from");
    Yuv420Frame output = allocate_yuv420_frame(
        impl_->options.width, impl_->options.height);
    impl_->decode_frame_into(encoded.bitstream, mutable_yuv420_view(output));
    return output;
}

void DecoderSession::decode_frame_into(const EncodedFrame& encoded,
                                       MutableYuv420FrameView output)
{
    if (!impl_)
        throw std::runtime_error("DecoderSession has been moved from");
    impl_->decode_frame_into(encoded.bitstream, output);
}

void DecoderSession::decode_frame_into(std::span<const std::byte> access_unit,
                                       MutableYuv420FrameView output)
{
    if (!impl_)
        throw std::runtime_error("DecoderSession has been moved from");
    impl_->decode_frame_into(access_unit, output);
}

std::vector<std::string> available_model_profiles()
{
    return embedded_model_profiles();
}

CodecStats encode_video(const CodecOptions& options)
{
    validate_file_options(options, true);

    EncoderSession session(make_session_options(options));
    validate_qp(options.q_index, session.model_config(), "--q-index");

    OpenedIoStreams io(options.input_path, "input YUV",
                       options.output_path, "output bitstream");

    if (options.debug_dir.empty()) {
        if (auto* buffered = dynamic_cast<BufferedCodecBackend*>(
                session.impl_->backend.get())) {
            return encode_buffered(options, session.impl_->config,
                                   session.impl_->padding,
                                   session.impl_->entropy,
                                   *session.impl_->backend, *buffered,
                                   *io.input, *io.output, io.stream_output);
        }
    }

    const auto started = std::chrono::steady_clock::now();
    CodecStats stats;
    Yuv420Frame frame;
    while ((options.frames == 0 || stats.frames < options.frames) &&
           read_yuv420_frame(*io.input, options.width, options.height, frame)) {
        const EncodedFrame encoded =
            session.encode_frame(yuv420_view(frame), options.q_index);
        write_access_unit(*io.output, encoded.bitstream);
        if (io.stream_output)
            io.output->flush();
        if (!*io.output)
            throw std::runtime_error("failed to flush output bitstream");
        ++stats.frames;
        stats.input_bytes += yuv420_frame_bytes(options.width, options.height);
        stats.output_bytes += encoded.bitstream.size();
    }
    io.output->flush();
    if (!*io.output)
        throw std::runtime_error("failed to finalize output bitstream");
    if (!io.stream_output)
        stats.output_bytes = regular_file_size(options.output_path);
    stats.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return stats;
}

CodecStats decode_video(const CodecOptions& options)
{
    validate_file_options(options, false);
    OpenedIoStreams io(options.input_path, "MLVC bitstream",
                       options.output_path, "output YUV");

    AccessUnitReader reader(*io.input);
    std::optional<std::vector<std::byte>> initial_au;
    CodecOptions resolved = options;
    if (resolved.width == 0 && resolved.height == 0) {
        std::vector<std::byte> first_au;
        if (!reader.read(first_au)) {
            io.output->flush();
            if (!*io.output)
                throw std::runtime_error("failed to finalize output YUV");
            return CodecStats{};
        }
        BitstreamDecoder probe_decoder;
        const FrameData first_frame = probe_decoder.decode(first_au);
        resolved.width = first_frame.display_width();
        resolved.height = first_frame.display_height();
        initial_au = std::move(first_au);
    }

    DecoderSession session(make_session_options(resolved));

    if (resolved.debug_dir.empty()) {
        if (auto* buffered = dynamic_cast<BufferedCodecBackend*>(
                session.impl_->backend.get())) {
            return decode_buffered(resolved, session.impl_->config,
                                   session.impl_->padding,
                                   session.impl_->entropy,
                                   *session.impl_->backend, *buffered,
                                   reader, std::move(initial_au),
                                   *io.output, io.stream_output);
        }
    }

    const auto started = std::chrono::steady_clock::now();
    CodecStats stats;
    Yuv420Frame frame = allocate_yuv420_frame(resolved.width, resolved.height);
    std::vector<std::byte> access_unit;
    auto next_access_unit = [&]() -> bool {
        if (initial_au.has_value()) {
            access_unit = std::move(*initial_au);
            initial_au.reset();
            return true;
        }
        return reader.read(access_unit);
    };

    EncodedFrame encoded;
    while ((resolved.frames == 0 || stats.frames < resolved.frames) &&
           next_access_unit()) {
        encoded.bitstream = std::move(access_unit);
        session.decode_frame_into(encoded, mutable_yuv420_view(frame));
        write_yuv420_frame(*io.output, yuv420_view(frame));
        if (io.stream_output)
            io.output->flush();
        if (!*io.output)
            throw std::runtime_error("failed to flush output YUV");
        ++stats.frames;
        stats.input_bytes += encoded.bitstream.size();
        stats.output_bytes += yuv420_frame_bytes(resolved.width, resolved.height);
    }
    io.output->flush();
    if (!*io.output)
        throw std::runtime_error("failed to finalize output YUV");
    stats.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return stats;
}

}  // namespace mlvc
