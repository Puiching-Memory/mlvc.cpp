#include "aot_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlvc::driver_cubin_backend {
namespace {

std::size_t yuv_bytes(const CodecIoConfig& config)
{
    if (config.width <= 0 || config.height <= 0 ||
        config.width % 2 != 0 || config.height % 2 != 0 ||
        config.model_width <= 0 || config.model_height <= 0 ||
        config.slots == 0 || !std::isfinite(config.pixel_range) ||
        config.pixel_range <= 0.0F) {
        throw std::runtime_error("driver-cubin: invalid codec I/O configuration");
    }
    return static_cast<std::size_t>(config.width) * config.height * 3U / 2U;
}

}  // namespace

void AotGraph::warmup_and_capture()
{
    for (const auto& buffer : input_buffers_)
        driver_.zero_async(buffer);
    if (driver_.device_info().compute_major >= 8 && !graph_.nodes.empty()) {
        constexpr std::size_t kSlotStride =
            driver_cubin::kCutlassPointwiseParamsStorageBytes;
        cutlass_init_scratch_ =
            driver_.allocate(graph_.nodes.size() * kSlotStride);
        pending_cutlass_inits_.clear();
        batch_initializing_cutlass_ = true;
        try {
            execute_schedule();
        } catch (...) {
            batch_initializing_cutlass_ = false;
            pending_cutlass_inits_.clear();
            cutlass_init_scratch_ = {};
            throw;
        }
        batch_initializing_cutlass_ = false;
        if (!pending_cutlass_inits_.empty()) {
            const std::size_t total_bytes =
                pending_cutlass_inits_.size() * kSlotStride;
            std::vector<std::byte> host_scratch(total_bytes);
            driver_.download(host_scratch.data(),
                             cutlass_init_scratch_.address(), total_bytes);
            for (const PendingCutlassInit& item : pending_cutlass_inits_) {
                const std::byte* src = host_scratch.data() + item.offset;
                if (item.is_spatial) {
                    auto& dst =
                        cutlass_spatial_host_parameters_[item.node_index];
                    std::memcpy(dst.data(), src, dst.size());
                } else {
                    auto& dst = cutlass_host_parameters_[item.node_index];
                    std::memcpy(dst.data(), src, dst.size());
                }
            }
        }
        pending_cutlass_inits_.clear();
        cutlass_init_scratch_ = {};
    }
    cutlass_parameters_ready_ = true;
    driver_.begin_capture();
    execute_schedule();
    executable_graph_ = driver_.end_capture();
    driver_.synchronize();
}

void AotGraph::cache_runtime_metadata()
{
    external_outputs_.clear();
    external_outputs_.reserve(output_ids_.size());
    for (std::size_t index = 0; index < output_ids_.size(); ++index) {
        if (is_state_output(index))
            continue;
        const TensorId id = output_ids_[index];
        const Value& result = value(id);
        const std::size_t bytes =
            element_count(result.shape) * dtype_bytes(result.dtype);
        external_outputs_.push_back({index, id, result.address, bytes});
    }

    q_input_index_.reset();
    decoder_latent_inputs_.clear();
    for (std::size_t index = 0; index < input_ids_.size(); ++index) {
        const Value& input = value(input_ids_[index]);
        if (input.name == "q_index_shifted") {
            if (input.dtype == DataType::INT32 && input.shape.size() == 1 &&
                input.shape[0] == 1) {
                q_input_index_ = index;
            }
            continue;
        }
        if (!is_state_input(index))
            decoder_latent_inputs_.push_back(index);
    }
}

void AotGraph::reset_state()
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    auto ctx_guard = driver_.bind_context();
    for (const StateBinding& binding : state_bindings_)
        driver_.zero_async(input_buffers_[binding.input_index]);
    state_initialized_ = true;
}

void AotGraph::enqueue_model()
{
    if (!cutlass_parameters_ready_ &&
        driver_.device_info().compute_major >= 8) {
        execute_schedule();
        cutlass_parameters_ready_ = true;
    } else if (!executable_graph_) {
        driver_.begin_capture();
        execute_schedule();
        executable_graph_ = driver_.end_capture();
        driver_.launch_graph(executable_graph_);
    } else {
        driver_.launch_graph(executable_graph_);
    }
}

void AotGraph::enqueue_outputs(HostSlot& slot)
{
    for (const ExternalOutput& output : external_outputs_) {
        const Value& result = value(output.id);
        if (needs_nhwc_boundary_transpose(result)) {
            ensure_boundary_transpose_scratch(output.bytes);
            const int rows = static_cast<int>(result.shape[2] * result.shape[3]);
            const int cols = static_cast<int>(result.shape[1]);
            launch_transpose_2d(
                output.address, boundary_transpose_scratch_.address(),
                rows, cols);
            driver_.download_async(
                slot.outputs[output.output_index].data(),
                boundary_transpose_scratch_.address(), output.bytes);
        } else {
            driver_.download_async(
                slot.outputs[output.output_index].data(),
                output.address, output.bytes);
        }
    }
}

void AotGraph::set_q_index(HostSlot& slot, int shifted_q)
{
    if (!q_input_index_)
        throw std::runtime_error("driver-cubin: graph has no valid q_index_shifted input");
    const std::size_t index = *q_input_index_;
    if (!slot.inputs[index])
        throw std::runtime_error("driver-cubin: invalid q_index_shifted input");
    std::memcpy(slot.inputs[index].data(), &shifted_q, sizeof(shifted_q));
    driver_.upload_async(input_buffers_[index], slot.inputs[index].data(),
                         sizeof(shifted_q));
}

void AotGraph::launch_yuv420_to_nchw(DeviceAddress yuv_device)
{
    DeviceAddress input = yuv_device;
    DeviceAddress lut = codec_byte_lut_device_.address();
    int width = codec_io_.width;
    int height = codec_io_.height;
    int model_width = codec_io_.model_width;
    int model_height = codec_io_.model_height;
    int rotated = codec_io_.padding.rotated ? 1 : 0;
    if (use_nhwc_ && codec_fused_s2d8_node_.has_value()) {
        DeviceAddress output = codec_s2d8_output_;
        void* parameters[] = {&input, &lut, &output, &width, &height,
                              &model_width, &model_height, &rotated};
        driver_.launch(
            yuv420_to_s2d8_nhwc_,
            {divide_up(static_cast<std::size_t>(model_width / 8), 8),
             static_cast<unsigned int>(model_height / 8), 1},
            {24, 8, 1}, 0, parameters);
        return;
    }
    DeviceAddress output = codec_image_address_;
    void* parameters[] = {&input, &lut, &output, &width, &height,
                          &model_width, &model_height, &rotated};
    if (use_nhwc_) {
        driver_.launch(
            yuv420_to_nhwc_,
            {divide_up(static_cast<std::size_t>(model_width), 32),
             divide_up(static_cast<std::size_t>(model_height), 8), 1},
            {32, 8, 1}, 0, parameters);
        return;
    }
    driver_.launch(
        yuv420_to_nchw_,
        {divide_up(static_cast<std::size_t>(model_width), 32),
         divide_up(static_cast<std::size_t>(model_height), 8), 3},
        {32, 8, 1}, 0, parameters);
}

void AotGraph::launch_nchw_to_yuv420(DeviceAddress yuv_device)
{
    DeviceAddress output = yuv_device;
    int width = codec_io_.width;
    int height = codec_io_.height;
    int model_width = codec_io_.model_width;
    int model_height = codec_io_.model_height;
    int pad_left = codec_io_.padding.left;
    int pad_top = codec_io_.padding.top;
    int rotated = codec_io_.padding.rotated ? 1 : 0;
    float inverse_pixel_range = 1.0F / codec_io_.pixel_range;
    if (use_nhwc_ && codec_fused_d2s8_node_.has_value()) {
        DeviceAddress input = codec_d2s8_input_;
        int has_clip = codec_d2s8_has_clip_ ? 1 : 0;
        float clip_min = codec_d2s8_clip_min_;
        float clip_max = codec_d2s8_clip_max_;
        void* parameters[] = {
            &input, &output, &width, &height, &model_width, &model_height,
            &pad_left, &pad_top, &rotated, &inverse_pixel_range,
            &has_clip, &clip_min, &clip_max};
        driver_.launch(
            d2s8_nhwc_to_yuv420_,
            {divide_up(static_cast<std::size_t>(width / 2), 16),
             divide_up(static_cast<std::size_t>(height / 2), 8), 1},
            {16, 8, 1}, 0, parameters);
        return;
    }
    DeviceAddress input = codec_image_address_;
    void* parameters[] = {
        &input, &output, &width, &height, &model_width, &model_height,
        &pad_left, &pad_top, &rotated, &inverse_pixel_range};
    driver_.launch(
        use_nhwc_ ? nhwc_to_yuv420_ : nchw_to_yuv420_,
        {divide_up(static_cast<std::size_t>(width / 2), 16),
         divide_up(static_cast<std::size_t>(height / 2), 8), 1},
        {16, 8, 1}, 0, parameters);
}

void AotGraph::capture_codec_graph(HostSlot& slot)
{
    if (!q_input_index_)
        throw std::runtime_error("driver-cubin: graph has no valid q_index_shifted input");
    if (!cutlass_parameters_ready_ && driver_.device_info().compute_major >= 8) {
        execute_schedule();
        cutlass_parameters_ready_ = true;
    }
    driver_.synchronize();
    codec_elided_nodes_.clear();
    active_codec_yuv_device_ = 0;
    if (use_nhwc_) {
        if (codec_fused_d2s8_clip_node_)
            codec_elided_nodes_.push_back(*codec_fused_d2s8_clip_node_);
        if (codec_fused_s2d8_node_ || codec_fused_d2s8_node_)
            active_codec_yuv_device_ = slot.yuv_device.address();
    }
    driver_.begin_capture();
    try {
        if (model_name_ == "MLVCEncoder") {
            if (!(use_nhwc_ && codec_fused_s2d8_node_.has_value()))
                launch_yuv420_to_nchw(slot.yuv_device.address());
            driver_.copy_dto_d_async(
                input_buffers_[*q_input_index_].address(),
                slot.q_device.address(), sizeof(int));
            execute_schedule();
            for (const ExternalOutput& output : external_outputs_) {
                const Value& result = value(output.id);
                if (needs_nhwc_boundary_transpose(result)) {
                    const int rows = static_cast<int>(
                        result.shape[2] * result.shape[3]);
                    const int cols = static_cast<int>(result.shape[1]);
                    launch_transpose_2d(
                        output.address,
                        slot.output_staging[output.output_index].address(),
                        rows, cols);
                } else {
                    driver_.copy_dto_d_async(
                        slot.output_staging[output.output_index].address(),
                        output.address, output.bytes);
                }
            }
        } else {
            for (const std::size_t index : decoder_latent_inputs_) {
                const Value& in_val = value(input_ids_[index]);
                if (needs_nhwc_boundary_transpose(in_val)) {
                    const int rows = static_cast<int>(in_val.shape[1]);
                    const int cols = static_cast<int>(
                        in_val.shape[2] * in_val.shape[3]);
                    launch_transpose_2d(
                        slot.input_staging[index].address(),
                        input_buffers_[index].address(), rows, cols);
                } else {
                    driver_.copy_dto_d_async(
                        input_buffers_[index].address(),
                        slot.input_staging[index].address(),
                        input_buffers_[index].size());
                }
            }
            driver_.copy_dto_d_async(
                input_buffers_[*q_input_index_].address(),
                slot.q_device.address(), sizeof(int));
            execute_schedule();
            if (!(use_nhwc_ && codec_fused_d2s8_node_.has_value()))
                launch_nchw_to_yuv420(slot.yuv_device.address());
        }
        slot.codec_graph = driver_.end_capture();
    } catch (...) {
        active_codec_yuv_device_ = 0;
        codec_elided_nodes_.clear();
        throw;
    }
    active_codec_yuv_device_ = 0;
    codec_elided_nodes_.clear();
}

void AotGraph::finish_submit(HostSlot& slot)
{
    driver_.record(slot.completion);
    slot.pending = true;
}

std::vector<Tensor> AotGraph::run(const std::vector<Tensor>& inputs)
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    auto ctx_guard = driver_.bind_context();
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error(
            "driver-cubin: reset_state() must be called before run()");
    if (inputs.size() + state_bindings_.size() != input_ids_.size())
        throw std::runtime_error("driver-cubin: graph input count mismatch");

    ensure_host_slots(1);
    for (std::size_t slot_index = 0; slot_index < host_slots_.size(); ++slot_index)
        wait_codec_slot_locked(slot_index);
    HostSlot& slot = host_slots_.front();
    std::size_t external_index = 0;
    for (std::size_t graph_index = 0; graph_index < input_ids_.size();
         ++graph_index) {
        if (is_state_input(graph_index))
            continue;
        const Tensor& input = inputs.at(external_index++);
        const Value& expected = value(input_ids_[graph_index]);
        if (input.shape != expected.shape ||
            input.data_type() != public_dtype(expected.dtype)) {
            throw std::runtime_error(
                "driver-cubin: graph input shape or dtype mismatch");
        }
        if (input.byte_size() != input_buffers_[graph_index].size())
            throw std::runtime_error(
                "driver-cubin: graph input byte size mismatch");
        std::memcpy(slot.inputs[graph_index].data(), input.raw_data(),
                    input.byte_size());
        if (needs_nhwc_boundary_transpose(expected)) {
            ensure_boundary_transpose_scratch(input.byte_size());
            driver_.upload_async(boundary_transpose_scratch_,
                                 slot.inputs[graph_index].data(),
                                 input.byte_size());
            const int rows = static_cast<int>(expected.shape[1]);
            const int cols =
                static_cast<int>(expected.shape[2] * expected.shape[3]);
            launch_transpose_2d(boundary_transpose_scratch_.address(),
                                input_buffers_[graph_index].address(),
                                rows, cols);
        } else {
            driver_.upload_async(input_buffers_[graph_index],
                                 slot.inputs[graph_index].data(),
                                 input.byte_size());
        }
    }

    enqueue_model();
    enqueue_outputs(slot);
    driver_.synchronize();

    std::vector<Tensor> outputs;
    outputs.reserve(external_outputs_.size());
    for (const ExternalOutput& output : external_outputs_) {
        const Value& result = value(output.id);
        Tensor tensor;
        tensor.name = result.name;
        tensor.shape = result.shape;
        if (result.dtype == DataType::FP16) {
            tensor.data =
                std::vector<Float16Storage>(output.bytes / sizeof(Float16Storage));
        } else if (result.dtype == DataType::INT32) {
            tensor.data =
                std::vector<std::int32_t>(output.bytes / sizeof(std::int32_t));
        } else {
            throw std::runtime_error(
                "driver-cubin: unsupported graph output dtype");
        }
        std::visit([&](auto& storage) {
            std::memcpy(storage.data(), slot.outputs[output.output_index].data(),
                        output.bytes);
        }, tensor.data);
        outputs.push_back(std::move(tensor));
    }
    return outputs;
}

void AotGraph::ensure_host_slots(std::size_t count)
{
    while (host_slots_.size() < count) {
        HostSlot slot;
        slot.inputs.resize(input_ids_.size());
        slot.outputs.resize(output_ids_.size());
        slot.htod_done = driver_.create_event();
        slot.compute_done = driver_.create_event();
        slot.completion = driver_.create_event();
        for (std::size_t index = 0; index < input_ids_.size(); ++index) {
            if (is_state_input(index))
                continue;
            slot.inputs[index] = driver_.allocate_host_pinned(
                input_buffers_[index].size());
        }
        for (std::size_t index = 0; index < output_ids_.size(); ++index) {
            if (is_state_output(index))
                continue;
            const Value& result = value(output_ids_[index]);
            const std::size_t bytes =
                element_count(result.shape) * dtype_bytes(result.dtype);
            slot.outputs[index] = driver_.allocate_host_pinned(bytes);
        }
        host_slots_.push_back(std::move(slot));
    }
    if (codec_io_configured_) {
        const std::size_t bytes = yuv_bytes(codec_io_);
        for (HostSlot& slot : host_slots_) {
            if (!slot.yuv)
                slot.yuv = driver_.allocate_host_pinned(bytes);
            if (!slot.yuv_device)
                slot.yuv_device = driver_.allocate(bytes);
            if (q_input_index_ && !slot.q_device)
                slot.q_device = driver_.allocate(sizeof(int));
            if (model_name_ == "MLVCEncoder") {
                if (slot.output_staging.empty()) {
                    slot.output_staging.resize(output_ids_.size());
                    for (const ExternalOutput& out : external_outputs_) {
                        slot.output_staging[out.output_index] =
                            driver_.allocate(out.bytes);
                    }
                }
            } else if (model_name_ == "MLVCDecoder") {
                if (slot.input_staging.empty()) {
                    slot.input_staging.resize(input_ids_.size());
                    for (const std::size_t idx : decoder_latent_inputs_) {
                        slot.input_staging[idx] =
                            driver_.allocate(input_buffers_[idx].size());
                    }
                }
            }
            if (!slot.codec_graph)
                capture_codec_graph(slot);
        }
    }
}

AotGraph::HostSlot& AotGraph::host_slot(std::size_t slot)
{
    if (slot >= host_slots_.size())
        throw std::runtime_error("driver-cubin: codec slot is out of range");
    return host_slots_[slot];
}

const AotGraph::HostSlot& AotGraph::host_slot(std::size_t slot) const
{
    if (slot >= host_slots_.size())
        throw std::runtime_error("driver-cubin: codec slot is out of range");
    return host_slots_[slot];
}

void AotGraph::configure_codec_io(const CodecIoConfig& config)
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    auto ctx_guard = driver_.bind_context();
    if (codec_io_configured_)
        throw std::runtime_error("driver-cubin: codec I/O is already configured");
    const std::size_t bytes = yuv_bytes(config);
    if (model_name_ != "MLVCEncoder" && model_name_ != "MLVCDecoder")
        throw std::runtime_error("driver-cubin: unsupported codec graph");
    const Value& image =
        value(model_name_ == "MLVCEncoder" ? "x" : "x_hat");
    if (image.shape.size() != 4 || image.shape[0] != 1 || image.shape[1] != 3 ||
        config.model_width != image.shape[3] ||
        config.model_height != image.shape[2]) {
        throw std::runtime_error(
            "driver-cubin: codec I/O model dimensions do not match graph");
    }

    codec_io_ = config;
    codec_io_configured_ = true;
    codec_image_address_ = image.address;
    codec_yuv_bytes_ = bytes;
    codec_fused_s2d8_node_.reset();
    codec_fused_d2s8_node_.reset();
    codec_fused_d2s8_clip_node_.reset();
    codec_s2d8_output_ = 0;
    codec_d2s8_input_ = 0;
    codec_d2s8_has_clip_ = false;
    if (use_nhwc_ && (config.model_width % 8) == 0 &&
        (config.model_height % 8) == 0) {
        const std::vector<int64_t> s2d_shape{
            1, 192, config.model_height / 8, config.model_width / 8};
        if (model_name_ == "MLVCEncoder") {
            const TensorId x_id = tensor_id("x");
            if (tensor_consumer_count(x_id) == 1) {
                for (const auto& node : graph_.nodes) {
                    if (node.op == Op::SPACE_TO_DEPTH &&
                        node.block_size() == 8 && node.inputs.size() == 1 &&
                        node.inputs[0] == x_id && node.outputs.size() == 1) {
                        const TensorId out_id = node.outputs[0];
                        const Value& out_v = value(out_id);
                        if (out_v.dtype == DataType::FP16 &&
                            out_v.shape == s2d_shape && is_packed_nhwc(out_v) &&
                            std::find(output_ids_.begin(), output_ids_.end(),
                                      out_id) == output_ids_.end()) {
                            codec_fused_s2d8_node_ = node.index;
                            codec_s2d8_output_ = out_v.address;
                        }
                        break;
                    }
                }
            }
        } else if (model_name_ == "MLVCDecoder") {
            const TensorId x_hat_id = tensor_id("x_hat");
            const Node* prod = nullptr;
            for (const auto& node : graph_.nodes) {
                if (node.outputs.size() == 1 && node.outputs[0] == x_hat_id) {
                    prod = &node;
                    break;
                }
            }
            if (prod != nullptr && prod->op == Op::DEPTH_TO_SPACE &&
                prod->block_size() == 8 && prod->inputs.size() == 1) {
                const Value& in_v = value(prod->inputs[0]);
                if (in_v.dtype == DataType::FP16 && in_v.shape == s2d_shape &&
                    is_packed_nhwc(in_v)) {
                    codec_fused_d2s8_node_ = prod->index;
                    codec_d2s8_input_ = in_v.address;
                    codec_d2s8_has_clip_ = false;
                }
            } else if (prod != nullptr && prod->op == Op::CLIP &&
                       !prod->inputs.empty() &&
                       tensor_consumer_count(prod->inputs[0]) == 1 &&
                       std::find(output_ids_.begin(), output_ids_.end(),
                                 prod->inputs[0]) == output_ids_.end()) {
                const TensorId mid_id = prod->inputs[0];
                const Node* d2s = nullptr;
                for (const auto& node : graph_.nodes) {
                    if (node.outputs.size() == 1 && node.outputs[0] == mid_id) {
                        d2s = &node;
                        break;
                    }
                }
                if (d2s != nullptr && d2s->op == Op::DEPTH_TO_SPACE &&
                    d2s->block_size() == 8 && d2s->inputs.size() == 1) {
                    const Value& in_v = value(d2s->inputs[0]);
                    if (in_v.dtype == DataType::FP16 && in_v.shape == s2d_shape &&
                        is_packed_nhwc(in_v)) {
                        codec_fused_d2s8_node_ = d2s->index;
                        codec_fused_d2s8_clip_node_ = prod->index;
                        codec_d2s8_input_ = in_v.address;
                        codec_d2s8_has_clip_ = true;
                        codec_d2s8_clip_min_ = -std::numeric_limits<float>::infinity();
                        codec_d2s8_clip_max_ = std::numeric_limits<float>::infinity();
                        if (prod->inputs.size() > 1 && prod->inputs[1] != 0)
                            codec_d2s8_clip_min_ = scalar_fp16(prod->inputs[1]);
                        if (prod->inputs.size() > 2 && prod->inputs[2] != 0)
                            codec_d2s8_clip_max_ = scalar_fp16(prod->inputs[2]);
                    }
                }
            }
        }
    }
    if (model_name_ == "MLVCEncoder") {
        std::vector<Float16Storage> byte_lut(256);
        for (std::size_t value = 0; value < byte_lut.size(); ++value) {
            byte_lut[value] = float_to_half(
                (static_cast<float>(value) / 255.0F) * config.pixel_range);
        }
        codec_byte_lut_device_ = driver_.allocate(
            byte_lut.size() * sizeof(Float16Storage));
        driver_.upload(codec_byte_lut_device_, byte_lut.data(),
                       byte_lut.size() * sizeof(Float16Storage));
    }
    ensure_host_slots(config.slots);
}

std::size_t AotGraph::codec_slot_count() const noexcept
{
    return codec_io_configured_ ? host_slots_.size() : 0;
}

MutableYuv420FrameView AotGraph::encoder_input_yuv(std::size_t slot_index)
{
    if (!codec_io_configured_ || model_name_ != "MLVCEncoder")
        throw std::runtime_error("driver-cubin: encoder codec I/O is unavailable");
    HostSlot& slot = host_slot(slot_index);
    if (slot.pending)
        throw std::runtime_error("driver-cubin: encoder codec slot is pending");
    auto* bytes = static_cast<std::uint8_t*>(slot.yuv.data());
    const std::size_t y_size =
        static_cast<std::size_t>(codec_io_.width) * codec_io_.height;
    const std::size_t uv_size = y_size / 4;
    const std::span<std::uint8_t> storage(bytes, slot.yuv.size());
    return MutableYuv420FrameView(
        codec_io_.width, codec_io_.height, storage.first(y_size),
        storage.subspan(y_size, uv_size),
        storage.subspan(y_size + uv_size, uv_size));
}

void AotGraph::submit_encoder(std::size_t slot_index, int shifted_q)
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    auto ctx_guard = driver_.bind_context();
    if (!codec_io_configured_ || model_name_ != "MLVCEncoder")
        throw std::runtime_error("driver-cubin: encoder codec I/O is unavailable");
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error(
            "driver-cubin: reset_state() must be called before submit_encoder()");
    HostSlot& slot = host_slot(slot_index);
    if (slot.pending)
        throw std::runtime_error("driver-cubin: encoder codec slot is pending");
    if (!q_input_index_)
        throw std::runtime_error("driver-cubin: graph has no valid q_index_shifted input");

    const std::size_t q_idx = *q_input_index_;
    std::memcpy(slot.inputs[q_idx].data(), &shifted_q, sizeof(shifted_q));

    // HtoD stream: upload frame YUV + q_index into slot-private device buffers
    // while the previous slot is computing on compute_stream.
    driver_.upload_async_htod(slot.yuv_device, slot.yuv.data(), codec_yuv_bytes_);
    driver_.upload_async_htod(slot.q_device, slot.inputs[q_idx].data(),
                              sizeof(shifted_q));
    driver_.record_htod(slot.htod_done);

    // Compute stream: wait for this slot's HtoD, launch full per-slot graph
    // (yuv2nchw + q DtoD + model + outputs DtoD into slot.output_staging).
    driver_.wait_compute_on(slot.htod_done);
    driver_.launch_graph(slot.codec_graph);
    driver_.record(slot.compute_done);

    // DtoH stream: download slot-private staged outputs while next slot computes.
    driver_.wait_dtoh_on(slot.compute_done);
    for (const ExternalOutput& output : external_outputs_) {
        driver_.download_async_dtoh(
            slot.outputs[output.output_index].data(),
            slot.output_staging[output.output_index], output.bytes);
    }
    driver_.record_dtoh(slot.completion);
    slot.pending = true;
}

std::vector<TensorView> AotGraph::encoder_outputs(
    std::size_t slot_index) const
{
    if (!codec_io_configured_ || model_name_ != "MLVCEncoder")
        throw std::runtime_error("driver-cubin: encoder codec I/O is unavailable");
    const HostSlot& slot = host_slot(slot_index);
    if (slot.pending)
        throw std::runtime_error("driver-cubin: encoder codec slot is pending");
    std::vector<TensorView> outputs;
    outputs.reserve(external_outputs_.size());
    for (const ExternalOutput& output : external_outputs_) {
        const Value& result = value(output.id);
        outputs.push_back(TensorView::from_raw(
            result.name, std::span<const int64_t>(result.shape),
            public_dtype(result.dtype), slot.outputs[output.output_index].data(),
            output.bytes));
    }
    return outputs;
}

std::vector<MutableTensorView> AotGraph::decoder_inputs(
    std::size_t slot_index)
{
    if (!codec_io_configured_ || model_name_ != "MLVCDecoder")
        throw std::runtime_error("driver-cubin: decoder codec I/O is unavailable");
    HostSlot& slot = host_slot(slot_index);
    if (slot.pending)
        throw std::runtime_error("driver-cubin: decoder codec slot is pending");
    std::vector<MutableTensorView> inputs;
    inputs.reserve(decoder_latent_inputs_.size());
    for (const std::size_t index : decoder_latent_inputs_) {
        const Value& input = value(input_ids_[index]);
        inputs.push_back(MutableTensorView::from_raw(
            input.name, std::span<const int64_t>(input.shape),
            public_dtype(input.dtype), slot.inputs[index].data(),
            input_buffers_[index].size()));
    }
    return inputs;
}

void AotGraph::submit_decoder(std::size_t slot_index, int shifted_q)
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    auto ctx_guard = driver_.bind_context();
    if (!codec_io_configured_ || model_name_ != "MLVCDecoder")
        throw std::runtime_error("driver-cubin: decoder codec I/O is unavailable");
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error(
            "driver-cubin: reset_state() must be called before submit_decoder()");
    HostSlot& slot = host_slot(slot_index);
    if (slot.pending)
        throw std::runtime_error("driver-cubin: decoder codec slot is pending");
    if (!q_input_index_)
        throw std::runtime_error("driver-cubin: graph has no valid q_index_shifted input");

    const std::size_t q_idx = *q_input_index_;
    std::memcpy(slot.inputs[q_idx].data(), &shifted_q, sizeof(shifted_q));

    // HtoD stream: upload latents + q_index into slot-private device staging.
    for (const std::size_t index : decoder_latent_inputs_) {
        driver_.upload_async_htod(slot.input_staging[index],
                                  slot.inputs[index].data(),
                                  input_buffers_[index].size());
    }
    driver_.upload_async_htod(slot.q_device, slot.inputs[q_idx].data(),
                              sizeof(shifted_q));
    driver_.record_htod(slot.htod_done);

    // Compute stream: wait for HtoD, launch full per-slot graph
    // (latents/q DtoD + model + nchw2yuv into slot.yuv_device).
    driver_.wait_compute_on(slot.htod_done);
    driver_.launch_graph(slot.codec_graph);
    driver_.record(slot.compute_done);

    // DtoH stream: download slot-private YUV while next slot computes.
    driver_.wait_dtoh_on(slot.compute_done);
    driver_.download_async_dtoh(slot.yuv.data(), slot.yuv_device,
                                codec_yuv_bytes_);
    driver_.record_dtoh(slot.completion);
    slot.pending = true;
}

Yuv420FrameView AotGraph::decoder_output_yuv(std::size_t slot_index) const
{
    if (!codec_io_configured_ || model_name_ != "MLVCDecoder")
        throw std::runtime_error("driver-cubin: decoder codec I/O is unavailable");
    const HostSlot& slot = host_slot(slot_index);
    if (slot.pending)
        throw std::runtime_error("driver-cubin: decoder codec slot is pending");
    const auto* bytes = static_cast<const std::uint8_t*>(slot.yuv.data());
    const std::size_t y_size =
        static_cast<std::size_t>(codec_io_.width) * codec_io_.height;
    const std::size_t uv_size = y_size / 4;
    const std::span<const std::uint8_t> storage(bytes, slot.yuv.size());
    return Yuv420FrameView(
        codec_io_.width, codec_io_.height, storage.first(y_size),
        storage.subspan(y_size, uv_size),
        storage.subspan(y_size + uv_size, uv_size));
}

void AotGraph::wait_codec_slot_locked(std::size_t slot_index)
{
    HostSlot& slot = host_slot(slot_index);
    if (!slot.pending)
        return;
    driver_.synchronize(slot.completion);
    slot.pending = false;
}

void AotGraph::wait_codec_slot(std::size_t slot_index)
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    auto ctx_guard = driver_.bind_context();
    wait_codec_slot_locked(slot_index);
}

AotGraph::~AotGraph()
{
    for (HostSlot& slot : host_slots_) {
        if (slot.pending) {
            try {
                driver_.synchronize(slot.completion);
            } catch (...) {
            }
        }
    }
    try {
        driver_.synchronize_all();
    } catch (...) {
    }
}

}  // namespace mlvc::driver_cubin_backend
