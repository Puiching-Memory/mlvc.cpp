#include "aot_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
        config.slots == 0 || config.pixel_range <= 0.0F) {
        throw std::runtime_error("driver-cubin: invalid codec I/O configuration");
    }
    return static_cast<std::size_t>(config.width) * config.height * 3U / 2U;
}

}  // namespace

void AotGraph::reset_state()
{
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
    for (std::size_t index = 0; index < output_names_.size(); ++index) {
        if (is_state_output(index))
            continue;
        const Value& result = value(output_names_[index]);
        const std::size_t bytes =
            element_count(result.shape) * dtype_bytes(result.dtype);
        driver_.download_async(slot.outputs[index], result.address, bytes);
    }
}

void AotGraph::set_q_index(HostSlot& slot, int shifted_q)
{
    const auto found = std::find(
        input_names_.begin(), input_names_.end(), "q_index_shifted");
    if (found == input_names_.end())
        throw std::runtime_error("driver-cubin: graph has no q_index_shifted input");
    const std::size_t index = static_cast<std::size_t>(
        std::distance(input_names_.begin(), found));
    const Value& q = value(*found);
    if (q.dtype != "int32" || q.shape != std::vector<int64_t>{1} ||
        slot.inputs[index] == nullptr) {
        throw std::runtime_error("driver-cubin: invalid q_index_shifted input");
    }
    std::memcpy(slot.inputs[index], &shifted_q, sizeof(shifted_q));
    driver_.upload_async(input_buffers_[index], slot.inputs[index],
                         sizeof(shifted_q));
}

void AotGraph::finish_submit(HostSlot& slot)
{
    driver_.record(slot.completion);
    slot.pending = true;
}

std::vector<Tensor> AotGraph::run(const std::vector<Tensor>& inputs)
{
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error(
            "driver-cubin: reset_state() must be called before run()");
    if (inputs.size() + state_bindings_.size() != input_names_.size())
        throw std::runtime_error("driver-cubin: graph input count mismatch");

    ensure_host_slots(1);
    HostSlot& slot = host_slots_.front();
    wait_codec_slot(0);
    std::size_t external_index = 0;
    for (std::size_t graph_index = 0; graph_index < input_names_.size();
         ++graph_index) {
        if (is_state_input(graph_index))
            continue;
        const Tensor& input = inputs.at(external_index++);
        const Value& expected = value(input_names_[graph_index]);
        if (input.shape != expected.shape ||
            input.data_type() != public_dtype(expected.dtype)) {
            throw std::runtime_error(
                "driver-cubin: graph input shape or dtype mismatch");
        }
        if (input.byte_size() != input_buffers_[graph_index].size())
            throw std::runtime_error(
                "driver-cubin: graph input byte size mismatch");
        std::memcpy(slot.inputs[graph_index], input.raw_data(),
                    input.byte_size());
        driver_.upload_async(input_buffers_[graph_index],
                             slot.inputs[graph_index], input.byte_size());
    }

    enqueue_model();
    enqueue_outputs(slot);
    driver_.synchronize();

    std::vector<Tensor> outputs;
    outputs.reserve(output_names_.size() - state_bindings_.size());
    for (std::size_t index = 0; index < output_names_.size(); ++index) {
        if (is_state_output(index))
            continue;
        const std::string& name = output_names_[index];
        const Value& result = value(name);
        Tensor tensor;
        tensor.name = name;
        tensor.shape = result.shape;
        if (result.dtype == "fp16") {
            tensor.data =
                std::vector<Float16Storage>(element_count(result.shape));
        } else if (result.dtype == "int32") {
            tensor.data =
                std::vector<std::int32_t>(element_count(result.shape));
        } else {
            throw std::runtime_error(
                "driver-cubin: unsupported graph output dtype");
        }
        std::visit([&](auto& storage) {
            std::memcpy(storage.data(), slot.outputs[index],
                        storage.size() * sizeof(
                            typename std::decay_t<decltype(storage)>::value_type));
        }, tensor.data);
        outputs.push_back(std::move(tensor));
    }
    return outputs;
}

void AotGraph::ensure_host_slots(std::size_t count)
{
    while (host_slots_.size() < count) {
        HostSlot slot;
        slot.inputs.resize(input_names_.size(), nullptr);
        slot.outputs.resize(output_names_.size(), nullptr);
        slot.completion = driver_.create_event();
        host_slots_.push_back(std::move(slot));
        HostSlot& inserted = host_slots_.back();
        for (std::size_t index = 0; index < input_names_.size(); ++index) {
            if (is_state_input(index))
                continue;
            inserted.inputs[index] =
                driver_.allocate_host_pinned(input_buffers_[index].size());
            if (!inserted.inputs[index])
                throw std::runtime_error(
                    "driver-cubin: failed to allocate pinned input staging");
        }
        for (std::size_t index = 0; index < output_names_.size(); ++index) {
            if (is_state_output(index))
                continue;
            const Value& result = value(output_names_[index]);
            const std::size_t bytes =
                element_count(result.shape) * dtype_bytes(result.dtype);
            inserted.outputs[index] = driver_.allocate_host_pinned(bytes);
            if (!inserted.outputs[index])
                throw std::runtime_error(
                    "driver-cubin: failed to allocate pinned output staging");
        }
    }
    if (codec_io_configured_) {
        const std::size_t bytes = yuv_bytes(codec_io_);
        for (HostSlot& slot : host_slots_) {
            if (slot.yuv)
                continue;
            slot.yuv = driver_.allocate_host_pinned(bytes);
            if (!slot.yuv)
                throw std::runtime_error(
                    "driver-cubin: failed to allocate pinned YUV staging");
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
    codec_yuv_device_ = driver_.allocate(bytes);
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
    auto* bytes = static_cast<std::uint8_t*>(slot.yuv);
    const std::size_t y_size =
        static_cast<std::size_t>(codec_io_.width) * codec_io_.height;
    const std::size_t uv_size = y_size / 4;
    return MutableYuv420FrameView{
        codec_io_.width, codec_io_.height, bytes,
        bytes + y_size, bytes + y_size + uv_size};
}

void AotGraph::submit_encoder(std::size_t slot_index, int shifted_q)
{
    if (!codec_io_configured_ || model_name_ != "MLVCEncoder")
        throw std::runtime_error("driver-cubin: encoder codec I/O is unavailable");
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error(
            "driver-cubin: reset_state() must be called before submit_encoder()");
    HostSlot& slot = host_slot(slot_index);
    if (slot.pending)
        throw std::runtime_error("driver-cubin: encoder codec slot is pending");

    driver_.upload_async(codec_yuv_device_, slot.yuv, yuv_bytes(codec_io_));
    DeviceAddress input = codec_yuv_device_.address();
    DeviceAddress lut = codec_byte_lut_device_.address();
    DeviceAddress output = value("x").address;
    int width = codec_io_.width;
    int height = codec_io_.height;
    int model_width = codec_io_.model_width;
    int model_height = codec_io_.model_height;
    int rotated = codec_io_.padding.rotated ? 1 : 0;
    void* parameters[] = {&input, &lut, &output, &width, &height,
                          &model_width, &model_height, &rotated};
    launch_linear(yuv420_to_nchw_,
                  static_cast<std::size_t>(3) * model_width * model_height,
                  parameters);
    set_q_index(slot, shifted_q);
    enqueue_model();
    enqueue_outputs(slot);
    finish_submit(slot);
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
    outputs.reserve(output_names_.size() - state_bindings_.size());
    for (std::size_t index = 0; index < output_names_.size(); ++index) {
        if (is_state_output(index))
            continue;
        const Value& result = value(output_names_[index]);
        outputs.push_back(TensorView{
            output_names_[index], std::span<const int64_t>(result.shape),
            public_dtype(result.dtype), slot.outputs[index],
            element_count(result.shape) * dtype_bytes(result.dtype)});
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
    inputs.reserve(input_names_.size() - state_bindings_.size() - 1);
    for (std::size_t index = 0; index < input_names_.size(); ++index) {
        if (is_state_input(index) || input_names_[index] == "q_index_shifted")
            continue;
        const Value& input = value(input_names_[index]);
        inputs.push_back(MutableTensorView{
            input_names_[index], std::span<const int64_t>(input.shape),
            public_dtype(input.dtype), slot.inputs[index],
            input_buffers_[index].size()});
    }
    return inputs;
}

void AotGraph::submit_decoder(std::size_t slot_index, int shifted_q)
{
    if (!codec_io_configured_ || model_name_ != "MLVCDecoder")
        throw std::runtime_error("driver-cubin: decoder codec I/O is unavailable");
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error(
            "driver-cubin: reset_state() must be called before submit_decoder()");
    HostSlot& slot = host_slot(slot_index);
    if (slot.pending)
        throw std::runtime_error("driver-cubin: decoder codec slot is pending");

    for (std::size_t index = 0; index < input_names_.size(); ++index) {
        if (is_state_input(index) || input_names_[index] == "q_index_shifted")
            continue;
        driver_.upload_async(input_buffers_[index], slot.inputs[index],
                             input_buffers_[index].size());
    }
    set_q_index(slot, shifted_q);
    enqueue_model();

    DeviceAddress input = value("x_hat").address;
    DeviceAddress output = codec_yuv_device_.address();
    int width = codec_io_.width;
    int height = codec_io_.height;
    int model_width = codec_io_.model_width;
    int model_height = codec_io_.model_height;
    int pad_left = codec_io_.padding.left;
    int pad_top = codec_io_.padding.top;
    int rotated = codec_io_.padding.rotated ? 1 : 0;
    float inverse_pixel_range = 1.0F / codec_io_.pixel_range;
    void* parameters[] = {
        &input, &output, &width, &height, &model_width, &model_height,
        &pad_left, &pad_top, &rotated, &inverse_pixel_range};
    launch_linear(nchw_to_yuv420_,
                  static_cast<std::size_t>(width) * height, parameters);
    driver_.download_async(slot.yuv, codec_yuv_device_, yuv_bytes(codec_io_));
    finish_submit(slot);
}

Yuv420FrameView AotGraph::decoder_output_yuv(std::size_t slot_index) const
{
    if (!codec_io_configured_ || model_name_ != "MLVCDecoder")
        throw std::runtime_error("driver-cubin: decoder codec I/O is unavailable");
    const HostSlot& slot = host_slot(slot_index);
    if (slot.pending)
        throw std::runtime_error("driver-cubin: decoder codec slot is pending");
    const auto* bytes = static_cast<const std::uint8_t*>(slot.yuv);
    const std::size_t y_size =
        static_cast<std::size_t>(codec_io_.width) * codec_io_.height;
    const std::size_t uv_size = y_size / 4;
    return Yuv420FrameView{
        codec_io_.width, codec_io_.height, bytes,
        bytes + y_size, bytes + y_size + uv_size};
}

void AotGraph::wait_codec_slot(std::size_t slot_index)
{
    HostSlot& slot = host_slot(slot_index);
    if (!slot.pending)
        return;
    driver_.synchronize(slot.completion);
    slot.pending = false;
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
        driver_.free_host_pinned(slot.yuv);
        for (void* pointer : slot.outputs)
            driver_.free_host_pinned(pointer);
        for (void* pointer : slot.inputs)
            driver_.free_host_pinned(pointer);
    }
}

}  // namespace mlvc::driver_cubin_backend
