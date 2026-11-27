#include "aot_graph.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlvc::driver_cubin_backend {

void AotGraph::configure_state(const ModelExecutionConfig& config)
{
    const auto final_outputs = graph_.nodes.back().outputs;
    for (const StateTensorBinding& requested : config.state_bindings) {
        if (requested.input_index >= input_ids_.size() ||
            requested.output_index >= output_ids_.size()) {
            throw std::runtime_error(
                "driver-cubin: state tensor index is out of range");
        }
        if (is_state_input(requested.input_index) ||
            is_state_output(requested.output_index)) {
            throw std::runtime_error(
                "driver-cubin: duplicate state tensor binding");
        }

        const Value& input = value(input_ids_[requested.input_index]);
        Value& output = values_.at(output_ids_[requested.output_index]);
        if (input.dtype != output.dtype || input.shape != output.shape ||
            requested.data_type != public_dtype(input.dtype) ||
            requested.shape != input.shape) {
            throw std::runtime_error(
                "driver-cubin: state tensor shape or dtype mismatch");
        }
        if (std::find(final_outputs.begin(), final_outputs.end(),
                      output_ids_[requested.output_index]) ==
            final_outputs.end()) {
            throw std::runtime_error(
                "driver-cubin: in-place state output must be produced by "
                "the final graph node");
        }

        output.address = input_buffers_[requested.input_index].address();
        state_bindings_.push_back(
            {requested.input_index, requested.output_index});
    }
}

bool AotGraph::is_state_input(std::size_t index) const
{
    return std::any_of(
        state_bindings_.begin(), state_bindings_.end(),
        [&](const StateBinding& binding) {
            return binding.input_index == index;
        });
}

bool AotGraph::is_state_output(std::size_t index) const
{
    return std::any_of(
        state_bindings_.begin(), state_bindings_.end(),
        [&](const StateBinding& binding) {
            return binding.output_index == index;
        });
}

bool AotGraph::is_channel_slice(const Node& node, int start, int end) const
{
    const auto inputs = node.inputs;
    if (inputs.size() < 4 || (inputs[1] == 0) || (inputs[2] == 0) ||
        (inputs[3] == 0)) {
        return false;
    }
    const auto starts = initializer_values<std::int64_t>(inputs[1]);
    const auto ends = initializer_values<std::int64_t>(inputs[2]);
    const auto axes = initializer_values<std::int64_t>(inputs[3]);
    const auto steps = inputs.size() > 4 && !(inputs[4] == 0)
        ? initializer_values<std::int64_t>(inputs[4])
        : std::vector<std::int64_t>{1};
    return starts.size() == 1 && ends.size() == 1 && axes.size() == 1 &&
           steps.size() == 1 && starts[0] == start && ends[0] == end &&
           (axes[0] == 1 || axes[0] == -3) && steps[0] == 1;
}

bool AotGraph::ranges_overlap(const Value& lhs, const Value& rhs)
{
    auto span_bytes = [](const Value& v) -> std::size_t {
        if (v.nhwc_stride > 0 && v.shape.size() == 4) {
            const std::size_t pixels = static_cast<std::size_t>(
                v.shape[0] * v.shape[2] * v.shape[3]);
            if (pixels == 0)
                return 0;
            return ((pixels - 1U) * static_cast<std::size_t>(v.nhwc_stride) +
                    static_cast<std::size_t>(v.shape[1])) *
                dtype_bytes(v.dtype);
        }
        return element_count(v.shape) * dtype_bytes(v.dtype);
    };
    const std::size_t lhs_bytes = span_bytes(lhs);
    const std::size_t rhs_bytes = span_bytes(rhs);
    return lhs.address < rhs.address + rhs_bytes &&
           rhs.address < lhs.address + lhs_bytes;
}

bool AotGraph::supports_strided_nhwc_input(const Node& consumer) const
{
    if (consumer.op == Op::CONV) {
        if (consumer.inputs.size() < 2)
            return false;
        const Value& weight = value(consumer.inputs[1]);
        const Value& input = value(consumer.inputs[0]);
        const auto& attrs = consumer.conv();
        const int kh = static_cast<int>(weight.shape.at(2));
        const int kw = static_cast<int>(weight.shape.at(3));
        const bool is_pw1x1 = attrs.groups == 1 && kh == 1 && kw == 1 &&
            attrs.strides[0] == 1 && attrs.strides[1] == 1 &&
            attrs.pads[0] == 0 && attrs.pads[1] == 0;
        const bool is_dw3x3 = attrs.groups == static_cast<int>(input.shape[1]) &&
            weight.shape[0] == input.shape[1] && weight.shape[1] == 1 &&
            kh == 3 && kw == 3 && attrs.strides[0] == 1 &&
            attrs.strides[1] == 1 && attrs.pads[0] == 1 && attrs.pads[1] == 1 &&
            (input.shape[1] % 4 == 0);
        return is_pw1x1 || is_dw3x3;
    }
    return consumer.op != Op::GATHER;
}

bool AotGraph::supports_strided_nhwc_output(const Node& producer) const
{
    if (producer.op == Op::CONV) {
        if (producer.inputs.size() < 2)
            return false;
        const Value& weight = value(producer.inputs[1]);
        const Value& input = value(producer.inputs[0]);
        const auto& attrs = producer.conv();
        const int kh = static_cast<int>(weight.shape.at(2));
        const int kw = static_cast<int>(weight.shape.at(3));
        const bool is_pw1x1 = attrs.groups == 1 && kh == 1 && kw == 1 &&
            attrs.strides[0] == 1 && attrs.strides[1] == 1 &&
            attrs.pads[0] == 0 && attrs.pads[1] == 0 &&
            (input.shape[1] % 8 == 0) && (weight.shape[0] % 8 == 0);
        const bool is_dw3x3 = attrs.groups == static_cast<int>(input.shape[1]) &&
            weight.shape[0] == input.shape[1] && weight.shape[1] == 1 &&
            kh == 3 && kw == 3 && attrs.strides[0] == 1 &&
            attrs.strides[1] == 1 && attrs.pads[0] == 1 && attrs.pads[1] == 1 &&
            (input.shape[1] % 4 == 0);
        return is_pw1x1 || is_dw3x3;
    }
    return producer.op != Op::GATHER && producer.op != Op::CONCAT;
}

bool AotGraph::is_nchw_external_tensor(const Value& v) const noexcept
{
    return v.name == "y_raw_0" || v.name == "y_raw_1";
}

bool AotGraph::needs_nhwc_boundary_transpose(const Value& v) const noexcept
{
    return use_nhwc_ && v.dtype == DataType::FP16 && v.shape.size() == 4 &&
           v.shape[0] == 1 && v.shape[1] > 1 &&
           (v.shape[2] * v.shape[3]) > 1 && !is_nchw_external_tensor(v);
}

void AotGraph::ensure_boundary_transpose_scratch(std::size_t bytes)
{
    if (!boundary_transpose_scratch_ ||
        boundary_transpose_scratch_.size() < bytes) {
        boundary_transpose_scratch_ = driver_.allocate(bytes);
    }
}

void AotGraph::launch_transpose_2d(DeviceAddress src, DeviceAddress dst,
                                   int rows, int cols)
{
    void* params[] = {&src, &dst, &rows, &cols};
    driver_.launch(transpose_,
                   {divide_up(static_cast<std::size_t>(cols), 32),
                    divide_up(static_cast<std::size_t>(rows), 32), 1},
                   {32, 8, 1}, 0, params);
}

DeviceAddress AotGraph::get_or_pack_dwconv_weight(std::size_t node_index,
                                                  DeviceAddress weight,
                                                  int channels)
{
    auto [it, inserted] = dwconv_weight_buffers_.try_emplace(node_index);
    if (inserted) {
        const std::size_t bytes =
            static_cast<std::size_t>(channels) * 9U * sizeof(Float16Storage);
        it->second = driver_.allocate(bytes);
        DeviceAddress dst = it->second.address();
        void* params[] = {&weight, &dst, &channels};
        driver_.launch(dwconv_weight_oihw_to_k9c_,
                       {divide_up(static_cast<std::size_t>(channels), 128), 9, 1},
                       {128, 1, 1}, 0, params);
    }
    return it->second.address();
}

}  // namespace mlvc::driver_cubin_backend

