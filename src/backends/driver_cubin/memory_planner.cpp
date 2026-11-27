#include "aot_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace mlvc::driver_cubin_backend {

void AotGraph::plan_input_slice_aliases()
{
    for (const auto& node : graph_.nodes) {
        if (node.op != Op::SLICE)
            continue;
        const auto inputs =
            node.inputs;
        const auto outputs =
            node.outputs;
        if (inputs.empty() || outputs.size() != 1 ||
            std::find(input_ids_.begin(), input_ids_.end(), inputs[0]) ==
                input_ids_.end()) {
            continue;
        }
        const Value& input = value(inputs[0]);
        Value& output = values_.at(outputs[0]);
        if (input.dtype != output.dtype || input.shape.size() != 4 ||
            output.shape.size() != 4 || input.shape[0] != 1 ||
            output.shape[0] != 1 || input.shape[2] != output.shape[2] ||
            input.shape[3] != output.shape[3] ||
            output.shape[1] > input.shape[1]) {
            continue;
        }
        const int output_channels = static_cast<int>(output.shape[1]);
        for (int start = 0;
             start + output_channels <= static_cast<int>(input.shape[1]);
             ++start) {
            if (!is_channel_slice(node, start, start + output_channels))
                continue;
            if (use_nhwc_) {
                if ((start % 8) != 0 || (output_channels % 8) != 0 ||
                    (value_stride(input) % 8) != 0) {
                    break;
                }
                bool all_supported = true;
                for (const auto& consumer : graph_.nodes) {
                    if (std::find(consumer.inputs.begin(), consumer.inputs.end(),
                                  outputs[0]) != consumer.inputs.end()) {
                        if (!supports_strided_nhwc_input(consumer)) {
                            all_supported = false;
                            break;
                        }
                    }
                }
                if (!all_supported)
                    break;
                output.address = input.address +
                    static_cast<std::size_t>(start) * dtype_bytes(input.dtype);
                output.nhwc_stride = value_stride(input);
                aliased_input_slices_.push_back(node.index);
                break;
            }
            const std::size_t channel_bytes =
                static_cast<std::size_t>(input.shape[2] * input.shape[3]) *
                dtype_bytes(input.dtype);
            output.address = input.address +
                static_cast<std::size_t>(start) * channel_bytes;
            aliased_input_slices_.push_back(
                node.index);
            break;
        }
    }
}

void AotGraph::plan_spatial_slice_aliases()
{
    if (!use_nhwc_)
        return;
    for (const auto& node : graph_.nodes) {
        if (node.op != Op::SLICE || node.inputs.size() < 2 ||
            node.outputs.size() != 1) {
            continue;
        }
        const TensorId in_id = node.inputs[0];
        const TensorId out_id = node.outputs[0];
        if (in_id == 0 || out_id == 0 ||
            tensor_consumer_count(in_id) != 1 ||
            std::find(input_ids_.begin(), input_ids_.end(), in_id) !=
                input_ids_.end() ||
            std::find(output_ids_.begin(), output_ids_.end(), in_id) !=
                output_ids_.end() ||
            std::find(output_ids_.begin(), output_ids_.end(), out_id) !=
                output_ids_.end()) {
            continue;
        }
        Value& input = values_.at(in_id);
        Value& output = values_.at(out_id);
        if (input.initializer || input.dtype != DataType::FP16 ||
            output.dtype != DataType::FP16 || input.shape.size() != 4 ||
            output.shape.size() != 4 || input.shape[0] != 1 ||
            output.shape[0] != 1 || input.shape[1] != output.shape[1] ||
            input.shape[3] != output.shape[3] ||
            output.shape[2] >= input.shape[2] || !is_packed_nhwc(input) ||
            !std::holds_alternative<ArenaSlot>(graph_.tensors.at(in_id).storage) ||
            !std::holds_alternative<ArenaSlot>(graph_.tensors.at(out_id).storage)) {
            continue;
        }
        const auto starts_vals = initializer_values<std::int64_t>(node.inputs[1]);
        const auto axes_vals = node.inputs.size() > 3 && node.inputs[3] != 0
            ? initializer_values<std::int64_t>(node.inputs[3])
            : std::vector<std::int64_t>{};
        const auto steps_vals = node.inputs.size() > 4 && node.inputs[4] != 0
            ? initializer_values<std::int64_t>(node.inputs[4])
            : std::vector<std::int64_t>{};
        bool prefix_h_slice = true;
        for (std::size_t i = 0; i < starts_vals.size(); ++i) {
            int axis = axes_vals.empty() ? static_cast<int>(i)
                                         : static_cast<int>(axes_vals.at(i));
            if (axis < 0)
                axis += 4;
            int step = steps_vals.empty() ? 1 : static_cast<int>(steps_vals.at(i));
            if (starts_vals[i] != 0 || step != 1 || axis != 2) {
                prefix_h_slice = false;
                break;
            }
        }
        if (!prefix_h_slice)
            continue;
        // Allocate a dedicated buffer shared by input and its height-prefix slice
        // so the slice becomes a 0 us no-op with zero arena lifetime conflicts.
        const std::size_t bytes =
            element_count(input.shape) * dtype_bytes(input.dtype);
        alias_buffers_.push_back(driver_.allocate(bytes));
        input.address = alias_buffers_.back().address();
        output.address = alias_buffers_.back().address();
        output.nhwc_stride = input.nhwc_stride;
        aliased_input_slices_.push_back(node.index);
        aliased_spatial_values_.push_back(in_id);
        aliased_spatial_values_.push_back(out_id);
    }
}

void AotGraph::plan_epilogue_buffers()
{
    if (driver_.device_info().compute_major >= 8) {
        struct EpilogueInterval {
            TensorId name;
            std::size_t birth;
            std::size_t death;
            std::size_t bytes;
            std::size_t slot;
        };
        std::vector<EpilogueInterval> intervals;
        const auto& nodes = graph_.nodes;
        for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
            const auto& convolution = nodes.at(index);
            const auto& epilogue = nodes.at(index + 1);
            const Op epilogue_op =
                epilogue.op;
            if (convolution.op != Op::CONV ||
                (epilogue_op != Op::ADD && epilogue_op != Op::LEAKY_RELU)) {
                continue;
            }
            const auto convolution_outputs = convolution.outputs;
            const auto epilogue_inputs = epilogue.inputs;
            const auto epilogue_outputs = epilogue.outputs;
            if (convolution_outputs.size() != 1 ||
                epilogue_outputs.size() != 1 ||
                std::find(epilogue_inputs.begin(), epilogue_inputs.end(),
                          convolution_outputs[0]) == epilogue_inputs.end() ||
                !std::holds_alternative<ArenaSlot>(graph_.tensors.at(epilogue_outputs[0]).storage)) {
                continue;
            }
            const auto& allocation =
                std::get<ArenaSlot>(graph_.tensors.at(epilogue_outputs[0]).storage);
            intervals.push_back({
                epilogue_outputs[0],
                allocation.birth,
                allocation.death,
                allocation.bytes, 0});
        }

        std::vector<std::size_t> slot_deaths;
        std::vector<std::size_t> slot_bytes;
        for (EpilogueInterval& interval : intervals) {
            std::size_t slot = 0;
            while (slot < slot_deaths.size() &&
                   slot_deaths[slot] >= interval.birth) {
                ++slot;
            }
            if (slot == slot_deaths.size()) {
                slot_deaths.push_back(interval.death);
                slot_bytes.push_back(interval.bytes);
            } else {
                slot_deaths[slot] = interval.death;
                slot_bytes[slot] = std::max(slot_bytes[slot], interval.bytes);
            }
            interval.slot = slot;
        }
        epilogue_buffers_.reserve(slot_bytes.size());
        for (std::size_t bytes : slot_bytes)
            epilogue_buffers_.push_back(driver_.allocate(bytes));
        for (const EpilogueInterval& interval : intervals)
            values_.at(interval.name).address =
                epilogue_buffers_.at(interval.slot).address();
    }
}

void AotGraph::plan_reglu_buffer()
{
    std::size_t reglu_bytes = 0;
    const auto& nodes = graph_.nodes;
    for (std::size_t index = 0; index + 3 < nodes.size(); ++index) {
        if (nodes.at(index).op == Op::SLICE &&
            nodes.at(index + 1).op == Op::SLICE &&
            nodes.at(index + 2).op == Op::CLIP &&
            nodes.at(index + 3).op == Op::MUL) {
            const auto outputs = nodes.at(index + 3).outputs;
            if (outputs.size() == 1) {
                const Value& output = value(outputs[0]);
                reglu_bytes = std::max(
                    reglu_bytes,
                    element_count(output.shape) * dtype_bytes(output.dtype));
            }
        }
    }
    if (reglu_bytes != 0)
        reglu_buffer_ = driver_.allocate(reglu_bytes);
}

void AotGraph::plan_spatial_buffers()
{
    if (use_nhwc_)
        return;
    const auto& nodes = graph_.nodes;
    if (driver_.device_info().compute_major >= 8) {
        std::size_t spatial_input_bytes = 0;
        std::size_t spatial_output_bytes = 0;
        for (const auto& node : nodes) {
            if (node.op != Op::CONV)
                continue;
            const auto inputs =
                node.inputs;
            const auto outputs =
                node.outputs;
            const auto& attributes = node.conv();
            if (inputs.size() < 3 || (inputs[2] == 0) ||
                outputs.size() != 1 || attributes.groups != 1) {
                continue;
            }
            const Value& input = value(inputs[0]);
            const Value& weight = value(inputs[1]);
            const Value& output = value(outputs[0]);
            if (input.dtype != DataType::FP16 || weight.dtype != DataType::FP16 ||
                output.dtype != DataType::FP16 || input.shape.size() != 4 ||
                weight.shape.size() != 4 || output.shape.size() != 4 ||
                input.shape[0] != 1 ||
                weight.shape[2] * weight.shape[3] == 1 ||
                input.shape[1] % 8 != 0 || output.shape[1] % 8 != 0) {
                continue;
            }
            spatial_input_bytes = std::max(
                spatial_input_bytes,
                element_count(input.shape) * dtype_bytes(input.dtype));
            spatial_output_bytes = std::max(
                spatial_output_bytes,
                element_count(output.shape) * dtype_bytes(output.dtype));
        }
        if (spatial_input_bytes != 0 && spatial_output_bytes != 0) {
            spatial_input_buffer_ = driver_.allocate(spatial_input_bytes);
            spatial_output_buffer_ = driver_.allocate(spatial_output_bytes);
        }
    }
}

void AotGraph::plan_direct_concat_buffers()
{
    const auto& nodes = graph_.nodes;
    concat_buffers_.reserve(nodes.size());
    for (auto iterator = nodes.rbegin(); iterator != nodes.rend(); ++iterator) {
        const Node& node = *iterator;
        if (node.op != Op::CONCAT ||
            node.axis() != 1) {
            continue;
        }
        const auto inputs =
            node.inputs;
        const auto outputs =
            node.outputs;
        if (outputs.size() != 1)
            continue;
        Value& output = values_.at(outputs[0]);
        if (output.dtype != DataType::FP16 || output.shape.size() != 4 ||
            output.shape[0] != 1) {
            continue;
        }

        std::vector<bool> redirect(inputs.size(), false);
        bool has_redirect = false;
        std::size_t scan_channel_offset = 0;
        for (std::size_t input_index = 0; input_index < inputs.size();
             ++input_index) {
            TensorId input_name = inputs[input_index];
            const Value& input = value(input_name);
            const std::size_t this_offset = scan_channel_offset;
            scan_channel_offset += static_cast<std::size_t>(input.shape[1]);
            if (input.initializer || input.dtype != output.dtype ||
                input.shape.size() != 4 || input.shape[0] != 1 ||
                input.shape[2] != output.shape[2] ||
                input.shape[3] != output.shape[3] ||
                std::find(input_ids_.begin(), input_ids_.end(),
                          input_name) != input_ids_.end() ||
                std::find(output_ids_.begin(), output_ids_.end(),
                          input_name) != output_ids_.end() ||
                std::find(aliased_spatial_values_.begin(),
                          aliased_spatial_values_.end(), input_name) !=
                    aliased_spatial_values_.end()) {
                continue;
            }
            if (use_nhwc_ &&
                ((this_offset % 8U) != 0 || (output.shape[1] % 8) != 0 ||
                 (input.shape[1] % 8) != 0)) {
                continue;
            }
            std::size_t consumers = 0;
            const Node* producer_node = nullptr;
            for (const auto& candidate : nodes) {
                const auto candidate_inputs = candidate.inputs;
                consumers += static_cast<std::size_t>(std::count(
                    candidate_inputs.begin(), candidate_inputs.end(),
                    input_name));
                const auto candidate_outputs = candidate.outputs;
                if (std::find(candidate_outputs.begin(),
                              candidate_outputs.end(),
                              input_name) != candidate_outputs.end()) {
                    producer_node = &candidate;
                }
            }
            bool can_redirect = producer_node != nullptr && consumers == 1;
            if (can_redirect && use_nhwc_) {
                can_redirect = supports_strided_nhwc_output(*producer_node);
            }
            redirect[input_index] = can_redirect;
            has_redirect = has_redirect || redirect[input_index];
        }
        if (!has_redirect)
            continue;

        concat_buffers_.push_back(driver_.allocate(
            element_count(output.shape) * dtype_bytes(output.dtype)));
        output.address = concat_buffers_.back().address();
        std::size_t channel_offset = 0;
        const std::size_t channel_bytes =
            static_cast<std::size_t>(output.shape[2] * output.shape[3]) *
            dtype_bytes(output.dtype);
        for (std::size_t input_index = 0; input_index < inputs.size();
             ++input_index) {
            Value& input = values_.at(inputs[input_index]);
            if (redirect[input_index]) {
                if (use_nhwc_) {
                    input.address = output.address +
                        channel_offset * dtype_bytes(output.dtype);
                    input.nhwc_stride = value_stride(output);
                } else {
                    input.address =
                        output.address + channel_offset * channel_bytes;
                }
                direct_concat_values_.push_back(inputs[input_index]);
            }
            channel_offset += static_cast<std::size_t>(input.shape[1]);
        }
    }
}

}  // namespace mlvc::driver_cubin_backend
