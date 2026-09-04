#include "aot_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace mlvc::driver_cubin_backend {

void AotGraph::plan_input_slice_aliases()
{
    for (const auto& node : manifest_.at("nodes")) {
        if (node.at("op") != "Slice")
            continue;
        const auto inputs =
            node.at("inputs").get<std::vector<std::string>>();
        const auto outputs =
            node.at("outputs").get<std::vector<std::string>>();
        if (inputs.empty() || outputs.size() != 1 ||
            std::find(input_names_.begin(), input_names_.end(), inputs[0]) ==
                input_names_.end()) {
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
            const std::size_t channel_bytes =
                static_cast<std::size_t>(input.shape[2] * input.shape[3]) *
                dtype_bytes(input.dtype);
            output.address = input.address +
                static_cast<std::size_t>(start) * channel_bytes;
            aliased_input_slices_.push_back(
                node.at("index").get<std::size_t>());
            break;
        }
    }
}

void AotGraph::plan_epilogue_buffers()
{
    if (driver_.device_info().compute_major >= 8) {
        struct EpilogueInterval {
            std::string name;
            std::size_t birth;
            std::size_t death;
            std::size_t bytes;
            std::size_t slot;
        };
        std::vector<EpilogueInterval> intervals;
        const auto& nodes = manifest_.at("nodes");
        for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
            const auto& convolution = nodes.at(index);
            const auto& epilogue = nodes.at(index + 1);
            const std::string epilogue_op =
                epilogue.at("op").get<std::string>();
            if (convolution.at("op") != "Conv" ||
                (epilogue_op != "Add" && epilogue_op != "LeakyRelu")) {
                continue;
            }
            const auto convolution_outputs = convolution.at("outputs")
                .get<std::vector<std::string>>();
            const auto epilogue_inputs = epilogue.at("inputs")
                .get<std::vector<std::string>>();
            const auto epilogue_outputs = epilogue.at("outputs")
                .get<std::vector<std::string>>();
            if (convolution_outputs.size() != 1 ||
                epilogue_outputs.size() != 1 ||
                std::find(epilogue_inputs.begin(), epilogue_inputs.end(),
                          convolution_outputs[0]) == epilogue_inputs.end() ||
                !manifest_.at("arena").contains(epilogue_outputs[0])) {
                continue;
            }
            const auto& allocation =
                manifest_.at("arena").at(epilogue_outputs[0]);
            intervals.push_back({
                epilogue_outputs[0],
                allocation.at("birth").get<std::size_t>(),
                allocation.at("death").get<std::size_t>(),
                allocation.at("bytes").get<std::size_t>(), 0});
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
    const auto& nodes = manifest_.at("nodes");
    for (std::size_t index = 0; index + 3 < nodes.size(); ++index) {
        if (nodes.at(index).at("op") == "Slice" &&
            nodes.at(index + 1).at("op") == "Slice" &&
            nodes.at(index + 2).at("op") == "Clip" &&
            nodes.at(index + 3).at("op") == "Mul") {
            const auto outputs = nodes.at(index + 3).at("outputs")
                .get<std::vector<std::string>>();
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
    const auto& nodes = manifest_.at("nodes");
    if (driver_.device_info().compute_major >= 8) {
        std::size_t spatial_input_bytes = 0;
        std::size_t spatial_output_bytes = 0;
        for (const auto& node : nodes) {
            if (node.at("op") != "Conv")
                continue;
            const auto inputs =
                node.at("inputs").get<std::vector<std::string>>();
            const auto outputs =
                node.at("outputs").get<std::vector<std::string>>();
            const auto& attributes = node.at("attributes");
            if (inputs.size() < 3 || inputs[2].empty() ||
                outputs.size() != 1 || attributes.value("group", 1) != 1) {
                continue;
            }
            const Value& input = value(inputs[0]);
            const Value& weight = value(inputs[1]);
            const Value& output = value(outputs[0]);
            if (input.dtype != "fp16" || weight.dtype != "fp16" ||
                output.dtype != "fp16" || input.shape.size() != 4 ||
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
    const auto& nodes = manifest_.at("nodes");
    concat_buffers_.reserve(nodes.size());
    for (auto iterator = nodes.rbegin(); iterator != nodes.rend(); ++iterator) {
        const json& node = *iterator;
        if (node.at("op") != "Concat" ||
            node.at("attributes").at("axis").get<int>() != 1) {
            continue;
        }
        const auto inputs =
            node.at("inputs").get<std::vector<std::string>>();
        const auto outputs =
            node.at("outputs").get<std::vector<std::string>>();
        if (outputs.size() != 1)
            continue;
        Value& output = values_.at(outputs[0]);
        if (output.dtype != "fp16" || output.shape.size() != 4 ||
            output.shape[0] != 1) {
            continue;
        }

        std::vector<bool> redirect(inputs.size(), false);
        bool has_redirect = false;
        for (std::size_t input_index = 0; input_index < inputs.size();
             ++input_index) {
            const std::string& input_name = inputs[input_index];
            const Value& input = value(input_name);
            if (input.initializer || input.dtype != output.dtype ||
                input.shape.size() != 4 || input.shape[0] != 1 ||
                input.shape[2] != output.shape[2] ||
                input.shape[3] != output.shape[3] ||
                std::find(input_names_.begin(), input_names_.end(),
                          input_name) != input_names_.end() ||
                std::find(output_names_.begin(), output_names_.end(),
                          input_name) != output_names_.end()) {
                continue;
            }
            std::size_t consumers = 0;
            bool produced = false;
            for (const auto& candidate : nodes) {
                const auto candidate_inputs = candidate.at("inputs")
                    .get<std::vector<std::string>>();
                consumers += static_cast<std::size_t>(std::count(
                    candidate_inputs.begin(), candidate_inputs.end(),
                    input_name));
                const auto candidate_outputs = candidate.at("outputs")
                    .get<std::vector<std::string>>();
                produced = produced || std::find(
                    candidate_outputs.begin(), candidate_outputs.end(),
                    input_name) != candidate_outputs.end();
            }
            redirect[input_index] = produced && consumers == 1;
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
                input.address = output.address + channel_offset * channel_bytes;
                direct_concat_values_.push_back(inputs[input_index]);
            }
            channel_offset += static_cast<std::size_t>(input.shape[1]);
        }
    }
}

}  // namespace mlvc::driver_cubin_backend
