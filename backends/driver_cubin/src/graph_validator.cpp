#include "aot_graph.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlvc::driver_cubin_backend {

void AotGraph::configure_state(const ModelExecutionConfig& config)
{
    const auto final_outputs = manifest_.at("nodes").back().at("outputs")
        .get<std::vector<std::string>>();
    for (const StateTensorBinding& requested : config.state_bindings) {
        if (requested.input_index >= input_names_.size() ||
            requested.output_index >= output_names_.size()) {
            throw std::runtime_error(
                "driver-cubin: state tensor index is out of range");
        }
        if (is_state_input(requested.input_index) ||
            is_state_output(requested.output_index)) {
            throw std::runtime_error(
                "driver-cubin: duplicate state tensor binding");
        }

        const Value& input = value(input_names_[requested.input_index]);
        Value& output = values_.at(output_names_[requested.output_index]);
        if (input.dtype != output.dtype || input.shape != output.shape ||
            requested.data_type != public_dtype(input.dtype) ||
            requested.shape != input.shape) {
            throw std::runtime_error(
                "driver-cubin: state tensor shape or dtype mismatch");
        }
        if (std::find(final_outputs.begin(), final_outputs.end(),
                      output_names_[requested.output_index]) ==
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

bool AotGraph::is_channel_slice(const json& node, int start, int end) const
{
    const auto inputs = node.at("inputs").get<std::vector<std::string>>();
    if (inputs.size() < 4 || inputs[1].empty() || inputs[2].empty() ||
        inputs[3].empty()) {
        return false;
    }
    const auto starts = initializer_values<std::int64_t>(inputs[1]);
    const auto ends = initializer_values<std::int64_t>(inputs[2]);
    const auto axes = initializer_values<std::int64_t>(inputs[3]);
    const auto steps = inputs.size() > 4 && !inputs[4].empty()
        ? initializer_values<std::int64_t>(inputs[4])
        : std::vector<std::int64_t>{1};
    return starts.size() == 1 && ends.size() == 1 && axes.size() == 1 &&
           steps.size() == 1 && starts[0] == start && ends[0] == end &&
           (axes[0] == 1 || axes[0] == -3) && steps[0] == 1;
}

bool AotGraph::ranges_overlap(const Value& lhs, const Value& rhs)
{
    const std::size_t lhs_bytes =
        element_count(lhs.shape) * dtype_bytes(lhs.dtype);
    const std::size_t rhs_bytes =
        element_count(rhs.shape) * dtype_bytes(rhs.dtype);
    return lhs.address < rhs.address + rhs_bytes &&
           rhs.address < lhs.address + lhs_bytes;
}

}  // namespace mlvc::driver_cubin_backend

