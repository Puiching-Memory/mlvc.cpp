#include "aot_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlvc::driver_cubin_backend {

void AotGraph::reset_state()
{
    for (const StateBinding& binding : state_bindings_)
        driver_.zero_async(input_buffers_[binding.input_index]);
    state_initialized_ = true;
}

std::vector<Tensor> AotGraph::run(const std::vector<Tensor>& inputs)
{
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error(
            "driver-cubin: reset_state() must be called before run()");
    if (inputs.size() + state_bindings_.size() != input_names_.size())
        throw std::runtime_error("driver-cubin: graph input count mismatch");
    ensure_input_staging();
    std::size_t external_index = 0;
    for (std::size_t graph_index = 0; graph_index < input_names_.size();
         ++graph_index) {
        if (is_state_input(graph_index))
            continue;
        const Tensor& input = inputs.at(external_index++);
        const Value& expected = value(input_names_[graph_index]);
        if (input.shape != expected.shape ||
            input.data_type() != public_dtype(expected.dtype))
            throw std::runtime_error("driver-cubin: graph input shape or dtype mismatch");
        if (input.byte_size() != input_buffers_[graph_index].size())
            throw std::runtime_error(
                "driver-cubin: graph input byte size mismatch");
        std::memcpy(input_staging_[graph_index], input.raw_data(),
                    input.byte_size());
        driver_.upload_async(input_buffers_[graph_index],
                             input_staging_[graph_index],
                             input.byte_size());
    }

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

    ensure_output_staging();

    std::vector<Tensor> outputs;
    outputs.reserve(output_names_.size() - state_bindings_.size());
    struct StagedCopy {
        void* destination;
        void* source;
        std::size_t bytes;
    };
    std::vector<StagedCopy> staged_copies;
    for (std::size_t index = 0; index < output_names_.size(); ++index) {
        if (is_state_output(index))
            continue;
        const std::string& name = output_names_[index];
        const Value& result = value(name);
        Tensor tensor;
        tensor.name = name;
        tensor.shape = result.shape;
        if (result.dtype == "fp16") {
            tensor.data = std::vector<Float16Storage>(element_count(result.shape));
        } else if (result.dtype == "int32") {
            tensor.data = std::vector<std::int32_t>(element_count(result.shape));
        } else {
            throw std::runtime_error("driver-cubin: unsupported graph output dtype");
        }
        std::visit([&](auto& storage) {
            const std::size_t bytes =
                storage.size() *
                sizeof(typename std::decay_t<decltype(storage)>::value_type);
            if (output_staging_[index] != nullptr) {
                driver_.download_async(output_staging_[index], result.address,
                                       bytes);
                staged_copies.push_back(
                    {storage.data(), output_staging_[index], bytes});
            } else {
                driver_.download(storage.data(), result.address, bytes);
            }
        }, tensor.data);
        outputs.push_back(std::move(tensor));
    }
    driver_.synchronize();
    for (const StagedCopy& copy : staged_copies)
        std::memcpy(copy.destination, copy.source, copy.bytes);
    return outputs;
}

void AotGraph::ensure_input_staging()
{
    if (input_staging_.size() == input_names_.size())
        return;
    if (!input_staging_.empty())
        throw std::runtime_error("driver-cubin: graph input count changed");
    input_staging_.resize(input_names_.size(), nullptr);
    for (std::size_t i = 0; i < input_names_.size(); ++i) {
        if (is_state_input(i))
            continue;
        input_staging_[i] =
            driver_.allocate_host_pinned(input_buffers_[i].size());
        if (!input_staging_[i])
            throw std::runtime_error(
                "driver-cubin: failed to allocate pinned input staging");
    }
}

void AotGraph::ensure_output_staging()
{
    if (!output_staging_.empty())
        return;
    output_staging_.resize(output_names_.size(), nullptr);
    for (std::size_t index = 0; index < output_names_.size(); ++index) {
        if (is_state_output(index))
            continue;
        const Value& result = value(output_names_[index]);
        const std::size_t element_bytes = result.dtype == "fp16" ? 2 : 4;
        const std::size_t bytes =
            element_count(result.shape) * element_bytes;
        output_staging_[index] = driver_.allocate_host_pinned(bytes);
    }
}

AotGraph::~AotGraph()
{
    for (void* pointer : output_staging_)
        driver_.free_host_pinned(pointer);
    for (void* pointer : input_staging_)
        driver_.free_host_pinned(pointer);
}

}  // namespace mlvc::driver_cubin_backend

