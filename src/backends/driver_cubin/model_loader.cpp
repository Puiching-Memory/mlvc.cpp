#include "aot_graph.hpp"

#include "core/env.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlvc::driver_cubin_backend {

AotGraph::AotGraph(const ModelAssetReader& assets,
                   const std::string& model_name,
                   const ModelExecutionConfig& execution_config,
                   driver_cubin::Driver& driver,
                   const driver_cubin::Module& module)
    : driver_(driver), module_(module), model_name_(model_name)
{
    auto ctx_guard = driver_.bind_context();
    // Opt-in L2 threadblock swizzle for the CUTLASS pointwise GEMMs.
    cutlass_log_tile_ =
        detail::parse_env_int("MLVC_CUTLASS_LOG_TILE", 0, 0, 3);
    load_model(assets, model_name);
    plan_input_slice_aliases();
    plan_epilogue_buffers();
    plan_reglu_buffer();
    plan_spatial_buffers();
    collect_graph_outputs();
    plan_direct_concat_buffers();
    register_kernels();
    configure_state(execution_config);
    warmup_and_capture();
    cache_runtime_metadata();
}

void AotGraph::load_model(const ModelAssetReader& assets,
                          const std::string& model_name)
{
    auto loaded = aot::load_aot_graph(assets, model_name);
    graph_ = std::move(loaded.descriptor);
    tensor_consumer_counts_.assign(graph_.tensors.size(), 0U);
    for (const auto& node : graph_.nodes) {
        for (const TensorId input_id : node.inputs) {
            if (input_id < tensor_consumer_counts_.size())
                ++tensor_consumer_counts_[input_id];
        }
    }
    weights_device_ = driver_.allocate(std::max<std::size_t>(loaded.weights.size(), 1));
    if (!loaded.weights.empty())
        driver_.upload(weights_device_, loaded.weights.data(), loaded.weights.size());
    arena_device_ = driver_.allocate(graph_.arena_bytes);
    values_.resize(graph_.tensors.size());
    for (std::size_t id = 1; id < graph_.tensors.size(); ++id) {
        const auto& item = graph_.tensors[id];
        auto& value = values_[id];
        value.name = item.name;
        value.dtype = item.dtype;
        value.shape = item.shape;
        if (const auto* weight = std::get_if<WeightLocation>(&item.storage)) {
            value.host_offset = weight->offset;
            value.address = weights_device_.address() + weight->offset;
            value.initializer = true;
            // Retain only small metadata/scalar initializers on the host instead
            // of keeping the entire multi-megabyte weights buffer in host RAM.
            const std::size_t count = element_count(item.shape);
            if (item.dtype == DataType::INT64 || count <= 16U) {
                const std::byte* begin = loaded.weights.data() + weight->offset;
                host_initializers_.emplace(
                    static_cast<TensorId>(id),
                    std::vector<std::byte>(begin, begin + weight->bytes));
            }
        } else if (const auto* arena = std::get_if<ArenaSlot>(&item.storage)) {
            value.address = arena_device_.address() + arena->offset;
        }
    }
    input_ids_ = graph_.inputs;
    for (const auto id : input_ids_) {
        auto& value = values_.at(id);
        input_buffers_.push_back(driver_.allocate(element_count(value.shape) * dtype_bytes(value.dtype)));
        value.address = input_buffers_.back().address();
    }
}

void AotGraph::collect_graph_outputs()
{
    output_ids_ = graph_.outputs;
}

}  // namespace mlvc::driver_cubin_backend
