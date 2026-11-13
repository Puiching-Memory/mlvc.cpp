#include "aot_graph.hpp"

#include <algorithm>
#include <cstdlib>
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
    // Opt-in L2 threadblock swizzle for the CUTLASS pointwise GEMMs.
    if (const char* env = std::getenv("MLVC_CUTLASS_LOG_TILE"))
        cutlass_log_tile_ = std::clamp(std::atoi(env), 0, 3);
    load_model(assets, model_name);
    plan_input_slice_aliases();
    plan_epilogue_buffers();
    plan_reglu_buffer();
    plan_spatial_buffers();
    collect_graph_outputs();
    plan_direct_concat_buffers();
    register_kernels();
    configure_state(execution_config);
}

void AotGraph::load_model(const ModelAssetReader& assets,
                          const std::string& model_name)
{
    auto loaded = aot::load_aot_graph(assets, model_name);
    graph_ = std::move(loaded.descriptor);
    weights_host_ = std::move(loaded.weights);
    weights_device_ = driver_.allocate(std::max<std::size_t>(weights_host_.size(), 1));
    if (!weights_host_.empty())
        driver_.upload(weights_device_, weights_host_.data(), weights_host_.size());
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
