#include "aot_graph.hpp"

#include "model_assets.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlvc::driver_cubin_backend {

AotGraph::AotGraph(const std::filesystem::path& model_dir,
                   const std::string& model_name,
                   const ModelExecutionConfig& execution_config,
                   driver_cubin::Driver& driver,
    const driver_cubin::Module& module)
    : driver_(driver), module_(module), model_name_(model_name)
{
    load_model(model_dir, model_name);
    plan_input_slice_aliases();
    plan_epilogue_buffers();
    plan_reglu_buffer();
    plan_spatial_buffers();
    collect_graph_outputs();
    plan_direct_concat_buffers();
    register_kernels();
    configure_state(execution_config);
}

void AotGraph::load_model(const std::filesystem::path& model_dir,
                          const std::string& model_name)
{
    const std::filesystem::path graph_dir =
        std::filesystem::path("aot") / model_name;
    manifest_ = json::parse(detail::read_model_text(
        model_dir, graph_dir / "graph.json"));
    if (manifest_.at("schema_version").get<int>() != 1)
        throw std::runtime_error("driver-cubin: unsupported AOT graph schema");
    weights_host_ = detail::read_model_binary(
        model_dir, graph_dir / "weights.bin");
    if (weights_host_.size() != manifest_.at("weights_bytes").get<std::size_t>())
        throw std::runtime_error("driver-cubin: weights size does not match graph");

    weights_device_ =
        driver_.allocate(std::max<std::size_t>(weights_host_.size(), 1));
    if (!weights_host_.empty())
        driver_.upload(weights_device_, weights_host_.data(),
                       weights_host_.size());
    arena_device_ =
        driver_.allocate(manifest_.at("arena_bytes").get<std::size_t>());

    for (const auto& item : manifest_.at("weights")) {
        Value value;
        value.dtype = item.at("dtype").get<std::string>();
        value.shape = item.at("shape").get<std::vector<int64_t>>();
        value.host_offset = item.at("offset").get<std::size_t>();
        value.address = weights_device_.address() + value.host_offset;
        value.initializer = true;
        values_.emplace(item.at("name").get<std::string>(), std::move(value));
    }

    for (auto it = manifest_.at("tensors").begin();
         it != manifest_.at("tensors").end(); ++it) {
        Value value;
        value.dtype = it.value().at("dtype").get<std::string>();
        value.shape = it.value().at("shape").get<std::vector<int64_t>>();
        if (manifest_.at("arena").contains(it.key())) {
            value.address = arena_device_.address() +
                manifest_.at("arena")
                    .at(it.key())
                    .at("offset")
                    .get<std::size_t>();
        }
        values_.emplace(it.key(), std::move(value));
    }

    for (const auto& input : manifest_.at("inputs")) {
        const std::string name = input.at("name").get<std::string>();
        Value& value = values_.at(name);
        input_names_.push_back(name);
        input_buffers_.push_back(driver_.allocate(
            element_count(value.shape) * dtype_bytes(value.dtype)));
        value.address = input_buffers_.back().address();
    }
}

void AotGraph::collect_graph_outputs()
{
    for (const auto& output : manifest_.at("outputs"))
        output_names_.push_back(output.at("name").get<std::string>());
}

}  // namespace mlvc::driver_cubin_backend
