// Public backend facade for the fixed-shape CUDA Driver implementation.

#include "aot_graph.hpp"

#include "mlvc/runtime/backend.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" const unsigned char mlvc_driver_kernels_fatbin[];
extern "C" const std::size_t mlvc_driver_kernels_fatbin_size;

namespace mlvc {
namespace {

class DriverCubinBackend final : public InferenceBackend {
public:
    explicit DriverCubinBackend(BackendOptions options)
        : options_(std::move(options)), driver_(options_.device_id),
          module_(driver_.load_module(std::span<const std::byte>(
              reinterpret_cast<const std::byte*>(mlvc_driver_kernels_fatbin),
              mlvc_driver_kernels_fatbin_size)))
    {
    }

    std::string_view name() const noexcept override { return "driver-cubin"; }

    void load(const std::string& model_name,
              const ModelExecutionConfig& config) override
    {
        graph_ = std::make_unique<driver_cubin_backend::AotGraph>(
            options_.model_dir, model_name, config, driver_, module_);
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (!graph_)
            throw std::runtime_error("driver-cubin: load() must be called first");
        return graph_->run(inputs);
    }

    void reset_state() override
    {
        if (!graph_)
            throw std::runtime_error("driver-cubin: load() must be called first");
        graph_->reset_state();
    }

private:
    BackendOptions options_;
    driver_cubin::Driver driver_;
    driver_cubin::Module module_;
    std::unique_ptr<driver_cubin_backend::AotGraph> graph_;
};

}  // namespace

std::string_view compiled_backend_name() noexcept
{
    return "driver-cubin";
}

std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options)
{
    return std::make_unique<DriverCubinBackend>(options);
}

}  // namespace mlvc
