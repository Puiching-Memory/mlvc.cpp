#include "execution_plan.hpp"

#include "mlvc/core/assets.hpp"
#include "mlvc/runtime/backend.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mlvc::cpu_backend {
namespace {

class CpuBackend final : public mlvc::InferenceBackend {
public:
    explicit CpuBackend(BackendOptions options)
        : options_(std::move(options))
    {
        if (!options_.allow_cpu_fallback &&
            (options_.execution_preference == ExecutionPreference::kGpu ||
             options_.execution_preference == ExecutionPreference::kNpu)) {
            throw std::runtime_error(
                "cpu: disallowed by strict non-CPU execution preference");
        }
    }

    std::string_view name() const noexcept override { return "cpu"; }

    void load(const std::string& model_name,
              const ModelExecutionConfig& config) override
    {
        const auto assets = resolve_model_asset_reader(
            options_.model_dir, options_.asset_reader);
        plan_ = std::make_unique<CpuExecutionPlan>(
            *assets, model_name, config, options_.num_threads);
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (!plan_)
            throw std::runtime_error("cpu: load() must be called before run()");
        return plan_->run(inputs);
    }

    void reset_state() override
    {
        if (!plan_)
            throw std::runtime_error("cpu: load() must be called before reset_state()");
        plan_->reset_state();
    }

private:
    BackendOptions options_;
    std::unique_ptr<CpuExecutionPlan> plan_;
};

}  // namespace

void register_cpu_backend(bool make_default)
{
    mlvc::register_backend(
        "cpu",
        [](const BackendOptions& options) {
            return std::make_unique<CpuBackend>(options);
        },
        make_default);
}

}  // namespace mlvc::cpu_backend
