#include "execution_plan.hpp"

#include "core/assets.hpp"
#include "runtime/backend.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mlvc::cpu_backend {
namespace {

class CpuBackend final : public mlvc::InferenceBackend,
                         public mlvc::BufferedCodecBackend {
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
            *assets, model_name, config, options_.num_threads,
            options_.share_thread_pool);
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        return require_plan("run()").run(inputs);
    }

    void reset_state() override
    {
        require_plan("reset_state()").reset_state();
    }

    void configure_codec_io(const CodecIoConfig& config) override
    {
        require_plan("configure_codec_io()").configure_codec_io(config);
    }

    std::size_t codec_slot_count() const noexcept override
    {
        return plan_ ? plan_->codec_slot_count() : 0;
    }

    MutableYuv420FrameView encoder_input_yuv(std::size_t slot) override
    {
        return require_plan("encoder_input_yuv()").encoder_input_yuv(slot);
    }

    void submit_encoder(std::size_t slot, int shifted_q) override
    {
        require_plan("submit_encoder()").submit_encoder(slot, shifted_q);
    }

    std::vector<TensorView> encoder_outputs(std::size_t slot) const override
    {
        return require_plan("encoder_outputs()").encoder_outputs(slot);
    }

    std::vector<MutableTensorView> decoder_inputs(std::size_t slot) override
    {
        return require_plan("decoder_inputs()").decoder_inputs(slot);
    }

    void submit_decoder(std::size_t slot, int shifted_q) override
    {
        require_plan("submit_decoder()").submit_decoder(slot, shifted_q);
    }

    Yuv420FrameView decoder_output_yuv(std::size_t slot) const override
    {
        return require_plan("decoder_output_yuv()").decoder_output_yuv(slot);
    }

    void wait_codec_slot(std::size_t slot) override
    {
        require_plan("wait_codec_slot()").wait_codec_slot(slot);
    }

private:
    CpuExecutionPlan& require_plan(std::string_view caller)
    {
        if (!plan_)
            throw std::runtime_error("cpu: load() must be called before " + std::string(caller));
        return *plan_;
    }

    const CpuExecutionPlan& require_plan(std::string_view caller) const
    {
        if (!plan_)
            throw std::runtime_error("cpu: load() must be called before " + std::string(caller));
        return *plan_;
    }

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
