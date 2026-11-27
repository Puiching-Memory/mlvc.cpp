#include "webgpu_surface.hpp"

#include "core/assets.hpp"
#include "runtime/backend.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mlvc::webgpu_backend {
namespace {

class WebGpuBackend final : public mlvc::InferenceBackend,
                            public mlvc::BufferedCodecBackend,
                            public WebGpuSurfaceBackend {
public:
    explicit WebGpuBackend(BackendOptions options)
        : options_(std::move(options)),
          ctx_(WgpuContext::create_default(options_.device_id))
    {
    }

    std::string_view name() const noexcept override { return "webgpu"; }

    void load(const std::string& model_name,
              const ModelExecutionConfig& config) override
    {
        const auto assets = resolve_model_asset_reader(
            options_.model_dir, options_.asset_reader);
        plan_ = std::make_unique<WebGpuExecutionPlan>(
            ctx_, *assets, model_name, config);
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
        return plan_ ? plan_->codec_slot_count() : 0U;
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

    // WebGpuSurfaceBackend
    std::shared_ptr<WgpuContext> wgpu_context() const override { return ctx_; }

    void configure_surface_output(const SurfaceRenderConfig& config) override
    {
        require_plan("configure_surface_output()").configure_surface_output(config);
    }

    void submit_decoder_to_texture(const std::vector<TensorView>& latents,
                                   int shifted_q,
                                   bool reset_state_before_run,
                                   WGPUTextureView target_view) override
    {
        require_plan("submit_decoder_to_texture()").submit_decoder_to_texture(
            latents, shifted_q, reset_state_before_run, target_view);
    }

    bool fused_d2s8_active() const noexcept override
    {
        return plan_ ? plan_->fused_d2s8_active() : false;
    }

    bool uses_f16() const noexcept override
    {
        return plan_ ? plan_->uses_f16() : ctx_->caps().has_shader_f16;
    }

    std::size_t step_count() const noexcept override
    {
        return plan_ ? plan_->step_count() : 0U;
    }

private:
    WebGpuExecutionPlan& require_plan(std::string_view caller)
    {
        if (!plan_) {
            throw std::runtime_error(
                "webgpu: load() must be called before " + std::string(caller));
        }
        return *plan_;
    }

    const WebGpuExecutionPlan& require_plan(std::string_view caller) const
    {
        if (!plan_) {
            throw std::runtime_error(
                "webgpu: load() must be called before " + std::string(caller));
        }
        return *plan_;
    }

    BackendOptions options_;
    std::shared_ptr<WgpuContext> ctx_;
    std::unique_ptr<WebGpuExecutionPlan> plan_;
};

}  // namespace

void register_webgpu_backend(bool make_default)
{
    mlvc::register_backend(
        "webgpu",
        [](const BackendOptions& options) {
            return std::make_unique<WebGpuBackend>(options);
        },
        make_default);
}

}  // namespace mlvc::webgpu_backend
