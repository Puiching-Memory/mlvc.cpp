// Public backend facade for the fixed-shape CUDA Driver implementation.

#include "aot_graph.hpp"

#include "mlvc/core/assets.hpp"
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

class DriverCubinBackend final : public InferenceBackend,
                                 public BufferedCodecBackend {
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
        const auto assets = resolve_model_asset_reader(
            options_.model_dir, options_.asset_reader);
        graph_ = std::make_unique<driver_cubin_backend::AotGraph>(
            *assets, model_name, config, driver_, module_);
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

    void configure_codec_io(const CodecIoConfig& config) override
    {
        require_graph().configure_codec_io(config);
    }

    std::size_t codec_slot_count() const noexcept override
    {
        return graph_ ? graph_->codec_slot_count() : 0;
    }

    MutableYuv420FrameView encoder_input_yuv(std::size_t slot) override
    {
        return require_graph().encoder_input_yuv(slot);
    }

    void submit_encoder(std::size_t slot, int shifted_q) override
    {
        require_graph().submit_encoder(slot, shifted_q);
    }

    std::vector<TensorView> encoder_outputs(
        std::size_t slot) const override
    {
        return require_graph().encoder_outputs(slot);
    }

    std::vector<MutableTensorView> decoder_inputs(
        std::size_t slot) override
    {
        return require_graph().decoder_inputs(slot);
    }

    void submit_decoder(std::size_t slot, int shifted_q) override
    {
        require_graph().submit_decoder(slot, shifted_q);
    }

    Yuv420FrameView decoder_output_yuv(std::size_t slot) const override
    {
        return require_graph().decoder_output_yuv(slot);
    }

    void wait_codec_slot(std::size_t slot) override
    {
        require_graph().wait_codec_slot(slot);
    }

private:
    driver_cubin_backend::AotGraph& require_graph()
    {
        if (!graph_)
            throw std::runtime_error("driver-cubin: load() must be called first");
        return *graph_;
    }

    const driver_cubin_backend::AotGraph& require_graph() const
    {
        if (!graph_)
            throw std::runtime_error("driver-cubin: load() must be called first");
        return *graph_;
    }

    BackendOptions options_;
    driver_cubin::Driver driver_;
    driver_cubin::Module module_;
    std::unique_ptr<driver_cubin_backend::AotGraph> graph_;
};

}  // namespace

namespace driver_cubin_backend {

void register_driver_cubin_backend()
{
    mlvc::register_backend(
        "driver-cubin",
        [](const BackendOptions& options) {
            return std::make_unique<DriverCubinBackend>(options);
        },
        true);
}

}  // namespace driver_cubin_backend

}  // namespace mlvc
