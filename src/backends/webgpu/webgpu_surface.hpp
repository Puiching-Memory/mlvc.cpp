#pragma once

#include "execution_plan.hpp"

#include "runtime/backend.hpp"

#include <memory>

namespace mlvc::webgpu_backend {

// Optional zero-readback GPU texture render extension implemented by the
// `"webgpu"` backend. Callers (such as the Emscripten browser decoder) can
// `dynamic_cast<WebGpuSurfaceBackend*>(backend.get())` to render decoded frames
// directly into a WebGPU swapchain texture without CPU readback.
class WebGpuSurfaceBackend {
public:
    virtual ~WebGpuSurfaceBackend() = default;

    virtual std::shared_ptr<WgpuContext> wgpu_context() const = 0;
    virtual void configure_surface_output(const SurfaceRenderConfig& config) = 0;
    virtual void submit_decoder_to_texture(const std::vector<TensorView>& latents,
                                           int shifted_q,
                                           bool reset_state_before_run,
                                           WGPUTextureView target_view) = 0;
    virtual bool fused_d2s8_active() const noexcept = 0;
    virtual bool uses_f16() const noexcept = 0;
    virtual std::size_t step_count() const noexcept = 0;
};

}  // namespace mlvc::webgpu_backend
