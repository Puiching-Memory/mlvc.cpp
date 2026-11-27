#pragma once

#if defined(__EMSCRIPTEN__)
#  include <webgpu/webgpu.h>
#elif defined(MLVC_WEBGPU_FLAT_HEADER)
#  include "webgpu.h"
#else
#  include <webgpu/webgpu.h>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace mlvc::webgpu_backend {

inline WGPUStringView to_wgpu_string(std::string_view sv) noexcept
{
    return WGPUStringView{sv.empty() ? nullptr : sv.data(), sv.size()};
}

inline std::string from_wgpu_string(WGPUStringView sv)
{
    if (!sv.data || sv.length == 0U)
        return {};
    if (sv.length == WGPU_STRLEN)
        return std::string(sv.data);
    return std::string(sv.data, sv.length);
}

inline std::uint64_t align_up_u64(std::uint64_t value, std::uint64_t align) noexcept
{
    return ((value + align - 1U) / align) * align;
}

template <typename Handle, void (*ReleaseFn)(Handle)>
class UniqueHandle {
public:
    UniqueHandle() = default;
    explicit UniqueHandle(Handle h) noexcept : handle_(h) {}
    ~UniqueHandle() { reset(); }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    UniqueHandle(UniqueHandle&& other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    UniqueHandle& operator=(UniqueHandle&& other) noexcept
    {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void reset(Handle h = nullptr) noexcept
    {
        if (handle_)
            ReleaseFn(handle_);
        handle_ = h;
    }

    Handle get() const noexcept { return handle_; }
    explicit operator bool() const noexcept { return handle_ != nullptr; }

    Handle release() noexcept
    {
        Handle out = handle_;
        handle_ = nullptr;
        return out;
    }

private:
    Handle handle_ = nullptr;
};

using UniqueInstance = UniqueHandle<WGPUInstance, wgpuInstanceRelease>;
using UniqueAdapter = UniqueHandle<WGPUAdapter, wgpuAdapterRelease>;
using UniqueDevice = UniqueHandle<WGPUDevice, wgpuDeviceRelease>;
using UniqueQueue = UniqueHandle<WGPUQueue, wgpuQueueRelease>;
using UniqueBuffer = UniqueHandle<WGPUBuffer, wgpuBufferRelease>;
using UniqueShaderModule = UniqueHandle<WGPUShaderModule, wgpuShaderModuleRelease>;
using UniqueBindGroupLayout = UniqueHandle<WGPUBindGroupLayout, wgpuBindGroupLayoutRelease>;
using UniqueBindGroup = UniqueHandle<WGPUBindGroup, wgpuBindGroupRelease>;
using UniquePipelineLayout = UniqueHandle<WGPUPipelineLayout, wgpuPipelineLayoutRelease>;
using UniqueComputePipeline = UniqueHandle<WGPUComputePipeline, wgpuComputePipelineRelease>;
using UniqueRenderPipeline = UniqueHandle<WGPURenderPipeline, wgpuRenderPipelineRelease>;
using UniqueCommandEncoder = UniqueHandle<WGPUCommandEncoder, wgpuCommandEncoderRelease>;
using UniqueCommandBuffer = UniqueHandle<WGPUCommandBuffer, wgpuCommandBufferRelease>;
using UniqueComputePassEncoder = UniqueHandle<WGPUComputePassEncoder, wgpuComputePassEncoderRelease>;
using UniqueRenderPassEncoder = UniqueHandle<WGPURenderPassEncoder, wgpuRenderPassEncoderRelease>;
using UniqueQuerySet = UniqueHandle<WGPUQuerySet, wgpuQuerySetRelease>;
using UniqueTexture = UniqueHandle<WGPUTexture, wgpuTextureRelease>;
using UniqueTextureView = UniqueHandle<WGPUTextureView, wgpuTextureViewRelease>;
using UniqueSurface = UniqueHandle<WGPUSurface, wgpuSurfaceRelease>;

struct WebGpuDeviceCaps {
    bool has_shader_f16 = false;
    bool has_timestamp_query = false;
    bool has_subgroups = false;
    std::uint32_t min_storage_offset_alignment = 256U;
    std::uint32_t min_uniform_offset_alignment = 256U;
    std::uint32_t max_storage_buffers_per_stage = 8U;
    std::uint32_t max_workgroup_storage_size = 16384U;
    std::uint32_t max_workgroup_invocations = 256U;
    std::uint32_t max_workgroups_per_dim = 65535U;
    std::uint64_t max_storage_binding_size = 128ULL << 20;
    std::uint64_t max_buffer_size = 256ULL << 20;
    std::string adapter_name;
    std::string backend_name;
};

class WgpuContext {
public:
    // Creates a standalone native WebGPU instance/adapter/device (Dawn).
    static std::shared_ptr<WgpuContext> create_default(int device_id = 0);

    // Wraps an externally created WebGPU device (e.g., browser Emscripten JS init).
    static std::shared_ptr<WgpuContext> from_external(
        WGPUInstance instance, WGPUAdapter adapter,
        WGPUDevice device, WGPUQueue queue);

    // Optional process-wide external device registration for browser WASM init.
    static void set_external_context(std::shared_ptr<WgpuContext> ctx);
    static std::shared_ptr<WgpuContext> external_context();

    ~WgpuContext();

    WGPUInstance instance() const noexcept { return instance_.get(); }
    WGPUAdapter adapter() const noexcept { return adapter_.get(); }
    WGPUDevice device() const noexcept { return device_.get(); }
    WGPUQueue queue() const noexcept { return queue_.get(); }
    const WebGpuDeviceCaps& caps() const noexcept { return caps_; }

    UniqueBuffer create_buffer(std::uint64_t size, WGPUBufferUsage usage,
                               std::string_view label = {}) const;

    // Writes arbitrary byte count (pads tail to 4-byte multiple cleanly).
    void write_buffer(WGPUBuffer buffer, std::uint64_t offset,
                      const void* data, std::size_t size) const;

    // Synchronous readback helper (used by native InferenceBackend / conformance tests).
    void read_buffer_sync(WGPUBuffer src, std::uint64_t src_offset,
                          void* dst, std::size_t size) const;

    void wait_queue_done() const;

    UniqueShaderModule create_shader_module(std::string_view wgsl,
                                            std::string_view label = {}) const;

    void check_error(std::string_view where) const;
    void clear_error() const;

    void record_error(std::string message) noexcept;

private:
    WgpuContext() = default;

    UniqueInstance instance_;
    UniqueAdapter adapter_;
    UniqueDevice device_;
    UniqueQueue queue_;
    WebGpuDeviceCaps caps_;

    mutable std::mutex error_mutex_;
    mutable std::string last_error_;
    mutable std::atomic<bool> has_error_{false};
};

}  // namespace mlvc::webgpu_backend
