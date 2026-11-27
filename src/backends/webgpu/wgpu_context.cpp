#include "wgpu_context.hpp"

#include "core/env.hpp"
#include "core/logging.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mlvc::webgpu_backend {
namespace {

#if !defined(__EMSCRIPTEN__)
std::string env_str(const char* name)
{
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string{};
}
#endif

std::mutex g_external_ctx_mutex;
std::shared_ptr<WgpuContext> g_external_ctx;

std::string backend_type_name(WGPUBackendType type)
{
    switch (type) {
    case WGPUBackendType_D3D11:
        return "D3D11";
    case WGPUBackendType_D3D12:
        return "D3D12";
    case WGPUBackendType_Metal:
        return "Metal";
    case WGPUBackendType_Vulkan:
        return "Vulkan";
    case WGPUBackendType_OpenGL:
        return "OpenGL";
    case WGPUBackendType_OpenGLES:
        return "OpenGLES";
    case WGPUBackendType_WebGPU:
        return "WebGPU";
    default:
        return "Unknown";
    }
}

#if !defined(__EMSCRIPTEN__)
void on_uncaptured_error(WGPUDevice const* /*device*/, WGPUErrorType type,
                         WGPUStringView message, void* userdata1,
                         void* /*userdata2*/)
{
    auto* ctx = static_cast<WgpuContext*>(userdata1);
    std::string msg = "WebGPU error (" + std::to_string(static_cast<int>(type)) +
                      "): " + from_wgpu_string(message);
    if (log_enabled(LogLevel::kError))
        log_message(LogLevel::kError, "webgpu", msg);
    if (ctx)
        ctx->record_error(std::move(msg));
}

void on_device_lost(WGPUDevice const* /*device*/, WGPUDeviceLostReason reason,
                    WGPUStringView message, void* userdata1,
                    void* /*userdata2*/)
{
    if (reason == WGPUDeviceLostReason_Destroyed)
        return;
    auto* ctx = static_cast<WgpuContext*>(userdata1);
    std::string msg = "WebGPU device lost (" + std::to_string(static_cast<int>(reason)) +
                      "): " + from_wgpu_string(message);
    if (log_enabled(LogLevel::kError))
        log_message(LogLevel::kError, "webgpu", msg);
    if (ctx)
        ctx->record_error(std::move(msg));
}
#endif

void populate_caps(WgpuContext* /*ctx*/, WGPUAdapter adapter, WGPUDevice device,
                   WebGpuDeviceCaps& caps)
{
    const bool force_f32 = mlvc::detail::env_flag_enabled("MLVC_WEBGPU_FORCE_F32");
    caps.has_shader_f16 = !force_f32 &&
        (wgpuDeviceHasFeature(device, WGPUFeatureName_ShaderF16) == WGPU_TRUE);
    caps.has_timestamp_query =
        (wgpuDeviceHasFeature(device, WGPUFeatureName_TimestampQuery) == WGPU_TRUE);
    caps.has_subgroups =
        (wgpuDeviceHasFeature(device, WGPUFeatureName_Subgroups) == WGPU_TRUE);

    WGPULimits limits = WGPU_LIMITS_INIT;
    if (wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success) {
        caps.min_storage_offset_alignment = limits.minStorageBufferOffsetAlignment;
        caps.min_uniform_offset_alignment = limits.minUniformBufferOffsetAlignment;
        caps.max_storage_buffers_per_stage = limits.maxStorageBuffersPerShaderStage;
        caps.max_workgroup_storage_size = limits.maxComputeWorkgroupStorageSize;
        caps.max_workgroup_invocations = limits.maxComputeInvocationsPerWorkgroup;
        caps.max_workgroups_per_dim = limits.maxComputeWorkgroupsPerDimension;
        caps.max_storage_binding_size = limits.maxStorageBufferBindingSize;
        caps.max_buffer_size = limits.maxBufferSize;
    }

    if (adapter) {
        WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
        if (wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success) {
            caps.adapter_name = from_wgpu_string(info.device);
            if (caps.adapter_name.empty())
                caps.adapter_name = from_wgpu_string(info.description);
            caps.backend_name = backend_type_name(info.backendType);
            wgpuAdapterInfoFreeMembers(info);
        }
    }
    if (caps.backend_name.empty())
        caps.backend_name = "WebGPU";
    if (caps.adapter_name.empty())
        caps.adapter_name = "Browser/External GPU";
}

}  // namespace

WgpuContext::~WgpuContext()
{
    // Release dependent GPU resources in reverse creation order.
    queue_.reset();
    device_.reset();
    adapter_.reset();
    instance_.reset();
}

void WgpuContext::record_error(std::string message) noexcept
{
    try {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (last_error_.empty())
            last_error_ = std::move(message);
        has_error_.store(true, std::memory_order_release);
    } catch (...) {
    }
}

void WgpuContext::clear_error() const
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_.clear();
    has_error_.store(false, std::memory_order_release);
}

void WgpuContext::check_error(std::string_view where) const
{
    if (has_error_.load(std::memory_order_acquire)) {
        std::string msg;
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            msg = last_error_;
            last_error_.clear();
            has_error_.store(false, std::memory_order_release);
        }
        throw std::runtime_error(
            "webgpu (" + std::string(where) + "): " + msg);
    }
}

void WgpuContext::set_external_context(std::shared_ptr<WgpuContext> ctx)
{
    std::lock_guard<std::mutex> lock(g_external_ctx_mutex);
    g_external_ctx = std::move(ctx);
}

std::shared_ptr<WgpuContext> WgpuContext::external_context()
{
    std::lock_guard<std::mutex> lock(g_external_ctx_mutex);
    return g_external_ctx;
}

std::shared_ptr<WgpuContext> WgpuContext::from_external(
    WGPUInstance instance, WGPUAdapter adapter,
    WGPUDevice device, WGPUQueue queue)
{
    if (!device)
        throw std::runtime_error("webgpu: external WGPUDevice must not be null");
    std::shared_ptr<WgpuContext> ctx(new WgpuContext());
    if (instance) {
        wgpuInstanceAddRef(instance);
        ctx->instance_.reset(instance);
    }
    if (adapter) {
        wgpuAdapterAddRef(adapter);
        ctx->adapter_.reset(adapter);
    }
    wgpuDeviceAddRef(device);
    ctx->device_.reset(device);
    if (queue) {
        wgpuQueueAddRef(queue);
        ctx->queue_.reset(queue);
    } else {
        ctx->queue_.reset(wgpuDeviceGetQueue(device));
    }
    populate_caps(ctx.get(), adapter, device, ctx->caps_);
    if (log_enabled(LogLevel::kInfo)) {
        log_message(LogLevel::kInfo, "webgpu",
                    "attached external device='" + ctx->caps_.adapter_name +
                        "' backend=" + ctx->caps_.backend_name +
                        " f16=" + (ctx->caps_.has_shader_f16 ? "on" : "off") +
                        " timestamp=" + (ctx->caps_.has_timestamp_query ? "on" : "off") +
                        " max_storage_bufs=" + std::to_string(ctx->caps_.max_storage_buffers_per_stage));
    }
    return ctx;
}

std::shared_ptr<WgpuContext> WgpuContext::create_default(int device_id)
{
    if (auto ext = external_context())
        return ext;

#if defined(__EMSCRIPTEN__)
    (void)device_id;
    throw std::runtime_error(
        "webgpu: in Emscripten/browser builds, initialize WebGPU asynchronously "
        "and register the device via WgpuContext::set_external_context()");
#else
    std::shared_ptr<WgpuContext> ctx(new WgpuContext());

    const WGPUInstanceFeatureName inst_features[] = {
        WGPUInstanceFeatureName_TimedWaitAny,
    };
    WGPUInstanceLimits inst_limits = WGPU_INSTANCE_LIMITS_INIT;
    inst_limits.timedWaitAnyMaxCount = 64U;

    WGPUInstanceDescriptor inst_desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
    inst_desc.requiredFeatureCount = 1U;
    inst_desc.requiredFeatures = inst_features;
    inst_desc.requiredLimits = &inst_limits;

    ctx->instance_.reset(wgpuCreateInstance(&inst_desc));
    if (!ctx->instance_)
        throw std::runtime_error("webgpu: wgpuCreateInstance failed");

    WGPURequestAdapterOptions adapter_opts = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    const std::string power_env = env_str("MLVC_WEBGPU_POWER");
    if (power_env == "low" || device_id == 1) {
        adapter_opts.powerPreference = WGPUPowerPreference_LowPower;
    } else {
        adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;
    }
    const std::string backend_env = env_str("MLVC_WEBGPU_BACKEND");
    if (backend_env == "d3d12") {
        adapter_opts.backendType = WGPUBackendType_D3D12;
    } else if (backend_env == "vulkan") {
        adapter_opts.backendType = WGPUBackendType_Vulkan;
    } else if (backend_env == "metal") {
        adapter_opts.backendType = WGPUBackendType_Metal;
    }

    struct AdapterState {
        WGPURequestAdapterStatus status = WGPURequestAdapterStatus_Error;
        WGPUAdapter adapter = nullptr;
        std::string message;
    } adapter_state;

    WGPURequestAdapterCallbackInfo adapter_cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    adapter_cb.mode = WGPUCallbackMode_WaitAnyOnly;
    adapter_cb.userdata1 = &adapter_state;
    adapter_cb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                             WGPUStringView message, void* ud1, void* /*ud2*/) {
        auto* st = static_cast<AdapterState*>(ud1);
        st->status = status;
        st->adapter = adapter;
        st->message = from_wgpu_string(message);
    };

    WGPUFutureWaitInfo wait_info{};
    wait_info.future = wgpuInstanceRequestAdapter(ctx->instance_.get(), &adapter_opts, adapter_cb);
    const WGPUWaitStatus wait_status =
        wgpuInstanceWaitAny(ctx->instance_.get(), 1U, &wait_info, UINT64_MAX);
    if (wait_status != WGPUWaitStatus_Success ||
        adapter_state.status != WGPURequestAdapterStatus_Success ||
        !adapter_state.adapter) {
        throw std::runtime_error(
            "webgpu: failed to request adapter: " + adapter_state.message);
    }
    ctx->adapter_.reset(adapter_state.adapter);

    std::vector<WGPUFeatureName> req_features;
    const bool force_f32 = mlvc::detail::env_flag_enabled("MLVC_WEBGPU_FORCE_F32");
    if (!force_f32 &&
        wgpuAdapterHasFeature(ctx->adapter_.get(), WGPUFeatureName_ShaderF16) == WGPU_TRUE) {
        req_features.push_back(WGPUFeatureName_ShaderF16);
    }
    if (wgpuAdapterHasFeature(ctx->adapter_.get(), WGPUFeatureName_TimestampQuery) == WGPU_TRUE) {
        req_features.push_back(WGPUFeatureName_TimestampQuery);
    }
    if (wgpuAdapterHasFeature(ctx->adapter_.get(), WGPUFeatureName_Subgroups) == WGPU_TRUE) {
        req_features.push_back(WGPUFeatureName_Subgroups);
    }

    WGPULimits adapter_limits = WGPU_LIMITS_INIT;
    WGPULimits req_limits = WGPU_LIMITS_INIT;
    if (wgpuAdapterGetLimits(ctx->adapter_.get(), &adapter_limits) == WGPUStatus_Success) {
        req_limits.maxStorageBufferBindingSize = adapter_limits.maxStorageBufferBindingSize;
        req_limits.maxBufferSize = adapter_limits.maxBufferSize;
        req_limits.maxStorageBuffersPerShaderStage = adapter_limits.maxStorageBuffersPerShaderStage;
        req_limits.maxComputeWorkgroupStorageSize = adapter_limits.maxComputeWorkgroupStorageSize;
        req_limits.maxComputeInvocationsPerWorkgroup = adapter_limits.maxComputeInvocationsPerWorkgroup;
        req_limits.maxComputeWorkgroupSizeX = adapter_limits.maxComputeWorkgroupSizeX;
        req_limits.maxComputeWorkgroupSizeY = adapter_limits.maxComputeWorkgroupSizeY;
        req_limits.maxComputeWorkgroupSizeZ = adapter_limits.maxComputeWorkgroupSizeZ;
        req_limits.maxComputeWorkgroupsPerDimension = adapter_limits.maxComputeWorkgroupsPerDimension;
        req_limits.maxBindGroups = adapter_limits.maxBindGroups;
        req_limits.maxBindingsPerBindGroup = adapter_limits.maxBindingsPerBindGroup;
    }

    WGPUDeviceDescriptor dev_desc = WGPU_DEVICE_DESCRIPTOR_INIT;
    dev_desc.label = to_wgpu_string("mlvc.webgpu.device");
    dev_desc.requiredFeatureCount = req_features.size();
    dev_desc.requiredFeatures = req_features.empty() ? nullptr : req_features.data();
    dev_desc.requiredLimits = &req_limits;
    dev_desc.uncapturedErrorCallbackInfo.callback = &on_uncaptured_error;
    dev_desc.uncapturedErrorCallbackInfo.userdata1 = ctx.get();
    dev_desc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    dev_desc.deviceLostCallbackInfo.callback = &on_device_lost;
    dev_desc.deviceLostCallbackInfo.userdata1 = ctx.get();

    struct DeviceState {
        WGPURequestDeviceStatus status = WGPURequestDeviceStatus_Error;
        WGPUDevice device = nullptr;
        std::string message;
    } dev_state;

    WGPURequestDeviceCallbackInfo dev_cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    dev_cb.mode = WGPUCallbackMode_WaitAnyOnly;
    dev_cb.userdata1 = &dev_state;
    dev_cb.callback = [](WGPURequestDeviceStatus status, WGPUDevice device,
                         WGPUStringView message, void* ud1, void* /*ud2*/) {
        auto* st = static_cast<DeviceState*>(ud1);
        st->status = status;
        st->device = device;
        st->message = from_wgpu_string(message);
    };

    wait_info = {};
    wait_info.future = wgpuAdapterRequestDevice(ctx->adapter_.get(), &dev_desc, dev_cb);
    if (wgpuInstanceWaitAny(ctx->instance_.get(), 1U, &wait_info, UINT64_MAX) != WGPUWaitStatus_Success ||
        dev_state.status != WGPURequestDeviceStatus_Success ||
        !dev_state.device) {
        throw std::runtime_error(
            "webgpu: failed to request device: " + dev_state.message);
    }
    ctx->device_.reset(dev_state.device);
    ctx->queue_.reset(wgpuDeviceGetQueue(ctx->device_.get()));
    populate_caps(ctx.get(), ctx->adapter_.get(), ctx->device_.get(), ctx->caps_);

    if (log_enabled(LogLevel::kInfo)) {
        log_message(LogLevel::kInfo, "webgpu",
                    "initialized device='" + ctx->caps_.adapter_name +
                        "' backend=" + ctx->caps_.backend_name +
                        " f16=" + (ctx->caps_.has_shader_f16 ? "on" : "off") +
                        " subgroups=" + (ctx->caps_.has_subgroups ? "on" : "off") +
                        " timestamp=" + (ctx->caps_.has_timestamp_query ? "on" : "off") +
                        " max_storage_bufs=" + std::to_string(ctx->caps_.max_storage_buffers_per_stage));
    }
    return ctx;
#endif
}

UniqueBuffer WgpuContext::create_buffer(std::uint64_t size, WGPUBufferUsage usage,
                                        std::string_view label) const
{
    const std::uint64_t aligned = align_up_u64(std::max<std::uint64_t>(size, 16U), 16U);
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.label = to_wgpu_string(label);
    desc.usage = usage;
    desc.size = aligned;
    desc.mappedAtCreation = WGPU_FALSE;
    UniqueBuffer buf(wgpuDeviceCreateBuffer(device_.get(), &desc));
    if (!buf)
        throw std::runtime_error("webgpu: failed to allocate buffer of " + std::to_string(aligned) + " bytes");
    return buf;
}

void WgpuContext::write_buffer(WGPUBuffer buffer, std::uint64_t offset,
                               const void* data, std::size_t size) const
{
    if (size == 0U)
        return;
    if ((size & 3U) == 0U) {
        wgpuQueueWriteBuffer(queue_.get(), buffer, offset, data, size);
        return;
    }
    const std::size_t padded = (size + 3U) & ~std::size_t{3U};
    std::vector<std::uint8_t> tmp(padded, 0U);
    std::memcpy(tmp.data(), data, size);
    wgpuQueueWriteBuffer(queue_.get(), buffer, offset, tmp.data(), padded);
}

void WgpuContext::read_buffer_sync(WGPUBuffer src, std::uint64_t src_offset,
                                   void* dst, std::size_t size) const
{
    if (size == 0U)
        return;
#if defined(__EMSCRIPTEN__)
    (void)src;
    (void)src_offset;
    (void)dst;
    throw std::runtime_error(
        "webgpu: synchronous GPU->CPU readback is not supported inside browser WASM; "
        "use the zero-readback surface/texture render path");
#else
    const std::uint64_t copy_size = align_up_u64(size, 4U);
    UniqueBuffer staging = create_buffer(
        copy_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, "readback_staging");

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    UniqueCommandEncoder enc(wgpuDeviceCreateCommandEncoder(device_.get(), &enc_desc));
    wgpuCommandEncoderCopyBufferToBuffer(enc.get(), src, src_offset, staging.get(), 0U, copy_size);
    WGPUCommandBufferDescriptor cb_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    UniqueCommandBuffer cmd(wgpuCommandEncoderFinish(enc.get(), &cb_desc));
    WGPUCommandBuffer cmd_raw = cmd.get();
    wgpuQueueSubmit(queue_.get(), 1U, &cmd_raw);

    struct MapState {
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
        std::string message;
    } map_state;

    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.userdata1 = &map_state;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView message,
                     void* ud1, void* /*ud2*/) {
        auto* st = static_cast<MapState*>(ud1);
        st->status = status;
        st->message = from_wgpu_string(message);
    };

    WGPUFutureWaitInfo wait_info{};
    wait_info.future = wgpuBufferMapAsync(staging.get(), WGPUMapMode_Read, 0U,
                                          static_cast<std::size_t>(copy_size), cb);
    if (wgpuInstanceWaitAny(instance_.get(), 1U, &wait_info, UINT64_MAX) != WGPUWaitStatus_Success ||
        map_state.status != WGPUMapAsyncStatus_Success) {
        throw std::runtime_error("webgpu: readback mapAsync failed: " + map_state.message);
    }
    const void* mapped = wgpuBufferGetConstMappedRange(
        staging.get(), 0U, static_cast<std::size_t>(copy_size));
    if (!mapped) {
        wgpuBufferUnmap(staging.get());
        throw std::runtime_error("webgpu: wgpuBufferGetConstMappedRange returned null");
    }
    std::memcpy(dst, mapped, size);
    wgpuBufferUnmap(staging.get());
    check_error("read_buffer_sync");
#endif
}

void WgpuContext::wait_queue_done() const
{
#if defined(__EMSCRIPTEN__)
    return;
#else
    struct DoneState {
        WGPUQueueWorkDoneStatus status = WGPUQueueWorkDoneStatus_Error;
    } done_state;
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.userdata1 = &done_state;
    cb.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView /*message*/,
                     void* ud1, void* /*ud2*/) {
        static_cast<DoneState*>(ud1)->status = status;
    };
    WGPUFutureWaitInfo wait_info{};
    wait_info.future = wgpuQueueOnSubmittedWorkDone(queue_.get(), cb);
    wgpuInstanceWaitAny(instance_.get(), 1U, &wait_info, UINT64_MAX);
    check_error("wait_queue_done");
#endif
}

UniqueShaderModule WgpuContext::create_shader_module(
    std::string_view wgsl, std::string_view label) const
{
    WGPUShaderSourceWGSL wgsl_src = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl_src.code = to_wgpu_string(wgsl);

    WGPUShaderModuleDescriptor desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    desc.nextInChain = &wgsl_src.chain;
    desc.label = to_wgpu_string(label);

    UniqueShaderModule mod(wgpuDeviceCreateShaderModule(device_.get(), &desc));
    if (!mod)
        throw std::runtime_error("webgpu: failed to compile WGSL shader: " + std::string(label));
    return mod;
}

}  // namespace mlvc::webgpu_backend
