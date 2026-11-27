#include "backends/driver_cubin/driver.hpp"

#include "core/logging.hpp"

#include <dlfcn.h>

#include <array>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace mlvc::driver_cubin {
namespace {

#define MLVC_CUDA_STR_IMPL(x) #x
#define MLVC_CUDA_STR(x) MLVC_CUDA_STR_IMPL(x)

struct CudaDriverTable {
    void* handle = nullptr;
    decltype(&::cuInit) init = nullptr;
    decltype(&::cuDeviceGetCount) device_get_count = nullptr;
    decltype(&::cuDriverGetVersion) driver_get_version = nullptr;
    decltype(&::cuDeviceGet) device_get = nullptr;
    decltype(&::cuDeviceGetName) device_get_name = nullptr;
    decltype(&::cuDeviceGetAttribute) device_get_attribute = nullptr;
    decltype(&::cuDevicePrimaryCtxRetain) device_primary_ctx_retain = nullptr;
    decltype(&::cuDevicePrimaryCtxRelease) device_primary_ctx_release = nullptr;
    decltype(&::cuCtxGetCurrent) ctx_get_current = nullptr;
    decltype(&::cuCtxPushCurrent) ctx_push_current = nullptr;
    decltype(&::cuCtxPopCurrent) ctx_pop_current = nullptr;
    decltype(&::cuStreamCreate) stream_create = nullptr;
    decltype(&::cuStreamDestroy) stream_destroy = nullptr;
    decltype(&::cuStreamSynchronize) stream_synchronize = nullptr;
    decltype(&::cuStreamWaitEvent) stream_wait_event = nullptr;
    decltype(&::cuStreamBeginCapture) stream_begin_capture = nullptr;
    decltype(&::cuStreamEndCapture) stream_end_capture = nullptr;
    decltype(&::cuGetErrorName) get_error_name = nullptr;
    decltype(&::cuGetErrorString) get_error_string = nullptr;
    decltype(&::cuEventCreate) event_create = nullptr;
    decltype(&::cuEventDestroy) event_destroy = nullptr;
    decltype(&::cuEventRecord) event_record = nullptr;
    decltype(&::cuEventSynchronize) event_synchronize = nullptr;
    decltype(&::cuGraphInstantiate) graph_instantiate = nullptr;
    decltype(&::cuGraphDestroy) graph_destroy = nullptr;
    decltype(&::cuGraphExecDestroy) graph_exec_destroy = nullptr;
    decltype(&::cuGraphLaunch) graph_launch = nullptr;
    decltype(&::cuMemAlloc) mem_alloc = nullptr;
    decltype(&::cuMemFree) mem_free = nullptr;
    decltype(&::cuMemHostAlloc) mem_host_alloc = nullptr;
    decltype(&::cuMemFreeHost) mem_free_host = nullptr;
    decltype(&::cuModuleLoadData) module_load_data = nullptr;
    decltype(&::cuModuleUnload) module_unload = nullptr;
    decltype(&::cuModuleGetFunction) module_get_function = nullptr;
    decltype(&::cuMemcpyHtoDAsync) memcpy_htod_async = nullptr;
    decltype(&::cuMemcpyDtoHAsync) memcpy_dtoh_async = nullptr;
    decltype(&::cuMemcpyDtoDAsync) memcpy_dtod_async = nullptr;
    decltype(&::cuMemcpyHtoD) memcpy_htod = nullptr;
    decltype(&::cuMemcpyDtoH) memcpy_dtoh = nullptr;
    decltype(&::cuMemsetD8Async) memset_d8_async = nullptr;
    decltype(&::cuLaunchKernel) launch_kernel = nullptr;
    decltype(&::cuFuncSetAttribute) func_set_attribute = nullptr;
};

template <typename Fn>
Fn resolve_cuda_symbol(void* handle, const char* versioned_name,
                       const char* fallback_name)
{
    void* sym = ::dlsym(handle, versioned_name);
    if (!sym && fallback_name)
        sym = ::dlsym(handle, fallback_name);
    if (!sym) {
        throw std::runtime_error(
            std::string("driver-cubin: missing CUDA driver symbol ") +
            versioned_name);
    }
    return reinterpret_cast<Fn>(sym);
}

const CudaDriverTable& load_cuda_driver()
{
    static std::mutex load_mutex;
    static CudaDriverTable table;
    static bool loaded = false;

    std::lock_guard<std::mutex> lock(load_mutex);
    if (loaded)
        return table;

    void* handle = ::dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!handle)
        handle = ::dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        const char* err = ::dlerror();
        throw std::runtime_error(
            std::string("driver-cubin: NVIDIA driver library libcuda.so.1 is unavailable") +
            (err ? std::string(": ") + err : std::string{}));
    }

    CudaDriverTable resolved;
    resolved.handle = handle;
    try {
#define MLVC_LOAD_CU(field, sym) \
    resolved.field = resolve_cuda_symbol<decltype(resolved.field)>( \
        handle, MLVC_CUDA_STR(sym), #sym)
        MLVC_LOAD_CU(init, cuInit);
        MLVC_LOAD_CU(device_get_count, cuDeviceGetCount);
        MLVC_LOAD_CU(driver_get_version, cuDriverGetVersion);
        MLVC_LOAD_CU(device_get, cuDeviceGet);
        MLVC_LOAD_CU(device_get_name, cuDeviceGetName);
        MLVC_LOAD_CU(device_get_attribute, cuDeviceGetAttribute);
        MLVC_LOAD_CU(device_primary_ctx_retain, cuDevicePrimaryCtxRetain);
        MLVC_LOAD_CU(device_primary_ctx_release, cuDevicePrimaryCtxRelease);
        MLVC_LOAD_CU(ctx_get_current, cuCtxGetCurrent);
        MLVC_LOAD_CU(ctx_push_current, cuCtxPushCurrent);
        MLVC_LOAD_CU(ctx_pop_current, cuCtxPopCurrent);
        MLVC_LOAD_CU(stream_create, cuStreamCreate);
        MLVC_LOAD_CU(stream_destroy, cuStreamDestroy);
        MLVC_LOAD_CU(stream_synchronize, cuStreamSynchronize);
        MLVC_LOAD_CU(stream_wait_event, cuStreamWaitEvent);
        MLVC_LOAD_CU(stream_begin_capture, cuStreamBeginCapture);
        MLVC_LOAD_CU(stream_end_capture, cuStreamEndCapture);
        MLVC_LOAD_CU(get_error_name, cuGetErrorName);
        MLVC_LOAD_CU(get_error_string, cuGetErrorString);
        MLVC_LOAD_CU(event_create, cuEventCreate);
        MLVC_LOAD_CU(event_destroy, cuEventDestroy);
        MLVC_LOAD_CU(event_record, cuEventRecord);
        MLVC_LOAD_CU(event_synchronize, cuEventSynchronize);
        MLVC_LOAD_CU(graph_instantiate, cuGraphInstantiate);
        MLVC_LOAD_CU(graph_destroy, cuGraphDestroy);
        MLVC_LOAD_CU(graph_exec_destroy, cuGraphExecDestroy);
        MLVC_LOAD_CU(graph_launch, cuGraphLaunch);
        MLVC_LOAD_CU(mem_alloc, cuMemAlloc);
        MLVC_LOAD_CU(mem_free, cuMemFree);
        MLVC_LOAD_CU(mem_host_alloc, cuMemHostAlloc);
        MLVC_LOAD_CU(mem_free_host, cuMemFreeHost);
        MLVC_LOAD_CU(module_load_data, cuModuleLoadData);
        MLVC_LOAD_CU(module_unload, cuModuleUnload);
        MLVC_LOAD_CU(module_get_function, cuModuleGetFunction);
        MLVC_LOAD_CU(memcpy_htod_async, cuMemcpyHtoDAsync);
        MLVC_LOAD_CU(memcpy_dtoh_async, cuMemcpyDtoHAsync);
        MLVC_LOAD_CU(memcpy_dtod_async, cuMemcpyDtoDAsync);
        MLVC_LOAD_CU(memcpy_htod, cuMemcpyHtoD);
        MLVC_LOAD_CU(memcpy_dtoh, cuMemcpyDtoH);
        MLVC_LOAD_CU(memset_d8_async, cuMemsetD8Async);
        MLVC_LOAD_CU(launch_kernel, cuLaunchKernel);
        MLVC_LOAD_CU(func_set_attribute, cuFuncSetAttribute);
#undef MLVC_LOAD_CU
    } catch (...) {
        ::dlclose(handle);
        throw;
    }

    table = resolved;
    loaded = true;
    return table;
}

class ScopedContext final {
public:
    ScopedContext(const CudaDriverTable& api, abi::Context ctx, bool throw_on_error)
        : api_(&api)
    {
        if (!ctx)
            return;
        abi::Context current = nullptr;
        if (api_->ctx_get_current(&current) == abi::kSuccess && current == ctx) {
            active_ = true;
            return;
        }
        const abi::Result res = api_->ctx_push_current(ctx);
        if (res == abi::kSuccess) {
            pushed_ = true;
            active_ = true;
        } else if (throw_on_error) {
            const char* name = nullptr;
            const char* description = nullptr;
            api_->get_error_name(res, &name);
            api_->get_error_string(res, &description);
            throw std::runtime_error(
                std::string("driver-cubin: cuCtxPushCurrent failed with ") +
                (name ? name : "CUDA driver error") + " (" +
                std::to_string(static_cast<int>(res)) + ")" +
                (description ? std::string(": ") + description : std::string{}));
        }
    }

    ~ScopedContext()
    {
        if (pushed_) {
            abi::Context popped = nullptr;
            api_->ctx_pop_current(&popped);
        }
    }

    ScopedContext(const ScopedContext&) = delete;
    ScopedContext& operator=(const ScopedContext&) = delete;

    explicit operator bool() const noexcept { return active_; }

private:
    const CudaDriverTable* api_ = nullptr;
    bool pushed_ = false;
    bool active_ = false;
};

}  // namespace

class DriverState final {
public:
    explicit DriverState(int device_ordinal)
        : api(&load_cuda_driver())
    {
        check(api->init(0), "cuInit");

        int device_count = 0;
        check(api->device_get_count(&device_count), "cuDeviceGetCount");
        if (device_ordinal < 0 || device_ordinal >= device_count) {
            throw std::runtime_error(
                "driver-cubin: device ordinal " + std::to_string(device_ordinal) +
                " is outside the available range [0, " +
                std::to_string(device_count) + ")");
        }

        info.ordinal = device_ordinal;
        check(api->driver_get_version(&info.driver_version), "cuDriverGetVersion");
        check(api->device_get(&device, device_ordinal), "cuDeviceGet");

        std::array<char, 256> name{};
        check(api->device_get_name(name.data(), static_cast<int>(name.size()), device),
              "cuDeviceGetName");
        info.name = name.data();
        check(api->device_get_attribute(&info.compute_major,
                                        abi::kComputeCapabilityMajor, device),
              "cuDeviceGetAttribute(major)");
        check(api->device_get_attribute(&info.compute_minor,
                                        abi::kComputeCapabilityMinor, device),
              "cuDeviceGetAttribute(minor)");
        check(api->device_get_attribute(&info.multiprocessor_count,
                                        abi::kMultiprocessorCount, device),
              "cuDeviceGetAttribute(multiprocessor count)");
        check(api->device_get_attribute(&info.max_shared_memory_per_sm,
                                        abi::kMaxSharedMemoryPerMultiprocessor,
                                        device),
              "cuDeviceGetAttribute(max shared memory per sm)");

        try {
            check(api->device_primary_ctx_retain(&context, device),
                  "cuDevicePrimaryCtxRetain");
            ScopedContext ctx_guard = push_context();
            check(api->stream_create(&stream, abi::kStreamNonBlocking),
                  "cuStreamCreate");
            check(api->stream_create(&htod_stream, abi::kStreamNonBlocking),
                  "cuStreamCreate(htod)");
            check(api->stream_create(&dtoh_stream, abi::kStreamNonBlocking),
                  "cuStreamCreate(dtoh)");
        } catch (...) {
            cleanup();
            throw;
        }
        if (log_enabled(LogLevel::kInfo)) {
            log_message(LogLevel::kInfo, "driver-cubin",
                        "initialized GPU " + std::to_string(info.ordinal) + " (" +
                            info.name + ", sm_" + std::to_string(info.compute_major) +
                            std::to_string(info.compute_minor) + ")");
        }
    }

    ~DriverState()
    {
        cleanup();
    }

    DriverState(const DriverState&) = delete;
    DriverState& operator=(const DriverState&) = delete;

    [[noreturn]] void throw_error(abi::Result result, const char* operation) const
    {
        const char* name = nullptr;
        const char* description = nullptr;
        api->get_error_name(result, &name);
        api->get_error_string(result, &description);
        throw std::runtime_error(
            std::string("driver-cubin: ") + operation + " failed with " +
            (name ? name : "CUDA driver error") + " (" +
            std::to_string(static_cast<int>(result)) + ")" +
            (description ? std::string(": ") + description : std::string{}));
    }

    void check(abi::Result result, const char* operation) const
    {
        if (result != abi::kSuccess)
            throw_error(result, operation);
    }

    ScopedContext push_context() const
    {
        return ScopedContext(*api, context, true);
    }

    ScopedContext try_push_context() const noexcept
    {
        return ScopedContext(*api, context, false);
    }

    void cleanup() noexcept
    {
        if (context) {
            if (auto guard = try_push_context()) {
                if (dtoh_stream)
                    api->stream_destroy(dtoh_stream);
                if (htod_stream)
                    api->stream_destroy(htod_stream);
                if (stream)
                    api->stream_destroy(stream);
            }
            dtoh_stream = nullptr;
            htod_stream = nullptr;
            stream = nullptr;
        }
        if (context) {
            api->device_primary_ctx_release(device);
            context = nullptr;
        }
    }

    const CudaDriverTable* api = nullptr;
    abi::Device device = 0;
    abi::Context context = nullptr;
    abi::Stream stream = nullptr;
    abi::Stream htod_stream = nullptr;
    abi::Stream dtoh_stream = nullptr;
    DeviceInfo info;
};

Event::Event(std::shared_ptr<DriverState> state, abi::Event event)
    : state_(std::move(state)), event_(event)
{
}

Event::~Event()
{
    reset();
}

Event::Event(Event&& other) noexcept
    : state_(std::move(other.state_)),
      event_(std::exchange(other.event_, nullptr))
{
}

Event& Event::operator=(Event&& other) noexcept
{
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
}

void Event::reset() noexcept
{
    if (state_ && event_) {
        if (auto guard = state_->try_push_context())
            state_->api->event_destroy(event_);
    }
    event_ = nullptr;
    state_.reset();
}

ExecutableGraph::ExecutableGraph(std::shared_ptr<DriverState> state,
                                 abi::GraphExec graph)
    : state_(std::move(state)), graph_(graph)
{
}

ExecutableGraph::~ExecutableGraph()
{
    reset();
}

ExecutableGraph::ExecutableGraph(ExecutableGraph&& other) noexcept
    : state_(std::move(other.state_)),
      graph_(std::exchange(other.graph_, nullptr))
{
}

ExecutableGraph& ExecutableGraph::operator=(ExecutableGraph&& other) noexcept
{
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        graph_ = std::exchange(other.graph_, nullptr);
    }
    return *this;
}

void ExecutableGraph::reset() noexcept
{
    if (state_ && graph_) {
        if (auto guard = state_->try_push_context())
            state_->api->graph_exec_destroy(graph_);
    }
    graph_ = nullptr;
    state_.reset();
}

DeviceBuffer::DeviceBuffer(std::shared_ptr<DriverState> state,
                           abi::DeviceAddress address, std::size_t size)
    : state_(std::move(state)), address_(address), size_(size)
{
}

DeviceBuffer::~DeviceBuffer()
{
    reset();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : state_(std::move(other.state_)),
      address_(std::exchange(other.address_, 0)),
      size_(std::exchange(other.size_, 0))
{
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept
{
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        address_ = std::exchange(other.address_, 0);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void DeviceBuffer::reset() noexcept
{
    if (state_ && address_) {
        if (auto guard = state_->try_push_context())
            state_->api->mem_free(address_);
    }
    address_ = 0;
    size_ = 0;
    state_.reset();
}

PinnedHostBuffer::PinnedHostBuffer(std::shared_ptr<DriverState> state,
                                   void* data, std::size_t size)
    : state_(std::move(state)), data_(data), size_(size)
{
}

PinnedHostBuffer::~PinnedHostBuffer()
{
    reset();
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : state_(std::move(other.state_)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0))
{
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept
{
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void PinnedHostBuffer::reset() noexcept
{
    if (state_ && data_) {
        if (auto guard = state_->try_push_context())
            state_->api->mem_free_host(data_);
    }
    data_ = nullptr;
    size_ = 0;
    state_.reset();
}

Module::Module(std::shared_ptr<DriverState> state, abi::Module module)
    : state_(std::move(state)), module_(module)
{
}

Module::~Module()
{
    reset();
}

Module::Module(Module&& other) noexcept
    : state_(std::move(other.state_)),
      module_(std::exchange(other.module_, nullptr))
{
}

Module& Module::operator=(Module&& other) noexcept
{
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        module_ = std::exchange(other.module_, nullptr);
    }
    return *this;
}

void Module::reset() noexcept
{
    if (state_ && module_) {
        if (auto guard = state_->try_push_context())
            state_->api->module_unload(module_);
    }
    module_ = nullptr;
    state_.reset();
}

abi::Function Module::function(std::string_view name) const
{
    if (!state_ || !module_)
        throw std::runtime_error("driver-cubin: module is not loaded");
    if (name.empty() || name.find('\0') != std::string_view::npos)
        throw std::runtime_error("driver-cubin: invalid kernel name");

    auto guard = state_->push_context();
    abi::Function function = nullptr;
    const std::string terminated(name);
    state_->check(
        state_->api->module_get_function(&function, module_, terminated.c_str()),
        "cuModuleGetFunction");
    return function;
}

Driver::ContextGuard::ContextGuard(std::shared_ptr<DriverState> state)
    : state_(std::move(state))
{
    if (!state_ || !state_->context)
        return;
    abi::Context current = nullptr;
    if (state_->api->ctx_get_current(&current) == abi::kSuccess &&
        current == state_->context) {
        return;
    }
    state_->check(state_->api->ctx_push_current(state_->context),
                  "cuCtxPushCurrent");
    pushed_ = true;
}

Driver::ContextGuard::~ContextGuard()
{
    reset();
}

Driver::ContextGuard::ContextGuard(ContextGuard&& other) noexcept
    : state_(std::move(other.state_)),
      pushed_(std::exchange(other.pushed_, false))
{
}

Driver::ContextGuard& Driver::ContextGuard::operator=(
    ContextGuard&& other) noexcept
{
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        pushed_ = std::exchange(other.pushed_, false);
    }
    return *this;
}

void Driver::ContextGuard::reset() noexcept
{
    if (state_ && pushed_) {
        abi::Context popped = nullptr;
        state_->api->ctx_pop_current(&popped);
    }
    pushed_ = false;
    state_.reset();
}

Driver::Driver(int device_ordinal)
    : state_(std::make_shared<DriverState>(device_ordinal))
{
}

const DeviceInfo& Driver::device_info() const noexcept
{
    return state_->info;
}

Driver::ContextGuard Driver::bind_context() const
{
    return ContextGuard(state_);
}

DeviceBuffer Driver::allocate(std::size_t bytes) const
{
    if (bytes == 0)
        throw std::runtime_error("driver-cubin: allocation size must be positive");
    auto guard = state_->push_context();
    abi::DeviceAddress address = 0;
    state_->check(state_->api->mem_alloc(&address, bytes), "cuMemAlloc");
    return DeviceBuffer(state_, address, bytes);
}

Event Driver::create_event() const
{
    auto guard = state_->push_context();
    abi::Event event = nullptr;
    state_->check(state_->api->event_create(&event, abi::kEventFlags),
                  "cuEventCreate");
    return Event(state_, event);
}

Module Driver::load_module(std::span<const std::byte> image) const
{
    if (image.empty())
        throw std::runtime_error("driver-cubin: module image is empty");
    auto guard = state_->push_context();
    abi::Module module = nullptr;
    state_->check(state_->api->module_load_data(&module, image.data()),
                  "cuModuleLoadData");
    return Module(state_, module);
}

void Driver::upload_async(const DeviceBuffer& destination,
                          const void* source, std::size_t bytes) const
{
    if (!source || bytes > destination.size())
        throw std::runtime_error("driver-cubin: invalid host-to-device copy");
    auto guard = state_->push_context();
    state_->check(
        state_->api->memcpy_htod_async(
            destination.address(), source, bytes, state_->stream),
        "cuMemcpyHtoDAsync");
}

void Driver::upload_async_htod(const DeviceBuffer& destination,
                               const void* source, std::size_t bytes) const
{
    if (!source || bytes > destination.size())
        throw std::runtime_error("driver-cubin: invalid host-to-device copy");
    auto guard = state_->push_context();
    state_->check(
        state_->api->memcpy_htod_async(
            destination.address(), source, bytes, state_->htod_stream),
        "cuMemcpyHtoDAsync(htod)");
}

void Driver::zero_async(const DeviceBuffer& destination) const
{
    if (!destination)
        throw std::runtime_error("driver-cubin: invalid device memset");
    auto guard = state_->push_context();
    state_->check(
        state_->api->memset_d8_async(
            destination.address(), 0, destination.size(), state_->stream),
        "cuMemsetD8Async");
}

void Driver::copy_dto_d_async(abi::DeviceAddress destination,
                              abi::DeviceAddress source,
                              std::size_t bytes) const
{
    if (!destination || !source)
        throw std::runtime_error("driver-cubin: invalid device-to-device copy");
    if (bytes == 0)
        return;
    auto guard = state_->push_context();
    state_->check(
        state_->api->memcpy_dtod_async(destination, source, bytes, state_->stream),
        "cuMemcpyDtoDAsync");
}

void Driver::download_async(void* destination, const DeviceBuffer& source,
                            std::size_t bytes) const
{
    if (!destination || bytes > source.size())
        throw std::runtime_error("driver-cubin: invalid device-to-host copy");
    download_async(destination, source.address(), bytes);
}

void Driver::download_async(void* destination, abi::DeviceAddress source,
                            std::size_t bytes) const
{
    if (!destination || !source)
        throw std::runtime_error("driver-cubin: invalid device-to-host copy");
    auto guard = state_->push_context();
    state_->check(
        state_->api->memcpy_dtoh_async(destination, source, bytes, state_->stream),
        "cuMemcpyDtoHAsync");
}

void Driver::download_async_dtoh(void* destination, const DeviceBuffer& source,
                                 std::size_t bytes) const
{
    if (!destination || bytes > source.size())
        throw std::runtime_error("driver-cubin: invalid device-to-host copy");
    download_async_dtoh(destination, source.address(), bytes);
}

void Driver::download_async_dtoh(void* destination, abi::DeviceAddress source,
                                 std::size_t bytes) const
{
    if (!destination || !source)
        throw std::runtime_error("driver-cubin: invalid device-to-host copy");
    auto guard = state_->push_context();
    state_->check(
        state_->api->memcpy_dtoh_async(
            destination, source, bytes, state_->dtoh_stream),
        "cuMemcpyDtoHAsync(dtoh)");
}

void Driver::upload(const DeviceBuffer& destination, const void* source,
                    std::size_t bytes) const
{
    if (!source || bytes > destination.size())
        throw std::runtime_error("driver-cubin: invalid host-to-device copy");
    auto guard = state_->push_context();
    state_->check(
        state_->api->memcpy_htod_async(
            destination.address(), source, bytes, state_->stream),
        "cuMemcpyHtoDAsync");
    state_->check(state_->api->stream_synchronize(state_->stream),
                  "cuStreamSynchronize");
}

void Driver::download(void* destination, abi::DeviceAddress source,
                      std::size_t bytes) const
{
    if (!destination || !source)
        throw std::runtime_error("driver-cubin: invalid device-to-host copy");
    auto guard = state_->push_context();
    state_->check(
        state_->api->memcpy_dtoh_async(
            destination, source, bytes, state_->stream),
        "cuMemcpyDtoHAsync");
    state_->check(state_->api->stream_synchronize(state_->stream),
                  "cuStreamSynchronize");
}

void Driver::launch(abi::Function function, Dim3 grid, Dim3 block,
                    unsigned int shared_memory_bytes,
                    std::span<void*> parameters) const
{
    if (!function || grid.x == 0 || grid.y == 0 || grid.z == 0 ||
        block.x == 0 || block.y == 0 || block.z == 0) {
        throw std::runtime_error("driver-cubin: invalid kernel launch dimensions");
    }
    auto guard = state_->push_context();
    state_->check(
        state_->api->launch_kernel(
            function, grid.x, grid.y, grid.z,
            block.x, block.y, block.z, shared_memory_bytes,
            state_->stream, parameters.data(), nullptr),
        "cuLaunchKernel");
}

void Driver::set_max_dynamic_shared_memory(abi::Function function,
                                           unsigned int bytes) const
{
    if (!function || bytes > static_cast<unsigned int>(
                               std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "driver-cubin: invalid dynamic shared-memory limit");
    }
    auto guard = state_->push_context();
    state_->check(
        state_->api->func_set_attribute(
            function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(bytes)),
        "cuFuncSetAttribute(max dynamic shared memory)");
}

void Driver::begin_capture() const
{
    auto guard = state_->push_context();
    state_->check(
        state_->api->stream_begin_capture(
            state_->stream, abi::kStreamCaptureThreadLocal),
        "cuStreamBeginCapture");
}

ExecutableGraph Driver::end_capture() const
{
    auto guard = state_->push_context();
    abi::Graph graph = nullptr;
    state_->check(state_->api->stream_end_capture(state_->stream, &graph),
                  "cuStreamEndCapture");

    abi::GraphExec executable = nullptr;
    const abi::Result instantiate_result =
        state_->api->graph_instantiate(&executable, graph, 0);
    const abi::Result destroy_result = state_->api->graph_destroy(graph);
    if (instantiate_result != abi::kSuccess)
        state_->throw_error(instantiate_result, "cuGraphInstantiate");
    if (destroy_result != abi::kSuccess) {
        state_->api->graph_exec_destroy(executable);
        state_->throw_error(destroy_result, "cuGraphDestroy");
    }
    return ExecutableGraph(state_, executable);
}

void Driver::launch_graph(const ExecutableGraph& graph) const
{
    if (!graph.graph_)
        throw std::runtime_error("driver-cubin: executable graph is empty");
    if (graph.state_.get() != state_.get())
        throw std::runtime_error("driver-cubin: executable graph belongs to another driver");
    auto guard = state_->push_context();
    state_->check(state_->api->graph_launch(graph.graph_, state_->stream),
                  "cuGraphLaunch");
}

void Driver::record(Event& event) const
{
    if (!event.event_ || event.state_.get() != state_.get())
        throw std::runtime_error("driver-cubin: invalid completion event");
    auto guard = state_->push_context();
    state_->check(state_->api->event_record(event.event_, state_->stream),
                  "cuEventRecord");
}

void Driver::record_htod(Event& event) const
{
    if (!event.event_ || event.state_.get() != state_.get())
        throw std::runtime_error("driver-cubin: invalid htod event");
    auto guard = state_->push_context();
    state_->check(state_->api->event_record(event.event_, state_->htod_stream),
                  "cuEventRecord(htod)");
}

void Driver::record_dtoh(Event& event) const
{
    if (!event.event_ || event.state_.get() != state_.get())
        throw std::runtime_error("driver-cubin: invalid dtoh event");
    auto guard = state_->push_context();
    state_->check(state_->api->event_record(event.event_, state_->dtoh_stream),
                  "cuEventRecord(dtoh)");
}

void Driver::wait_compute_on(const Event& event) const
{
    if (!event.event_ || event.state_.get() != state_.get())
        throw std::runtime_error("driver-cubin: invalid wait event");
    auto guard = state_->push_context();
    state_->check(
        state_->api->stream_wait_event(state_->stream, event.event_, 0),
        "cuStreamWaitEvent(compute)");
}

void Driver::wait_dtoh_on(const Event& event) const
{
    if (!event.event_ || event.state_.get() != state_.get())
        throw std::runtime_error("driver-cubin: invalid wait event");
    auto guard = state_->push_context();
    state_->check(
        state_->api->stream_wait_event(state_->dtoh_stream, event.event_, 0),
        "cuStreamWaitEvent(dtoh)");
}

void Driver::synchronize(const Event& event) const
{
    if (!event.event_ || event.state_.get() != state_.get())
        throw std::runtime_error("driver-cubin: invalid completion event");
    auto guard = state_->push_context();
    state_->check(state_->api->event_synchronize(event.event_),
                  "cuEventSynchronize");
}

void Driver::synchronize() const
{
    auto guard = state_->push_context();
    state_->check(state_->api->stream_synchronize(state_->stream),
                  "cuStreamSynchronize");
}

void Driver::synchronize_all() const
{
    auto guard = state_->push_context();
    state_->check(state_->api->stream_synchronize(state_->htod_stream),
                  "cuStreamSynchronize(htod)");
    state_->check(state_->api->stream_synchronize(state_->stream),
                  "cuStreamSynchronize(compute)");
    state_->check(state_->api->stream_synchronize(state_->dtoh_stream),
                  "cuStreamSynchronize(dtoh)");
}

PinnedHostBuffer Driver::allocate_host_pinned(std::size_t bytes) const
{
    if (bytes == 0)
        throw std::runtime_error(
            "driver-cubin: pinned allocation size must be positive");
    auto guard = state_->push_context();
    void* pointer = nullptr;
    state_->check(
        state_->api->mem_host_alloc(&pointer, bytes, CU_MEMHOSTALLOC_PORTABLE),
        "cuMemHostAlloc");
    return PinnedHostBuffer(state_, pointer, bytes);
}

}  // namespace mlvc::driver_cubin
