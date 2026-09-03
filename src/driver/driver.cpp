#include "mlvc/driver/driver.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mlvc::driver {

class DriverState final {
public:
    explicit DriverState(int device_ordinal)
    {
        check(cuInit(0), "cuInit");

        int device_count = 0;
        check(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
        if (device_ordinal < 0 || device_ordinal >= device_count) {
            throw std::runtime_error(
                "driver-cubin: device ordinal " + std::to_string(device_ordinal) +
                " is outside the available range [0, " +
                std::to_string(device_count) + ")");
        }

        info.ordinal = device_ordinal;
        check(cuDriverGetVersion(&info.driver_version), "cuDriverGetVersion");
        check(cuDeviceGet(&device, device_ordinal), "cuDeviceGet");

        std::array<char, 256> name{};
        check(cuDeviceGetName(name.data(), static_cast<int>(name.size()), device),
              "cuDeviceGetName");
        info.name = name.data();
        check(cuDeviceGetAttribute(&info.compute_major,
                                   abi::kComputeCapabilityMajor, device),
              "cuDeviceGetAttribute(major)");
        check(cuDeviceGetAttribute(&info.compute_minor,
                                   abi::kComputeCapabilityMinor, device),
              "cuDeviceGetAttribute(minor)");
        check(cuDeviceGetAttribute(&info.multiprocessor_count,
                                   abi::kMultiprocessorCount, device),
              "cuDeviceGetAttribute(multiprocessor count)");

        try {
            check(cuDevicePrimaryCtxRetain(&context, device),
                  "cuDevicePrimaryCtxRetain");
            check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
            check(cuStreamCreate(&stream, abi::kStreamNonBlocking),
                  "cuStreamCreate");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~DriverState()
    {
        cleanup();
    }

    DriverState(const DriverState&) = delete;
    DriverState& operator=(const DriverState&) = delete;

    [[noreturn]] static void throw_error(abi::Result result,
                                         const char* operation)
    {
        const char* name = nullptr;
        const char* description = nullptr;
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &description);
        throw std::runtime_error(
            std::string("driver-cubin: ") + operation + " failed with " +
            (name ? name : "CUDA driver error") + " (" +
            std::to_string(static_cast<int>(result)) + ")" +
            (description ? std::string(": ") + description : std::string{}));
    }

    static void check(abi::Result result, const char* operation)
    {
        if (result != abi::kSuccess)
            throw_error(result, operation);
    }

    void make_current() const
    {
        check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    }

    void cleanup() noexcept
    {
        if (context)
            cuCtxSetCurrent(context);
        if (stream) {
            cuStreamDestroy(stream);
            stream = nullptr;
        }
        if (context) {
            cuDevicePrimaryCtxRelease(device);
            context = nullptr;
        }
    }

    abi::Device device = 0;
    abi::Context context = nullptr;
    abi::Stream stream = nullptr;
    DeviceInfo info;
};

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
        if (cuCtxSetCurrent(state_->context) == CUDA_SUCCESS)
            cuGraphExecDestroy(graph_);
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
        if (cuCtxSetCurrent(state_->context) == CUDA_SUCCESS)
            cuMemFree(address_);
    }
    address_ = 0;
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
        if (cuCtxSetCurrent(state_->context) == CUDA_SUCCESS)
            cuModuleUnload(module_);
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

    state_->make_current();
    abi::Function function = nullptr;
    const std::string terminated(name);
    DriverState::check(
        cuModuleGetFunction(&function, module_, terminated.c_str()),
        "cuModuleGetFunction");
    return function;
}

Driver::Driver(int device_ordinal)
    : state_(std::make_shared<DriverState>(device_ordinal))
{
}

const DeviceInfo& Driver::device_info() const noexcept
{
    return state_->info;
}

DeviceBuffer Driver::allocate(std::size_t bytes) const
{
    if (bytes == 0)
        throw std::runtime_error("driver-cubin: allocation size must be positive");
    state_->make_current();
    abi::DeviceAddress address = 0;
    DriverState::check(cuMemAlloc(&address, bytes), "cuMemAlloc");
    return DeviceBuffer(state_, address, bytes);
}

Module Driver::load_module(std::span<const std::byte> image) const
{
    if (image.empty())
        throw std::runtime_error("driver-cubin: module image is empty");
    state_->make_current();
    abi::Module module = nullptr;
    DriverState::check(cuModuleLoadData(&module, image.data()),
                       "cuModuleLoadData");
    return Module(state_, module);
}

void Driver::upload_async(const DeviceBuffer& destination,
                          const void* source, std::size_t bytes) const
{
    if (!source || bytes > destination.size())
        throw std::runtime_error("driver-cubin: invalid host-to-device copy");
    state_->make_current();
    DriverState::check(
        cuMemcpyHtoDAsync(destination.address(), source, bytes, state_->stream),
        "cuMemcpyHtoDAsync");
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
    state_->make_current();
    DriverState::check(
        cuMemcpyDtoHAsync(destination, source, bytes, state_->stream),
        "cuMemcpyDtoHAsync");
}

void Driver::launch(abi::Function function, Dim3 grid, Dim3 block,
                    unsigned int shared_memory_bytes,
                    std::span<void*> parameters) const
{
    if (!function || grid.x == 0 || grid.y == 0 || grid.z == 0 ||
        block.x == 0 || block.y == 0 || block.z == 0) {
        throw std::runtime_error("driver-cubin: invalid kernel launch dimensions");
    }
    state_->make_current();
    DriverState::check(
        cuLaunchKernel(function, grid.x, grid.y, grid.z,
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
    state_->make_current();
    DriverState::check(
        cuFuncSetAttribute(
            function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(bytes)),
        "cuFuncSetAttribute(max dynamic shared memory)");
}

void Driver::begin_capture() const
{
    state_->make_current();
    DriverState::check(
        cuStreamBeginCapture(state_->stream, abi::kStreamCaptureThreadLocal),
        "cuStreamBeginCapture");
}

ExecutableGraph Driver::end_capture() const
{
    state_->make_current();
    abi::Graph graph = nullptr;
    DriverState::check(cuStreamEndCapture(state_->stream, &graph),
                       "cuStreamEndCapture");

    abi::GraphExec executable = nullptr;
    const abi::Result instantiate_result =
        cuGraphInstantiate(&executable, graph, 0);
    const abi::Result destroy_result = cuGraphDestroy(graph);
    if (instantiate_result != abi::kSuccess)
        DriverState::throw_error(instantiate_result, "cuGraphInstantiate");
    if (destroy_result != abi::kSuccess) {
        cuGraphExecDestroy(executable);
        DriverState::throw_error(destroy_result, "cuGraphDestroy");
    }
    return ExecutableGraph(state_, executable);
}

void Driver::launch_graph(const ExecutableGraph& graph) const
{
    if (!graph.graph_)
        throw std::runtime_error("driver-cubin: executable graph is empty");
    if (graph.state_.get() != state_.get())
        throw std::runtime_error("driver-cubin: executable graph belongs to another driver");
    state_->make_current();
    DriverState::check(cuGraphLaunch(graph.graph_, state_->stream),
                       "cuGraphLaunch");
}

void Driver::synchronize() const
{
    state_->make_current();
    DriverState::check(cuStreamSynchronize(state_->stream),
                       "cuStreamSynchronize");
}

void* Driver::allocate_host_pinned(std::size_t bytes) const
{
    if (bytes == 0)
        return nullptr;
    state_->make_current();
    void* pointer = nullptr;
    CUresult result = cuMemHostAlloc(&pointer, bytes, CU_MEMHOSTALLOC_PORTABLE);
    return result == CUDA_SUCCESS ? pointer : nullptr;
}

void Driver::free_host_pinned(void* pointer) const
{
    if (!pointer)
        return;
    state_->make_current();
    cuMemFreeHost(pointer);
}

bool Driver::pin_host(const void* host_pointer, std::size_t bytes) const
{
    if (!host_pointer || bytes == 0)
        return true;
    state_->make_current();
    CUresult result = cuMemHostRegister(
        const_cast<void*>(host_pointer), bytes,
        CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
    return result == CUDA_SUCCESS ||
           result == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
}

}  // namespace mlvc::driver
