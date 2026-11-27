#pragma once

#include "backends/driver_cubin/abi.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace mlvc::driver_cubin {

struct Dim3 {
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;
};

struct DeviceInfo {
    int ordinal = 0;
    int driver_version = 0;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessor_count = 0;
    int max_shared_memory_per_sm = 0;
    std::string name;
};

class DriverState;

class Event final {
public:
    Event() = default;
    ~Event();

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;

    explicit operator bool() const noexcept { return event_ != nullptr; }

private:
    friend class Driver;
    Event(std::shared_ptr<DriverState> state, abi::Event event);
    void reset() noexcept;

    std::shared_ptr<DriverState> state_;
    abi::Event event_ = nullptr;
};

class ExecutableGraph final {
public:
    ExecutableGraph() = default;
    ~ExecutableGraph();

    ExecutableGraph(const ExecutableGraph&) = delete;
    ExecutableGraph& operator=(const ExecutableGraph&) = delete;
    ExecutableGraph(ExecutableGraph&& other) noexcept;
    ExecutableGraph& operator=(ExecutableGraph&& other) noexcept;

    explicit operator bool() const noexcept { return graph_ != nullptr; }

private:
    friend class Driver;
    ExecutableGraph(std::shared_ptr<DriverState> state, abi::GraphExec graph);
    void reset() noexcept;

    std::shared_ptr<DriverState> state_;
    abi::GraphExec graph_ = nullptr;
};

class DeviceBuffer final {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    abi::DeviceAddress address() const noexcept { return address_; }
    std::size_t size() const noexcept { return size_; }
    explicit operator bool() const noexcept { return address_ != 0; }

private:
    friend class Driver;
    DeviceBuffer(std::shared_ptr<DriverState> state,
                 abi::DeviceAddress address, std::size_t size);
    void reset() noexcept;

    std::shared_ptr<DriverState> state_;
    abi::DeviceAddress address_ = 0;
    std::size_t size_ = 0;
};

class PinnedHostBuffer final {
public:
    PinnedHostBuffer() = default;
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    explicit operator bool() const noexcept { return data_ != nullptr; }

private:
    friend class Driver;
    PinnedHostBuffer(std::shared_ptr<DriverState> state, void* data,
                     std::size_t size);
    void reset() noexcept;

    std::shared_ptr<DriverState> state_;
    void* data_ = nullptr;
    std::size_t size_ = 0;
};

class Module final {
public:
    Module() = default;
    ~Module();

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&& other) noexcept;
    Module& operator=(Module&& other) noexcept;

    abi::Function function(std::string_view name) const;
    explicit operator bool() const noexcept { return module_ != nullptr; }

private:
    friend class Driver;
    Module(std::shared_ptr<DriverState> state, abi::Module module);
    void reset() noexcept;

    std::shared_ptr<DriverState> state_;
    abi::Module module_ = nullptr;
};

class Driver final {
public:
    class ContextGuard final {
    public:
        ContextGuard() = default;
        ~ContextGuard();

        ContextGuard(const ContextGuard&) = delete;
        ContextGuard& operator=(const ContextGuard&) = delete;
        ContextGuard(ContextGuard&& other) noexcept;
        ContextGuard& operator=(ContextGuard&& other) noexcept;

    private:
        friend class Driver;
        explicit ContextGuard(std::shared_ptr<DriverState> state);
        void reset() noexcept;

        std::shared_ptr<DriverState> state_;
        bool pushed_ = false;
    };

    explicit Driver(int device_ordinal = 0);

    Driver(const Driver&) = delete;
    Driver& operator=(const Driver&) = delete;
    Driver(Driver&&) noexcept = default;
    Driver& operator=(Driver&&) noexcept = default;

    const DeviceInfo& device_info() const noexcept;
    ContextGuard bind_context() const;

    DeviceBuffer allocate(std::size_t bytes) const;
    PinnedHostBuffer allocate_host_pinned(std::size_t bytes) const;
    Event create_event() const;
    Module load_module(std::span<const std::byte> image) const;

    void upload_async(const DeviceBuffer& destination,
                      const void* source, std::size_t bytes) const;
    void upload_async_htod(const DeviceBuffer& destination,
                           const void* source, std::size_t bytes) const;
    void zero_async(const DeviceBuffer& destination) const;
    void copy_dto_d_async(abi::DeviceAddress destination,
                          abi::DeviceAddress source, std::size_t bytes) const;
    void download_async(void* destination, const DeviceBuffer& source,
                        std::size_t bytes) const;
    void download_async(void* destination, abi::DeviceAddress source,
                        std::size_t bytes) const;
    void download_async_dtoh(void* destination, const DeviceBuffer& source,
                             std::size_t bytes) const;
    void download_async_dtoh(void* destination, abi::DeviceAddress source,
                             std::size_t bytes) const;

    void upload(const DeviceBuffer& destination,
                const void* source, std::size_t bytes) const;
    void download(void* destination, abi::DeviceAddress source,
                  std::size_t bytes) const;
    void launch(abi::Function function, Dim3 grid, Dim3 block,
                unsigned int shared_memory_bytes,
                std::span<void*> parameters) const;
    void set_max_dynamic_shared_memory(abi::Function function,
                                       unsigned int bytes) const;
    void begin_capture() const;
    ExecutableGraph end_capture() const;
    void launch_graph(const ExecutableGraph& graph) const;
    void record(Event& event) const;
    void record_htod(Event& event) const;
    void record_dtoh(Event& event) const;
    void wait_compute_on(const Event& event) const;
    void wait_dtoh_on(const Event& event) const;
    void synchronize(const Event& event) const;
    void synchronize() const;
    void synchronize_all() const;

private:
    std::shared_ptr<DriverState> state_;
};

}  // namespace mlvc::driver_cubin
