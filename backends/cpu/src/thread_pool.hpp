#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace mlvc::cpu_backend {

// Detects optimal worker thread count when requested <= 0. On heterogeneous
// ARM64 SoCs (Android/OpenHarmony/Linux), skips low-clock LITTLE cores based
// on cpufreq sysfs maximum frequencies so synchronization barriers do not
// stall fast big/prime cores.
int detect_optimal_thread_count(int requested) noexcept;

class CpuThreadPool final {
public:
    explicit CpuThreadPool(int num_threads);
    ~CpuThreadPool();

    CpuThreadPool(const CpuThreadPool&) = delete;
    CpuThreadPool& operator=(const CpuThreadPool&) = delete;

    std::size_t thread_count() const noexcept { return total_threads_; }

    template <typename Fn>
    void parallel_for(std::size_t total_items, std::size_t grain_size, Fn&& fn)
    {
        if (total_items == 0)
            return;
        const std::size_t effective_grain = grain_size > 0 ? grain_size : 1;
        if (total_threads_ <= 1 || total_items <= effective_grain) {
            fn(0, total_items);
            return;
        }

        TaskContext task;
        task.fn_ptr = &fn;
        task.invoke = [](void* ptr, std::size_t begin, std::size_t end) {
            (*static_cast<Fn*>(ptr))(begin, end);
        };
        task.total_items = total_items;
        task.grain_size = effective_grain;
        task.remaining_workers.store(workers_.size(), std::memory_order_relaxed);
        dispatch_task(&task);
    }

private:
    struct TaskContext {
        void (*invoke)(void*, std::size_t, std::size_t) = nullptr;
        void* fn_ptr = nullptr;
        std::size_t total_items = 0;
        std::size_t grain_size = 1;
        std::atomic<std::size_t> next_item{0};
        std::atomic<std::size_t> remaining_workers{0};
        std::exception_ptr first_exception;
        std::mutex error_mutex;
    };

    void worker_loop(std::size_t worker_index);
    void dispatch_task(TaskContext* task);
    static void run_task_chunks(TaskContext* task);

    std::size_t total_threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable worker_cv_;
    std::condition_variable done_cv_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<TaskContext*> active_task_{nullptr};
    std::atomic<bool> shutdown_{false};
};

}  // namespace mlvc::cpu_backend
