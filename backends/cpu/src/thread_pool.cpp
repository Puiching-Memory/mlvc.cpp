#include "thread_pool.hpp"

#include <algorithm>
#include <fstream>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace mlvc::cpu_backend {
namespace {

inline void cpu_relax() noexcept
{
#if defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

}  // namespace

int detect_optimal_thread_count(int requested) noexcept
{
    if (requested > 0)
        return requested;

    const unsigned int hw = std::max(1U, std::thread::hardware_concurrency());

#if defined(__linux__) || defined(__ANDROID__) || defined(__OHOS__)
    std::vector<unsigned long> max_freqs;
    max_freqs.reserve(hw);
    for (unsigned int cpu = 0; cpu < hw; ++cpu) {
        const std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/cpufreq/cpuinfo_max_freq";
        std::ifstream input(path);
        unsigned long freq = 0;
        if (input >> freq && freq > 0)
            max_freqs.push_back(freq);
    }
    if (!max_freqs.empty()) {
        const unsigned long peak =
            *std::max_element(max_freqs.begin(), max_freqs.end());
        int fast_cores = 0;
        for (const unsigned long freq : max_freqs) {
            // Keep prime + performance cores (>= 60% of peak frequency).
            if (freq * 100 >= peak * 60)
                ++fast_cores;
        }
        if (fast_cores > 0)
            return std::min(fast_cores, 16);
    }
#endif

    return static_cast<int>(std::min<unsigned int>(hw, 16U));
}

CpuThreadPool::CpuThreadPool(int num_threads)
    : total_threads_(static_cast<std::size_t>(
          std::max(1, detect_optimal_thread_count(num_threads))))
{
    if (total_threads_ <= 1)
        return;

    workers_.reserve(total_threads_ - 1);
    try {
        for (std::size_t i = 0; i < total_threads_ - 1; ++i) {
            workers_.emplace_back(&CpuThreadPool::worker_loop, this, i);
        }
    } catch (...) {
        shutdown_.store(true, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(mutex_);
        }
        worker_cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable())
                w.join();
        }
        throw;
    }
}

CpuThreadPool::~CpuThreadPool()
{
    if (workers_.empty())
        return;
    shutdown_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(mutex_);
    }
    worker_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable())
            w.join();
    }
}

void CpuThreadPool::run_task_chunks(TaskContext* task)
{
    try {
        for (;;) {
            const std::size_t start =
                task->next_item.fetch_add(task->grain_size, std::memory_order_relaxed);
            if (start >= task->total_items)
                break;
            const std::size_t end =
                std::min(start + task->grain_size, task->total_items);
            task->invoke(task->fn_ptr, start, end);
        }
    } catch (...) {
        std::lock_guard<std::mutex> err_lock(task->error_mutex);
        if (!task->first_exception)
            task->first_exception = std::current_exception();
    }
}

void CpuThreadPool::dispatch_task(TaskContext* task)
{
    active_task_.store(task, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_acq_rel);
    // Always wake any parked workers while avoiding locks when workers are spinning.
    {
        std::lock_guard<std::mutex> lock(mutex_);
    }
    worker_cv_.notify_all();

    // Caller thread participates as worker 0 to eliminate master/worker idle waste.
    run_task_chunks(task);

    // Wait for all worker threads to finish this task.
    constexpr int kCallerSpinIters = 1024;
    for (int iter = 0; iter < kCallerSpinIters; ++iter) {
        if (task->remaining_workers.load(std::memory_order_acquire) == 0)
            break;
        cpu_relax();
    }
    if (task->remaining_workers.load(std::memory_order_acquire) != 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&]() {
            return task->remaining_workers.load(std::memory_order_acquire) == 0;
        });
    }

    active_task_.store(nullptr, std::memory_order_release);
    if (task->first_exception)
        std::rethrow_exception(task->first_exception);
}

void CpuThreadPool::worker_loop(std::size_t /*worker_index*/)
{
    std::uint64_t seen_generation = 0;

    for (;;) {
        TaskContext* task = nullptr;
        // Brief user-space spin catches back-to-back layers inside one frame without futex wakeup cost.
        constexpr int kWorkerSpinIters = 2000;
        for (int iter = 0; iter < kWorkerSpinIters; ++iter) {
            const std::uint64_t cur_gen = generation_.load(std::memory_order_acquire);
            if (cur_gen != seen_generation) {
                task = active_task_.load(std::memory_order_acquire);
                seen_generation = cur_gen;
                break;
            }
            if (shutdown_.load(std::memory_order_acquire))
                return;
            cpu_relax();
        }

        if (!task && !shutdown_.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(mutex_);
            worker_cv_.wait(lock, [&]() {
                return shutdown_.load(std::memory_order_acquire) ||
                       generation_.load(std::memory_order_acquire) != seen_generation;
            });
            if (shutdown_.load(std::memory_order_acquire))
                return;
            seen_generation = generation_.load(std::memory_order_acquire);
            task = active_task_.load(std::memory_order_acquire);
        }

        if (shutdown_.load(std::memory_order_acquire))
            return;
        if (!task)
            continue;

        run_task_chunks(task);

        if (task->remaining_workers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(mutex_);
            done_cv_.notify_one();
        }
    }
}

}  // namespace mlvc::cpu_backend
