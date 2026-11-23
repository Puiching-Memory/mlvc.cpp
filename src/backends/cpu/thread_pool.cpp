#include "thread_pool.hpp"

#include "core/env.hpp"

#include <array>
#include <cstdint>
#include <cstdio>

#if defined(__linux__)
#include <sched.h>
#endif

#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>

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

int configured_worker_spin_iters() noexcept
{
    constexpr int kDefaultSpinIters = 4096;
    constexpr int kMaxSpinIters = 65536;
    return detail::parse_env_int(
        "MLVC_CPU_SPIN_ITERS", kDefaultSpinIters, 0, kMaxSpinIters);
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
          std::max(1, detect_optimal_thread_count(num_threads)))),
      worker_spin_iters_(configured_worker_spin_iters())
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

namespace {

struct CpuTopology {
    static constexpr int kMaxCpus = 256;
    bool has_hetero = false;
    std::array<bool, kMaxCpus> is_little{};
    std::array<bool, kMaxCpus> is_prime{};
};

CpuTopology probe_cpu_topology() noexcept
{
    CpuTopology topo{};
#if defined(__linux__)
    std::array<unsigned long long, CpuTopology::kMaxCpus> freqs{};
    unsigned long long min_freq = 0;
    unsigned long long max_freq = 0;
    int seen = 0;
    for (int cpu = 0; cpu < CpuTopology::kMaxCpus; ++cpu) {
        char path[96];
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        FILE* f = std::fopen(path, "r");
        if (f == nullptr)
            continue;
        unsigned long long freq = 0;
        if (std::fscanf(f, "%llu", &freq) == 1 && freq > 0) {
            freqs[static_cast<std::size_t>(cpu)] = freq;
            min_freq = (seen == 0) ? freq : std::min(min_freq, freq);
            max_freq = std::max(max_freq, freq);
            ++seen;
        }
        std::fclose(f);
    }
    if (seen >= 2 && min_freq * 100ULL < max_freq * 85ULL) {
        topo.has_hetero = true;
        for (int cpu = 0; cpu < CpuTopology::kMaxCpus; ++cpu) {
            const unsigned long long freq = freqs[static_cast<std::size_t>(cpu)];
            if (freq == 0)
                continue;
            if (freq * 100ULL < max_freq * 80ULL)
                topo.is_little[static_cast<std::size_t>(cpu)] = true;
            else if (freq * 100ULL >= max_freq * 95ULL)
                topo.is_prime[static_cast<std::size_t>(cpu)] = true;
        }
    }
#endif
    return topo;
}

const CpuTopology& cpu_topology() noexcept
{
    static const CpuTopology kTopology = probe_cpu_topology();
    return kTopology;
}

inline int current_cpu_id() noexcept
{
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

}  // namespace

void CpuThreadPool::run_task_chunks(TaskContext* task)
{
    try {
        const CpuTopology& topo = cpu_topology();
        if (!topo.has_hetero) {
            for (;;) {
                const std::size_t start =
                    task->next_item.fetch_add(task->grain_size, std::memory_order_relaxed);
                if (start >= task->total_items)
                    break;
                const std::size_t end =
                    std::min(start + task->grain_size, task->total_items);
                task->invoke(task->fn_ptr, start, end);
            }
        } else {
            const int cpu = current_cpu_id();
            const bool is_little =
                (cpu >= 0 && cpu < CpuTopology::kMaxCpus) &&
                topo.is_little[static_cast<std::size_t>(cpu)];
            const bool is_prime =
                (cpu >= 0 && cpu < CpuTopology::kMaxCpus) &&
                topo.is_prime[static_cast<std::size_t>(cpu)];

            struct FastThreadGuard {
                std::atomic<int>* counter = nullptr;
                ~FastThreadGuard()
                {
                    if (counter != nullptr)
                        counter->fetch_sub(1, std::memory_order_release);
                }
            } fast_guard;
            if (!is_little) {
                task->active_fast_threads.fetch_add(1, std::memory_order_acq_rel);
                fast_guard.counter = &task->active_fast_threads;
            }

            const std::size_t total = task->total_items;
            const std::size_t grain = std::max<std::size_t>(1U, task->grain_size);
            const std::size_t tail_reserve =
                std::min<std::size_t>(total / 3U, grain * 12U);

            for (;;) {
                std::size_t cur = task->next_item.load(std::memory_order_relaxed);
                std::size_t start = 0;
                std::size_t end = 0;
                for (;;) {
                    if (cur >= total) {
                        start = total;
                        break;
                    }
                    const std::size_t remaining = total - cur;
                    if (is_little && remaining <= tail_reserve &&
                        task->active_fast_threads.load(std::memory_order_acquire) > 0) {
                        start = total;
                        break;
                    }
                    std::size_t chunk = grain;
                    if (!is_little && remaining > grain * 24U) {
                        const std::size_t mult =
                            (is_prime && remaining > grain * 48U) ? 3U : 2U;
                        chunk = grain * mult;
                    }
                    end = (chunk >= remaining) ? total : (cur + chunk);
                    if (task->next_item.compare_exchange_weak(
                            cur, end, std::memory_order_relaxed,
                            std::memory_order_relaxed)) {
                        start = cur;
                        break;
                    }
                }
                if (start >= total)
                    break;
                task->invoke(task->fn_ptr, start, end);
            }
        }
    } catch (...) {
        std::lock_guard<std::mutex> err_lock(task->error_mutex);
        if (!task->first_exception)
            task->first_exception = std::current_exception();
    }
}

void CpuThreadPool::dispatch_task(TaskContext* task)
{
    std::lock_guard<std::mutex> dispatch_lock(dispatch_mutex_);
    active_task_.store(task, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_seq_cst);
    // Skip mutex+condvar wakeup syscall when all workers are spinning in user space.
    if (sleeping_workers_.load(std::memory_order_seq_cst) > 0) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
        }
        worker_cv_.notify_all();
    }

    // Caller thread participates as worker 0 to eliminate master/worker idle waste.
    run_task_chunks(task);

    // Wait for all worker threads to finish this task.
    const int caller_spin_iters = std::min(worker_spin_iters_, 1024);
    for (int iter = 0; iter < caller_spin_iters; ++iter) {
        if (task->remaining_workers.load(std::memory_order_acquire) == 0)
            break;
        cpu_relax();
    }
    if (task->remaining_workers.load(std::memory_order_acquire) != 0) {
        caller_waiting_.store(true, std::memory_order_seq_cst);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [&]() {
                return task->remaining_workers.load(std::memory_order_acquire) == 0;
            });
        }
        caller_waiting_.store(false, std::memory_order_release);
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
        for (int iter = 0; iter < worker_spin_iters_; ++iter) {
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
            sleeping_workers_.fetch_add(1, std::memory_order_seq_cst);
            {
                std::unique_lock<std::mutex> lock(mutex_);
                worker_cv_.wait(lock, [&]() {
                    return shutdown_.load(std::memory_order_acquire) ||
                           generation_.load(std::memory_order_acquire) != seen_generation;
                });
            }
            sleeping_workers_.fetch_sub(1, std::memory_order_release);
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

        if (task->remaining_workers.fetch_sub(1, std::memory_order_seq_cst) == 1 &&
            caller_waiting_.load(std::memory_order_seq_cst)) {
            std::lock_guard<std::mutex> lock(mutex_);
            done_cv_.notify_one();
        }
    }
}

std::shared_ptr<CpuThreadPool> acquire_shared_thread_pool(int requested_threads)
{
    const int resolved = std::max(1, detect_optimal_thread_count(requested_threads));
    static std::mutex cache_mutex;
    static std::unordered_map<int, std::weak_ptr<CpuThreadPool>> cache;

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto& entry = cache[resolved];
    if (auto existing = entry.lock())
        return existing;
    auto created = std::make_shared<CpuThreadPool>(resolved);
    entry = created;
    return created;
}

}  // namespace mlvc::cpu_backend
