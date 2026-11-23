#include "core/logging.hpp"

#include <atomic>
#include <mutex>
#include <string>

namespace mlvc {
namespace {

struct LogState {
    std::mutex mutex;
    LogCallback callback = nullptr;
    void* user_data = nullptr;
    std::atomic<int> min_level{static_cast<int>(LogLevel::kOff)};
};

LogState& log_state() noexcept
{
    static LogState state;
    return state;
}

}  // namespace

void set_log_callback(LogCallback callback, void* user_data,
                      LogLevel min_level) noexcept
{
    try {
        auto& state = log_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.callback = callback;
        state.user_data = callback ? user_data : nullptr;
        const LogLevel effective = callback ? min_level : LogLevel::kOff;
        state.min_level.store(static_cast<int>(effective), std::memory_order_release);
    } catch (...) {
    }
}

LogLevel log_level() noexcept
{
    return static_cast<LogLevel>(
        log_state().min_level.load(std::memory_order_acquire));
}

bool log_enabled(LogLevel level) noexcept
{
    const int min_lvl = log_state().min_level.load(std::memory_order_acquire);
    return static_cast<int>(level) >= min_lvl &&
           min_lvl < static_cast<int>(LogLevel::kOff);
}

void log_message(LogLevel level, const char* component,
                 std::string_view message) noexcept
{
    if (!log_enabled(level))
        return;
    try {
        LogCallback callback = nullptr;
        void* user_data = nullptr;
        {
            auto& state = log_state();
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!state.callback ||
                static_cast<int>(level) <
                    state.min_level.load(std::memory_order_relaxed)) {
                return;
            }
            callback = state.callback;
            user_data = state.user_data;
        }
        const std::string owned(message);
        callback(level, component ? component : "mlvc", owned.c_str(), user_data);
    } catch (...) {
    }
}

}  // namespace mlvc
