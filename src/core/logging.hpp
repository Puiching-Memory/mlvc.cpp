#pragma once

#include <string_view>

namespace mlvc {

enum class LogLevel : int {
    kTrace = 0,
    kDebug = 1,
    kInfo = 2,
    kWarn = 3,
    kError = 4,
    kOff = 5,
};

using LogCallback = void (*)(LogLevel level, const char* component,
                             const char* message, void* user_data) noexcept;

// Registers a process-wide log callback. Passing nullptr disables logging.
void set_log_callback(LogCallback callback, void* user_data = nullptr,
                      LogLevel min_level = LogLevel::kInfo) noexcept;

LogLevel log_level() noexcept;
bool log_enabled(LogLevel level) noexcept;

void log_message(LogLevel level, const char* component,
                 std::string_view message) noexcept;

}  // namespace mlvc
