#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdlib>

namespace mlvc::detail {

inline int parse_env_int(const char* name, int default_val, int min_val,
                         int max_val) noexcept
{
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0')
        return default_val;
    errno = 0;
    char* end = nullptr;
    const long val = std::strtol(env, &end, 10);
    if (errno == ERANGE || !end || end == env || *end != '\0')
        return default_val;
    if (val < static_cast<long>(min_val))
        return min_val;
    if (val > static_cast<long>(max_val))
        return max_val;
    return static_cast<int>(val);
}

inline bool env_flag_enabled(const char* name) noexcept
{
    const char* env = std::getenv(name);
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

}  // namespace mlvc::detail
