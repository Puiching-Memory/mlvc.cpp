#pragma once

#include "core/assets.hpp"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mlvc::test {

inline void require(bool condition, std::string_view message, const char* file,
                    int line)
{
    if (!condition) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " +
            std::string(message));
    }
}

template <typename Operation>
inline void require_throws(Operation&& operation, std::string_view message,
                           const char* file, int line)
{
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + ": " +
        std::string(message));
}

inline std::vector<std::byte> to_bytes(std::string_view data)
{
    std::vector<std::byte> out(data.size());
    if (!data.empty())
        std::memcpy(out.data(), data.data(), data.size());
    return out;
}

class InMemoryAssetReader final : public ModelAssetReader {
public:
    explicit InMemoryAssetReader(std::string label = "in-memory-test")
        : label_(std::move(label))
    {
    }

    void put(const std::filesystem::path& relative_path,
             std::vector<std::byte> bytes)
    {
        files_[canonical_key(relative_path)] = std::move(bytes);
    }

    std::string location_label() const override { return label_; }

    bool exists(const std::filesystem::path& relative_path) const override
    {
        return files_.find(canonical_key(relative_path)) != files_.end();
    }

    std::vector<std::byte> read_binary(
        const std::filesystem::path& relative_path) const override
    {
        const auto found = files_.find(canonical_key(relative_path));
        if (found == files_.end()) {
            throw std::runtime_error(
                "missing in-memory asset: " + relative_path.generic_string());
        }
        return found->second;
    }

private:
    static std::string canonical_key(const std::filesystem::path& path)
    {
        return path.lexically_normal().generic_string();
    }

    std::string label_;
    std::map<std::string, std::vector<std::byte>> files_;
};

class ScopedEnvVar final {
public:
    ScopedEnvVar(std::string name, const char* value) : name_(std::move(name))
    {
        if (const char* current = std::getenv(name_.c_str())) {
            had_previous_ = true;
            previous_ = current;
        }
        set(value);
    }

    ~ScopedEnvVar()
    {
        if (had_previous_)
            set(previous_.c_str());
        else
            clear();
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    void set(const char* value) const
    {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value && value[0] != '\0')
            setenv(name_.c_str(), value, 1);
        else
            unsetenv(name_.c_str());
#endif
    }

    void clear() const
    {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), "");
#else
        unsetenv(name_.c_str());
#endif
    }

    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
};

class ScopedTempDir final {
public:
    explicit ScopedTempDir(std::string_view prefix)
    {
        const auto stamp =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                (std::string(prefix) + "-" + std::to_string(stamp));
        if (!std::filesystem::create_directory(path_)) {
            throw std::runtime_error(
                "failed to create temporary directory: " + path_.string());
        }
    }

    ~ScopedTempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    ScopedTempDir(const ScopedTempDir&) = delete;
    ScopedTempDir& operator=(const ScopedTempDir&) = delete;

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace mlvc::test

#define MLVC_REQUIRE(cond, msg) \
    ::mlvc::test::require(static_cast<bool>(cond), (msg), __FILE__, __LINE__)

#define MLVC_REQUIRE_THROWS(expr, msg) \
    ::mlvc::test::require_throws([&] { expr; }, (msg), __FILE__, __LINE__)
