#include "runtime/backend.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlvc {
namespace {

struct BackendEntry {
    const std::string* name = nullptr;
    BackendFactory factory;
};

struct BackendRegistry {
    std::mutex mutex;
    // Append-only interned strings so returned std::string_view / const char*
    // pointers remain valid for the lifetime of the process even if new
    // backends are registered concurrently.
    std::vector<std::unique_ptr<std::string>> interned_names;
    std::vector<BackendEntry> entries;
    std::atomic<const std::string*> default_name{nullptr};

    const std::string* intern(const std::string& value)
    {
        for (const auto& existing : interned_names) {
            if (*existing == value)
                return existing.get();
        }
        interned_names.push_back(std::make_unique<std::string>(value));
        return interned_names.back().get();
    }
};

BackendRegistry& registry()
{
    static BackendRegistry instance;
    return instance;
}

bool is_cpu_backend_name(std::string_view name) noexcept
{
    return name == "cpu";
}

bool is_gpu_backend_name(std::string_view name) noexcept
{
    return name == "driver-cubin" ||
           name.find("gpu") != std::string_view::npos ||
           name.find("cuda") != std::string_view::npos;
}

bool is_npu_backend_name(std::string_view name) noexcept
{
    return name.find("npu") != std::string_view::npos;
}

struct Candidate {
    std::string name;
    BackendFactory factory;
};

void append_unique(std::vector<Candidate>& candidates, const BackendEntry& entry)
{
    if (!entry.name || !entry.factory)
        return;
    for (const Candidate& existing : candidates) {
        if (existing.name == *entry.name)
            return;
    }
    candidates.push_back({*entry.name, entry.factory});
}

}  // namespace

void register_backend(std::string name, BackendFactory factory,
                      bool make_default)
{
    if (name.empty())
        throw std::runtime_error("backend name must not be empty");
    if (!factory)
        throw std::runtime_error("backend factory must not be null");

    auto& state = registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    const std::string* stable_name = state.intern(name);
    const auto found = std::find_if(
        state.entries.begin(), state.entries.end(),
        [&](const BackendEntry& entry) { return entry.name == stable_name; });
    if (found != state.entries.end()) {
        found->factory = std::move(factory);
    } else {
        state.entries.push_back({stable_name, std::move(factory)});
    }
    if (make_default ||
        state.default_name.load(std::memory_order_relaxed) == nullptr) {
        state.default_name.store(stable_name, std::memory_order_release);
    }
}

std::vector<std::string> available_backends()
{
    auto& state = registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::vector<std::string> names;
    names.reserve(state.entries.size());
    for (const BackendEntry& entry : state.entries)
        names.push_back(*entry.name);
    return names;
}

std::size_t copy_available_backend_c_strs(const char** out_names,
                                          std::size_t capacity)
{
    auto& state = registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (out_names && capacity > 0U) {
        const std::size_t count = std::min(capacity, state.entries.size());
        for (std::size_t i = 0; i < count; ++i)
            out_names[i] = state.entries[i].name->c_str();
    }
    return state.entries.size();
}

std::string_view default_backend_name() noexcept
{
    const std::string* current =
        registry().default_name.load(std::memory_order_acquire);
    if (!current || current->empty())
        return "none";
    return *current;
}

const char* compiled_backend_name_c_str() noexcept
{
    const std::string* current =
        registry().default_name.load(std::memory_order_acquire);
    if (!current || current->empty())
        return "none";
    return current->c_str();
}

std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options)
{
    std::vector<Candidate> candidates;
    {
        auto& state = registry();
        std::lock_guard<std::mutex> lock(state.mutex);

        auto find_entry = [&](std::string_view target) -> const BackendEntry* {
            const auto found = std::find_if(
                state.entries.begin(), state.entries.end(),
                [&](const BackendEntry& entry) { return *entry.name == target; });
            return found == state.entries.end() ? nullptr : &(*found);
        };

        if (!options.backend_name.empty()) {
            const BackendEntry* entry = find_entry(options.backend_name);
            if (!entry) {
                throw std::runtime_error(
                    "unknown inference backend: " + options.backend_name);
            }
            append_unique(candidates, *entry);
        } else {
            const std::string* def =
                state.default_name.load(std::memory_order_acquire);
            if (!def || state.entries.empty()) {
                throw std::runtime_error(
                    "no inference backend is registered in this build");
            }

            switch (options.execution_preference) {
            case ExecutionPreference::kCpu: {
                if (const BackendEntry* cpu = find_entry("cpu")) {
                    append_unique(candidates, *cpu);
                } else {
                    throw std::runtime_error(
                        "CPU execution requested, but no 'cpu' backend is registered");
                }
                break;
            }
            case ExecutionPreference::kGpu: {
                if (def && is_gpu_backend_name(*def)) {
                    if (const BackendEntry* entry = find_entry(*def))
                        append_unique(candidates, *entry);
                }
                for (const BackendEntry& entry : state.entries) {
                    if (is_gpu_backend_name(*entry.name))
                        append_unique(candidates, entry);
                }
                if (options.allow_cpu_fallback) {
                    if (const BackendEntry* cpu = find_entry("cpu"))
                        append_unique(candidates, *cpu);
                }
                if (candidates.empty()) {
                    throw std::runtime_error(
                        "GPU execution requested with CPU fallback disabled, "
                        "but no GPU backend is registered");
                }
                break;
            }
            case ExecutionPreference::kNpu: {
                for (const BackendEntry& entry : state.entries) {
                    if (is_npu_backend_name(*entry.name))
                        append_unique(candidates, entry);
                }
                if (options.allow_cpu_fallback) {
                    if (const BackendEntry* cpu = find_entry("cpu"))
                        append_unique(candidates, *cpu);
                }
                if (candidates.empty()) {
                    throw std::runtime_error(
                        "NPU execution requested with CPU fallback disabled, "
                        "but no NPU backend is registered");
                }
                break;
            }
            case ExecutionPreference::kAuto: {
                if (const BackendEntry* entry = find_entry(*def))
                    append_unique(candidates, *entry);
                if (options.allow_cpu_fallback && !is_cpu_backend_name(*def)) {
                    if (const BackendEntry* cpu = find_entry("cpu"))
                        append_unique(candidates, *cpu);
                }
                break;
            }
            }
        }
    }

    if (candidates.empty()) {
        throw std::runtime_error(
            "no inference backend is available for the requested configuration");
    }

    BackendOptions base = options;
    if (!base.asset_reader) {
        base.asset_reader =
            resolve_model_asset_reader(base.model_dir, nullptr);
    }

    std::string last_error;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        BackendOptions attempt = base;
        attempt.backend_name = candidates[i].name;
        const bool is_fallback = (i > 0);
        if (is_fallback && is_cpu_backend_name(attempt.backend_name)) {
            // Preserve the caller's allow_cpu_fallback=true intent on fallback.
            attempt.execution_preference = ExecutionPreference::kCpu;
        }
        try {
            auto backend = candidates[i].factory(attempt);
            if (is_fallback && log_enabled(LogLevel::kWarn)) {
                log_message(LogLevel::kWarn, "runtime",
                            "fell back to backend '" + candidates[i].name +
                                "' after primary failure: " + last_error);
            } else if (log_enabled(LogLevel::kInfo)) {
                log_message(LogLevel::kInfo, "runtime",
                            "created inference backend '" + candidates[i].name + "'");
            }
            return backend;
        } catch (const std::exception& error) {
            last_error = error.what();
            if (i + 1U == candidates.size())
                throw;
            if (log_enabled(LogLevel::kWarn)) {
                log_message(LogLevel::kWarn, "runtime",
                            "backend '" + candidates[i].name +
                                "' initialization failed (" + last_error +
                                "); trying fallback");
            }
        }
    }

    throw std::runtime_error("failed to create inference backend: " + last_error);
}

}  // namespace mlvc
