#include "mlvc/runtime/backend.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlvc {
namespace {

struct BackendEntry {
    std::string name;
    BackendFactory factory;
};

struct BackendRegistry {
    std::mutex mutex;
    std::vector<BackendEntry> entries;
    std::string default_name;
};

BackendRegistry& registry()
{
    static BackendRegistry instance;
    return instance;
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
    const auto found = std::find_if(
        state.entries.begin(), state.entries.end(),
        [&](const BackendEntry& entry) { return entry.name == name; });
    if (found != state.entries.end()) {
        found->factory = std::move(factory);
    } else {
        state.entries.push_back({name, std::move(factory)});
    }
    if (make_default || state.default_name.empty())
        state.default_name = std::move(name);
}

std::vector<std::string> available_backends()
{
    auto& state = registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::vector<std::string> names;
    names.reserve(state.entries.size());
    for (const BackendEntry& entry : state.entries)
        names.push_back(entry.name);
    return names;
}

std::string_view default_backend_name() noexcept
{
    auto& state = registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.default_name.empty())
        return "none";
    return state.default_name;
}

const char* compiled_backend_name_c_str() noexcept
{
    auto& state = registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.default_name.empty())
        return "none";
    return state.default_name.c_str();
}

std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options)
{
    BackendFactory factory;
    std::string target_name;
    {
        auto& state = registry();
        std::lock_guard<std::mutex> lock(state.mutex);
        target_name = options.backend_name.empty() ? state.default_name
                                                   : options.backend_name;
        if (target_name.empty()) {
            throw std::runtime_error(
                "no inference backend is registered in this build");
        }
        const auto found = std::find_if(
            state.entries.begin(), state.entries.end(),
            [&](const BackendEntry& entry) { return entry.name == target_name; });
        if (found == state.entries.end()) {
            throw std::runtime_error(
                "unknown inference backend: " + target_name);
        }
        factory = found->factory;
    }

    BackendOptions resolved = options;
    resolved.backend_name = std::move(target_name);
    if (!resolved.asset_reader) {
        resolved.asset_reader =
            resolve_model_asset_reader(resolved.model_dir, nullptr);
    }
    return factory(resolved);
}

}  // namespace mlvc
