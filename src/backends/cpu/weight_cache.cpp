#include "weight_cache.hpp"

#include <functional>
#include <mutex>

namespace mlvc::cpu_backend {
namespace {

struct WeightCacheKeyHash {
    std::size_t operator()(const WeightCacheKey& k) const noexcept
    {
        std::size_t h = 0;
        for (std::size_t i = 0; i < 8U; ++i)
            h = (h << 8) | static_cast<std::size_t>(k.weights_sha256[i]);
        h ^= std::hash<std::string>{}(k.model_name) + 0x9e3779b9U + (h << 6) + (h >> 2);
        h ^= static_cast<std::size_t>(k.gemm_mode) * 2654435761U;
        h ^= static_cast<std::size_t>(k.fuse);
        return h;
    }
};

struct WeightBankCache {
    std::mutex mutex;
    std::unordered_map<WeightCacheKey, std::weak_ptr<const SharedWeightBank>,
                       WeightCacheKeyHash> entries;
};

WeightBankCache& weight_bank_cache()
{
    static WeightBankCache cache;
    return cache;
}

void prune_expired_entries(WeightBankCache& cache)
{
    for (auto it = cache.entries.begin(); it != cache.entries.end();) {
        if (it->second.expired())
            it = cache.entries.erase(it);
        else
            ++it;
    }
}

}  // namespace

std::shared_ptr<const SharedWeightBank> lookup_shared_weight_bank(
    const WeightCacheKey& key)
{
    auto& cache = weight_bank_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (const auto it = cache.entries.find(key); it != cache.entries.end())
        return it->second.lock();
    return nullptr;
}

std::shared_ptr<const SharedWeightBank> publish_shared_weight_bank(
    const WeightCacheKey& key,
    std::shared_ptr<SharedWeightBank> candidate)
{
    auto& cache = weight_bank_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (auto existing = cache.entries[key].lock())
        return existing;
    std::shared_ptr<const SharedWeightBank> published = std::move(candidate);
    cache.entries[key] = published;
    if (cache.entries.size() > 32U)
        prune_expired_entries(cache);
    return published;
}

}  // namespace mlvc::cpu_backend
