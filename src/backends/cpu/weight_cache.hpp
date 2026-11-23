#pragma once

#include "cpu_features.hpp"

#include "core/tensor.hpp"
#include "runtime/aot_graph.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlvc::cpu_backend {

struct SharedWeightBank {
    std::unordered_map<std::size_t, std::vector<Float16Storage>> conv_packed_by_node;
    std::unordered_map<aot::TensorId, std::vector<Float16Storage>> init_fp16;
    std::unordered_map<aot::TensorId, std::vector<std::int32_t>> init_int32;
    std::unordered_map<aot::TensorId, std::vector<std::int64_t>> init_int64;
};

struct WeightCacheKey {
    std::array<std::byte, 32> weights_sha256{};
    std::string model_name;
    GemmMode gemm_mode = GemmMode::kScalarFp32;
    bool fuse = true;

    bool operator==(const WeightCacheKey& other) const = default;
};

std::shared_ptr<const SharedWeightBank> lookup_shared_weight_bank(
    const WeightCacheKey& key);

// Publishes a newly built weight bank into the weak-reference cache.
// Returns the canonical shared bank (which may be a concurrently inserted bank
// if another thread raced on the same key).
std::shared_ptr<const SharedWeightBank> publish_shared_weight_bank(
    const WeightCacheKey& key,
    std::shared_ptr<SharedWeightBank> candidate);

}  // namespace mlvc::cpu_backend
