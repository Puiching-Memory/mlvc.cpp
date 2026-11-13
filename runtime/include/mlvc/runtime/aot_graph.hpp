#pragma once

#include "mlvc/core/assets.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace mlvc::aot {

using TensorId = std::uint32_t;

enum class Op {
    OP_UNSPECIFIED = 0,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    CLIP = 4,
    CONCAT = 5,
    CONV = 6,
    DEPTH_TO_SPACE = 7,
    GATHER = 8,
    LEAKY_RELU = 9,
    RECIPROCAL = 10,
    ROUND = 11,
    SIGMOID = 12,
    SLICE = 13,
    SPACE_TO_DEPTH = 14,
};

enum class DataType {
    DATA_TYPE_UNSPECIFIED = 0,
    FP16 = 1,
    INT32 = 2,
    INT64 = 3,
};

struct ConvAttributes {
    std::array<int, 2> strides;
    std::array<int, 4> pads;
    std::array<int, 2> dilations;
    int groups;
};

struct AxisAttributes {
    int axis;
};

struct ActivationAttributes {
    float alpha;
};

struct PixelAttributes {
    int block_size;
};

struct Node {
    std::size_t index = 0;
    Op op = Op::OP_UNSPECIFIED;
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    std::variant<std::monostate, ConvAttributes, AxisAttributes,
                 ActivationAttributes, PixelAttributes> attributes;

    const ConvAttributes& conv() const { return std::get<ConvAttributes>(attributes); }
    int axis() const { return std::get<AxisAttributes>(attributes).axis; }
    float alpha() const { return std::get<ActivationAttributes>(attributes).alpha; }
    int axis_or_zero() const
    {
        const auto* value = std::get_if<AxisAttributes>(&attributes);
        return value ? value->axis : 0;
    }
    float alpha_or_default() const
    {
        const auto* value = std::get_if<ActivationAttributes>(&attributes);
        return value ? value->alpha : 0.01F;
    }
    int block_size() const { return std::get<PixelAttributes>(attributes).block_size; }
};

struct ArenaSlot {
    std::size_t offset = 0;
    std::size_t bytes = 0;
    std::size_t birth = 0;
    std::size_t death = 0;
};

struct WeightLocation {
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

struct TensorDescriptor {
    std::string name;
    DataType dtype = DataType::DATA_TYPE_UNSPECIFIED;
    std::vector<std::int64_t> shape;
    std::variant<std::monostate, ArenaSlot, WeightLocation> storage;
};

struct GraphDescriptor {
    std::vector<Node> nodes;
    std::vector<TensorDescriptor> tensors;  // Dense ID table; slot zero is invalid.
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    std::size_t arena_bytes = 0;
    std::size_t weights_bytes = 0;
};

struct LoadedGraph {
    GraphDescriptor descriptor;
    std::vector<std::byte> weights;
};

GraphDescriptor decode_graph_binary(std::span<const std::byte> bytes);
LoadedGraph load_aot_graph(const ModelAssetReader& assets,
                           const std::string& model_name);
LoadedGraph load_aot_graph(const std::filesystem::path& model_dir,
                           const std::string& model_name);

}  // namespace mlvc::aot
