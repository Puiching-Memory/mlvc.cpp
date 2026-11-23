#pragma once

#include "core/assets.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
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
    AssetView weights;
    std::array<std::byte, 32> weights_sha256{};
};

GraphDescriptor decode_graph_binary(std::span<const std::byte> bytes);
LoadedGraph load_aot_graph(const ModelAssetReader& assets,
                           const std::string& model_name);
LoadedGraph load_aot_graph(const std::filesystem::path& model_dir,
                           const std::string& model_name);

inline std::size_t dtype_byte_size(DataType dtype)
{
    switch (dtype) {
    case DataType::FP16:
        return sizeof(Float16Storage);
    case DataType::INT32:
        return sizeof(std::int32_t);
    case DataType::INT64:
        return sizeof(std::int64_t);
    default:
        throw std::runtime_error(
            "unsupported AOT tensor data type " +
            std::to_string(static_cast<int>(dtype)));
    }
}

inline TensorDataType to_public_dtype(DataType dtype)
{
    switch (dtype) {
    case DataType::FP16:
        return TensorDataType::kFloat16;
    case DataType::INT32:
        return TensorDataType::kInt32;
    default:
        throw std::runtime_error(
            "unsupported AOT graph I/O data type " +
            std::to_string(static_cast<int>(dtype)));
    }
}

inline std::size_t checked_element_count(std::span<const std::int64_t> shape)
{
    return checked_tensor_element_count(shape);
}

inline std::array<int, 4> dims_to_4d(std::span<const std::int64_t> shape)
{
    if (shape.size() != 4U)
        throw std::runtime_error("expected 4D NCHW tensor");
    std::array<int, 4> dims{};
    for (std::size_t i = 0; i < 4U; ++i) {
        if (shape[i] <= 0 || shape[i] > std::numeric_limits<int>::max())
            throw std::runtime_error("tensor dimension out of int range");
        dims[i] = static_cast<int>(shape[i]);
    }
    return dims;
}

inline std::array<int, 4> pad_dims_to_4d(std::span<const std::int64_t> shape)
{
    if (shape.size() > 4U)
        throw std::runtime_error("tensors with rank > 4 are unsupported");
    std::array<int, 4> dims{1, 1, 1, 1};
    const std::size_t offset = 4U - shape.size();
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0 || shape[i] > std::numeric_limits<int>::max())
            throw std::runtime_error("tensor dimension out of int range");
        dims[offset + i] = static_cast<int>(shape[i]);
    }
    return dims;
}

inline unsigned int divide_up(std::size_t value, unsigned int divisor) noexcept
{
    return static_cast<unsigned int>((value + divisor - 1U) / divisor);
}

}  // namespace mlvc::aot
