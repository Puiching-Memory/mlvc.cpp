#pragma once

#include "mlvc_model.pb.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace mlvc::driver_cubin_backend {

using TensorId = std::uint32_t;
using Op = wire::OpKind;
using DataType = wire::DataType;
struct ConvAttributes {
    std::array<int, 2> strides;
    std::array<int, 4> pads;
    std::array<int, 2> dilations;
    int groups;
};
struct AxisAttributes { int axis; };
struct ActivationAttributes { float alpha; };
struct PixelAttributes { int block_size; };
struct Node {
    std::size_t index;
    Op op;
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    std::variant<std::monostate, ConvAttributes, AxisAttributes,
                 ActivationAttributes, PixelAttributes> attributes;
    const ConvAttributes& conv() const { return std::get<ConvAttributes>(attributes); }
    int axis() const { return std::get<AxisAttributes>(attributes).axis; }
    float alpha() const { return std::get<ActivationAttributes>(attributes).alpha; }
    int axis_or_zero() const {
        const auto* value = std::get_if<AxisAttributes>(&attributes);
        return value ? value->axis : 0;
    }
    float alpha_or_default() const {
        const auto* value = std::get_if<ActivationAttributes>(&attributes);
        return value ? value->alpha : 0.01F;
    }
    int block_size() const { return std::get<PixelAttributes>(attributes).block_size; }
};
struct ArenaSlot {
    std::size_t offset, bytes, birth, death;
};
struct WeightLocation { std::size_t offset, bytes; };
struct TensorDescriptor {
    std::string name;
    DataType dtype = wire::DATA_TYPE_UNSPECIFIED;
    std::vector<std::int64_t> shape;
    std::variant<std::monostate, ArenaSlot, WeightLocation> storage;
};
struct GraphDescriptor {
    std::vector<Node> nodes;
    std::vector<TensorDescriptor> tensors; // Dense ID table; slot zero is invalid.
    std::vector<TensorId> inputs, outputs;
    std::size_t arena_bytes, weights_bytes;
};

GraphDescriptor decode_graph(const wire::Graph& source);

}  // namespace mlvc::driver_cubin_backend
