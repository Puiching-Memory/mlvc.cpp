#include "graph_descriptor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace mlvc::driver_cubin_backend {
namespace {
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(std::string("invalid AOT Protobuf graph: ") + message);
}
int positive_int(std::uint32_t value)
{
    require(value > 0 && value <= std::numeric_limits<int>::max(), "invalid positive attribute");
    return static_cast<int>(value);
}
template <std::size_t N>
std::array<int, N> dimensions(const google::protobuf::RepeatedField<std::uint32_t>& values,
                              bool positive)
{
    require(values.size() == static_cast<int>(N), "attribute dimension count");
    std::array<int, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        const auto value = values.Get(static_cast<int>(i));
        require(value <= std::numeric_limits<int>::max() && (!positive || value > 0),
                "invalid attribute dimension");
        result[i] = static_cast<int>(value);
    }
    return result;
}
}  // namespace

GraphDescriptor decode_graph(const wire::Graph& source)
{
    require(source.schema_version() == 3, "unsupported schema");
    require(source.checkpoint_sha256().size() == 32 && source.weights_sha256().size() == 32,
            "missing SHA-256 provenance");
    require(source.tensors_size() > 0 && source.tensors_size() <= 1000000, "tensor count");
    require(source.nodes_size() > 0 && source.inputs_size() > 0 && source.outputs_size() > 0,
            "empty graph");
    require(source.arena_alignment() == 256 && source.arena_bytes() > 0, "arena configuration");
    GraphDescriptor result;
    result.arena_bytes = source.arena_bytes();
    result.weights_bytes = source.weights_bytes();
    result.tensors.resize(static_cast<std::size_t>(source.tensors_size()) + 1);
    std::unordered_set<std::string> names;
    for (const auto& input : source.tensors()) {
        require(input.id() > 0 && input.id() < result.tensors.size(), "tensor ID range");
        auto& tensor = result.tensors[input.id()];
        require(tensor.name.empty() && !input.name().empty() && names.insert(input.name()).second,
                "duplicate tensor ID or name");
        tensor.name = input.name();
        tensor.dtype = input.dtype();
        require(tensor.dtype == wire::FP16 || tensor.dtype == wire::INT32 || tensor.dtype == wire::INT64,
                "unsupported dtype");
        tensor.shape.assign(input.shape().begin(), input.shape().end());
        require(tensor.shape.size() <= 4, "tensor rank");
        std::size_t bytes = tensor.dtype == wire::FP16 ? 2 : tensor.dtype == wire::INT32 ? 4 : 8;
        for (const auto dimension : tensor.shape) {
            require(dimension > 0 && static_cast<std::uint64_t>(dimension) <=
                std::numeric_limits<std::size_t>::max() / bytes, "tensor extent overflow");
            bytes *= static_cast<std::size_t>(dimension);
        }
        if (input.has_weight()) {
            const auto& weight = input.weight();
            require(tensor.dtype != wire::INT32 && weight.offset() % 256 == 0 &&
                weight.size_bytes() == bytes && weight.offset() <= result.weights_bytes &&
                weight.size_bytes() <= result.weights_bytes - weight.offset(), "weight extent");
            tensor.storage = WeightLocation{weight.offset(), weight.size_bytes()};
        } else {
            require(tensor.dtype != wire::INT64, "runtime int64 tensor");
            if (input.has_arena()) {
                const auto& arena = input.arena();
                require(arena.offset() % 256 == 0 && arena.size_bytes() >= bytes &&
                    arena.offset() <= result.arena_bytes &&
                    arena.size_bytes() <= result.arena_bytes - arena.offset() &&
                    arena.birth() <= arena.death() &&
                    arena.death() <= static_cast<std::uint32_t>(source.nodes_size()), "arena extent");
                tensor.storage = ArenaSlot{arena.offset(), arena.size_bytes(), arena.birth(), arena.death()};
            }
        }
    }
    auto valid = [&](TensorId id) {
        return id > 0 && id < result.tensors.size() && !result.tensors[id].name.empty();
    };
    std::unordered_set<TensorId> available;
    for (const auto& input : source.tensors())
        if (input.has_weight()) available.insert(input.id());
    for (const auto id : source.inputs()) {
        require(valid(id) && std::holds_alternative<std::monostate>(result.tensors[id].storage) &&
                available.insert(id).second, "input ID or storage");
        result.inputs.push_back(id);
    }
    for (const auto& input : source.nodes()) {
        Node node;
        node.index = result.nodes.size();
        node.op = input.op();
        node.inputs.assign(input.inputs().begin(), input.inputs().end());
        node.outputs.assign(input.outputs().begin(), input.outputs().end());
        const auto count = node.inputs.size();
        switch (node.op) {
        case wire::CONV:
            require(input.has_conv() && (count == 2 || count == 3), "convolution contract");
            node.attributes = ConvAttributes{dimensions<2>(input.conv().strides(), true),
                dimensions<4>(input.conv().pads(), false), dimensions<2>(input.conv().dilations(), true),
                positive_int(input.conv().groups())};
            require(node.conv().dilations == std::array<int, 2>{1, 1}, "unsupported dilation");
            require(node.conv().pads[0] == node.conv().pads[2] &&
                    node.conv().pads[1] == node.conv().pads[3], "asymmetric convolution padding");
            break;
        case wire::CONCAT: case wire::GATHER:
            require(input.has_axis() && (node.op == wire::CONCAT ? count > 0 : count == 2), "axis contract");
            node.attributes = AxisAttributes{input.axis().axis()};
            break;
        case wire::LEAKY_RELU:
            require(input.has_activation() && count == 1 && std::isfinite(input.activation().alpha()), "activation contract");
            node.attributes = ActivationAttributes{input.activation().alpha()};
            break;
        case wire::SPACE_TO_DEPTH: case wire::DEPTH_TO_SPACE:
            require(input.has_pixel() && count == 1, "pixel layout contract");
            node.attributes = PixelAttributes{positive_int(input.pixel().block_size())};
            break;
        case wire::ADD: case wire::SUB: case wire::MUL:
            require(input.attributes_case() == wire::Node::ATTRIBUTES_NOT_SET && count == 2, "binary contract");
            break;
        case wire::CLIP:
            require(input.attributes_case() == wire::Node::ATTRIBUTES_NOT_SET && count >= 1 && count <= 3, "clip contract");
            break;
        case wire::SLICE:
            require(input.attributes_case() == wire::Node::ATTRIBUTES_NOT_SET && count >= 3 && count <= 5, "slice contract");
            break;
        case wire::RECIPROCAL: case wire::ROUND: case wire::SIGMOID:
            require(input.attributes_case() == wire::Node::ATTRIBUTES_NOT_SET && count == 1, "unary contract");
            break;
        default:
            require(false, "unknown operator");
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (node.inputs[i] == 0) {
                require((node.op == wire::CONV && i == 2) || (node.op == wire::CLIP && i > 0) ||
                        (node.op == wire::SLICE && i > 2), "absent required operand");
            } else {
                require(valid(node.inputs[i]) && available.contains(node.inputs[i]), "unresolved operand");
            }
        }
        require(node.outputs.size() == 1 && valid(node.outputs[0]), "output ID");
        const auto id = node.outputs[0];
        require(std::holds_alternative<ArenaSlot>(result.tensors[id].storage) && available.insert(id).second,
                "duplicate output or missing arena");
        require(std::get<ArenaSlot>(result.tensors[id].storage).birth == node.index, "tensor birth");
        result.nodes.push_back(std::move(node));
    }
    std::unordered_set<TensorId> outputs;
    for (const auto id : source.outputs()) {
        require(valid(id) && available.contains(id) && outputs.insert(id).second, "graph output");
        result.outputs.push_back(id);
    }
    return result;
}

}  // namespace mlvc::driver_cubin_backend
