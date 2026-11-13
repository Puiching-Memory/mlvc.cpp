#include "mlvc/runtime/aot_graph_proto.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace mlvc::aot {
namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(std::string("invalid AOT Protobuf graph: ") + message);
}

int positive_int(std::uint32_t value)
{
    require(value > 0 &&
                value <= static_cast<std::uint32_t>(std::numeric_limits<int>::max()),
            "invalid positive attribute");
    return static_cast<int>(value);
}

template <std::size_t N>
std::array<int, N> dimensions(
    const google::protobuf::RepeatedField<std::uint32_t>& values, bool positive)
{
    require(values.size() == static_cast<int>(N), "attribute dimension count");
    std::array<int, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        const auto value = values.Get(static_cast<int>(i));
        require(value <= static_cast<std::uint32_t>(std::numeric_limits<int>::max()) &&
                    (!positive || value > 0),
                "invalid attribute dimension");
        result[i] = static_cast<int>(value);
    }
    return result;
}

DataType convert_dtype(wire::DataType value)
{
    switch (value) {
    case wire::FP16:
        return DataType::FP16;
    case wire::INT32:
        return DataType::INT32;
    case wire::INT64:
        return DataType::INT64;
    default:
        require(false, "unsupported dtype");
        return DataType::DATA_TYPE_UNSPECIFIED;
    }
}

Op convert_op(wire::OpKind value)
{
    switch (value) {
    case wire::ADD:
        return Op::ADD;
    case wire::SUB:
        return Op::SUB;
    case wire::MUL:
        return Op::MUL;
    case wire::CLIP:
        return Op::CLIP;
    case wire::CONCAT:
        return Op::CONCAT;
    case wire::CONV:
        return Op::CONV;
    case wire::DEPTH_TO_SPACE:
        return Op::DEPTH_TO_SPACE;
    case wire::GATHER:
        return Op::GATHER;
    case wire::LEAKY_RELU:
        return Op::LEAKY_RELU;
    case wire::RECIPROCAL:
        return Op::RECIPROCAL;
    case wire::ROUND:
        return Op::ROUND;
    case wire::SIGMOID:
        return Op::SIGMOID;
    case wire::SLICE:
        return Op::SLICE;
    case wire::SPACE_TO_DEPTH:
        return Op::SPACE_TO_DEPTH;
    default:
        require(false, "unknown operator");
        return Op::OP_UNSPECIFIED;
    }
}

template <typename Message>
Message parse_proto_bytes(std::span<const std::byte> bytes,
                          const std::string& label)
{
    constexpr std::size_t kLimit = 64U * 1024U * 1024U;
    if (bytes.empty() || bytes.size() > kLimit)
        throw std::runtime_error("invalid Protobuf model message size: " + label);
    Message result;
    if (!result.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())))
        throw std::runtime_error("invalid Protobuf model message: " + label);
    return result;
}

}  // namespace

GraphDescriptor decode_graph(const wire::Graph& source)
{
    require(source.schema_version() == 3, "unsupported schema");
    require(source.checkpoint_sha256().size() == 32 &&
                source.weights_sha256().size() == 32,
            "missing SHA-256 provenance");
    require(source.tensors_size() > 0 && source.tensors_size() <= 1000000,
            "tensor count");
    require(source.nodes_size() > 0 && source.inputs_size() > 0 &&
                source.outputs_size() > 0,
            "empty graph");
    require(source.arena_alignment() == 256 && source.arena_bytes() > 0,
            "arena configuration");

    GraphDescriptor result;
    result.arena_bytes = source.arena_bytes();
    result.weights_bytes = source.weights_bytes();
    result.tensors.resize(static_cast<std::size_t>(source.tensors_size()) + 1);
    std::unordered_set<std::string> names;
    for (const auto& input : source.tensors()) {
        require(input.id() > 0 && input.id() < result.tensors.size(),
                "tensor ID range");
        auto& tensor = result.tensors[input.id()];
        require(tensor.name.empty() && !input.name().empty() &&
                    names.insert(input.name()).second,
                "duplicate tensor ID or name");
        tensor.name = input.name();
        tensor.dtype = convert_dtype(input.dtype());
        tensor.shape.assign(input.shape().begin(), input.shape().end());
        require(tensor.shape.size() <= 4, "tensor rank");
        std::size_t bytes =
            tensor.dtype == DataType::FP16 ? 2 : tensor.dtype == DataType::INT32 ? 4 : 8;
        for (const auto dimension : tensor.shape) {
            require(dimension > 0 &&
                        static_cast<std::uint64_t>(dimension) <=
                            std::numeric_limits<std::size_t>::max() / bytes,
                    "tensor extent overflow");
            bytes *= static_cast<std::size_t>(dimension);
        }
        if (input.has_weight()) {
            const auto& weight = input.weight();
            require(tensor.dtype != DataType::INT32 && weight.offset() % 256 == 0 &&
                        weight.size_bytes() == bytes &&
                        weight.offset() <= result.weights_bytes &&
                        weight.size_bytes() <= result.weights_bytes - weight.offset(),
                    "weight extent");
            tensor.storage = WeightLocation{weight.offset(), weight.size_bytes()};
        } else {
            require(tensor.dtype != DataType::INT64, "runtime int64 tensor");
            if (input.has_arena()) {
                const auto& arena = input.arena();
                require(arena.offset() % 256 == 0 && arena.size_bytes() >= bytes &&
                            arena.offset() <= result.arena_bytes &&
                            arena.size_bytes() <= result.arena_bytes - arena.offset() &&
                            arena.birth() <= arena.death() &&
                            arena.death() <=
                                static_cast<std::uint32_t>(source.nodes_size()),
                        "arena extent");
                tensor.storage = ArenaSlot{
                    arena.offset(), arena.size_bytes(), arena.birth(), arena.death()};
            }
        }
    }

    auto valid = [&](TensorId id) {
        return id > 0 && id < result.tensors.size() && !result.tensors[id].name.empty();
    };
    std::unordered_set<TensorId> available;
    for (const auto& item : source.tensors())
        if (item.has_weight())
            available.insert(item.id());
    for (const auto id : source.inputs()) {
        require(valid(id) &&
                    std::holds_alternative<std::monostate>(result.tensors[id].storage) &&
                    available.insert(id).second,
                "input ID or storage");
        result.inputs.push_back(id);
    }

    for (const auto& input : source.nodes()) {
        Node node;
        node.index = result.nodes.size();
        node.op = convert_op(input.op());
        node.inputs.assign(input.inputs().begin(), input.inputs().end());
        node.outputs.assign(input.outputs().begin(), input.outputs().end());
        const auto count = node.inputs.size();
        switch (node.op) {
        case Op::CONV:
            require(input.has_conv() && (count == 2 || count == 3),
                    "convolution contract");
            node.attributes = ConvAttributes{
                dimensions<2>(input.conv().strides(), true),
                dimensions<4>(input.conv().pads(), false),
                dimensions<2>(input.conv().dilations(), true),
                positive_int(input.conv().groups())};
            require(node.conv().dilations == std::array<int, 2>{1, 1},
                    "unsupported dilation");
            require(node.conv().pads[0] == node.conv().pads[2] &&
                        node.conv().pads[1] == node.conv().pads[3],
                    "asymmetric convolution padding");
            break;
        case Op::CONCAT:
        case Op::GATHER:
            require(input.has_axis() &&
                        (node.op == Op::CONCAT ? count > 0 : count == 2),
                    "axis contract");
            node.attributes = AxisAttributes{input.axis().axis()};
            break;
        case Op::LEAKY_RELU:
            require(input.has_activation() && count == 1 &&
                        std::isfinite(input.activation().alpha()),
                    "activation contract");
            node.attributes = ActivationAttributes{input.activation().alpha()};
            break;
        case Op::SPACE_TO_DEPTH:
        case Op::DEPTH_TO_SPACE:
            require(input.has_pixel() && count == 1, "pixel layout contract");
            node.attributes = PixelAttributes{positive_int(input.pixel().block_size())};
            break;
        case Op::ADD:
        case Op::SUB:
        case Op::MUL:
            require(input.attributes_case() == wire::Node::ATTRIBUTES_NOT_SET &&
                        count == 2,
                    "binary contract");
            break;
        case Op::CLIP:
            require(input.attributes_case() == wire::Node::ATTRIBUTES_NOT_SET &&
                        count >= 1 && count <= 3,
                    "clip contract");
            break;
        case Op::SLICE:
            require(input.attributes_case() == wire::Node::ATTRIBUTES_NOT_SET &&
                        count >= 3 && count <= 5,
                    "slice contract");
            break;
        case Op::RECIPROCAL:
        case Op::ROUND:
        case Op::SIGMOID:
            require(input.attributes_case() == wire::Node::ATTRIBUTES_NOT_SET &&
                        count == 1,
                    "unary contract");
            break;
        default:
            require(false, "unknown operator");
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (node.inputs[i] == 0) {
                require((node.op == Op::CONV && i == 2) ||
                            (node.op == Op::CLIP && i > 0) ||
                            (node.op == Op::SLICE && i > 2),
                        "absent required operand");
            } else {
                require(valid(node.inputs[i]) && available.contains(node.inputs[i]),
                        "unresolved operand");
            }
        }
        require(node.outputs.size() == 1 && valid(node.outputs[0]), "output ID");
        const auto id = node.outputs[0];
        require(std::holds_alternative<ArenaSlot>(result.tensors[id].storage) &&
                    available.insert(id).second,
                "duplicate output or missing arena");
        require(std::get<ArenaSlot>(result.tensors[id].storage).birth == node.index,
                "tensor birth");
        result.nodes.push_back(std::move(node));
    }

    std::unordered_set<TensorId> outputs;
    for (const auto id : source.outputs()) {
        require(valid(id) && available.contains(id) && outputs.insert(id).second,
                "graph output");
        result.outputs.push_back(id);
    }
    return result;
}

GraphDescriptor decode_graph_binary(std::span<const std::byte> bytes)
{
    return decode_graph(parse_proto_bytes<wire::Graph>(bytes, "graph.pb"));
}

LoadedGraph load_aot_graph(const ModelAssetReader& assets,
                           const std::string& model_name)
{
    const std::filesystem::path graph_dir =
        std::filesystem::path("aot") / model_name;
    const auto graph_bytes = assets.read_binary(graph_dir / "graph.pb");
    const auto graph_message =
        parse_proto_bytes<wire::Graph>(graph_bytes, (graph_dir / "graph.pb").string());
    const auto metadata_bytes = assets.read_binary("metadata.pb");
    const auto metadata =
        parse_proto_bytes<wire::ModelMetadata>(metadata_bytes, "metadata.pb");
    if (graph_message.checkpoint_sha256() != metadata.checkpoint_sha256()) {
        throw std::runtime_error(
            "AOT graph checkpoint differs from metadata (" +
            assets.location_label() + ")");
    }

    LoadedGraph loaded;
    loaded.descriptor = decode_graph(graph_message);
    loaded.weights = assets.read_binary(graph_dir / "weights.bin");
    if (loaded.weights.size() != loaded.descriptor.weights_bytes) {
        throw std::runtime_error(
            "AOT weights size does not match graph (" +
            assets.location_label() + ")");
    }
    return loaded;
}

LoadedGraph load_aot_graph(const std::filesystem::path& model_dir,
                           const std::string& model_name)
{
    const auto assets = make_model_asset_reader(model_dir);
    return load_aot_graph(*assets, model_name);
}

}  // namespace mlvc::aot
