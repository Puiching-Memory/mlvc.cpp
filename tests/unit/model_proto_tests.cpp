#include "graph_descriptor.hpp"
#include "model_proto.hpp"
#include "mlvc/core/model.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
using namespace mlvc;
using namespace mlvc::driver_cubin_backend;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
template <typename Operation>
void rejects(Operation operation)
{
    try { operation(); } catch (const std::exception&) { return; }
    throw std::runtime_error("malformed Protobuf model was accepted");
}

wire::Graph example()
{
    wire::Graph graph;
    graph.set_schema_version(3);
    graph.set_checkpoint_sha256(std::string(32, 'a'));
    graph.set_weights_sha256(std::string(32, 'b'));
    graph.set_weights_bytes(2);
    graph.set_arena_bytes(256);
    graph.set_arena_alignment(256);
    for (std::uint32_t id = 1; id <= 3; ++id) {
        auto* tensor = graph.add_tensors();
        tensor->set_id(id);
        tensor->set_name("tensor-" + std::to_string(id));
        tensor->set_dtype(wire::FP16);
        tensor->add_shape(1);
    }
    graph.mutable_tensors(0)->mutable_weight()->set_size_bytes(2);
    auto* arena = graph.mutable_tensors(2)->mutable_arena();
    arena->set_size_bytes(256);
    arena->set_death(1);
    graph.add_inputs(2);
    graph.add_outputs(3);
    auto* node = graph.add_nodes();
    node->set_op(wire::ADD);
    node->add_inputs(1);
    node->add_inputs(2);
    node->add_outputs(3);
    return graph;
}

void test_graph()
{
    const auto graph = example();
    wire::Graph decoded;
    require(decoded.ParseFromString(graph.SerializeAsString()), "Protobuf round trip");
    const auto runtime = decode_graph(decoded);
    require(runtime.nodes.at(0).op == wire::ADD && runtime.nodes.at(0).inputs.at(0) == 1,
            "typed graph decoding");
    auto invalid = graph;
    invalid.set_schema_version(2);
    rejects([&] { decode_graph(invalid); });
    invalid = graph;
    invalid.mutable_tensors(1)->set_id(1);
    rejects([&] { decode_graph(invalid); });
    invalid = graph;
    invalid.mutable_nodes(0)->set_inputs(0, 99);
    rejects([&] { decode_graph(invalid); });
    invalid = graph;
    invalid.mutable_nodes(0)->set_inputs(0, 3); // Forward/self reference.
    rejects([&] { decode_graph(invalid); });
    invalid = graph;
    invalid.mutable_nodes(0)->set_op(static_cast<wire::OpKind>(999));
    rejects([&] { decode_graph(invalid); });
    invalid = graph;
    invalid.mutable_nodes(0)->mutable_conv(); // Incorrect attributes for Add.
    rejects([&] { decode_graph(invalid); });
    invalid = graph;
    invalid.mutable_tensors(0)->mutable_weight()->set_offset(256);
    rejects([&] { decode_graph(invalid); });
    invalid = graph;
    invalid.mutable_tensors(2)->mutable_arena()->set_offset(256);
    rejects([&] { decode_graph(invalid); });
}

void test_file_decoder()
{
    const auto directory = std::filesystem::temp_directory_path() /
        ("mlvc-proto-" + std::to_string(getpid()));
    require(std::filesystem::create_directory(directory), "temporary directory already exists");
    const auto path = directory / "graph.pb";
    try {
        {
            std::ofstream file(path, std::ios::binary);
            require(example().SerializeToOstream(&file), "serialize model file");
        }
        decode_graph(detail::read_model_proto<wire::Graph>(directory, "graph.pb"));
        {
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            file << "{\"schema_version\":2}";
        }
        rejects([&] { detail::read_model_proto<wire::Graph>(directory, "graph.pb"); });
        std::filesystem::remove(path);
        std::filesystem::remove(directory);
    } catch (...) {
        std::filesystem::remove(path);
        std::filesystem::remove(directory);
        throw;
    }
}
}  // namespace

int main()
{
    try {
        test_graph();
        test_file_decoder();
        std::cout << "Protobuf model tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
