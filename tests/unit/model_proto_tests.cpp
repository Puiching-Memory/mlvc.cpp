#include "test_support.hpp"

#include "core/assets.hpp"
#include "core/model.hpp"
#include "runtime/aot_graph_proto.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace mlvc;
using namespace mlvc::aot;

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
    MLVC_REQUIRE(decoded.ParseFromString(graph.SerializeAsString()),
                 "Protobuf round trip");
    const auto runtime = decode_graph(decoded);
    MLVC_REQUIRE(runtime.nodes.at(0).op == Op::ADD &&
                     runtime.nodes.at(0).inputs.at(0) == 1,
                 "typed graph decoding");

    auto invalid = graph;
    invalid.set_schema_version(2);
    MLVC_REQUIRE_THROWS(static_cast<void>(decode_graph(invalid)),
                        "malformed schema version was accepted");
    invalid = graph;
    invalid.mutable_tensors(1)->set_id(1);
    MLVC_REQUIRE_THROWS(static_cast<void>(decode_graph(invalid)),
                        "duplicate tensor id was accepted");
    invalid = graph;
    invalid.mutable_nodes(0)->set_inputs(0, 99);
    MLVC_REQUIRE_THROWS(static_cast<void>(decode_graph(invalid)),
                        "out-of-range input id was accepted");
    invalid = graph;
    invalid.mutable_nodes(0)->set_inputs(0, 3);  // Forward/self reference.
    MLVC_REQUIRE_THROWS(static_cast<void>(decode_graph(invalid)),
                        "forward reference was accepted");
    invalid = graph;
    invalid.mutable_nodes(0)->set_op(static_cast<wire::OpKind>(999));
    MLVC_REQUIRE_THROWS(static_cast<void>(decode_graph(invalid)),
                        "unknown op kind was accepted");
    invalid = graph;
    invalid.mutable_nodes(0)->mutable_conv();  // Incorrect attributes for Add.
    MLVC_REQUIRE_THROWS(static_cast<void>(decode_graph(invalid)),
                        "mismatched node attributes were accepted");
    invalid = graph;
    invalid.mutable_tensors(0)->mutable_weight()->set_offset(256);
    MLVC_REQUIRE_THROWS(static_cast<void>(decode_graph(invalid)),
                        "out-of-bounds weight offset was accepted");
    invalid = graph;
    invalid.mutable_tensors(2)->mutable_arena()->set_offset(256);
    MLVC_REQUIRE_THROWS(static_cast<void>(decode_graph(invalid)),
                        "out-of-bounds arena offset was accepted");
}

void test_file_decoder()
{
    mlvc::test::ScopedTempDir temp("mlvc-proto");
    const auto path = temp.path() / "graph.pb";
    {
        std::ofstream file(path, std::ios::binary);
        MLVC_REQUIRE(example().SerializeToOstream(&file), "serialize model file");
    }
    const auto reader = make_directory_asset_reader(temp.path());
    const auto binary = reader->read_binary("graph.pb");
    const auto runtime = decode_graph_binary(binary);
    MLVC_REQUIRE(runtime.nodes.at(0).op == Op::ADD, "binary graph decoding");
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << "{\"schema_version\":2}";
    }
    const auto bad_binary = reader->read_binary("graph.pb");
    MLVC_REQUIRE_THROWS(static_cast<void>(decode_graph_binary(bad_binary)),
                        "corrupt binary graph was accepted");
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
