#include "backends/webgpu/webgpu_surface.hpp"
#include "core/assets.hpp"
#include "core/half.hpp"
#include "core/sha256.hpp"
#include "mlvc_model.pb.h"
#include "runtime/backend.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace mlvc;

class InMemoryAssetReader final : public ModelAssetReader {
public:
    void add(std::string path, std::vector<std::byte> bytes)
    {
        files_.emplace(std::move(path), std::move(bytes));
    }

    std::string location_label() const override { return "memory:webgpu_test"; }

    bool exists(const std::filesystem::path& rel) const override
    {
        return files_.find(rel.generic_string()) != files_.end();
    }

    std::vector<std::byte> read_binary(const std::filesystem::path& rel) const override
    {
        const auto it = files_.find(rel.generic_string());
        if (it == files_.end())
            throw std::runtime_error("missing asset: " + rel.generic_string());
        return it->second;
    }

private:
    std::map<std::string, std::vector<std::byte>> files_;
};

void append_aligned_weights(std::vector<std::byte>& blob,
                            const void* data, std::size_t bytes,
                            std::uint64_t& out_offset)
{
    const std::size_t aligned = (blob.size() + 255U) & ~std::size_t{255U};
    blob.resize(aligned, std::byte{0});
    out_offset = static_cast<std::uint64_t>(blob.size());
    const auto* src = static_cast<const std::byte*>(data);
    blob.insert(blob.end(), src, src + bytes);
}

std::vector<std::byte> serialize_proto(const google::protobuf::MessageLite& msg)
{
    std::string raw;
    if (!msg.SerializeToString(&raw))
        throw std::runtime_error("failed to serialize test proto");
    std::vector<std::byte> out(raw.size());
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

void add_arena_tensor(wire::Graph& g, std::uint32_t id, const std::string& name,
                      wire::DataType dt, std::initializer_list<std::int64_t> shape,
                      std::uint64_t offset, std::uint32_t birth, std::uint32_t death)
{
    auto* t = g.add_tensors();
    t->set_id(id);
    t->set_name(name);
    t->set_dtype(dt);
    std::uint64_t elems = 1;
    for (const auto d : shape) {
        t->add_shape(d);
        elems *= static_cast<std::uint64_t>(d);
    }
    const std::uint64_t bpe = (dt == wire::FP16) ? 2U : (dt == wire::INT32 ? 4U : 8U);
    auto* a = t->mutable_arena();
    a->set_offset(offset);
    a->set_size_bytes(elems * bpe);
    a->set_birth(birth);
    a->set_death(death);
}

void add_weight_tensor(wire::Graph& g, std::vector<std::byte>& wblob,
                       std::uint32_t id, const std::string& name,
                       wire::DataType dt, std::initializer_list<std::int64_t> shape,
                       const void* data, std::size_t bytes)
{
    std::uint64_t off = 0;
    append_aligned_weights(wblob, data, bytes, off);
    auto* t = g.add_tensors();
    t->set_id(id);
    t->set_name(name);
    t->set_dtype(dt);
    for (const auto d : shape)
        t->add_shape(d);
    auto* wl = t->mutable_weight();
    wl->set_offset(off);
    wl->set_size_bytes(bytes);
}

void test_pointwise_reglu_and_d2s_surface()
{
    // Builds a synthetic graph:
    //   input x [1, 8, 4, 4] -> Conv 1x1 (8 -> 16) -> ReGLU (Slice/Slice/Clip/Mul -> [1, 8, 4, 4])
    //   -> Depthwise 3x3 s1p1 (8 -> 8) -> output
    wire::Graph g;
    g.set_schema_version(3);
    g.set_arena_alignment(256);
    g.set_arena_bytes(4096);
    std::vector<std::byte> wblob;

    {
        auto* t = g.add_tensors();
        t->set_id(1);
        t->set_name("x");
        t->set_dtype(wire::FP16);
        for (const auto d : {1, 8, 4, 4})
            t->add_shape(d);
    }

    // Conv 1x1 weights [16, 8, 1, 1] + bias [16]
    std::vector<Float16Storage> w1(16 * 8, 0);
    for (std::size_t o = 0; o < 16; ++o) {
        for (std::size_t c = 0; c < 8; ++c) {
            const float val = ((o % 8) == c) ? (o < 8 ? 0.5F : 1.0F) : 0.0F;
            w1[o * 8 + c] = float_to_half(val);
        }
    }
    std::vector<Float16Storage> b1(16, float_to_half(0.25F));
    add_weight_tensor(g, wblob, 2, "w1", wire::FP16, {16, 8, 1, 1}, w1.data(), w1.size() * 2);
    add_weight_tensor(g, wblob, 3, "b1", wire::FP16, {16}, b1.data(), b1.size() * 2);

    add_arena_tensor(g, 4, "conv_out", wire::FP16, {1, 16, 4, 4}, 256, 0, 2);
    std::int64_t zero64 = 0, eight64 = 8, sixteen64 = 16, one64 = 1;
    add_weight_tensor(g, wblob, 5, "s0", wire::INT64, {1}, &zero64, 8);
    add_weight_tensor(g, wblob, 6, "e0", wire::INT64, {1}, &eight64, 8);
    add_weight_tensor(g, wblob, 7, "e1", wire::INT64, {1}, &sixteen64, 8);
    add_weight_tensor(g, wblob, 8, "ax1", wire::INT64, {1}, &one64, 8);
    Float16Storage cmin = float_to_half(0.0F);
    Float16Storage cmax = float_to_half(1.0F);
    add_weight_tensor(g, wblob, 9, "cmin", wire::FP16, {1}, &cmin, 2);
    add_weight_tensor(g, wblob, 10, "cmax", wire::FP16, {1}, &cmax, 2);

    add_arena_tensor(g, 11, "gate", wire::FP16, {1, 8, 4, 4}, 1024, 1, 3);
    add_arena_tensor(g, 12, "lin", wire::FP16, {1, 8, 4, 4}, 1536, 2, 4);
    add_arena_tensor(g, 13, "clipped", wire::FP16, {1, 8, 4, 4}, 2048, 3, 4);
    add_arena_tensor(g, 14, "reglu_out", wire::FP16, {1, 8, 4, 4}, 2560, 4, 5);

    // Depthwise 3x3 s1p1 identity weights [8, 1, 3, 3] (center = 1.0)
    std::vector<Float16Storage> dw(8 * 9, float_to_half(0.0F));
    for (std::size_t c = 0; c < 8; ++c)
        dw[c * 9 + 4] = float_to_half(1.0F);
    std::vector<Float16Storage> db(8, float_to_half(0.0F));
    add_weight_tensor(g, wblob, 15, "dw", wire::FP16, {8, 1, 3, 3}, dw.data(), dw.size() * 2);
    add_weight_tensor(g, wblob, 16, "db", wire::FP16, {8}, db.data(), db.size() * 2);
    add_arena_tensor(g, 17, "y", wire::FP16, {1, 8, 4, 4}, 3072, 5, 5);

    g.add_inputs(1);
    g.add_outputs(17);

    // Node 0: Conv 1x1
    {
        auto* n = g.add_nodes();
        n->set_op(wire::CONV);
        n->add_inputs(1); n->add_inputs(2); n->add_inputs(3);
        n->add_outputs(4);
        auto* c = n->mutable_conv();
        c->add_strides(1); c->add_strides(1);
        c->add_pads(0); c->add_pads(0); c->add_pads(0); c->add_pads(0);
        c->add_dilations(1); c->add_dilations(1);
        c->set_groups(1);
    }
    // Node 1: Slice 0..8
    {
        auto* n = g.add_nodes();
        n->set_op(wire::SLICE);
        n->add_inputs(4); n->add_inputs(5); n->add_inputs(6); n->add_inputs(8);
        n->add_outputs(11);
    }
    // Node 2: Slice 8..16
    {
        auto* n = g.add_nodes();
        n->set_op(wire::SLICE);
        n->add_inputs(4); n->add_inputs(6); n->add_inputs(7); n->add_inputs(8);
        n->add_outputs(12);
    }
    // Node 3: Clip
    {
        auto* n = g.add_nodes();
        n->set_op(wire::CLIP);
        n->add_inputs(11); n->add_inputs(9); n->add_inputs(10);
        n->add_outputs(13);
    }
    // Node 4: Mul
    {
        auto* n = g.add_nodes();
        n->set_op(wire::MUL);
        n->add_inputs(13); n->add_inputs(12);
        n->add_outputs(14);
    }
    // Node 5: Depthwise 3x3
    {
        auto* n = g.add_nodes();
        n->set_op(wire::CONV);
        n->add_inputs(14); n->add_inputs(15); n->add_inputs(16);
        n->add_outputs(17);
        auto* c = n->mutable_conv();
        c->add_strides(1); c->add_strides(1);
        c->add_pads(1); c->add_pads(1); c->add_pads(1); c->add_pads(1);
        c->add_dilations(1); c->add_dilations(1);
        c->set_groups(8);
    }

    const std::string fake_ckpt_sha(32, 'k');
    const auto wsha = sha256_digest(wblob);
    g.set_checkpoint_sha256(fake_ckpt_sha);
    g.set_weights_bytes(wblob.size());
    g.set_weights_sha256(wsha.data(), wsha.size());

    wire::ModelMetadata meta;
    meta.set_schema_version(3);
    meta.set_profile("webgpu-unit-test");
    meta.set_model_version("v1");
    meta.set_checkpoint_sha256(fake_ckpt_sha);
    meta.set_width(4);
    meta.set_height(4);
    meta.set_pixel_range(255.0F);
    meta.set_qp_num(1);
    meta.set_total_qp_num(1);
    meta.add_frame_index_map(0);
    meta.add_qp_shift(0);
    meta.set_feature_channels(8);
    meta.set_latent_channels(8);
    meta.set_hyperprior_channels(8);
    meta.set_downsample_feature(2);
    meta.set_downsample_hyperprior(4);

    auto reader = std::make_shared<InMemoryAssetReader>();
    reader->add("metadata.pb", serialize_proto(meta));
    reader->add("aot/TestGraph/graph.pb", serialize_proto(g));
    reader->add("aot/TestGraph/weights.bin", std::move(wblob));

    BackendOptions opts;
    opts.backend_name = "webgpu";
    opts.asset_reader = reader;
    auto backend = create_backend(opts);
    backend->load("TestGraph", ModelExecutionConfig{});

    auto* surf = dynamic_cast<webgpu_backend::WebGpuSurfaceBackend*>(backend.get());
    if (!surf)
        throw std::runtime_error("WebGpuBackend does not expose WebGpuSurfaceBackend");
    // Fused Conv+ReGLU + Depthwise 3x3 should compile down to exactly 2 dispatches!
    if (surf->step_count() != 2U) {
        throw std::runtime_error("expected 2 fused dispatches, got " +
                                 std::to_string(surf->step_count()));
    }

    std::vector<Float16Storage> in_vals(8 * 4 * 4);
    for (std::size_t i = 0; i < in_vals.size(); ++i)
        in_vals[i] = float_to_half(-1.0F + 0.05F * static_cast<float>(i % 40));

    Tensor in_t{"x", {1, 8, 4, 4}, in_vals};
    auto outs = backend->run({in_t});
    if (outs.size() != 1U)
        throw std::runtime_error("expected 1 output tensor");
    const auto& out_fp16 = std::get<std::vector<Float16Storage>>(outs[0].data);
    float max_err = 0.0F;
    for (std::size_t i = 0; i < in_vals.size(); ++i) {
        const float x = half_to_float(in_vals[i]);
        const float gate = std::min(std::max(half_to_float(float_to_half(0.5F * x + 0.25F)), 0.0F), 1.0F);
        const float lin = half_to_float(float_to_half(x + 0.25F));
        const float expected = half_to_float(float_to_half(gate * lin));
        const float actual = half_to_float(out_fp16[i]);
        max_err = std::max(max_err, std::fabs(expected - actual));
    }
    const float tol = surf->uses_f16() ? 1e-3F : 2e-3F;
    if (max_err > tol) {
        throw std::runtime_error("pointwise+ReGLU+DW3x3 max_err=" + std::to_string(max_err));
    }
    std::cout << "[PASS] test_pointwise_reglu_and_dw3x3 (f16="
              << (surf->uses_f16() ? "on" : "off")
              << ", dispatches=" << surf->step_count()
              << ", max_err=" << max_err << ")\n";
}

}  // namespace

int main()
{
    try {
        test_pointwise_reglu_and_d2s_surface();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
