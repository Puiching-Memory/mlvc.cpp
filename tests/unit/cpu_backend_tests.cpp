#include "mlvc/core/assets.hpp"
#include "mlvc/core/half.hpp"
#include "mlvc/core/tensor.hpp"
#include "mlvc/runtime/backend.hpp"
#include "mlvc_model.pb.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace mlvc;

void require(bool cond, const char* msg)
{
    if (!cond)
        throw std::runtime_error(msg);
}

template <typename Fn>
void expect_throw(Fn&& fn, const char* msg)
{
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(msg);
}

class MemoryAssetReader final : public ModelAssetReader {
public:
    void add(const std::filesystem::path& rel, std::vector<std::byte> data)
    {
        files_[rel.generic_string()] = std::move(data);
    }

    std::string location_label() const override { return "in-memory-test"; }

    bool exists(const std::filesystem::path& rel) const override
    {
        return files_.find(rel.generic_string()) != files_.end();
    }

    std::vector<std::byte> read_binary(const std::filesystem::path& rel) const override
    {
        const auto it = files_.find(rel.generic_string());
        if (it == files_.end())
            throw std::runtime_error("missing in-memory asset: " + rel.generic_string());
        return it->second;
    }

private:
    std::map<std::string, std::vector<std::byte>> files_;
};

std::vector<std::byte> to_bytes(const std::string& s)
{
    std::vector<std::byte> out(s.size());
    std::memcpy(out.data(), s.data(), s.size());
    return out;
}

class WireGraphBuilder {
public:
    explicit WireGraphBuilder(std::string model_name) : model_name_(std::move(model_name))
    {
        sha_ = std::string(32, 'k');
        wsha_ = std::string(32, 'w');
        graph_.set_schema_version(3);
        graph_.set_checkpoint_sha256(sha_);
        graph_.set_weights_sha256(wsha_);
        graph_.set_arena_alignment(256);
    }

    std::uint32_t add_input(const std::string& name, wire::DataType dt,
                            std::initializer_list<std::int64_t> shape)
    {
        const std::uint32_t id = next_tensor(name, dt, shape);
        graph_.add_inputs(id);
        return id;
    }

    std::uint32_t add_weight_fp16(const std::string& name,
                                  std::initializer_list<std::int64_t> shape,
                                  const std::vector<float>& vals)
    {
        std::vector<Float16Storage> h(vals.size());
        for (std::size_t i = 0; i < vals.size(); ++i)
            h[i] = float_to_half(vals[i]);
        return add_weight_raw(name, wire::FP16, shape, h.data(), h.size() * sizeof(Float16Storage));
    }

    std::uint32_t add_weight_fp16_raw(const std::string& name,
                                      std::initializer_list<std::int64_t> shape,
                                      const std::vector<Float16Storage>& vals)
    {
        return add_weight_raw(name, wire::FP16, shape, vals.data(), vals.size() * sizeof(Float16Storage));
    }

    std::uint32_t add_weight_int64(const std::string& name,
                                   std::initializer_list<std::int64_t> shape,
                                   const std::vector<std::int64_t>& vals)
    {
        return add_weight_raw(name, wire::INT64, shape, vals.data(), vals.size() * sizeof(std::int64_t));
    }

    std::uint32_t add_activation(const std::string& name, wire::DataType dt,
                                 std::initializer_list<std::int64_t> shape,
                                 std::uint32_t birth_node)
    {
        const std::uint32_t id = next_tensor(name, dt, shape);
        auto* td = graph_.mutable_tensors(static_cast<int>(id - 1));
        std::size_t elems = 1;
        for (const auto d : shape)
            elems *= static_cast<std::size_t>(d);
        const std::size_t elem_sz = dt == wire::FP16 ? 2U : 4U;
        const std::size_t sz = ((elems * elem_sz + 255U) / 256U) * 256U;
        auto* ar = td->mutable_arena();
        ar->set_offset(arena_cursor_);
        ar->set_size_bytes(sz);
        ar->set_birth(birth_node);
        ar->set_death(0);  // patched in build()
        arena_cursor_ += sz;
        return id;
    }

    std::uint32_t add_unary(wire::OpKind op, const std::string& out_name,
                            std::uint32_t in_id, std::initializer_list<std::int64_t> out_shape)
    {
        const std::uint32_t node_idx = static_cast<std::uint32_t>(graph_.nodes_size());
        const std::uint32_t out_id = add_activation(out_name, wire::FP16, out_shape, node_idx);
        auto* n = graph_.add_nodes();
        n->set_op(op);
        n->add_inputs(in_id);
        n->add_outputs(out_id);
        return out_id;
    }

    std::uint32_t add_leaky_relu(const std::string& out_name, std::uint32_t in_id,
                                 float alpha, std::initializer_list<std::int64_t> out_shape)
    {
        const std::uint32_t node_idx = static_cast<std::uint32_t>(graph_.nodes_size());
        const std::uint32_t out_id = add_activation(out_name, wire::FP16, out_shape, node_idx);
        auto* n = graph_.add_nodes();
        n->set_op(wire::LEAKY_RELU);
        n->add_inputs(in_id);
        n->add_outputs(out_id);
        n->mutable_activation()->set_alpha(alpha);
        return out_id;
    }

    std::uint32_t add_clip(const std::string& out_name, std::uint32_t in_id,
                           std::uint32_t min_id, std::uint32_t max_id,
                           std::initializer_list<std::int64_t> out_shape)
    {
        const std::uint32_t node_idx = static_cast<std::uint32_t>(graph_.nodes_size());
        const std::uint32_t out_id = add_activation(out_name, wire::FP16, out_shape, node_idx);
        auto* n = graph_.add_nodes();
        n->set_op(wire::CLIP);
        n->add_inputs(in_id);
        n->add_inputs(min_id);
        n->add_inputs(max_id);
        n->add_outputs(out_id);
        return out_id;
    }

    std::uint32_t add_binary(wire::OpKind op, const std::string& out_name,
                             std::uint32_t a, std::uint32_t b,
                             std::initializer_list<std::int64_t> out_shape)
    {
        const std::uint32_t node_idx = static_cast<std::uint32_t>(graph_.nodes_size());
        const std::uint32_t out_id = add_activation(out_name, wire::FP16, out_shape, node_idx);
        auto* n = graph_.add_nodes();
        n->set_op(op);
        n->add_inputs(a);
        n->add_inputs(b);
        n->add_outputs(out_id);
        return out_id;
    }

    std::uint32_t add_conv(const std::string& out_name, std::uint32_t x,
                           std::uint32_t w, std::uint32_t b,
                           std::array<std::uint32_t, 2> strides,
                           std::array<std::uint32_t, 4> pads,
                           std::uint32_t groups,
                           std::initializer_list<std::int64_t> out_shape)
    {
        const std::uint32_t node_idx = static_cast<std::uint32_t>(graph_.nodes_size());
        const std::uint32_t out_id = add_activation(out_name, wire::FP16, out_shape, node_idx);
        auto* n = graph_.add_nodes();
        n->set_op(wire::CONV);
        n->add_inputs(x);
        n->add_inputs(w);
        n->add_inputs(b);
        n->add_outputs(out_id);
        auto* ca = n->mutable_conv();
        ca->add_strides(strides[0]);
        ca->add_strides(strides[1]);
        for (const auto p : pads)
            ca->add_pads(p);
        ca->add_dilations(1);
        ca->add_dilations(1);
        ca->set_groups(groups);
        return out_id;
    }

    std::uint32_t add_slice_1d_axis(const std::string& out_name, std::uint32_t x,
                                    std::int64_t start, std::int64_t end, std::int64_t axis,
                                    std::initializer_list<std::int64_t> out_shape)
    {
        const std::string prefix = out_name + "_sl_";
        const auto s_id = add_weight_int64(prefix + "s", {1}, {start});
        const auto e_id = add_weight_int64(prefix + "e", {1}, {end});
        const auto a_id = add_weight_int64(prefix + "a", {1}, {axis});
        const auto st_id = add_weight_int64(prefix + "st", {1}, {1});
        const std::uint32_t node_idx = static_cast<std::uint32_t>(graph_.nodes_size());
        const std::uint32_t out_id = add_activation(out_name, wire::FP16, out_shape, node_idx);
        auto* n = graph_.add_nodes();
        n->set_op(wire::SLICE);
        n->add_inputs(x);
        n->add_inputs(s_id);
        n->add_inputs(e_id);
        n->add_inputs(a_id);
        n->add_inputs(st_id);
        n->add_outputs(out_id);
        return out_id;
    }

    std::uint32_t add_concat(const std::string& out_name,
                             std::initializer_list<std::uint32_t> ins,
                             std::int32_t axis,
                             std::initializer_list<std::int64_t> out_shape)
    {
        const std::uint32_t node_idx = static_cast<std::uint32_t>(graph_.nodes_size());
        const std::uint32_t out_id = add_activation(out_name, wire::FP16, out_shape, node_idx);
        auto* n = graph_.add_nodes();
        n->set_op(wire::CONCAT);
        for (const auto id : ins)
            n->add_inputs(id);
        n->add_outputs(out_id);
        n->mutable_axis()->set_axis(axis);
        return out_id;
    }

    std::uint32_t add_gather(const std::string& out_name, std::uint32_t data,
                             std::uint32_t idx, std::int32_t axis,
                             std::initializer_list<std::int64_t> out_shape)
    {
        const std::uint32_t node_idx = static_cast<std::uint32_t>(graph_.nodes_size());
        const std::uint32_t out_id = add_activation(out_name, wire::FP16, out_shape, node_idx);
        auto* n = graph_.add_nodes();
        n->set_op(wire::GATHER);
        n->add_inputs(data);
        n->add_inputs(idx);
        n->add_outputs(out_id);
        n->mutable_axis()->set_axis(axis);
        return out_id;
    }

    std::uint32_t add_pixel(wire::OpKind op, const std::string& out_name,
                            std::uint32_t in_id, std::uint32_t block_size,
                            std::initializer_list<std::int64_t> out_shape)
    {
        const std::uint32_t node_idx = static_cast<std::uint32_t>(graph_.nodes_size());
        const std::uint32_t out_id = add_activation(out_name, wire::FP16, out_shape, node_idx);
        auto* n = graph_.add_nodes();
        n->set_op(op);
        n->add_inputs(in_id);
        n->add_outputs(out_id);
        n->mutable_pixel()->set_block_size(block_size);
        return out_id;
    }

    void mark_output(std::uint32_t id) { graph_.add_outputs(id); }

    std::shared_ptr<const ModelAssetReader> build()
    {
        const std::uint32_t n_nodes = static_cast<std::uint32_t>(graph_.nodes_size());
        for (int i = 0; i < graph_.tensors_size(); ++i) {
            auto* td = graph_.mutable_tensors(i);
            if (td->has_arena())
                td->mutable_arena()->set_death(n_nodes);
        }
        graph_.set_arena_bytes(std::max<std::uint64_t>(arena_cursor_, 256U));
        if (weights_.empty())
            weights_.resize(256, std::byte{0});
        graph_.set_weights_bytes(weights_.size());

        wire::ModelMetadata meta;
        meta.set_schema_version(3);
        meta.set_profile("test");
        meta.set_model_version("v1");
        meta.set_checkpoint_sha256(sha_);
        meta.set_width(16);
        meta.set_height(16);
        meta.set_pixel_range(255.0F);
        meta.set_qp_num(1);
        meta.set_total_qp_num(1);
        meta.add_frame_index_map(0);
        meta.add_qp_shift(0);
        meta.set_feature_channels(16);
        meta.set_latent_channels(8);
        meta.set_hyperprior_channels(8);
        meta.set_downsample_feature(2);
        meta.set_downsample_hyperprior(4);

        auto reader = std::make_shared<MemoryAssetReader>();
        reader->add("metadata.pb", to_bytes(meta.SerializeAsString()));
        const std::filesystem::path dir = std::filesystem::path("aot") / model_name_;
        reader->add(dir / "graph.pb", to_bytes(graph_.SerializeAsString()));
        reader->add(dir / "weights.bin", weights_);
        return reader;
    }

private:
    std::uint32_t next_tensor(const std::string& name, wire::DataType dt,
                              std::initializer_list<std::int64_t> shape)
    {
        const std::uint32_t id = static_cast<std::uint32_t>(graph_.tensors_size() + 1);
        auto* td = graph_.add_tensors();
        td->set_id(id);
        td->set_name(name);
        td->set_dtype(dt);
        for (const auto d : shape)
            td->add_shape(d);
        return id;
    }

    std::uint32_t add_weight_raw(const std::string& name, wire::DataType dt,
                                 std::initializer_list<std::int64_t> shape,
                                 const void* data, std::size_t bytes)
    {
        const std::uint32_t id = next_tensor(name, dt, shape);
        const std::size_t aligned_off = ((weights_.size() + 255U) / 256U) * 256U;
        weights_.resize(aligned_off + bytes, std::byte{0});
        if (bytes > 0U)
            std::memcpy(weights_.data() + aligned_off, data, bytes);
        auto* td = graph_.mutable_tensors(static_cast<int>(id - 1));
        auto* wl = td->mutable_weight();
        wl->set_offset(aligned_off);
        wl->set_size_bytes(bytes);
        return id;
    }

    std::string model_name_;
    std::string sha_;
    std::string wsha_;
    wire::Graph graph_;
    std::vector<std::byte> weights_;
    std::uint64_t arena_cursor_ = 0;
};

Tensor make_fp16_tensor(std::string name, std::vector<std::int64_t> shape,
                        const std::vector<float>& vals)
{
    Tensor t;
    t.name = std::move(name);
    t.shape = std::move(shape);
    std::vector<Float16Storage> h(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i)
        h[i] = float_to_half(vals[i]);
    t.data = std::move(h);
    return t;
}

Tensor make_int32_tensor(std::string name, std::vector<std::int64_t> shape,
                         std::vector<std::int32_t> vals)
{
    Tensor t;
    t.name = std::move(name);
    t.shape = std::move(shape);
    t.data = std::move(vals);
    return t;
}

std::vector<float> to_floats(const Tensor& t)
{
    const auto& vec = std::get<std::vector<Float16Storage>>(t.data);
    std::vector<float> out(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i)
        out[i] = half_to_float(vec[i]);
    return out;
}

void check_close(const Tensor& got, const std::vector<float>& expected, float tol, const char* msg)
{
    require(got.data_type() == TensorDataType::kFloat16, "expected fp16 output");
    const auto vals = to_floats(got);
    require(vals.size() == expected.size(), msg);
    for (std::size_t i = 0; i < vals.size(); ++i) {
        if (std::fabs(vals[i] - expected[i]) > tol || std::isnan(vals[i])) {
            throw std::runtime_error(std::string(msg) + " at index " + std::to_string(i) +
                                     " got=" + std::to_string(vals[i]) +
                                     " expected=" + std::to_string(expected[i]));
        }
    }
}

void test_backend_lifecycle()
{
    const auto backends = available_backends();
    require(std::find(backends.begin(), backends.end(), "cpu") != backends.end(),
            "cpu backend must be registered");

    BackendOptions strict_gpu;
    strict_gpu.backend_name = "cpu";
    strict_gpu.asset_reader = std::make_shared<MemoryAssetReader>();
    strict_gpu.execution_preference = ExecutionPreference::kGpu;
    strict_gpu.allow_cpu_fallback = false;
    expect_throw([&] { create_backend(strict_gpu); },
                 "strict GPU non-fallback should reject cpu backend");

    BackendOptions opts;
    opts.backend_name = "cpu";
    opts.asset_reader = std::make_shared<MemoryAssetReader>();
    opts.num_threads = 2;
    auto b = create_backend(opts);
    require(b->name() == "cpu", "backend name must be cpu");
    expect_throw([&] { b->run({}); }, "run before load must throw");
    expect_throw([&] { b->reset_state(); }, "reset_state before load must throw");
}

void test_conv_and_epilogues()
{
    // Test 1x1 conv + ReGLU (fused), 1x1 conv + LeakyReLU (fused),
    // 1x1 conv + residual ADD (fused), 3x3 spatial conv, and DW3x3 conv.
    WireGraphBuilder gb("ConvNet");
    const std::int64_t H = 4, W = 5, IC = 16, OC = 32;
    const auto x_id = gb.add_input("x", wire::FP16, {1, IC, H, W});

    // 1x1 conv -> OC=32 -> slice(0..16), slice(16..32), clip(0..1), mul => out_reglu [1, 16, 4, 5]
    std::vector<float> w1(OC * IC, 0.0F);
    std::vector<float> b1(OC, 0.0F);
    for (std::size_t oc = 0; oc < OC; ++oc) {
        b1[oc] = (oc < 16U) ? 0.25F : -0.5F;
        w1[oc * IC + (oc % IC)] = (oc % 2 == 0) ? 0.5F : -0.25F;
    }
    const auto w1_id = gb.add_weight_fp16("w1", {OC, IC, 1, 1}, w1);
    const auto b1_id = gb.add_weight_fp16("b1", {OC}, b1);
    const auto zero_id = gb.add_weight_fp16("zero", {1}, {0.0F});
    const auto one_id = gb.add_weight_fp16("one", {1}, {1.0F});

    const auto c1 = gb.add_conv("c1", x_id, w1_id, b1_id, {1, 1}, {0, 0, 0, 0}, 1, {1, OC, H, W});
    const auto s_gate = gb.add_slice_1d_axis("gate", c1, 0, 16, 1, {1, 16, H, W});
    const auto s_lin = gb.add_slice_1d_axis("lin", c1, 16, 32, 1, {1, 16, H, W});
    const auto clipped = gb.add_clip("clip_g", s_gate, zero_id, one_id, {1, 16, H, W});
    const auto reglu_out = gb.add_binary(wire::MUL, "reglu", clipped, s_lin, {1, 16, H, W});

    // DW 3x3 s1p1 on reglu_out [1, 16, 4, 5]
    std::vector<float> wdw(16 * 9, 0.0F);
    std::vector<float> bdw(16, 0.1F);
    for (int c = 0; c < 16; ++c) {
        wdw[static_cast<std::size_t>(c) * 9U + 4U] = 1.0F;  // center tap
        wdw[static_cast<std::size_t>(c) * 9U + 0U] = 0.25F; // top-left tap
    }
    const auto wdw_id = gb.add_weight_fp16("wdw", {16, 1, 3, 3}, wdw);
    const auto bdw_id = gb.add_weight_fp16("bdw", {16}, bdw);
    const auto dw_out = gb.add_conv("dw", reglu_out, wdw_id, bdw_id, {1, 1}, {1, 1, 1, 1}, 16, {1, 16, H, W});

    // 1x1 conv -> LeakyReLU -> 3x3 stride 2 conv -> residual/bias check
    std::vector<float> w2(16 * 16, 0.0F);
    for (int c = 0; c < 16; ++c)
        w2[static_cast<std::size_t>(c) * 16U + static_cast<std::size_t>(c)] = 1.0F;
    const auto w2_id = gb.add_weight_fp16("w2", {16, 16, 1, 1}, w2);
    const auto c2 = gb.add_conv("c2", dw_out, w2_id, 0, {1, 1}, {0, 0, 0, 0}, 1, {1, 16, H, W});
    const auto lrelu = gb.add_leaky_relu("lrelu", c2, 0.01F, {1, 16, H, W});

    // Residual add after another 1x1 conv (identity + dw_out)
    const auto c3 = gb.add_conv("c3", lrelu, w2_id, 0, {1, 1}, {0, 0, 0, 0}, 1, {1, 16, H, W});
    const auto res_out = gb.add_binary(wire::ADD, "res_out", c3, x_id, {1, 16, H, W});

    // Spatial 3x3 conv (stride=1, pad=1) from 16 -> 16
    std::vector<float> w3(16 * 16 * 9, 0.0F);
    for (int c = 0; c < 16; ++c)
        w3[(static_cast<std::size_t>(c) * 16U + static_cast<std::size_t>(c)) * 9U + 4U] = 0.5F;
    const auto w3_id = gb.add_weight_fp16("w3", {16, 16, 3, 3}, w3);
    const auto spatial_out = gb.add_conv("sp3", res_out, w3_id, 0, {1, 1}, {1, 1, 1, 1}, 1, {1, 16, H, W});

    // Wide tiled 1x1 conv (OC=128 > kColBlock=64) + ResidualAdd check.
    const std::int64_t WOC = 128;
    const auto res_wide_id = gb.add_input("res_wide", wire::FP16, {1, WOC, H, W});
    std::vector<float> w_wide(WOC * 16, 0.0F);
    for (int oc = 0; oc < WOC; ++oc)
        w_wide[static_cast<std::size_t>(oc) * 16U + static_cast<std::size_t>(oc % 16)] = 0.25F;
    const auto w_wide_id = gb.add_weight_fp16("w_wide", {WOC, 16, 1, 1}, w_wide);
    const auto c_wide = gb.add_conv("c_wide", spatial_out, w_wide_id, 0, {1, 1}, {0, 0, 0, 0}, 1, {1, WOC, H, W});
    const auto wide_res_out = gb.add_binary(wire::ADD, "wide_res_out", c_wide, res_wide_id, {1, WOC, H, W});

    gb.mark_output(reglu_out);
    gb.mark_output(spatial_out);
    gb.mark_output(wide_res_out);

    BackendOptions opts;
    opts.backend_name = "cpu";
    opts.asset_reader = gb.build();
    opts.num_threads = 4;
    auto backend = create_backend(opts);
    backend->load("ConvNet", {});

    std::vector<float> x_vals(IC * H * W);
    for (std::size_t i = 0; i < x_vals.size(); ++i)
        x_vals[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25F;
    std::vector<float> rw_vals(WOC * H * W);
    for (std::size_t i = 0; i < rw_vals.size(); ++i)
        rw_vals[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.125F;
    const auto x_tensor = make_fp16_tensor("x", {1, IC, H, W}, x_vals);
    const auto rw_tensor = make_fp16_tensor("res_wide", {1, WOC, H, W}, rw_vals);
    const auto outs = backend->run({x_tensor, rw_tensor});
    require(outs.size() == 3U, "expected 3 outputs");

    // Compute scalar ground truth.
    std::vector<float> exp_reglu(16 * H * W, 0.0F);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < 16; ++c) {
                const float xv = half_to_float(float_to_half(x_vals[(c * H + y) * W + x]));
                const float g_raw = xv * ((c % 2 == 0) ? 0.5F : -0.25F) + 0.25F;
                const float l_raw = xv * (((c + 16) % 2 == 0) ? 0.5F : -0.25F) - 0.5F;
                const float g = std::min(std::max(g_raw, 0.0F), 1.0F);
                exp_reglu[(c * H + y) * W + x] = half_to_float(float_to_half(g * l_raw));
            }
        }
    }
    check_close(outs[0], exp_reglu, 1e-2F, "ReGLU output mismatch");

    std::vector<float> exp_spatial(16 * H * W, 0.0F);
    for (int c = 0; c < 16; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float dw = 0.1F + exp_reglu[(c * H + y) * W + x];
                if (y > 0 && x > 0)
                    dw += 0.25F * exp_reglu[(c * H + (y - 1)) * W + (x - 1)];
                const float lr = dw < 0.0F ? dw * 0.01F : dw;
                const float add = lr + half_to_float(float_to_half(x_vals[(c * H + y) * W + x]));
                exp_spatial[(c * H + y) * W + x] = add * 0.5F;
            }
        }
    }
    check_close(outs[1], exp_spatial, 2e-2F, "Spatial Conv/DW/Residual chain mismatch");

    std::vector<float> exp_wide(WOC * H * W, 0.0F);
    for (int oc = 0; oc < WOC; ++oc) {
        for (int p = 0; p < H * W; ++p) {
            const std::size_t idx = static_cast<std::size_t>(oc) * (H * W) + static_cast<std::size_t>(p);
            const float sp = exp_spatial[static_cast<std::size_t>(oc % 16) * (H * W) + static_cast<std::size_t>(p)];
            const float rv = half_to_float(float_to_half(rw_vals[idx]));
            exp_wide[idx] = sp * 0.25F + rv;
        }
    }
    check_close(outs[2], exp_wide, 2e-2F, "Wide tiled ResidualAdd (OC>64) mismatch");

    // Multi-thread 1 vs 4 determinism.
    BackendOptions opts1 = opts;
    opts1.num_threads = 1;
    auto b1_single = create_backend(opts1);
    b1_single->load("ConvNet", {});
    const auto outs1 = b1_single->run({x_tensor, rw_tensor});
    require(outs[0].data == outs1[0].data && outs[1].data == outs1[1].data &&
            outs[2].data == outs1[2].data,
            "1-thread and 4-thread execution must be bit-identical");
}

void test_feature_update_and_state()
{
    WireGraphBuilder gb("FeatureStateNet");
    const std::int64_t C = 8, H = 3, W = 4;
    const auto chunk_id = gb.add_input("chunk", wire::FP16, {1, C * 3, H, W});
    const auto hist_id = gb.add_input("hist_in", wire::FP16, {1, C, H, W});
    const auto one_id = gb.add_weight_fp16("one", {1}, {1.0F});
    const auto scale_id = gb.add_weight_fp16("scale", {1, C, 1, 1}, std::vector<float>(C, 2.0F));

    const auto a = gb.add_slice_1d_axis("a", chunk_id, 0, C, 1, {1, C, H, W});
    const auto b = gb.add_slice_1d_axis("b", chunk_id, C, C * 2, 1, {1, C, H, W});
    const auto cg = gb.add_slice_1d_axis("cg", chunk_id, C * 2, C * 3, 1, {1, C, H, W});
    const auto sig_b = gb.add_unary(wire::SIGMOID, "sig_b", b, {1, C, H, W});
    const auto sig_c = gb.add_unary(wire::SIGMOID, "sig_c", cg, {1, C, H, W});
    const auto wb = gb.add_binary(wire::MUL, "wb", sig_b, hist_id, {1, C, H, W});
    const auto inv = gb.add_binary(wire::SUB, "inv", one_id, sig_b, {1, C, H, W});
    const auto wa = gb.add_binary(wire::MUL, "wa", inv, a, {1, C, H, W});
    const auto blended = gb.add_binary(wire::ADD, "blended", wb, wa, {1, C, H, W});
    const auto gated = gb.add_binary(wire::MUL, "gated", sig_c, blended, {1, C, H, W});
    const auto scaled = gb.add_binary(wire::MUL, "scaled", gated, scale_id, {1, C, H, W});
    // Deferred concat after an identity Mul using scaled (triggers FeatureUpdateOutputs + deferred CONCAT)
    const auto used_scaled = gb.add_binary(wire::MUL, "used", scaled, one_id, {1, C, H, W});
    const auto feat_out = gb.add_concat("feat_out", {scaled, blended}, 1, {1, C * 2, H, W});
    const auto next_hist = gb.add_slice_1d_axis("next_h", feat_out, C, C * 2, 1, {1, C, H, W});

    gb.mark_output(used_scaled);
    gb.mark_output(next_hist);

    ModelExecutionConfig cfg;
    cfg.state_bindings.push_back({1U, 1U, TensorDataType::kFloat16, {1, C, H, W}});

    BackendOptions opts;
    opts.backend_name = "cpu";
    opts.asset_reader = gb.build();
    auto backend = create_backend(opts);
    backend->load("FeatureStateNet", cfg);

    std::vector<float> in_vals(C * 3 * H * W);
    for (std::size_t i = 0; i < in_vals.size(); ++i)
        in_vals[i] = (i % 7 == 0) ? 1.0F : 0.5F;
    const auto chunk_tensor = make_fp16_tensor("chunk", {1, C * 3, H, W}, in_vals);

    expect_throw([&] { backend->run({chunk_tensor}); },
                 "run before reset_state must throw when state bindings exist");
    backend->reset_state();

    const auto f1 = backend->run({chunk_tensor});
    const auto f2 = backend->run({chunk_tensor});
    require(f1.size() == 1U && f2.size() == 1U, "state output should be excluded");
    require(f1[0].data != f2[0].data, "frame 2 should differ from frame 1 due to persisted history");

    backend->reset_state();
    const auto f3 = backend->run({chunk_tensor});
    require(f1[0].data == f3[0].data, "reset_state must restore exact initial zero state");
}

void test_pixel_shuffle_and_quant_pipeline()
{
    WireGraphBuilder gb("PixelQuantNet");
    const std::int64_t Q = 8, H = 2, W = 4;
    const auto lat_id = gb.add_input("lat", wire::FP16, {1, 2 * Q, H, W});
    const auto pri0_id = gb.add_input("pri0", wire::FP16, {1, 4 * Q, H, W});
    const auto pri1_id = gb.add_input("pri1", wire::FP16, {1, 2 * Q, H, W});
    const auto qidx_id = gb.add_input("qidx", wire::INT32, {1});

    std::vector<float> lut_vals(4 * 2 * Q, 0.0F);
    for (int i = 0; i < 4; ++i)
        for (int c = 0; c < 2 * Q; ++c)
            lut_vals[static_cast<std::size_t>(i * 2 * Q + c)] = static_cast<float>(i + 1) * 0.5F;
    const auto lut_id = gb.add_weight_fp16("lut", {4, 2 * Q, 1, 1}, lut_vals);
    const auto gathered = gb.add_gather("g_scale", lut_id, qidx_id, 0, {1, 2 * Q, 1, 1});

    const auto qs_id = gb.add_weight_fp16("qs", {1, 2 * Q, H, W}, std::vector<float>(2 * Q * H * W, 1.0F));
    const auto us0_id = gb.add_weight_fp16("us0", {1, Q, H, W}, std::vector<float>(Q * H * W, 0.5F));
    const auto us1_id = gb.add_weight_fp16("us1", {1, Q, H, W}, std::vector<float>(Q * H * W, 0.5F));
    const auto cmin_id = gb.add_weight_fp16("cmin", {1}, {0.5F});
    const auto cmax_id = gb.add_weight_fp16("cmax", {1}, {65504.0F});
    const auto one_id = gb.add_weight_fp16("one", {1}, {1.0F});

    const auto p_head = gb.add_slice_1d_axis("ph", pri0_id, 0, 2 * Q, 1, {1, 2 * Q, H, W});
    const auto p_tail = gb.add_slice_1d_axis("pt", pri0_id, 2 * Q, 4 * Q, 1, {1, 2 * Q, H, W});
    const auto clipped = gb.add_clip("clp", p_head, cmin_id, cmax_id, {1, 2 * Q, H, W});
    const auto rec = gb.add_unary(wire::RECIPROCAL, "rec", clipped, {1, 2 * Q, H, W});
    const auto inv = gb.add_binary(wire::MUL, "inv", rec, one_id, {1, 2 * Q, H, W});
    const auto norm = gb.add_binary(wire::MUL, "norm", lat_id, inv, {1, 2 * Q, H, W});
    const auto diff = gb.add_binary(wire::SUB, "diff", norm, p_tail, {1, 2 * Q, H, W});
    const auto scaled = gb.add_binary(wire::MUL, "sc", diff, qs_id, {1, 2 * Q, H, W});
    const auto s0 = gb.add_slice_1d_axis("s0", scaled, 0, Q, 1, {1, Q, H, W});
    const auto s1 = gb.add_slice_1d_axis("s1", scaled, Q, 2 * Q, 1, {1, Q, H, W});
    const auto sum_s = gb.add_binary(wire::ADD, "sums", s0, s1, {1, Q, H, W});
    const auto y0_raw = gb.add_unary(wire::ROUND, "y0raw", sum_s, {1, Q, H, W});
    const auto pt0 = gb.add_slice_1d_axis("pt0", p_tail, 0, Q, 1, {1, Q, H, W});
    const auto pt1 = gb.add_slice_1d_axis("pt1", p_tail, Q, 2 * Q, 1, {1, Q, H, W});
    const auto a0 = gb.add_binary(wire::ADD, "a0", y0_raw, pt0, {1, Q, H, W});
    const auto u0 = gb.add_binary(wire::MUL, "u0", a0, us0_id, {1, Q, H, W});
    const auto a1 = gb.add_binary(wire::ADD, "a1", y0_raw, pt1, {1, Q, H, W});
    const auto u1 = gb.add_binary(wire::MUL, "u1", a1, us1_id, {1, Q, H, W});
    const auto c2 = gb.add_concat("c2", {u0, u1}, 1, {1, 2 * Q, H, W});
    const auto c3 = gb.add_concat("c3", {c2, pri0_id}, 1, {1, 6 * Q, H, W});

    // Y1Tail chain
    const auto y1_diff = gb.add_binary(wire::SUB, "y1d", norm, pri1_id, {1, 2 * Q, H, W});
    const auto y1_sc = gb.add_binary(wire::MUL, "y1sc", y1_diff, qs_id, {1, 2 * Q, H, W});
    const auto y1_s0 = gb.add_slice_1d_axis("y1s0", y1_sc, 0, Q, 1, {1, Q, H, W});
    const auto y1_s1 = gb.add_slice_1d_axis("y1s1", y1_sc, Q, 2 * Q, 1, {1, Q, H, W});
    const auto y1_sum = gb.add_binary(wire::ADD, "y1sum", y1_s0, y1_s1, {1, Q, H, W});
    const auto y1_raw = gb.add_unary(wire::ROUND, "y1raw", y1_sum, {1, Q, H, W});
    const auto y1_p0 = gb.add_slice_1d_axis("y1p0", pri1_id, 0, Q, 1, {1, Q, H, W});
    const auto y1_p1 = gb.add_slice_1d_axis("y1p1", pri1_id, Q, 2 * Q, 1, {1, Q, H, W});
    const auto y1_a0 = gb.add_binary(wire::ADD, "y1a0", y1_raw, y1_p0, {1, Q, H, W});
    const auto y1_u0 = gb.add_binary(wire::MUL, "y1u0", y1_a0, us0_id, {1, Q, H, W});
    const auto y1_a1 = gb.add_binary(wire::ADD, "y1a1", y1_raw, y1_p1, {1, Q, H, W});
    const auto y1_u1 = gb.add_binary(wire::MUL, "y1u1", y1_a1, us1_id, {1, Q, H, W});
    const auto y1_c2 = gb.add_concat("y1c2", {y1_u0, y1_u1}, 1, {1, 2 * Q, H, W});
    const auto y1_add = gb.add_binary(wire::ADD, "y1add", c2, y1_c2, {1, 2 * Q, H, W});
    const auto y1_out = gb.add_binary(wire::MUL, "y1out", y1_add, clipped, {1, 2 * Q, H, W});

    // SpaceToDepth -> DepthToSpace + H-concat + Gather broadcast
    const auto s2d = gb.add_pixel(wire::SPACE_TO_DEPTH, "s2d", lat_id, 2, {1, 8 * Q, 1, 2});
    const auto d2s = gb.add_pixel(wire::DEPTH_TO_SPACE, "d2s", s2d, 2, {1, 2 * Q, H, W});
    const auto h0 = gb.add_slice_1d_axis("h0", c3, 0, 1, 2, {1, 6 * Q, 1, W});
    const auto h1 = gb.add_slice_1d_axis("h1", c3, 1, 2, 2, {1, 6 * Q, 1, W});
    const auto hcat = gb.add_concat("hcat", {h0, h1}, 2, {1, 6 * Q, 2, W});
    const auto gnorm = gb.add_binary(wire::MUL, "gnorm", d2s, gathered, {1, 2 * Q, H, W});

    gb.mark_output(y0_raw);
    gb.mark_output(y1_raw);
    gb.mark_output(y1_out);
    gb.mark_output(hcat);
    gb.mark_output(gnorm);

    BackendOptions opts;
    opts.backend_name = "cpu";
    opts.asset_reader = gb.build();
    auto backend = create_backend(opts);
    backend->load("PixelQuantNet", {});

    std::vector<float> lat_v(2 * Q * H * W, 3.0F);
    std::vector<float> pri0_v(4 * Q * H * W, 0.25F);
    for (std::size_t i = 0; i < 2 * Q * H * W; ++i)
        pri0_v[i] = 0.5F;  // clipped head = 0.5 -> inv = 2.0 -> norm = 6.0; tail = 0.25
    std::vector<float> pri1_v(2 * Q * H * W, 0.25F);

    const auto outs = backend->run({
        make_fp16_tensor("lat", {1, 2 * Q, H, W}, lat_v),
        make_fp16_tensor("pri0", {1, 4 * Q, H, W}, pri0_v),
        make_fp16_tensor("pri1", {1, 2 * Q, H, W}, pri1_v),
        make_int32_tensor("qidx", {1}, {2}),
    });
    require(outs.size() == 5U, "expected 5 outputs");

    // norm = 3.0 / 0.5 = 6.0; diff = 6.0 - 0.25 = 5.75; s0 + s1 = 11.5 -> nearbyintf(11.5) = 12.0!
    check_close(outs[0], std::vector<float>(Q * H * W, 12.0F), 1e-3F, "y0_raw mismatch");
    // y1: diff = 6.0 - 0.25 = 5.75 -> 12.0
    check_close(outs[1], std::vector<float>(Q * H * W, 12.0F), 1e-3F, "y1_raw mismatch");
    // upd = (12.0 + 0.25)*0.5 = 6.125 -> (6.125 + 6.125) * 0.5 = 6.125
    check_close(outs[2], std::vector<float>(2 * Q * H * W, 6.125F), 1e-2F, "y1_out mismatch");
    std::vector<float> expected_hcat(6 * Q * H * W, 0.25F);
    for (std::size_t i = 0; i < 2 * Q * H * W; ++i)
        expected_hcat[i] = 6.125F;
    for (std::size_t i = 2 * Q * H * W; i < 4 * Q * H * W; ++i)
        expected_hcat[i] = 0.5F;
    check_close(outs[3], expected_hcat, 1e-3F, "H-concat/slice mismatch");
    // qidx=2 -> lut scale = 1.5; d2s == lat (3.0) -> gnorm = 4.5
    check_close(outs[4], std::vector<float>(2 * Q * H * W, 4.5F), 1e-3F, "SpaceToDepth/DepthToSpace/Gather mismatch");

#if defined(_WIN32)
    _putenv_s("MLVC_CPU_DISABLE_FUSION", "1");
#else
    setenv("MLVC_CPU_DISABLE_FUSION", "1", 1);
#endif
    auto unfused_backend = create_backend(opts);
    unfused_backend->load("PixelQuantNet", {});
    const auto unfused_outs = unfused_backend->run({
        make_fp16_tensor("lat", {1, 2 * Q, H, W}, lat_v),
        make_fp16_tensor("pri0", {1, 4 * Q, H, W}, pri0_v),
        make_fp16_tensor("pri1", {1, 2 * Q, H, W}, pri1_v),
        make_int32_tensor("qidx", {1}, {2}),
    });
#if defined(_WIN32)
    _putenv_s("MLVC_CPU_DISABLE_FUSION", "");
#else
    unsetenv("MLVC_CPU_DISABLE_FUSION");
#endif
    require(outs.size() == unfused_outs.size(), "fused vs unfused output count mismatch");
    for (std::size_t k = 0; k < outs.size(); ++k)
        check_close(outs[k], to_floats(unfused_outs[k]), 1e-3F, "fused vs unfused mismatch");
}

void test_small_channel_layout_roundtrip()
{
    // Covers C < 8 (such as C = 3 RGB reconstruction) and non-multiple-of-8
    // C / HW layout conversions across SIMD/scalar transpose tails.
    WireGraphBuilder gb("SmallChannelLayoutNet");
    const std::int64_t c3 = 3, h3 = 9, w3 = 11;
    const auto x3_id = gb.add_input("x3", wire::FP16, {1, c3, h3, w3});
    std::vector<float> w_id3(c3 * c3, 0.0F);
    std::vector<float> b_id3(c3, 0.0F);
    for (std::int64_t c = 0; c < c3; ++c) {
        w_id3[static_cast<std::size_t>(c) * c3 + static_cast<std::size_t>(c)] = 1.0F;
        b_id3[static_cast<std::size_t>(c)] = 0.5F + 0.25F * static_cast<float>(c);
    }
    const auto w3_id = gb.add_weight_fp16("w3", {c3, c3, 1, 1}, w_id3);
    const auto b3_id = gb.add_weight_fp16("b3", {c3}, b_id3);
    const auto out3 = gb.add_conv("conv3", x3_id, w3_id, b3_id, {1, 1}, {0, 0, 0, 0}, 1, {1, c3, h3, w3});

    const std::int64_t c11 = 11, h11 = 5, w11 = 7;
    const auto x11_id = gb.add_input("x11", wire::FP16, {1, c11, h11, w11});
    std::vector<float> w_id11(c11 * c11, 0.0F);
    std::vector<float> b_id11(c11, 0.0F);
    for (std::int64_t c = 0; c < c11; ++c) {
        w_id11[static_cast<std::size_t>(c) * c11 + static_cast<std::size_t>(c)] = 1.0F;
        b_id11[static_cast<std::size_t>(c)] = -0.25F * static_cast<float>(c);
    }
    const auto w11_id = gb.add_weight_fp16("w11", {c11, c11, 1, 1}, w_id11);
    const auto b11_id = gb.add_weight_fp16("b11", {c11}, b_id11);
    const auto out11 = gb.add_conv("conv11", x11_id, w11_id, b11_id, {1, 1}, {0, 0, 0, 0}, 1, {1, c11, h11, w11});

    gb.mark_output(out3);
    gb.mark_output(out11);

    BackendOptions opts;
    opts.backend_name = "cpu";
    opts.asset_reader = gb.build();
    opts.num_threads = 4;
    auto backend = create_backend(opts);
    backend->load("SmallChannelLayoutNet", {});

    const std::size_t hw3 = static_cast<std::size_t>(h3 * w3);
    std::vector<float> x3_vals(static_cast<std::size_t>(c3) * hw3);
    std::vector<float> exp3_vals(static_cast<std::size_t>(c3) * hw3);
    for (std::size_t c = 0; c < static_cast<std::size_t>(c3); ++c) {
        for (std::size_t s = 0; s < hw3; ++s) {
            const float v = 0.01F * static_cast<float>((c * 31U + s) % 40U) - 0.1F;
            x3_vals[c * hw3 + s] = v;
            exp3_vals[c * hw3 + s] = v + b_id3[c];
        }
    }

    const std::size_t hw11 = static_cast<std::size_t>(h11 * w11);
    std::vector<float> x11_vals(static_cast<std::size_t>(c11) * hw11);
    std::vector<float> exp11_vals(static_cast<std::size_t>(c11) * hw11);
    for (std::size_t c = 0; c < static_cast<std::size_t>(c11); ++c) {
        for (std::size_t s = 0; s < hw11; ++s) {
            const float v = 0.02F * static_cast<float>((c * 17U + s) % 25U) - 0.2F;
            x11_vals[c * hw11 + s] = v;
            exp11_vals[c * hw11 + s] = v + b_id11[c];
        }
    }

    const auto outs = backend->run({
        make_fp16_tensor("x3", {1, c3, h3, w3}, x3_vals),
        make_fp16_tensor("x11", {1, c11, h11, w11}, x11_vals),
    });
    require(outs.size() == 2U, "expected 2 layout roundtrip outputs");
    check_close(outs[0], exp3_vals, 5e-3F, "C=3 layout roundtrip mismatch");
    check_close(outs[1], exp11_vals, 5e-3F, "C=11 layout roundtrip mismatch");
}

struct ScopedCpuIsaEnv {
    std::string prev;
    bool had_prev = false;
    explicit ScopedCpuIsaEnv(const char* mode)
    {
        if (const char* cur = std::getenv("MLVC_CPU_ISA")) {
            had_prev = true;
            prev = cur;
        }
        set(mode);
    }
    ~ScopedCpuIsaEnv()
    {
        if (had_prev)
            set(prev.c_str());
        else
            clear();
    }
    static void set(const char* mode)
    {
#if defined(_WIN32)
        _putenv_s("MLVC_CPU_ISA", mode ? mode : "");
#else
        if (mode && mode[0] != '\0')
            setenv("MLVC_CPU_ISA", mode, 1);
        else
            unsetenv("MLVC_CPU_ISA");
#endif
    }
    static void clear()
    {
#if defined(_WIN32)
        _putenv_s("MLVC_CPU_ISA", "");
#else
        unsetenv("MLVC_CPU_ISA");
#endif
    }
};

void test_gemm_tail_shapes()
{
    // Exercises odd M/N/K tail blocks: M=7 (non-4, non-6), K=19 (8+8+3),
    // OC=28 (16+12) + ReGLU gate=14 (non-8, non-4), Clamp, LeakyReLU, ResidualAdd,
    // and C=3 spatial/1x1 output, comparing AUTO and FP32 against scalar truth.
    WireGraphBuilder gb("TailShapesNet");
    const std::int64_t H = 1, W = 7, IC = 19, OC_HALF = 14, OC_REG = 28;
    const auto x_id = gb.add_input("x", wire::FP16, {1, IC, H, W});
    const auto res_id = gb.add_input("res", wire::FP16, {1, OC_REG, H, W});
    const auto zero_id = gb.add_weight_fp16("zero", {1}, {-0.2F});
    const auto one_id = gb.add_weight_fp16("one", {1}, {0.8F});

    std::vector<float> w_reg(OC_REG * IC, 0.0F), b_reg(OC_REG, 0.0F);
    for (std::size_t oc = 0; oc < static_cast<std::size_t>(OC_REG); ++oc) {
        b_reg[oc] = (static_cast<float>(oc % 7U) - 3.0F) * 0.05F;
        for (std::size_t ic = 0; ic < static_cast<std::size_t>(IC); ++ic)
            w_reg[oc * IC + ic] = ((oc + ic) % 5 == 0) ? 0.12F : -0.06F;
    }
    const auto w_reg_id = gb.add_weight_fp16("w_reg", {OC_REG, IC, 1, 1}, w_reg);
    const auto b_reg_id = gb.add_weight_fp16("b_reg", {OC_REG}, b_reg);
    const auto c_reg = gb.add_conv("c_reg", x_id, w_reg_id, b_reg_id, {1, 1}, {0, 0, 0, 0}, 1, {1, OC_REG, H, W});
    const auto s_gate = gb.add_slice_1d_axis("g", c_reg, 0, OC_HALF, 1, {1, OC_HALF, H, W});
    const auto s_lin = gb.add_slice_1d_axis("l", c_reg, OC_HALF, OC_REG, 1, {1, OC_HALF, H, W});
    const auto clp_g = gb.add_clip("cg", s_gate, zero_id, one_id, {1, OC_HALF, H, W});
    const auto out_reglu = gb.add_binary(wire::MUL, "reglu", clp_g, s_lin, {1, OC_HALF, H, W});

    std::vector<float> w_lr(OC_REG * OC_HALF, 0.0F), b_lr(OC_REG, 0.0F);
    for (std::size_t oc = 0; oc < static_cast<std::size_t>(OC_REG); ++oc) {
        b_lr[oc] = (oc % 2U == 0U) ? 0.1F : -0.15F;
        w_lr[oc * OC_HALF + (oc % static_cast<std::size_t>(OC_HALF))] = (oc % 3U == 0U) ? 0.5F : -0.4F;
    }
    const auto w_lr_id = gb.add_weight_fp16("w_lr", {OC_REG, OC_HALF, 1, 1}, w_lr);
    const auto b_lr_id = gb.add_weight_fp16("b_lr", {OC_REG}, b_lr);
    const auto c_lr = gb.add_conv("c_lr", out_reglu, w_lr_id, b_lr_id, {1, 1}, {0, 0, 0, 0}, 1, {1, OC_REG, H, W});
    const auto out_lr = gb.add_leaky_relu("lr", c_lr, 0.2F, {1, OC_REG, H, W});

    std::vector<float> w_res(OC_REG * OC_REG, 0.0F);
    for (std::size_t oc = 0; oc < static_cast<std::size_t>(OC_REG); ++oc)
        w_res[oc * OC_REG + oc] = 0.75F;
    const auto w_res_id = gb.add_weight_fp16("w_res", {OC_REG, OC_REG, 1, 1}, w_res);
    const auto c_res = gb.add_conv("c_res", out_lr, w_res_id, 0, {1, 1}, {0, 0, 0, 0}, 1, {1, OC_REG, H, W});
    const auto out_res = gb.add_binary(wire::ADD, "res_add", c_res, res_id, {1, OC_REG, H, W});

    std::vector<float> w_clp(3 * OC_REG, 0.0F);
    for (std::size_t oc = 0; oc < 3U; ++oc)
        w_clp[oc * OC_REG + oc] = 1.0F;
    const auto w_clp_id = gb.add_weight_fp16("w_clp", {3, OC_REG, 1, 1}, w_clp);
    const auto c_clp = gb.add_conv("c_clp", out_res, w_clp_id, 0, {1, 1}, {0, 0, 0, 0}, 1, {1, 3, H, W});
    const auto out_clp = gb.add_clip("out_clp", c_clp, zero_id, one_id, {1, 3, H, W});

    gb.mark_output(out_reglu);
    gb.mark_output(out_lr);
    gb.mark_output(out_res);
    gb.mark_output(out_clp);

    BackendOptions opts;
    opts.backend_name = "cpu";
    opts.asset_reader = gb.build();
    opts.num_threads = 2;

    std::vector<float> x_v(IC * H * W);
    for (std::size_t i = 0; i < x_v.size(); ++i)
        x_v[i] = static_cast<float>(static_cast<int>(i % 17U) - 8) * 0.125F;
    std::vector<float> r_v(OC_REG * H * W);
    for (std::size_t i = 0; i < r_v.size(); ++i)
        r_v[i] = static_cast<float>(static_cast<int>(i % 11U) - 5) * 0.0625F;
    const Tensor xt = make_fp16_tensor("x", {1, IC, H, W}, x_v);
    const Tensor rt = make_fp16_tensor("res", {1, OC_REG, H, W}, r_v);

    std::vector<Tensor> ref_outs;
    {
        ScopedCpuIsaEnv guard("scalar");
        auto sb = create_backend(opts);
        sb->load("TailShapesNet", {});
        ref_outs = sb->run({xt, rt});
    }

    const char* isa_modes[] = {
        "auto",
#if defined(__aarch64__)
        "fp32",
#endif
    };
    for (const char* mode : isa_modes) {
        ScopedCpuIsaEnv guard(mode);
        auto b = create_backend(opts);
        b->load("TailShapesNet", {});
        const auto outs = b->run({xt, rt});
        require(outs.size() == ref_outs.size(), "tail shape output count mismatch");
        for (std::size_t idx = 0; idx < ref_outs.size(); ++idx)
            check_close(outs[idx], to_floats(ref_outs[idx]), 5e-3F, "tail shape ISA mismatch");
    }
}

}  // namespace

int main()
{
    try {
        test_backend_lifecycle();
        test_conv_and_epilogues();
        test_feature_update_and_state();
        test_pixel_shuffle_and_quant_pipeline();
        test_small_channel_layout_roundtrip();
        test_gemm_tail_shapes();
#if defined(__aarch64__)
        {
            ScopedCpuIsaEnv fp32_guard("fp32");
            test_conv_and_epilogues();
            test_small_channel_layout_roundtrip();
        }
#endif
        {
            ScopedCpuIsaEnv scalar_guard("scalar");
            test_conv_and_epilogues();
            test_small_channel_layout_roundtrip();
        }
        std::cout << "cpu_backend_tests passed\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "cpu_backend_tests failed: " << err.what() << '\n';
        return 1;
    }
}
