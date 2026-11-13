#include "execution_plan.hpp"

#include "mlvc/core/half.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mlvc::cpu_backend {
namespace {

std::size_t checked_elems(const std::vector<std::int64_t>& shape)
{
    std::size_t n = 1;
    for (const std::int64_t d : shape) {
        if (d <= 0)
            throw std::runtime_error("cpu: tensor dimensions must be positive");
        const auto ext = static_cast<std::size_t>(d);
        if (n > std::numeric_limits<std::size_t>::max() / ext)
            throw std::runtime_error("cpu: tensor element count overflow");
        n *= ext;
    }
    return n;
}

std::size_t dtype_size(aot::DataType dt)
{
    switch (dt) {
    case aot::DataType::FP16:
        return sizeof(Float16Storage);
    case aot::DataType::INT32:
        return sizeof(std::int32_t);
    case aot::DataType::INT64:
        return sizeof(std::int64_t);
    default:
        throw std::runtime_error("cpu: unsupported AOT tensor data type");
    }
}

TensorDataType to_public_dtype(aot::DataType dt)
{
    switch (dt) {
    case aot::DataType::FP16:
        return TensorDataType::kFloat16;
    case aot::DataType::INT32:
        return TensorDataType::kInt32;
    default:
        throw std::runtime_error("cpu: graph I/O only supports fp16 and int32");
    }
}

std::array<int, 4> dims_to_4d(const std::vector<std::int64_t>& shape)
{
    if (shape.size() != 4U)
        throw std::runtime_error("cpu: expected 4D NCHW tensor");
    std::array<int, 4> dims{};
    for (std::size_t i = 0; i < 4U; ++i) {
        if (shape[i] <= 0 || shape[i] > std::numeric_limits<int>::max())
            throw std::runtime_error("cpu: tensor dimension out of int range");
        dims[i] = static_cast<int>(shape[i]);
    }
    return dims;
}

bool needs_nhwc_transpose(const std::vector<std::int64_t>& shape, aot::DataType dt) noexcept
{
    return dt == aot::DataType::FP16 && shape.size() == 4U &&
           shape[1] > 1 && (shape[2] * shape[3]) > 1;
}

std::size_t align_up_64(std::size_t v) noexcept
{
    return (v + 63U) & ~std::size_t{63U};
}

bool fusion_enabled_from_env() noexcept
{
    const char* env = std::getenv("MLVC_CPU_DISABLE_FUSION");
    return !(env != nullptr && env[0] == '1' && env[1] == '\0');
}

void add_unique_id(std::vector<aot::TensorId>& vec, aot::TensorId id)
{
    if (id != 0 && std::find(vec.begin(), vec.end(), id) == vec.end())
        vec.push_back(id);
}

void execute_gather_general(
    const Float16Storage* data, const std::vector<std::int64_t>& data_shape,
    const std::int32_t* idx32, const std::int64_t* idx64,
    std::size_t idx_count, int axis, Float16Storage* output)
{
    const int rank = static_cast<int>(data_shape.size());
    const int norm_axis = axis < 0 ? axis + rank : axis;
    if (norm_axis < 0 || norm_axis >= rank)
        throw std::runtime_error("cpu: GATHER axis out of range");

    // Fast path for [72, C, 1, 1] indexed at axis 0 (NCHW == NHWC when H=W=1).
    if (norm_axis == 0 &&
        (data_shape.size() != 4U || (data_shape[2] == 1 && data_shape[3] == 1))) {
        const std::size_t outer_dim = static_cast<std::size_t>(data_shape[0]);
        std::size_t inner = 1;
        for (std::size_t d = 1; d < data_shape.size(); ++d)
            inner *= static_cast<std::size_t>(data_shape[d]);
        const std::size_t row_bytes = inner * sizeof(Float16Storage);
        for (std::size_t i = 0; i < idx_count; ++i) {
            std::int64_t raw = idx32 ? static_cast<std::int64_t>(idx32[i]) : idx64[i];
            if (raw < 0)
                raw += static_cast<std::int64_t>(outer_dim);
            if (raw < 0 || static_cast<std::size_t>(raw) >= outer_dim)
                throw std::runtime_error("cpu: GATHER index out of range");
            std::memcpy(output + i * inner,
                        data + static_cast<std::size_t>(raw) * inner, row_bytes);
        }
        return;
    }

    // General path (handles 4D NHWC buffers via NCHW->NHWC axis mapping).
    auto to_phys_axis = [&](int logical_axis) -> int {
        if (data_shape.size() != 4U)
            return logical_axis;
        static constexpr int kMap[4] = {0, 3, 1, 2};
        return kMap[logical_axis];
    };
    std::array<std::int64_t, 4> phys_shape{1, 1, 1, 1};
    const int phys_rank = rank;
    if (rank == 4) {
        phys_shape = {data_shape[0], data_shape[2], data_shape[3], data_shape[1]};
    } else {
        for (int d = 0; d < rank; ++d)
            phys_shape[static_cast<std::size_t>(d)] = data_shape[static_cast<std::size_t>(d)];
    }
    const int p_ax = to_phys_axis(norm_axis);
    std::size_t outer = 1;
    for (int d = 0; d < p_ax; ++d)
        outer *= static_cast<std::size_t>(phys_shape[static_cast<std::size_t>(d)]);
    const std::size_t axis_dim = static_cast<std::size_t>(phys_shape[static_cast<std::size_t>(p_ax)]);
    std::size_t inner = 1;
    for (int d = p_ax + 1; d < phys_rank; ++d)
        inner *= static_cast<std::size_t>(phys_shape[static_cast<std::size_t>(d)]);

    const std::size_t row_bytes = inner * sizeof(Float16Storage);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < idx_count; ++i) {
            std::int64_t raw = idx32 ? static_cast<std::int64_t>(idx32[i]) : idx64[i];
            if (raw < 0)
                raw += static_cast<std::int64_t>(axis_dim);
            if (raw < 0 || static_cast<std::size_t>(raw) >= axis_dim)
                throw std::runtime_error("cpu: GATHER index out of range");
            const std::size_t src_off = (o * axis_dim + static_cast<std::size_t>(raw)) * inner;
            const std::size_t dst_off = (o * idx_count + i) * inner;
            std::memcpy(output + dst_off, data + src_off, row_bytes);
        }
    }
}

void execute_binary_broadcast_fallback_nhwc(
    const Float16Storage* a, const std::vector<std::int64_t>& a_shape,
    const Float16Storage* b, const std::vector<std::int64_t>& b_shape,
    Float16Storage* out, const std::vector<std::int64_t>& out_shape,
    BinaryKind op, CpuThreadPool& pool)
{
    if (out_shape.size() != 4U) {
        const std::size_t total = checked_elems(out_shape);
        const std::size_t na = checked_elems(a_shape);
        const std::size_t nb = checked_elems(b_shape);
        for (std::size_t i = 0; i < total; ++i) {
            const float va = half_to_float(a[na == 1U ? 0U : (i % na)]);
            const float vb = half_to_float(b[nb == 1U ? 0U : (i % nb)]);
            float r = va + vb;
            if (op == BinaryKind::kSub) r = va - vb;
            else if (op == BinaryKind::kMul) r = va * vb;
            out[i] = float_to_half(r);
        }
        return;
    }

    auto bcast_dim = [](const std::vector<std::int64_t>& s) -> std::array<std::int64_t, 4> {
        std::array<std::int64_t, 4> r{1, 1, 1, 1};
        const std::size_t off = 4U - s.size();
        for (std::size_t i = 0; i < s.size(); ++i)
            r[off + i] = s[i];
        return r;
    };
    const auto ad = bcast_dim(a_shape);
    const auto bd = bcast_dim(b_shape);
    const auto od = bcast_dim(out_shape);
    const std::size_t n = static_cast<std::size_t>(od[0]);
    const std::size_t c = static_cast<std::size_t>(od[1]);
    const std::size_t h = static_cast<std::size_t>(od[2]);
    const std::size_t w = static_cast<std::size_t>(od[3]);

    pool.parallel_for(n * h, 4U, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t row = r0; row < r1; ++row) {
            const std::size_t bi = row / h;
            const std::size_t yi = row % h;
            const std::size_t abi = (ad[0] == 1 ? 0 : bi);
            const std::size_t ayi = (ad[2] == 1 ? 0 : yi);
            const std::size_t bbi = (bd[0] == 1 ? 0 : bi);
            const std::size_t byi = (bd[2] == 1 ? 0 : yi);
            for (std::size_t xi = 0; xi < w; ++xi) {
                const std::size_t axi = (ad[3] == 1 ? 0 : xi);
                const std::size_t bxi = (bd[3] == 1 ? 0 : xi);
                const Float16Storage* ap =
                    a + (((abi * static_cast<std::size_t>(ad[2]) + ayi) *
                          static_cast<std::size_t>(ad[3]) + axi) *
                         static_cast<std::size_t>(ad[1]));
                const Float16Storage* bp =
                    b + (((bbi * static_cast<std::size_t>(bd[2]) + byi) *
                          static_cast<std::size_t>(bd[3]) + bxi) *
                         static_cast<std::size_t>(bd[1]));
                Float16Storage* dst_px = out + (row * w + xi) * c;
                for (std::size_t ci = 0; ci < c; ++ci) {
                    const float va = half_to_float(ap[ad[1] == 1 ? 0 : ci]);
                    const float vb = half_to_float(bp[bd[1] == 1 ? 0 : ci]);
                    float res = va + vb;
                    if (op == BinaryKind::kSub) res = va - vb;
                    else if (op == BinaryKind::kMul) res = va * vb;
                    dst_px[ci] = float_to_half(res);
                }
            }
        }
    });
}

}  // namespace

CpuExecutionPlan::CpuExecutionPlan(const ModelAssetReader& assets,
                                   const std::string& model_name,
                                   const ModelExecutionConfig& config,
                                   int num_threads)
    : kernels_(&select_kernels(detect_cpu_features())),
      pool_(num_threads)
{
    const auto loaded = aot::load_aot_graph(assets, model_name);
    compile_steps(loaded, config);
    allocate_arena();
    if (state_bindings_.empty())
        state_initialized_ = true;
}

const CpuExecutionPlan::TensorMeta* CpuExecutionPlan::find_meta(aot::TensorId id) const noexcept
{
    if (id >= id_to_meta_.size() || id_to_meta_[id] < 0)
        return nullptr;
    return &metas_[static_cast<std::size_t>(id_to_meta_[id])];
}

CpuExecutionPlan::TensorMeta& CpuExecutionPlan::require_meta(aot::TensorId id)
{
    if (id >= id_to_meta_.size() || id_to_meta_[id] < 0)
        throw std::runtime_error("cpu: unknown tensor ID " + std::to_string(id));
    return metas_[static_cast<std::size_t>(id_to_meta_[id])];
}

const CpuExecutionPlan::TensorMeta& CpuExecutionPlan::require_meta(aot::TensorId id) const
{
    if (id >= id_to_meta_.size() || id_to_meta_[id] < 0)
        throw std::runtime_error("cpu: unknown tensor ID " + std::to_string(id));
    return metas_[static_cast<std::size_t>(id_to_meta_[id])];
}

void* CpuExecutionPlan::resolve_mut_ptr(aot::TensorId id)
{
    TensorMeta& m = require_meta(id);
    if (m.state_output_slot >= 0)
        return state_output_buffers_[static_cast<std::size_t>(m.state_output_slot)].data();
    if (m.is_initializer || m.is_graph_input)
        throw std::runtime_error("cpu: cannot obtain mutable pointer to constant or input tensor");
    return arena_.data() + m.arena_offset;
}

const void* CpuExecutionPlan::resolve_const_ptr(aot::TensorId id) const
{
    const TensorMeta& m = require_meta(id);
    if (m.is_graph_input)
        return input_buffers_[m.graph_input_pos].data();
    if (m.state_output_slot >= 0)
        return state_output_buffers_[static_cast<std::size_t>(m.state_output_slot)].data();
    if (m.is_initializer) {
        if (m.dtype == aot::DataType::FP16)
            return m.const_fp16.data();
        if (m.dtype == aot::DataType::INT32)
            return m.const_int32.data();
        return m.const_int64.data();
    }
    return arena_.data() + m.arena_offset;
}

Float16Storage* CpuExecutionPlan::fp16_mut(aot::TensorId id)
{
    return reinterpret_cast<Float16Storage*>(resolve_mut_ptr(id));
}

const Float16Storage* CpuExecutionPlan::fp16_const(aot::TensorId id) const
{
    return reinterpret_cast<const Float16Storage*>(resolve_const_ptr(id));
}

void CpuExecutionPlan::compile_steps(const aot::LoadedGraph& loaded,
                                     const ModelExecutionConfig& config)
{
    const auto& graph = loaded.descriptor;
    id_to_meta_.assign(graph.tensors.size(), -1);
    metas_.reserve(graph.tensors.size());

    for (std::size_t id = 1; id < graph.tensors.size(); ++id) {
        const auto& desc = graph.tensors[id];
        if (desc.name.empty())
            continue;
        TensorMeta m;
        m.id = static_cast<aot::TensorId>(id);
        m.name = desc.name;
        m.dtype = desc.dtype;
        m.shape = desc.shape;
        m.elem_count = checked_elems(desc.shape);
        m.byte_size = m.elem_count * dtype_size(desc.dtype);
        m.is_initializer = std::holds_alternative<aot::WeightLocation>(desc.storage);
        id_to_meta_[id] = static_cast<int>(metas_.size());
        metas_.push_back(std::move(m));
    }

    input_ids_ = graph.inputs;
    output_ids_ = graph.outputs;
    input_buffers_.resize(input_ids_.size());
    for (std::size_t i = 0; i < input_ids_.size(); ++i) {
        TensorMeta& m = require_meta(input_ids_[i]);
        m.is_graph_input = true;
        m.graph_input_pos = i;
        input_buffers_[i].assign(std::max<std::size_t>(m.byte_size, 1U), std::byte{0});
    }
    for (const aot::TensorId out_id : output_ids_)
        require_meta(out_id).is_graph_output = true;

    for (const StateTensorBinding& binding : config.state_bindings) {
        if (binding.input_index >= input_ids_.size() ||
            binding.output_index >= output_ids_.size()) {
            throw std::runtime_error("cpu: state binding index out of range");
        }
        TensorMeta& in_m = require_meta(input_ids_[binding.input_index]);
        TensorMeta& out_m = require_meta(output_ids_[binding.output_index]);
        if (in_m.state_input_slot >= 0 || out_m.state_output_slot >= 0)
            throw std::runtime_error("cpu: duplicate state tensor binding");
        if (in_m.dtype != out_m.dtype || in_m.shape != out_m.shape ||
            binding.data_type != to_public_dtype(in_m.dtype) ||
            binding.shape != in_m.shape) {
            throw std::runtime_error("cpu: state binding shape or dtype mismatch");
        }
        const int slot = static_cast<int>(state_bindings_.size());
        in_m.state_input_slot = slot;
        out_m.state_output_slot = slot;
        state_bindings_.push_back({binding.input_index, binding.output_index,
                                   in_m.id, out_m.id, in_m.byte_size});
        state_output_buffers_.emplace_back(std::max<std::size_t>(out_m.byte_size, 1U), std::byte{0});
    }

    for (std::size_t i = 0; i < input_ids_.size(); ++i) {
        if (require_meta(input_ids_[i]).state_input_slot < 0)
            external_input_indices_.push_back(i);
    }

    auto read_weight_span = [&](aot::TensorId tid) -> std::span<const std::byte> {
        if (tid == 0 || tid >= graph.tensors.size())
            return {};
        const auto* wl = std::get_if<aot::WeightLocation>(&graph.tensors[tid].storage);
        if (!wl || wl->offset > loaded.weights.size() ||
            wl->bytes > loaded.weights.size() - wl->offset) {
            throw std::runtime_error("cpu: invalid weight location");
        }
        return std::span<const std::byte>(loaded.weights.data() + wl->offset, wl->bytes);
    };

    auto read_fp16_raw = [&](aot::TensorId tid) -> std::vector<Float16Storage> {
        const auto bytes = read_weight_span(tid);
        std::vector<Float16Storage> vals(bytes.size() / sizeof(Float16Storage));
        if (!bytes.empty())
            std::memcpy(vals.data(), bytes.data(), bytes.size());
        return vals;
    };

    auto read_int64_raw = [&](aot::TensorId tid) -> std::vector<std::int64_t> {
        const auto bytes = read_weight_span(tid);
        std::vector<std::int64_t> vals(bytes.size() / sizeof(std::int64_t));
        if (!bytes.empty())
            std::memcpy(vals.data(), bytes.data(), bytes.size());
        return vals;
    };

    auto read_fp16_scalar = [&](aot::TensorId tid) -> float {
        const auto vals = read_fp16_raw(tid);
        if (vals.size() != 1U)
            throw std::runtime_error("cpu: expected scalar FP16 initializer");
        return half_to_float(vals[0]);
    };

    auto build_conv_info = [&](const aot::Node& node, ConvExecInfo& conv) {
        const auto& attr = node.conv();
        const auto& in_m = require_meta(node.inputs[0]);
        const auto& w_m = require_meta(node.inputs[1]);
        const auto& out_m = require_meta(node.outputs[0]);
        const auto id4 = dims_to_4d(in_m.shape);
        const auto wd4 = dims_to_4d(w_m.shape);
        const auto od4 = dims_to_4d(out_m.shape);

        conv.n = id4[0];
        conv.ic = id4[1];
        conv.ih = id4[2];
        conv.iw = id4[3];
        conv.oc = od4[1];
        conv.oh = od4[2];
        conv.ow = od4[3];
        conv.kh = wd4[2];
        conv.kw = wd4[3];
        conv.sh = attr.strides[0];
        conv.sw = attr.strides[1];
        conv.ph = attr.pads[0];
        conv.pw = attr.pads[1];
        conv.groups = attr.groups;

        const auto w_raw = read_fp16_raw(node.inputs[1]);
        std::vector<Float16Storage> b_raw;
        const Float16Storage* b_ptr = nullptr;
        if (node.inputs.size() >= 3U && node.inputs[2] != 0) {
            b_raw = read_fp16_raw(node.inputs[2]);
            if (static_cast<int>(b_raw.size()) != conv.oc)
                throw std::runtime_error("cpu: conv bias length mismatch");
            b_ptr = b_raw.data();
        }

        if (conv.groups == 1) {
            conv.is_1x1 = (conv.kh == 1 && conv.kw == 1 && conv.sh == 1 &&
                           conv.sw == 1 && conv.ph == 0 && conv.pw == 0);
            conv.packed_weights = pack_conv_weights_f16p16x1bias(
                w_raw.data(), b_ptr, static_cast<std::size_t>(conv.oc),
                static_cast<std::size_t>(conv.ic), static_cast<std::size_t>(conv.kh),
                static_cast<std::size_t>(conv.kw));
        } else if (conv.groups == conv.ic && conv.oc == conv.ic && wd4[1] == 1) {
            conv.is_depthwise = true;
            conv.is_dw_3x3_s1p1 = (conv.kh == 3 && conv.kw == 3 && conv.sh == 1 &&
                                   conv.sw == 1 && conv.ph == 1 && conv.pw == 1 &&
                                   conv.oh == conv.ih && conv.ow == conv.iw);
            conv.packed_weights = pack_depthwise_weights(
                w_raw.data(), b_ptr, static_cast<std::size_t>(conv.ic),
                static_cast<std::size_t>(conv.kh), static_cast<std::size_t>(conv.kw));
        } else {
            throw std::runtime_error("cpu: general grouped convolution (1 < groups < IC) unsupported");
        }
    };

    const bool fuse = fusion_enabled_from_env();
    auto schedule = build_fused_schedule(loaded, fuse);
    steps_.reserve(schedule.size());

    std::vector<aot::TensorId> materialized_inits;
    auto mark_init_data = [&](aot::TensorId tid) {
        if (tid != 0 && require_meta(tid).is_initializer)
            add_unique_id(materialized_inits, tid);
    };

    for (const FusedStep& fs : schedule) {
        CompiledStep cs;
        cs.fused = fs;
        if (fs.kind == FusedStepKind::kConvWithEpilogue) {
            build_conv_info(graph.nodes[fs.first_node], cs.fused_conv);
            mark_init_data(fs.conv_input);
            mark_init_data(fs.residual_input);
        } else if (fs.kind == FusedStepKind::kReglu) {
            mark_init_data(fs.reglu_input);
        } else if (fs.kind == FusedStepKind::kFeatureUpdate ||
                   fs.kind == FusedStepKind::kFeatureUpdateOutputs) {
            mark_init_data(fs.fu_input);
            mark_init_data(fs.fu_history);
            mark_init_data(fs.fu_scale);
        } else if (fs.kind == FusedStepKind::kY0Tail ||
                   fs.kind == FusedStepKind::kY1Tail) {
            mark_init_data(fs.y_latent_or_norm);
            mark_init_data(fs.y_prior);
            mark_init_data(fs.y_quant_scale);
            mark_init_data(fs.y_update_scale0);
            mark_init_data(fs.y_update_scale1);
            mark_init_data(fs.y_concat2_io);
            mark_init_data(fs.y_clipped_io);
        } else {
            const aot::Node& node = graph.nodes[fs.first_node];
            cs.single.op = node.op;
            cs.single.inputs = node.inputs;
            cs.single.output = node.outputs[0];

            switch (node.op) {
            case aot::Op::CONV:
                mark_init_data(node.inputs[0]);
                build_conv_info(node, cs.single.conv);
                break;
            case aot::Op::LEAKY_RELU:
                mark_init_data(node.inputs[0]);
                cs.single.alpha = std::get<aot::ActivationAttributes>(node.attributes).alpha;
                break;
            case aot::Op::CLIP:
                mark_init_data(node.inputs[0]);
                cs.single.clamp_min = node.inputs.size() > 1U && node.inputs[1] != 0
                    ? read_fp16_scalar(node.inputs[1])
                    : -std::numeric_limits<float>::infinity();
                cs.single.clamp_max = node.inputs.size() > 2U && node.inputs[2] != 0
                    ? read_fp16_scalar(node.inputs[2])
                    : std::numeric_limits<float>::infinity();
                break;
            case aot::Op::SIGMOID:
            case aot::Op::RECIPROCAL:
            case aot::Op::ROUND:
                mark_init_data(node.inputs[0]);
                break;
            case aot::Op::ADD:
            case aot::Op::SUB:
            case aot::Op::MUL: {
                mark_init_data(node.inputs[0]);
                mark_init_data(node.inputs[1]);
                cs.single.binary_kind = node.op == aot::Op::ADD ? BinaryKind::kAdd
                    : (node.op == aot::Op::SUB ? BinaryKind::kSub : BinaryKind::kMul);
                const auto& m0 = require_meta(node.inputs[0]);
                const auto& m1 = require_meta(node.inputs[1]);
                const auto& mo = require_meta(node.outputs[0]);
                if (m0.shape == mo.shape && m1.shape == mo.shape) {
                    cs.single.scalar_or_chan_bcast = false;
                } else if (m1.elem_count == 1U) {
                    cs.single.scalar_or_chan_bcast = true;
                    cs.single.is_scalar_bcast = true;
                    cs.single.bcast_on_lhs = false;
                } else if (m0.elem_count == 1U) {
                    cs.single.scalar_or_chan_bcast = true;
                    cs.single.is_scalar_bcast = true;
                    cs.single.bcast_on_lhs = true;
                } else if (mo.shape.size() == 4U && m0.shape == mo.shape &&
                           m1.shape == std::vector<std::int64_t>{1, mo.shape[1], 1, 1}) {
                    cs.single.scalar_or_chan_bcast = true;
                    cs.single.is_chan_bcast = true;
                    cs.single.bcast_on_lhs = false;
                    cs.single.bcast_channels = static_cast<std::size_t>(mo.shape[1]);
                    cs.single.bcast_spatial =
                        static_cast<std::size_t>(mo.shape[0] * mo.shape[2] * mo.shape[3]);
                } else if (mo.shape.size() == 4U && m1.shape == mo.shape &&
                           m0.shape == std::vector<std::int64_t>{1, mo.shape[1], 1, 1}) {
                    cs.single.scalar_or_chan_bcast = true;
                    cs.single.is_chan_bcast = true;
                    cs.single.bcast_on_lhs = true;
                    cs.single.bcast_channels = static_cast<std::size_t>(mo.shape[1]);
                    cs.single.bcast_spatial =
                        static_cast<std::size_t>(mo.shape[0] * mo.shape[2] * mo.shape[3]);
                }
                break;
            }
            case aot::Op::SLICE: {
                mark_init_data(node.inputs[0]);
                const auto& in_m = require_meta(node.inputs[0]);
                const auto& out_m = require_meta(node.outputs[0]);
                const auto id4 = dims_to_4d(in_m.shape);
                const auto od4 = dims_to_4d(out_m.shape);
                for (int d = 0; d < 4; ++d) {
                    cs.single.slice_id[d] = id4[static_cast<std::size_t>(d)];
                    cs.single.slice_od[d] = od4[static_cast<std::size_t>(d)];
                    cs.single.slice_starts[d] = 0;
                    cs.single.slice_steps[d] = 1;
                }
                const auto starts = read_int64_raw(node.inputs[1]);
                const auto ends = read_int64_raw(node.inputs[2]);
                std::vector<std::int64_t> axes;
                if (node.inputs.size() > 3U && node.inputs[3] != 0) {
                    axes = read_int64_raw(node.inputs[3]);
                } else {
                    axes.resize(starts.size());
                    std::iota(axes.begin(), axes.end(), 0);
                }
                std::vector<std::int64_t> steps(starts.size(), 1);
                if (node.inputs.size() > 4U && node.inputs[4] != 0)
                    steps = read_int64_raw(node.inputs[4]);
                for (std::size_t a = 0; a < starts.size(); ++a) {
                    int ax = static_cast<int>(axes[a]);
                    if (ax < 0)
                        ax += 4;
                    if (ax < 0 || ax >= 4)
                        throw std::runtime_error("cpu: SLICE axis out of range");
                    const std::int64_t dim = in_m.shape[static_cast<std::size_t>(ax)];
                    std::int64_t st = starts[a];
                    if (st < 0)
                        st += dim;
                    st = std::clamp<std::int64_t>(st, 0, dim);
                    (void)ends;
                    cs.single.slice_starts[ax] = static_cast<int>(st);
                    cs.single.slice_steps[ax] = static_cast<int>(steps[a]);
                }
                break;
            }
            case aot::Op::CONCAT: {
                cs.single.axis = std::get<aot::AxisAttributes>(node.attributes).axis;
                const auto& out_m = require_meta(node.outputs[0]);
                const auto od4 = dims_to_4d(out_m.shape);
                for (int d = 0; d < 4; ++d)
                    cs.single.concat_od[d] = od4[static_cast<std::size_t>(d)];
                if (cs.single.axis < 0)
                    cs.single.axis += 4;
                cs.single.concat_in_dims.reserve(node.inputs.size());
                for (const aot::TensorId cid : node.inputs) {
                    mark_init_data(cid);
                    const auto id4 = dims_to_4d(require_meta(cid).shape);
                    cs.single.concat_in_dims.push_back({id4[0], id4[1], id4[2], id4[3]});
                }
                break;
            }
            case aot::Op::GATHER:
                mark_init_data(node.inputs[0]);
                mark_init_data(node.inputs[1]);
                cs.single.axis = std::get<aot::AxisAttributes>(node.attributes).axis;
                break;
            case aot::Op::DEPTH_TO_SPACE:
            case aot::Op::SPACE_TO_DEPTH:
                mark_init_data(node.inputs[0]);
                cs.single.block_size = std::get<aot::PixelAttributes>(node.attributes).block_size;
                break;
            default:
                throw std::runtime_error("cpu: unsupported AOT operator");
            }
        }
        steps_.push_back(std::move(cs));
    }

    for (const aot::TensorId tid : materialized_inits) {
        TensorMeta& m = require_meta(tid);
        const auto bytes = read_weight_span(tid);
        if (m.dtype == aot::DataType::FP16) {
            std::vector<Float16Storage> raw(m.elem_count);
            if (!bytes.empty())
                std::memcpy(raw.data(), bytes.data(), bytes.size());
            if (needs_nhwc_transpose(m.shape, m.dtype)) {
                m.const_fp16.resize(m.elem_count);
                kernels_->nchw_to_nhwc(
                    raw.data(), m.const_fp16.data(),
                    static_cast<std::size_t>(m.shape[0]),
                    static_cast<std::size_t>(m.shape[1]),
                    static_cast<std::size_t>(m.shape[2] * m.shape[3]), pool_);
            } else {
                m.const_fp16 = std::move(raw);
            }
        } else if (m.dtype == aot::DataType::INT32) {
            m.const_int32.resize(m.elem_count);
            if (!bytes.empty())
                std::memcpy(m.const_int32.data(), bytes.data(), bytes.size());
        } else {
            m.const_int64.resize(m.elem_count);
            if (!bytes.empty())
                std::memcpy(m.const_int64.data(), bytes.data(), bytes.size());
        }
    }
}

void CpuExecutionPlan::allocate_arena()
{
    const std::size_t n_steps = steps_.size();
    const std::size_t n_metas = metas_.size();
    std::vector<std::size_t> first_def(n_metas, n_steps + 1U);
    std::vector<std::size_t> last_use(n_metas, 0U);
    std::vector<bool> needs_arena(n_metas, false);

    auto mark_def = [&](aot::TensorId tid, std::size_t step_idx) {
        if (tid == 0)
            return;
        const std::size_t mi = static_cast<std::size_t>(id_to_meta_[tid]);
        const TensorMeta& m = metas_[mi];
        if (m.is_initializer || m.is_graph_input || m.state_output_slot >= 0)
            return;
        needs_arena[mi] = true;
        first_def[mi] = std::min(first_def[mi], step_idx);
        last_use[mi] = std::max(last_use[mi], step_idx);
    };

    auto mark_use = [&](aot::TensorId tid, std::size_t step_idx) {
        if (tid == 0)
            return;
        const std::size_t mi = static_cast<std::size_t>(id_to_meta_[tid]);
        const TensorMeta& m = metas_[mi];
        if (m.is_initializer || m.is_graph_input || m.state_output_slot >= 0)
            return;
        last_use[mi] = std::max(last_use[mi], step_idx);
    };

    for (std::size_t s = 0; s < n_steps; ++s) {
        const FusedStep& fs = steps_[s].fused;
        switch (fs.kind) {
        case FusedStepKind::kConvWithEpilogue:
            mark_def(fs.output_tensor, s);
            mark_use(fs.conv_input, s);
            mark_use(fs.residual_input, s);
            break;
        case FusedStepKind::kReglu:
            mark_def(fs.output_tensor, s);
            mark_use(fs.reglu_input, s);
            break;
        case FusedStepKind::kFeatureUpdate:
            mark_def(fs.fu_feature_out, s);
            mark_use(fs.fu_input, s);
            mark_use(fs.fu_history, s);
            mark_use(fs.fu_scale, s);
            break;
        case FusedStepKind::kFeatureUpdateOutputs:
            mark_def(fs.fu_scaled_out, s);
            mark_def(fs.fu_blended_out, s);
            mark_def(fs.fu_feature_out, s);
            mark_use(fs.fu_input, s);
            mark_use(fs.fu_history, s);
            mark_use(fs.fu_scale, s);
            break;
        case FusedStepKind::kY0Tail:
            mark_def(fs.y_normalized_out, s);
            mark_def(fs.y_clipped_io, s);
            mark_def(fs.y_raw_out, s);
            mark_def(fs.y_concat2_io, s);
            mark_def(fs.y_concat3_or_out, s);
            mark_use(fs.y_latent_or_norm, s);
            mark_use(fs.y_prior, s);
            mark_use(fs.y_quant_scale, s);
            mark_use(fs.y_update_scale0, s);
            mark_use(fs.y_update_scale1, s);
            break;
        case FusedStepKind::kY1Tail:
            mark_def(fs.y_raw_out, s);
            mark_def(fs.y_concat3_or_out, s);
            mark_use(fs.y_latent_or_norm, s);
            mark_use(fs.y_prior, s);
            mark_use(fs.y_quant_scale, s);
            mark_use(fs.y_update_scale0, s);
            mark_use(fs.y_update_scale1, s);
            mark_use(fs.y_concat2_io, s);
            mark_use(fs.y_clipped_io, s);
            break;
        case FusedStepKind::kSingleNode:
            mark_def(steps_[s].single.output, s);
            for (const aot::TensorId iid : steps_[s].single.inputs)
                mark_use(iid, s);
            break;
        }
    }

    for (std::size_t mi = 0; mi < n_metas; ++mi) {
        if (needs_arena[mi] && metas_[mi].is_graph_output)
            last_use[mi] = n_steps;
    }

    struct PlacedBlock {
        std::size_t offset = 0;
        std::size_t size = 0;
        std::size_t first_def = 0;
        std::size_t last_use = 0;
    };
    std::vector<std::size_t> order;
    for (std::size_t mi = 0; mi < n_metas; ++mi) {
        if (needs_arena[mi])
            order.push_back(mi);
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (first_def[a] != first_def[b])
            return first_def[a] < first_def[b];
        return metas_[a].byte_size > metas_[b].byte_size;
    });

    std::vector<PlacedBlock> placed;
    std::size_t peak = 0;
    for (const std::size_t mi : order) {
        const std::size_t sz = align_up_64(std::max<std::size_t>(metas_[mi].byte_size, 64U));
        const std::size_t d0 = first_def[mi];
        const std::size_t d1 = last_use[mi];

        std::vector<std::pair<std::size_t, std::size_t>> conflicts;
        for (const auto& blk : placed) {
            if (blk.first_def <= d1 && d0 <= blk.last_use)
                conflicts.emplace_back(blk.offset, blk.offset + blk.size);
        }
        std::sort(conflicts.begin(), conflicts.end());

        std::size_t candidate = 0;
        for (const auto& iv : conflicts) {
            if (candidate + sz <= iv.first)
                break;
            if (iv.second > candidate)
                candidate = iv.second;
        }
        metas_[mi].arena_offset = candidate;
        placed.push_back({candidate, sz, d0, d1});
        peak = std::max(peak, candidate + sz);
    }

    arena_.assign(std::max<std::size_t>(peak, 64U), std::byte{0});
}

void CpuExecutionPlan::reset_state()
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    for (const ResolvedStateBinding& binding : state_bindings_) {
        auto& buf = input_buffers_[binding.input_index];
        std::fill(buf.begin(), buf.end(), std::byte{0});
    }
    for (auto& out_buf : state_output_buffers_)
        std::fill(out_buf.begin(), out_buf.end(), std::byte{0});
    state_initialized_ = true;
}

void CpuExecutionPlan::execute_step(const CompiledStep& step)
{
    const FusedStep& fs = step.fused;
    auto run_conv = [&](const ConvExecInfo& c, aot::TensorId in_id,
                        aot::TensorId out_id, const GemmEpilogue& ep) {
        const Float16Storage* in_ptr = fp16_const(in_id);
        Float16Storage* out_ptr = fp16_mut(out_id);
        if (c.is_depthwise) {
            if (c.is_dw_3x3_s1p1) {
                kernels_->depthwise3x3_s1p1(
                    in_ptr, c.packed_weights.data(), out_ptr,
                    c.n, c.ih, c.iw, c.ic, pool_);
            } else {
                execute_depthwise_generic_nhwc(
                    in_ptr, c.packed_weights.data(), out_ptr,
                    c.n, c.ic, c.ih, c.iw, c.oh, c.ow,
                    c.kh, c.kw, c.sh, c.sw, c.ph, c.pw, pool_);
            }
        } else if (c.is_1x1) {
            const std::size_t m =
                static_cast<std::size_t>(c.n) * static_cast<std::size_t>(c.oh) *
                static_cast<std::size_t>(c.ow);
            execute_gemm_nhwc(*kernels_, m, static_cast<std::size_t>(c.oc),
                              static_cast<std::size_t>(c.ic), in_ptr,
                              c.packed_weights.data(), out_ptr, ep, pool_);
        } else {
            execute_spatial_conv_nhwc(*kernels_, in_ptr, c.packed_weights.data(),
                                      out_ptr, c.n, c.ic, c.ih, c.iw, c.oc,
                                      c.oh, c.ow, c.kh, c.kw, c.sh, c.sw,
                                      c.ph, c.pw, ep, pool_);
        }
    };

    switch (fs.kind) {
    case FusedStepKind::kConvWithEpilogue: {
        GemmEpilogue ep;
        ep.kind = fs.epilogue_kind;
        ep.alpha = fs.alpha;
        ep.clamp_min = fs.clamp_min;
        ep.clamp_max = fs.clamp_max;
        if (fs.residual_input != 0)
            ep.residual = fp16_const(fs.residual_input);
        run_conv(step.fused_conv, fs.conv_input, fs.output_tensor, ep);
        return;
    }
    case FusedStepKind::kReglu: {
        const auto& m = require_meta(fs.output_tensor);
        const auto od = dims_to_4d(m.shape);
        const std::size_t spatial = static_cast<std::size_t>(od[0]) * od[2] * od[3];
        kernels_->reglu_nhwc(
            fp16_const(fs.reglu_input), fp16_mut(fs.output_tensor),
            spatial, fs.reglu_gate_channels, fs.clamp_min, fs.clamp_max, pool_);
        return;
    }
    case FusedStepKind::kFeatureUpdate: {
        const auto& m = require_meta(fs.fu_history);
        const auto hd = dims_to_4d(m.shape);
        const std::size_t spatial = static_cast<std::size_t>(hd[0]) * hd[2] * hd[3];
        kernels_->feature_update_nhwc(
            fp16_const(fs.fu_input), fp16_const(fs.fu_history),
            fp16_const(fs.fu_scale), fp16_mut(fs.fu_feature_out),
            spatial, fs.fu_channels, pool_);
        return;
    }
    case FusedStepKind::kFeatureUpdateOutputs: {
        const auto& m = require_meta(fs.fu_history);
        const auto hd = dims_to_4d(m.shape);
        const std::size_t spatial = static_cast<std::size_t>(hd[0]) * hd[2] * hd[3];
        kernels_->feature_update_outputs_nhwc(
            fp16_const(fs.fu_input), fp16_const(fs.fu_history),
            fp16_const(fs.fu_scale), fp16_mut(fs.fu_scaled_out),
            fp16_mut(fs.fu_blended_out), fp16_mut(fs.fu_feature_out),
            spatial, fs.fu_channels, pool_);
        return;
    }
    case FusedStepKind::kY0Tail: {
        const auto& m = require_meta(fs.y_raw_out);
        const auto yd = dims_to_4d(m.shape);
        const std::size_t spatial = static_cast<std::size_t>(yd[0]) * yd[2] * yd[3];
        kernels_->y0_tail_nhwc(
            fp16_const(fs.y_latent_or_norm), fp16_const(fs.y_prior),
            fp16_const(fs.y_quant_scale), fp16_const(fs.y_update_scale0),
            fp16_const(fs.y_update_scale1),
            fp16_mut(fs.y_normalized_out), fp16_mut(fs.y_clipped_io),
            fp16_mut(fs.y_raw_out), fp16_mut(fs.y_concat2_io),
            fp16_mut(fs.y_concat3_or_out),
            fs.y_clip_min, spatial, fs.y_quant_channels, pool_);
        return;
    }
    case FusedStepKind::kY1Tail: {
        const auto& m = require_meta(fs.y_raw_out);
        const auto yd = dims_to_4d(m.shape);
        const std::size_t spatial = static_cast<std::size_t>(yd[0]) * yd[2] * yd[3];
        kernels_->y1_tail_nhwc(
            fp16_const(fs.y_latent_or_norm), fp16_const(fs.y_prior),
            fp16_const(fs.y_quant_scale), fp16_const(fs.y_update_scale0),
            fp16_const(fs.y_update_scale1),
            fp16_const(fs.y_concat2_io), fp16_const(fs.y_clipped_io),
            fp16_mut(fs.y_raw_out), fp16_mut(fs.y_concat3_or_out),
            spatial, fs.y_quant_channels, pool_);
        return;
    }
    case FusedStepKind::kSingleNode:
        break;
    }

    const SingleOpInfo& op = step.single;
    const auto& out_m = require_meta(op.output);

    switch (op.op) {
    case aot::Op::CONV: {
        GemmEpilogue none;
        run_conv(op.conv, op.inputs[0], op.output, none);
        return;
    }
    case aot::Op::LEAKY_RELU:
        kernels_->unary(fp16_const(op.inputs[0]), fp16_mut(op.output),
                        out_m.elem_count, UnaryKind::kLeakyRelu, op.alpha,
                        0.0F, 0.0F, pool_);
        return;
    case aot::Op::CLIP:
        kernels_->unary(fp16_const(op.inputs[0]), fp16_mut(op.output),
                        out_m.elem_count, UnaryKind::kClip, 0.0F,
                        op.clamp_min, op.clamp_max, pool_);
        return;
    case aot::Op::SIGMOID:
        kernels_->unary(fp16_const(op.inputs[0]), fp16_mut(op.output),
                        out_m.elem_count, UnaryKind::kSigmoid, 0.0F,
                        0.0F, 0.0F, pool_);
        return;
    case aot::Op::RECIPROCAL:
        kernels_->unary(fp16_const(op.inputs[0]), fp16_mut(op.output),
                        out_m.elem_count, UnaryKind::kReciprocal, 0.0F,
                        0.0F, 0.0F, pool_);
        return;
    case aot::Op::ROUND:
        kernels_->unary(fp16_const(op.inputs[0]), fp16_mut(op.output),
                        out_m.elem_count, UnaryKind::kRound, 0.0F,
                        0.0F, 0.0F, pool_);
        return;
    case aot::Op::ADD:
    case aot::Op::SUB:
    case aot::Op::MUL: {
        const auto& m0 = require_meta(op.inputs[0]);
        const auto& m1 = require_meta(op.inputs[1]);
        const Float16Storage* a = fp16_const(op.inputs[0]);
        const Float16Storage* b = fp16_const(op.inputs[1]);
        Float16Storage* out = fp16_mut(op.output);
        if (!op.scalar_or_chan_bcast) {
            if (m0.shape == out_m.shape && m1.shape == out_m.shape) {
                kernels_->binary_contiguous(a, b, out, out_m.elem_count, op.binary_kind, pool_);
            } else {
                execute_binary_broadcast_fallback_nhwc(
                    a, m0.shape, b, m1.shape, out, out_m.shape, op.binary_kind, pool_);
            }
        } else if (op.is_scalar_bcast) {
            const Float16Storage* vec = op.bcast_on_lhs ? b : a;
            const Float16Storage scalar = op.bcast_on_lhs ? a[0] : b[0];
            kernels_->binary_scalar_bcast(vec, scalar, out, out_m.elem_count,
                                          op.binary_kind, op.bcast_on_lhs, pool_);
        } else {
            const Float16Storage* vec = op.bcast_on_lhs ? b : a;
            const Float16Storage* chan = op.bcast_on_lhs ? a : b;
            kernels_->binary_channel_bcast(vec, chan, out, op.bcast_spatial,
                                           op.bcast_channels, op.binary_kind,
                                           op.bcast_on_lhs, pool_);
        }
        return;
    }
    case aot::Op::SLICE:
        execute_slice_nhwc(
            fp16_const(op.inputs[0]), fp16_mut(op.output),
            op.slice_id, op.slice_od, op.slice_starts, op.slice_steps, pool_);
        return;
    case aot::Op::CONCAT: {
        std::vector<const Float16Storage*> ptrs(op.inputs.size());
        std::vector<const int*> dims(op.inputs.size());
        for (std::size_t idx = 0; idx < op.inputs.size(); ++idx) {
            ptrs[idx] = fp16_const(op.inputs[idx]);
            dims[idx] = op.concat_in_dims[idx].data();
        }
        execute_concat_nhwc(ptrs.data(), dims.data(), op.inputs.size(),
                            fp16_mut(op.output), op.concat_od, op.axis, pool_);
        return;
    }
    case aot::Op::GATHER: {
        const auto& data_m = require_meta(op.inputs[0]);
        const auto& idx_m = require_meta(op.inputs[1]);
        const void* idx_ptr = resolve_const_ptr(op.inputs[1]);
        execute_gather_general(
            fp16_const(op.inputs[0]), data_m.shape,
            idx_m.dtype == aot::DataType::INT32 ? reinterpret_cast<const std::int32_t*>(idx_ptr) : nullptr,
            idx_m.dtype == aot::DataType::INT64 ? reinterpret_cast<const std::int64_t*>(idx_ptr) : nullptr,
            idx_m.elem_count, op.axis, fp16_mut(op.output));
        return;
    }
    case aot::Op::DEPTH_TO_SPACE: {
        const auto id4 = dims_to_4d(require_meta(op.inputs[0]).shape);
        execute_depth_to_space_nhwc(
            fp16_const(op.inputs[0]), fp16_mut(op.output),
            id4[0], id4[1], id4[2], id4[3], op.block_size, pool_);
        return;
    }
    case aot::Op::SPACE_TO_DEPTH: {
        const auto id4 = dims_to_4d(require_meta(op.inputs[0]).shape);
        execute_space_to_depth_nhwc(
            fp16_const(op.inputs[0]), fp16_mut(op.output),
            id4[0], id4[1], id4[2], id4[3], op.block_size, pool_);
        return;
    }
    default:
        throw std::runtime_error("cpu: unsupported AOT operator during execution");
    }
}

std::vector<Tensor> CpuExecutionPlan::run(const std::vector<Tensor>& inputs)
{
    std::lock_guard<std::mutex> lock(exec_mutex_);
    if (!state_bindings_.empty() && !state_initialized_)
        throw std::runtime_error("cpu: reset_state() must be called before run()");
    if (inputs.size() != external_input_indices_.size())
        throw std::runtime_error("cpu: graph input count mismatch");

    for (std::size_t ext_i = 0; ext_i < external_input_indices_.size(); ++ext_i) {
        const std::size_t graph_idx = external_input_indices_[ext_i];
        const Tensor& input = inputs[ext_i];
        input.validate();
        const TensorMeta& expected = require_meta(input_ids_[graph_idx]);
        if (input.shape != expected.shape ||
            input.data_type() != to_public_dtype(expected.dtype) ||
            input.byte_size() != expected.byte_size) {
            throw std::runtime_error("cpu: graph input shape, dtype, or byte size mismatch");
        }
        if (needs_nhwc_transpose(expected.shape, expected.dtype)) {
            kernels_->nchw_to_nhwc(
                reinterpret_cast<const Float16Storage*>(input.raw_data()),
                reinterpret_cast<Float16Storage*>(input_buffers_[graph_idx].data()),
                static_cast<std::size_t>(expected.shape[0]),
                static_cast<std::size_t>(expected.shape[1]),
                static_cast<std::size_t>(expected.shape[2] * expected.shape[3]),
                pool_);
        } else {
            std::memcpy(input_buffers_[graph_idx].data(), input.raw_data(), expected.byte_size);
        }
    }

    for (const CompiledStep& step : steps_)
        execute_step(step);

    for (std::size_t slot = 0; slot < state_bindings_.size(); ++slot) {
        const ResolvedStateBinding& binding = state_bindings_[slot];
        std::memcpy(input_buffers_[binding.input_index].data(),
                    state_output_buffers_[slot].data(),
                    binding.byte_size);
    }

    std::vector<Tensor> outputs;
    outputs.reserve(output_ids_.size() - state_bindings_.size());
    for (const aot::TensorId out_id : output_ids_) {
        const TensorMeta& out_m = require_meta(out_id);
        if (out_m.state_output_slot >= 0)
            continue;
        Tensor t;
        t.name = out_m.name;
        t.shape = out_m.shape;
        if (out_m.dtype == aot::DataType::FP16) {
            std::vector<Float16Storage> buf(out_m.elem_count);
            const auto* src = reinterpret_cast<const Float16Storage*>(resolve_const_ptr(out_id));
            if (needs_nhwc_transpose(out_m.shape, out_m.dtype)) {
                kernels_->nhwc_to_nchw(
                    src, buf.data(),
                    static_cast<std::size_t>(out_m.shape[0]),
                    static_cast<std::size_t>(out_m.shape[2] * out_m.shape[3]),
                    static_cast<std::size_t>(out_m.shape[1]),
                    pool_);
            } else {
                std::memcpy(buf.data(), src, out_m.byte_size);
            }
            t.data = std::move(buf);
        } else if (out_m.dtype == aot::DataType::INT32) {
            std::vector<std::int32_t> buf(out_m.elem_count);
            std::memcpy(buf.data(), resolve_const_ptr(out_id), out_m.byte_size);
            t.data = std::move(buf);
        } else {
            throw std::runtime_error("cpu: unsupported graph output dtype");
        }
        outputs.push_back(std::move(t));
    }
    return outputs;
}

}  // namespace mlvc::cpu_backend
