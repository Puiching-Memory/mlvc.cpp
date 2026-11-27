#include "execution_plan.hpp"

#include "shaders.hpp"

#include "core/env.hpp"
#include "core/half.hpp"
#include "core/logging.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mlvc::webgpu_backend {
namespace {

std::uint32_t div_up_u32(std::size_t value, std::uint32_t divisor) noexcept
{
    return static_cast<std::uint32_t>((value + divisor - 1U) / divisor);
}

void nchw_to_nhwc_host(const Float16Storage* src, Float16Storage* dst,
                       std::size_t n, std::size_t c, std::size_t hw)
{
    for (std::size_t b = 0; b < n; ++b) {
        const Float16Storage* sb = src + b * c * hw;
        Float16Storage* db = dst + b * hw * c;
        for (std::size_t s = 0; s < hw; ++s) {
            Float16Storage* dp = db + s * c;
            for (std::size_t ch = 0; ch < c; ++ch)
                dp[ch] = sb[ch * hw + s];
        }
    }
}

std::vector<Float16Storage> pack_pointwise_kn(
    const Float16Storage* oihw, std::size_t oc, std::size_t ic)
{
    std::vector<Float16Storage> out(ic * oc, 0);
    for (std::size_t k = 0; k < ic; ++k) {
        Float16Storage* dst = out.data() + k * oc;
        for (std::size_t n = 0; n < oc; ++n)
            dst[n] = oihw[n * ic + k];
    }
    return out;
}

std::vector<Float16Storage> pack_pointwise_reglu_kn(
    const Float16Storage* oihw, std::size_t oc, std::size_t ic)
{
    const std::size_t g = oc / 2U;
    const std::size_t g4 = g / 4U;
    std::vector<Float16Storage> out(ic * oc, 0);
    for (std::size_t k = 0; k < ic; ++k) {
        Float16Storage* dst = out.data() + k * oc;
        for (std::size_t col4 = 0; col4 < g4; ++col4) {
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                const std::size_t ch = col4 * 4U + lane;
                dst[(col4 * 2U + 0U) * 4U + lane] = oihw[ch * ic + k];
                dst[(col4 * 2U + 1U) * 4U + lane] = oihw[(g + ch) * ic + k];
            }
        }
    }
    return out;
}

std::vector<Float16Storage> pack_reglu_bias(
    const Float16Storage* bias, std::size_t oc)
{
    const std::size_t g = oc / 2U;
    const std::size_t g4 = g / 4U;
    std::vector<Float16Storage> out(oc, 0);
    for (std::size_t col4 = 0; col4 < g4; ++col4) {
        for (std::size_t lane = 0; lane < 4U; ++lane) {
            const std::size_t ch = col4 * 4U + lane;
            out[(col4 * 2U + 0U) * 4U + lane] = bias[ch];
            out[(col4 * 2U + 1U) * 4U + lane] = bias[g + ch];
        }
    }
    return out;
}

std::vector<Float16Storage> pack_spatial_kn(
    const Float16Storage* oihw, std::size_t oc, std::size_t ic,
    std::size_t kh, std::size_t kw)
{
    const std::size_t k_total = kh * kw * ic;
    const std::size_t kplane = kh * kw;
    std::vector<Float16Storage> out(k_total * oc, 0);
    for (std::size_t ky = 0; ky < kh; ++ky) {
        for (std::size_t kx = 0; kx < kw; ++kx) {
            const std::size_t ksp = ky * kw + kx;
            for (std::size_t c = 0; c < ic; ++c) {
                const std::size_t k_row = ksp * ic + c;
                Float16Storage* dst = out.data() + k_row * oc;
                for (std::size_t n = 0; n < oc; ++n) {
                    dst[n] = oihw[(n * ic + c) * kplane + ksp];
                }
            }
        }
    }
    return out;
}

std::vector<Float16Storage> pack_depthwise_3x3(
    const Float16Storage* c1hw, std::size_t c)
{
    std::vector<Float16Storage> out(9U * c, 0);
    for (std::size_t k = 0; k < 9U; ++k) {
        Float16Storage* dst = out.data() + k * c;
        for (std::size_t ch = 0; ch < c; ++ch)
            dst[ch] = c1hw[ch * 9U + k];
    }
    return out;
}

std::vector<WGPUBufferBindingType> make_binding_types(
    std::size_t read_storage_count, std::size_t rw_storage_count)
{
    std::vector<WGPUBufferBindingType> types;
    types.reserve(1U + read_storage_count + rw_storage_count);
    types.push_back(WGPUBufferBindingType_Uniform);
    for (std::size_t i = 0; i < read_storage_count; ++i)
        types.push_back(WGPUBufferBindingType_ReadOnlyStorage);
    for (std::size_t i = 0; i < rw_storage_count; ++i)
        types.push_back(WGPUBufferBindingType_Storage);
    return types;
}

}  // namespace

bool WebGpuExecutionPlan::needs_nhwc_transpose(
    const std::vector<std::int64_t>& shape, aot::DataType dt) noexcept
{
    return dt == aot::DataType::FP16 && shape.size() == 4U &&
           shape[1] > 1 && (shape[2] * shape[3]) > 1;
}

WebGpuExecutionPlan::WebGpuExecutionPlan(
    std::shared_ptr<WgpuContext> ctx,
    const ModelAssetReader& assets,
    const std::string& model_name,
    const ModelExecutionConfig& config)
    : ctx_(std::move(ctx)),
      model_name_(model_name),
      is_encoder_(model_name == "MLVCEncoder")
{
    if (!ctx_)
        throw std::runtime_error("webgpu: WgpuContext must not be null");
    profile_enabled_ = mlvc::detail::env_flag_enabled("MLVC_WEBGPU_PROFILE") &&
                       ctx_->caps().has_timestamp_query;

    dummy_storage_buffer_ = ctx_->create_buffer(
        64U, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "dummy_storage");

    const auto loaded = aot::load_aot_graph(assets, model_name);
    compile_graph(loaded, config);

    if (log_enabled(LogLevel::kInfo)) {
        log_message(
            LogLevel::kInfo, "webgpu",
            "loaded " + model_name_ + " dispatches=" + std::to_string(dispatches_.size()) +
                " pools=" + std::to_string(pool_buffers_.size()) +
                " f16=" + (uses_f16() ? "on" : "off") +
                " d2s8_fusion=" + (fused_d2s8_active() ? "on" : "off"));
    }
}

WebGpuExecutionPlan::~WebGpuExecutionPlan() = default;

const WebGpuExecutionPlan::TensorMeta& WebGpuExecutionPlan::require_meta(
    aot::TensorId id) const
{
    if (id >= id_to_meta_.size() || id_to_meta_[id] < 0)
        throw std::runtime_error("webgpu: unknown tensor ID " + std::to_string(id));
    return metas_[static_cast<std::size_t>(id_to_meta_[id])];
}

WebGpuExecutionPlan::TensorMeta& WebGpuExecutionPlan::require_meta(aot::TensorId id)
{
    if (id >= id_to_meta_.size() || id_to_meta_[id] < 0)
        throw std::runtime_error("webgpu: unknown tensor ID " + std::to_string(id));
    return metas_[static_cast<std::size_t>(id_to_meta_[id])];
}

WGPUBuffer WebGpuExecutionPlan::resolve_buffer(aot::TensorId id, int parity) const
{
    if (id == 0)
        return dummy_storage_buffer_.get();
    const TensorMeta& m = require_meta(id);
    if (m.state_input_slot >= 0)
        return state_buffers_[static_cast<std::size_t>(m.state_input_slot)][static_cast<std::size_t>(parity)].get();
    if (m.state_output_slot >= 0)
        return state_buffers_[static_cast<std::size_t>(m.state_output_slot)][static_cast<std::size_t>(parity ^ 1)].get();
    if (m.is_graph_input)
        return input_buffers_[m.graph_input_pos].get();
    if (m.is_initializer) {
        if (!m.const_buffer)
            throw std::runtime_error("webgpu: unmaterialized initializer tensor " + m.name);
        return m.const_buffer;
    }
    if (m.pool_id < 0)
        throw std::runtime_error("webgpu: unassigned pool for tensor " + m.name);
    return pool_buffers_[static_cast<std::size_t>(m.pool_id)].get();
}

const WebGpuExecutionPlan::PipelineCacheEntry& WebGpuExecutionPlan::get_or_create_pipeline(
    const std::string& key,
    const std::string& wgsl_source,
    const std::vector<WGPUBufferBindingType>& binding_types)
{
    const auto it = pipeline_cache_.find(key);
    if (it != pipeline_cache_.end())
        return it->second;

    std::vector<WGPUBindGroupLayoutEntry> entries(binding_types.size());
    for (std::size_t i = 0; i < binding_types.size(); ++i) {
        entries[i] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entries[i].binding = static_cast<std::uint32_t>(i);
        entries[i].visibility = WGPUShaderStage_Compute;
        entries[i].buffer.type = binding_types[i];
    }

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.label = to_wgpu_string(key + "_bgl");
    bgl_desc.entryCount = entries.size();
    bgl_desc.entries = entries.data();
    UniqueBindGroupLayout bgl(
        wgpuDeviceCreateBindGroupLayout(ctx_->device(), &bgl_desc));

    WGPUBindGroupLayout bgl_raw = bgl.get();
    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.label = to_wgpu_string(key + "_pl");
    pl_desc.bindGroupLayoutCount = 1U;
    pl_desc.bindGroupLayouts = &bgl_raw;
    UniquePipelineLayout layout(
        wgpuDeviceCreatePipelineLayout(ctx_->device(), &pl_desc));

    UniqueShaderModule mod = ctx_->create_shader_module(wgsl_source, key);

    WGPUComputePipelineDescriptor cp_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    cp_desc.label = to_wgpu_string(key);
    cp_desc.layout = layout.get();
    cp_desc.compute.module = mod.get();
    cp_desc.compute.entryPoint = to_wgpu_string("main");
    UniqueComputePipeline pipeline(
        wgpuDeviceCreateComputePipeline(ctx_->device(), &cp_desc));
    if (!pipeline)
        throw std::runtime_error("webgpu: failed to create compute pipeline " + key);
    ctx_->check_error("create_pipeline:" + key);

    PipelineCacheEntry entry;
    entry.bgl = std::move(bgl);
    entry.layout = std::move(layout);
    entry.pipeline = std::move(pipeline);
    auto [inserted, _] = pipeline_cache_.emplace(key, std::move(entry));
    return inserted->second;
}

UniqueBuffer WebGpuExecutionPlan::create_uniform_buffer(
    const void* data, std::size_t size, std::string_view label)
{
    UniqueBuffer buf = ctx_->create_buffer(
        size, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, label);
    ctx_->write_buffer(buf.get(), 0U, data, size);
    return buf;
}

UniqueBindGroup WebGpuExecutionPlan::create_bind_group(
    WGPUBindGroupLayout bgl, const std::vector<WGPUBuffer>& buffers)
{
    std::vector<WGPUBindGroupEntry> entries(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        entries[i] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[i].binding = static_cast<std::uint32_t>(i);
        entries[i].buffer = buffers[i];
        entries[i].offset = 0U;
        entries[i].size = WGPU_WHOLE_SIZE;
    }
    WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    desc.layout = bgl;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    UniqueBindGroup bg(wgpuDeviceCreateBindGroup(ctx_->device(), &desc));
    if (!bg)
        throw std::runtime_error("webgpu: failed to create bind group");
    return bg;
}

void WebGpuExecutionPlan::compile_graph(
    const aot::LoadedGraph& loaded, const ModelExecutionConfig& config)
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
        m.elem_count = aot::checked_element_count(desc.shape);
        m.byte_size = m.elem_count * aot::dtype_byte_size(desc.dtype);
        m.is_initializer = std::holds_alternative<aot::WeightLocation>(desc.storage);
        id_to_meta_[id] = static_cast<int>(metas_.size());
        metas_.push_back(std::move(m));
    }

    input_ids_ = graph.inputs;
    output_ids_ = graph.outputs;
    input_buffers_.resize(input_ids_.size());
    input_nchw_buffers_.resize(input_ids_.size());
    output_nchw_buffers_.resize(output_ids_.size());

    for (std::size_t i = 0; i < input_ids_.size(); ++i) {
        TensorMeta& m = require_meta(input_ids_[i]);
        m.is_graph_input = true;
        m.graph_input_pos = i;
        if (m.name == "q_index_shifted")
            q_input_pos_ = i;
        else if (m.name == "x")
            x_input_pos_ = i;
    }
    for (std::size_t i = 0; i < output_ids_.size(); ++i) {
        TensorMeta& m = require_meta(output_ids_[i]);
        m.is_graph_output = true;
        if (m.name == "x_hat")
            x_hat_output_pos_ = i;
    }

    for (const StateTensorBinding& binding : config.state_bindings) {
        if (binding.input_index >= input_ids_.size() ||
            binding.output_index >= output_ids_.size()) {
            throw std::runtime_error("webgpu: state binding index out of range");
        }
        TensorMeta& in_m = require_meta(input_ids_[binding.input_index]);
        TensorMeta& out_m = require_meta(output_ids_[binding.output_index]);
        if (in_m.state_input_slot >= 0 || out_m.state_output_slot >= 0)
            throw std::runtime_error("webgpu: duplicate state tensor binding");
        if (in_m.dtype != out_m.dtype || in_m.shape != out_m.shape ||
            binding.data_type != aot::to_public_dtype(in_m.dtype) ||
            binding.shape != in_m.shape) {
            throw std::runtime_error("webgpu: state binding shape or dtype mismatch");
        }
        const int slot = static_cast<int>(state_bindings_.size());
        in_m.state_input_slot = slot;
        out_m.state_output_slot = slot;
        state_bindings_.push_back(
            {binding.input_index, binding.output_index, in_m.id, out_m.id, in_m.byte_size});
        std::array<UniqueBuffer, 2> pair;
        for (int p = 0; p < 2; ++p) {
            pair[static_cast<std::size_t>(p)] = ctx_->create_buffer(
                std::max<std::size_t>(in_m.byte_size, 16U),
                WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
                "state_" + std::to_string(slot) + "_" + std::to_string(p));
        }
        state_buffers_.push_back(std::move(pair));
    }

    for (std::size_t i = 0; i < input_ids_.size(); ++i) {
        const TensorMeta& m = require_meta(input_ids_[i]);
        if (m.state_input_slot >= 0)
            continue;
        external_input_indices_.push_back(i);
        input_buffers_[i] = ctx_->create_buffer(
            std::max<std::size_t>(m.byte_size, 16U),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            "input_" + m.name);
        if (needs_nhwc_transpose(m.shape, m.dtype)) {
            input_nchw_buffers_[i] = ctx_->create_buffer(
                std::max<std::size_t>(m.byte_size, 16U),
                WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                "input_nchw_" + m.name);
        }
    }

    for (std::size_t i = 0; i < output_ids_.size(); ++i) {
        const TensorMeta& m = require_meta(output_ids_[i]);
        if (m.state_output_slot >= 0)
            continue;
        if (needs_nhwc_transpose(m.shape, m.dtype)) {
            output_nchw_buffers_[i] = ctx_->create_buffer(
                std::max<std::size_t>(m.byte_size, 16U),
                WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                "output_nchw_" + m.name);
        }
    }

    const bool enable_fusion = !mlvc::detail::env_flag_enabled("MLVC_WEBGPU_DISABLE_FUSION");
    const auto schedule = build_fused_schedule(
        loaded, enable_fusion, ctx_->caps().max_storage_buffers_per_stage);

    // Pre-detect MLVCDecoder tail DepthToSpace(block=8) [+ Clip(0, 1)] -> x_hat
    // BEFORE pool coloring so d2s8_input_id_'s pool stays alive until the end of the graph.
    if (!is_encoder_ && !schedule.empty() && x_hat_output_pos_ != kInvalidIndex) {
        const aot::TensorId x_hat_id = output_ids_[x_hat_output_pos_];
        auto read_scalar_f16 = [&](aot::TensorId tid, float& val) -> bool {
            if (tid == 0 || tid >= graph.tensors.size())
                return false;
            const auto& td = graph.tensors[tid];
            const auto* wl = std::get_if<aot::WeightLocation>(&td.storage);
            if (!wl || wl->bytes != sizeof(Float16Storage) ||
                wl->offset + wl->bytes > loaded.weights.size())
                return false;
            Float16Storage raw = 0;
            std::memcpy(&raw, loaded.weights.data() + wl->offset, sizeof(raw));
            val = half_to_float(raw);
            return true;
        };
        // Find the schedule step that produces x_hat (may be followed by un-elided CONCAT).
        for (std::size_t si = schedule.size(); si-- > 0;) {
            const FusedStep& fs = schedule[si];
            if (fs.kind != FusedStepKind::kSingleNode)
                continue;
            const aot::Node& prod = graph.nodes[fs.first_node];
            if (prod.outputs.empty() || prod.outputs[0] != x_hat_id)
                continue;
            if (prod.op == aot::Op::DEPTH_TO_SPACE && prod.block_size() == 8) {
                const TensorMeta& d2s_in = require_meta(prod.inputs[0]);
                const TensorMeta& d2s_out = require_meta(prod.outputs[0]);
                if (d2s_in.shape.size() == 4U && d2s_in.shape[1] == 192 &&
                    d2s_out.shape.size() == 4U && d2s_out.shape[1] == 3) {
                    fused_d2s8_output_ = true;
                    d2s8_input_id_ = prod.inputs[0];
                }
            } else if (prod.op == aot::Op::CLIP && si > 0 &&
                       schedule[si - 1].kind == FusedStepKind::kSingleNode) {
                float cmin = -65504.0F;
                float cmax = 65504.0F;
                const bool ok_min = (prod.inputs.size() > 1U && read_scalar_f16(prod.inputs[1], cmin));
                const bool ok_max = (prod.inputs.size() > 2U && read_scalar_f16(prod.inputs[2], cmax));
                const aot::Node& prev = graph.nodes[schedule[si - 1].first_node];
                if (ok_min && ok_max && std::fabs(cmin) < 1e-4F && std::fabs(cmax - 1.0F) < 1e-4F &&
                    prev.op == aot::Op::DEPTH_TO_SPACE && prev.block_size() == 8 &&
                    !prev.outputs.empty() && prev.outputs[0] == prod.inputs[0]) {
                    const TensorMeta& d2s_in = require_meta(prev.inputs[0]);
                    const TensorMeta& d2s_out = require_meta(prod.outputs[0]);
                    if (d2s_in.shape.size() == 4U && d2s_in.shape[1] == 192 &&
                        d2s_out.shape.size() == 4U && d2s_out.shape[1] == 3) {
                        fused_d2s8_output_ = true;
                        d2s8_input_id_ = prev.inputs[0];
                    }
                }
            }
            break;
        }
    }

    build_dispatches(loaded, schedule);

    // Record exact dispatch indices to skip when fused_d2s8_output_ is active.
    if (fused_d2s8_output_ && x_hat_output_pos_ != kInvalidIndex) {
        const aot::TensorId x_hat_id = output_ids_[x_hat_output_pos_];
        // Note: every non-CONCAT FusedStep emits exactly 1 DispatchOp; map schedule -> dispatch index.
        std::size_t d_idx = 0;
        for (std::size_t si = 0; si < schedule.size(); ++si) {
            const FusedStep& fs = schedule[si];
            std::size_t emitted = 1U;
            if (fs.kind == FusedStepKind::kSingleNode) {
                const aot::Node& node = graph.nodes[fs.first_node];
                if (node.op == aot::Op::CONCAT) {
                    emitted = node.inputs.size();
                } else if (node.op == aot::Op::DEPTH_TO_SPACE &&
                           !node.inputs.empty() && node.inputs[0] == d2s8_input_id_) {
                    d2s8_step_index_ = d_idx;
                } else if (node.op == aot::Op::CLIP &&
                           !node.outputs.empty() && node.outputs[0] == x_hat_id) {
                    d2s8_clip_step_index_ = d_idx;
                }
            }
            d_idx += emitted;
        }
        if (d2s8_step_index_ == kInvalidIndex)
            fused_d2s8_output_ = false;
    }

    if (profile_enabled_ && !dispatches_.empty()) {
        const std::uint32_t qcount = static_cast<std::uint32_t>(dispatches_.size() * 2U);
        WGPUQuerySetDescriptor qd = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        qd.label = to_wgpu_string("mlvc_profile_queries");
        qd.type = WGPUQueryType_Timestamp;
        qd.count = qcount;
        profile_query_set_.reset(wgpuDeviceCreateQuerySet(ctx_->device(), &qd));
        const std::uint64_t qbytes = static_cast<std::uint64_t>(qcount) * sizeof(std::uint64_t);
        profile_resolve_buf_ = ctx_->create_buffer(
            qbytes, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc, "profile_resolve");
        profile_readback_buf_ = ctx_->create_buffer(
            qbytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead, "profile_readback");
        profile_pass_labels_.reserve(dispatches_.size());
        profile_pass_cats_.reserve(dispatches_.size());
        for (const auto& d : dispatches_) {
            profile_pass_labels_.push_back(d.label);
            profile_pass_cats_.push_back(d.op_category);
        }
    }

    ctx_->check_error("compile_graph");
}

void WebGpuExecutionPlan::build_dispatches(
    const aot::LoadedGraph& loaded, const std::vector<FusedStep>& schedule)
{
    const auto& graph = loaded.descriptor;
    const bool use_f16 = uses_f16();

    // -------------------------------------------------------------------------
    // 1. Interval-graph coloring into alias-free WGPUBuffer pools.
    //    Strict `<` inequality guarantees no two tensors live during the same
    //    step share a WGPUBuffer (preventing WebGPU read/write aliasing errors).
    // -------------------------------------------------------------------------
    const std::size_t n_steps = schedule.size();
    const std::size_t n_metas = metas_.size();
    std::vector<std::size_t> first_def(n_metas, n_steps + 1U);
    std::vector<std::size_t> last_use(n_metas, 0U);
    std::vector<bool> needs_pool(n_metas, false);

    auto mark_def = [&](aot::TensorId tid, std::size_t s) {
        if (tid == 0)
            return;
        const std::size_t mi = static_cast<std::size_t>(id_to_meta_[tid]);
        const TensorMeta& m = metas_[mi];
        if (m.is_initializer || m.is_graph_input || m.state_output_slot >= 0)
            return;
        needs_pool[mi] = true;
        first_def[mi] = std::min(first_def[mi], s);
        last_use[mi] = std::max(last_use[mi], s);
    };

    auto mark_use = [&](aot::TensorId tid, std::size_t s) {
        if (tid == 0)
            return;
        const std::size_t mi = static_cast<std::size_t>(id_to_meta_[tid]);
        const TensorMeta& m = metas_[mi];
        if (m.is_initializer || m.is_graph_input || m.state_output_slot >= 0)
            return;
        last_use[mi] = std::max(last_use[mi], s);
    };

    for (std::size_t s = 0; s < n_steps; ++s) {
        const FusedStep& fs = schedule[s];
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
            mark_def(fs.y_raw_io, s);
            mark_def(fs.y_concat2_io, s);
            mark_def(fs.y_concat3_or_out, s);
            mark_use(fs.y_latent_or_norm, s);
            mark_use(fs.y_prior, s);
            mark_use(fs.y_quant_scale, s);
            mark_use(fs.y_update_scale0, s);
            mark_use(fs.y_update_scale1, s);
            break;
        case FusedStepKind::kY1Tail:
            mark_def(fs.y_raw_io, s);
            mark_def(fs.y_concat3_or_out, s);
            mark_use(fs.y_latent_or_norm, s);
            mark_use(fs.y_prior, s);
            mark_use(fs.y_quant_scale, s);
            mark_use(fs.y_update_scale0, s);
            mark_use(fs.y_update_scale1, s);
            mark_use(fs.y_concat2_io, s);
            mark_use(fs.y_clipped_io, s);
            break;
        case FusedStepKind::kDecY0Tail:
            mark_def(fs.y_clipped_io, s);
            mark_def(fs.y_concat2_io, s);
            mark_def(fs.y_concat3_or_out, s);
            mark_use(fs.y_prior, s);
            mark_use(fs.y_raw_io, s);
            mark_use(fs.y_update_scale0, s);
            mark_use(fs.y_update_scale1, s);
            break;
        case FusedStepKind::kDecY1Tail:
            mark_def(fs.y_concat3_or_out, s);
            mark_use(fs.y_prior, s);
            mark_use(fs.y_raw_io, s);
            mark_use(fs.y_update_scale0, s);
            mark_use(fs.y_update_scale1, s);
            mark_use(fs.y_concat2_io, s);
            mark_use(fs.y_clipped_io, s);
            break;
        case FusedStepKind::kSingleNode: {
            const aot::Node& node = graph.nodes[fs.first_node];
            for (const aot::TensorId oid : node.outputs)
                mark_def(oid, s);
            for (const aot::TensorId iid : node.inputs)
                mark_use(iid, s);
            break;
        }
        }
    }

    for (std::size_t mi = 0; mi < n_metas; ++mi) {
        if (needs_pool[mi] && (metas_[mi].is_graph_output ||
                               (fused_d2s8_output_ && metas_[mi].id == d2s8_input_id_))) {
            last_use[mi] = n_steps;
        }
    }

    std::vector<std::size_t> order;
    for (std::size_t mi = 0; mi < n_metas; ++mi) {
        if (needs_pool[mi])
            order.push_back(mi);
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (first_def[a] != first_def[b])
            return first_def[a] < first_def[b];
        return metas_[a].byte_size > metas_[b].byte_size;
    });

    struct PoolTrack {
        std::size_t last_use = 0;
        std::size_t max_bytes = 0;
    };
    std::vector<PoolTrack> pools;
    for (const std::size_t mi : order) {
        const std::size_t d0 = first_def[mi];
        const std::size_t d1 = last_use[mi];
        const std::size_t bytes = metas_[mi].byte_size;
        int chosen = -1;
        // Prefer reusing a pool whose last_use < d0 and whose size is closest >= bytes.
        std::size_t best_size = std::numeric_limits<std::size_t>::max();
        for (std::size_t pi = 0; pi < pools.size(); ++pi) {
            if (pools[pi].last_use < d0) {
                if (pools[pi].max_bytes >= bytes && pools[pi].max_bytes < best_size) {
                    best_size = pools[pi].max_bytes;
                    chosen = static_cast<int>(pi);
                }
            }
        }
        if (chosen < 0) {
            for (std::size_t pi = 0; pi < pools.size(); ++pi) {
                if (pools[pi].last_use < d0) {
                    chosen = static_cast<int>(pi);
                    break;
                }
            }
        }
        if (chosen < 0) {
            chosen = static_cast<int>(pools.size());
            pools.push_back({d1, bytes});
        } else {
            pools[static_cast<std::size_t>(chosen)].last_use = d1;
            pools[static_cast<std::size_t>(chosen)].max_bytes =
                std::max(pools[static_cast<std::size_t>(chosen)].max_bytes, bytes);
        }
        metas_[mi].pool_id = chosen;
    }

    pool_buffers_.reserve(pools.size());
    for (std::size_t pi = 0; pi < pools.size(); ++pi) {
        pool_buffers_.push_back(ctx_->create_buffer(
            std::max<std::size_t>(pools[pi].max_bytes, 16U),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
            "pool_" + std::to_string(pi)));
    }

    // -------------------------------------------------------------------------
    // 2. Helper closures to read weight spans & upload constant GPU buffers.
    // -------------------------------------------------------------------------
    auto read_weight_span = [&](aot::TensorId tid) -> std::span<const std::byte> {
        const auto& desc = graph.tensors[tid];
        const auto* wl = std::get_if<aot::WeightLocation>(&desc.storage);
        if (!wl || wl->offset > loaded.weights.size() ||
            wl->bytes > loaded.weights.size() - wl->offset) {
            throw std::runtime_error("webgpu: invalid weight location for tensor " + desc.name);
        }
        return std::span<const std::byte>(loaded.weights.data() + wl->offset, wl->bytes);
    };

    auto read_int64_init = [&](aot::TensorId tid) -> std::vector<std::int64_t> {
        const auto sp = read_weight_span(tid);
        std::vector<std::int64_t> out(sp.size() / sizeof(std::int64_t));
        if (!out.empty())
            std::memcpy(out.data(), sp.data(), sp.size());
        return out;
    };

    auto read_fp16_scalar = [&](aot::TensorId tid) -> float {
        const auto sp = read_weight_span(tid);
        Float16Storage raw = 0;
        std::memcpy(&raw, sp.data(), sizeof(raw));
        return half_to_float(raw);
    };

    auto upload_buffer = [&](const void* data, std::size_t bytes, std::string_view label) -> WGPUBuffer {
        UniqueBuffer b = ctx_->create_buffer(
            std::max<std::size_t>(bytes, 16U),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, label);
        ctx_->write_buffer(b.get(), 0U, data, bytes);
        WGPUBuffer raw = b.get();
        weight_buffers_.push_back(std::move(b));
        return raw;
    };

    auto materialize_init = [&](aot::TensorId tid) -> WGPUBuffer {
        if (tid == 0)
            return dummy_storage_buffer_.get();
        TensorMeta& m = require_meta(tid);
        if (!m.is_initializer)
            return nullptr;
        if (m.const_buffer)
            return m.const_buffer;
        const auto sp = read_weight_span(tid);
        if (m.dtype == aot::DataType::FP16) {
            std::vector<Float16Storage> raw(m.elem_count, 0);
            if (!sp.empty())
                std::memcpy(raw.data(), sp.data(), sp.size());
            if (needs_nhwc_transpose(m.shape, m.dtype)) {
                std::vector<Float16Storage> nhwc(m.elem_count, 0);
                nchw_to_nhwc_host(
                    raw.data(), nhwc.data(),
                    static_cast<std::size_t>(m.shape[0]),
                    static_cast<std::size_t>(m.shape[1]),
                    static_cast<std::size_t>(m.shape[2] * m.shape[3]));
                m.const_buffer = upload_buffer(nhwc.data(), nhwc.size() * sizeof(Float16Storage), m.name);
            } else {
                m.const_buffer = upload_buffer(raw.data(), raw.size() * sizeof(Float16Storage), m.name);
            }
        } else if (m.dtype == aot::DataType::INT32) {
            m.const_buffer = upload_buffer(sp.data(), sp.size(), m.name);
        } else {
            // Convert INT64 -> INT32 for WGSL storage buffer compatibility.
            const auto v64 = read_int64_init(tid);
            std::vector<std::int32_t> v32(v64.size());
            for (std::size_t i = 0; i < v64.size(); ++i)
                v32[i] = static_cast<std::int32_t>(v64[i]);
            m.const_buffer = upload_buffer(v32.data(), v32.size() * sizeof(std::int32_t), m.name);
        }
        return m.const_buffer;
    };

    auto push_dispatch = [&](std::string label, std::string cat,
                             const PipelineCacheEntry& pipe,
                             WGPUBuffer uniform_buf,
                             const std::vector<aot::TensorId>& tensor_ids,
                             const std::vector<WGPUBuffer>& explicit_bufs,
                             std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) {
        DispatchOp op;
        op.label = std::move(label);
        op.op_category = std::move(cat);
        op.pipeline = pipe.pipeline.get();
        op.wg_x = gx;
        op.wg_y = gy;
        op.wg_z = gz;
        for (int parity = 0; parity < 2; ++parity) {
            std::vector<WGPUBuffer> bufs;
            bufs.reserve(1U + tensor_ids.size() + explicit_bufs.size());
            bufs.push_back(uniform_buf);
            for (const auto& eb : explicit_bufs)
                bufs.push_back(eb);
            for (const aot::TensorId tid : tensor_ids)
                bufs.push_back(resolve_buffer(tid, parity));
            op.bind_groups[static_cast<std::size_t>(parity)] =
                create_bind_group(pipe.bgl.get(), bufs);
        }
        dispatches_.push_back(std::move(op));
    };

    // Helper for compiling a CONV (fused or unfused) into optimal WGSL pipeline.
    auto emit_conv = [&](const aot::Node& node, aot::TensorId in_id,
                         aot::TensorId out_id, EpilogueKind epi,
                         float alpha, float cmin, float cmax,
                         aot::TensorId res_id) {
        materialize_init(in_id);
        materialize_init(res_id);
        const TensorMeta& in_m = require_meta(in_id);
        const TensorMeta& w_m = require_meta(node.inputs[1]);
        const TensorMeta& out_m = require_meta(out_id);
        const auto& conv = node.conv();

        const auto id4 = aot::dims_to_4d(in_m.shape);
        const auto od4 = aot::dims_to_4d(out_m.shape);
        const auto wd4 = aot::dims_to_4d(w_m.shape);
        const std::uint32_t n = static_cast<std::uint32_t>(id4[0]);
        const std::uint32_t ic = static_cast<std::uint32_t>(id4[1]);
        const std::uint32_t ih = static_cast<std::uint32_t>(id4[2]);
        const std::uint32_t iw = static_cast<std::uint32_t>(id4[3]);
        const std::uint32_t oc_conv = static_cast<std::uint32_t>(wd4[0]);
        const std::uint32_t oh = static_cast<std::uint32_t>(od4[2]);
        const std::uint32_t ow = static_cast<std::uint32_t>(od4[3]);
        const std::uint32_t kh = static_cast<std::uint32_t>(wd4[2]);
        const std::uint32_t kw = static_cast<std::uint32_t>(wd4[3]);
        const std::uint32_t sh = static_cast<std::uint32_t>(conv.strides[0]);
        const std::uint32_t sw = static_cast<std::uint32_t>(conv.strides[1]);
        const std::uint32_t ph = static_cast<std::uint32_t>(conv.pads[0]);
        const std::uint32_t pw = static_cast<std::uint32_t>(conv.pads[1]);
        const std::uint32_t dh = static_cast<std::uint32_t>(conv.dilations[0]);
        const std::uint32_t dw = static_cast<std::uint32_t>(conv.dilations[1]);
        const std::uint32_t groups = static_cast<std::uint32_t>(conv.groups);

        const auto w_span = read_weight_span(node.inputs[1]);
        std::vector<Float16Storage> w_raw(w_m.elem_count, 0);
        std::memcpy(w_raw.data(), w_span.data(), w_span.size());

        const bool has_bias = node.inputs.size() >= 3U && node.inputs[2] != 0;
        std::vector<Float16Storage> b_raw;
        if (has_bias) {
            const auto b_span = read_weight_span(node.inputs[2]);
            b_raw.resize(oc_conv, 0);
            std::memcpy(b_raw.data(), b_span.data(), b_span.size());
        }

        const bool is_dw_3x3_s1p1 =
            (epi == EpilogueKind::kNone) && (groups == ic) && (oc_conv == ic) &&
            (kh == 3U && kw == 3U && sh == 1U && sw == 1U &&
             ph == 1U && pw == 1U && dh == 1U && dw == 1U) &&
            use_f16 && ((ic % 4U) == 0U);

        const bool is_pointwise_fast =
            (groups == 1U) && (kh == 1U && kw == 1U && sh == 1U && sw == 1U &&
                               ph == 0U && pw == 0U && dh == 1U && dw == 1U) &&
            ((ic % 4U) == 0U) && ((oc_conv % 4U) == 0U) &&
            (epi != EpilogueKind::kReglu || (oc_conv % 8U) == 0U);

        const bool is_spatial_fast =
            !is_pointwise_fast && (groups == 1U) && (epi != EpilogueKind::kReglu) &&
            use_f16 && ((oc_conv % 4U) == 0U);

        if (is_dw_3x3_s1p1) {
            const auto packed_w = pack_depthwise_3x3(w_raw.data(), ic);
            WGPUBuffer w_buf = upload_buffer(
                packed_w.data(), packed_w.size() * sizeof(Float16Storage),
                "dw3x3_w_" + std::to_string(node.index));
            WGPUBuffer b_buf = has_bias
                ? upload_buffer(b_raw.data(), b_raw.size() * sizeof(Float16Storage),
                                "dw3x3_b_" + std::to_string(node.index))
                : dummy_storage_buffer_.get();
            const std::uint32_t c4 = ic / 4U;
            const std::uint32_t c_chunks = div_up_u32(c4, 16U);
            struct DwParams {
                std::uint32_t n, h, w, c4, has_bias, c_chunks, p0, p1;
            } params{n, ih, iw, c4, has_bias ? 1U : 0U, c_chunks, 0U, 0U};
            UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "dw3x3_u");
            WGPUBuffer ub_raw = ub.get();
            uniform_buffers_.push_back(std::move(ub));
            const auto& pipe = get_or_create_pipeline(
                "depthwise_3x3_s1p1", wgsl_depthwise_3x3_s1p1(use_f16),
                make_binding_types(3U, 1U));
            DispatchOp op;
            op.label = "dw3x3#" + std::to_string(node.index) +
                "[" + std::to_string(ih) + "x" + std::to_string(iw) + "x" + std::to_string(ic) + "]";
            op.op_category = "depthwise_3x3";
            op.pipeline = pipe.pipeline.get();
            const std::uint32_t strips_w = div_up_u32(iw, 4U);
            op.wg_x = div_up_u32(strips_w, 4U);
            op.wg_y = div_up_u32(ih, 2U);
            op.wg_z = n * c_chunks;
            for (int p = 0; p < 2; ++p) {
                op.bind_groups[static_cast<std::size_t>(p)] = create_bind_group(
                    pipe.bgl.get(),
                    {ub_raw, resolve_buffer(in_id, p), w_buf, b_buf, resolve_buffer(out_id, p)});
            }
            dispatches_.push_back(std::move(op));
            return;
        }

        if (is_pointwise_fast) {
            const bool is_reglu = (epi == EpilogueKind::kReglu);
            const std::uint32_t m_rows = n * oh * ow;
            // Empirically tuned on RTX 4060 Ti:
            //  - Standard pointwise: 64x64 keeps register pressure & occupancy optimal across N=96..384.
            //  - Fused ReGLU: 64x128 (BG=64, 4 rows/thread) halves global A traffic when N%128==0 (N=256/384/768/1024).
            const bool allow_wide = !mlvc::detail::env_flag_enabled("MLVC_WEBGPU_DISABLE_WIDE_TILE");
            const bool use_wide = allow_wide && is_reglu && (oc_conv >= 128U) && ((oc_conv % 128U) == 0U);
            const int tile_variant = !use_f16 ? 1 : (m_rows < 64U ? 0 : (use_wide ? 2 : 1));
            const auto packed_w = is_reglu
                ? pack_pointwise_reglu_kn(w_raw.data(), oc_conv, ic)
                : pack_pointwise_kn(w_raw.data(), oc_conv, ic);
            WGPUBuffer w_buf = upload_buffer(
                packed_w.data(), packed_w.size() * sizeof(Float16Storage),
                "pw_w_" + std::to_string(node.index));
            WGPUBuffer b_buf = dummy_storage_buffer_.get();
            if (has_bias) {
                if (is_reglu) {
                    const auto pb = pack_reglu_bias(b_raw.data(), oc_conv);
                    b_buf = upload_buffer(pb.data(), pb.size() * sizeof(Float16Storage),
                                          "pw_b_" + std::to_string(node.index));
                } else {
                    b_buf = upload_buffer(b_raw.data(), b_raw.size() * sizeof(Float16Storage),
                                          "pw_b_" + std::to_string(node.index));
                }
            }
            struct GemmParams {
                std::uint32_t m, k, n, has_bias;
                std::uint32_t epilogue, p0, p1, p2;
                float alpha, clamp_min, clamp_max, p3;
            } params{m_rows, ic, oc_conv, has_bias ? 1U : 0U,
                     static_cast<std::uint32_t>(epi), 0U, 0U, 0U,
                     alpha, cmin, cmax, 0.0F};
            UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "pw_u");
            WGPUBuffer ub_raw = ub.get();
            uniform_buffers_.push_back(std::move(ub));
            static constexpr const char* kTileTag[3] = {"sm", "md", "wide"};
            const std::string key = std::string("pw_gemm_") +
                (is_reglu ? "reglu_" : "std_") + kTileTag[tile_variant];
            const auto& pipe = get_or_create_pipeline(
                key, wgsl_pointwise_gemm(use_f16, tile_variant, is_reglu),
                make_binding_types(4U, 1U));
            DispatchOp op;
            op.label = (is_reglu ? "pw_reglu#" : "pw_conv#") + std::to_string(node.index) +
                "[" + std::to_string(m_rows) + "x" + std::to_string(ic) + "x" + std::to_string(oc_conv) + "]";
            op.op_category = is_reglu ? "pointwise_reglu" : "pointwise_1x1";
            op.pipeline = pipe.pipeline.get();
            if (!use_f16) {
                if (is_reglu) {
                    op.wg_x = div_up_u32(oc_conv / 8U, 8U);
                    op.wg_y = div_up_u32(m_rows, 16U);
                } else {
                    op.wg_x = div_up_u32(oc_conv / 4U, 16U);
                    op.wg_y = div_up_u32(m_rows, 16U);
                }
            } else if (is_reglu) {
                const std::uint32_t g4 = oc_conv / 8U;
                op.wg_x = div_up_u32(g4, tile_variant == 2 ? 16U : 8U);
                op.wg_y = div_up_u32(m_rows, tile_variant == 0 ? 16U : 64U);
            } else {
                const std::uint32_t n4 = oc_conv / 4U;
                op.wg_x = div_up_u32(n4, tile_variant == 2 ? 32U : 16U);
                op.wg_y = div_up_u32(m_rows, tile_variant == 0 ? 16U : 64U);
            }
            op.wg_z = 1U;
            for (int p = 0; p < 2; ++p) {
                op.bind_groups[static_cast<std::size_t>(p)] = create_bind_group(
                    pipe.bgl.get(),
                    {ub_raw, resolve_buffer(in_id, p), w_buf, b_buf,
                     resolve_buffer(res_id, p), resolve_buffer(out_id, p)});
            }
            dispatches_.push_back(std::move(op));
            return;
        }

        if (is_spatial_fast) {
            const auto packed_w = pack_spatial_kn(w_raw.data(), oc_conv, ic, kh, kw);
            WGPUBuffer w_buf = upload_buffer(
                packed_w.data(), packed_w.size() * sizeof(Float16Storage),
                "sp_w_" + std::to_string(node.index));
            WGPUBuffer b_buf = has_bias
                ? upload_buffer(b_raw.data(), b_raw.size() * sizeof(Float16Storage),
                                "sp_b_" + std::to_string(node.index))
                : dummy_storage_buffer_.get();
            const std::uint32_t m_rows = n * oh * ow;
            const std::uint32_t k_total = kh * kw * ic;
            struct SpParams {
                std::uint32_t n, ic, ih, iw;
                std::uint32_t oc, oh, ow, k_total;
                std::uint32_t kh, kw, sh, sw;
                std::uint32_t ph, pw, dh, dw;
                std::uint32_t has_bias, epilogue, p0, p1;
                float alpha, clamp_min, clamp_max, p2;
            } params{n, ic, ih, iw, oc_conv, oh, ow, k_total,
                     kh, kw, sh, sw, ph, pw, dh, dw,
                     has_bias ? 1U : 0U, static_cast<std::uint32_t>(epi), 0U, 0U,
                     alpha, cmin, cmax, 0.0F};
            UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "sp_u");
            WGPUBuffer ub_raw = ub.get();
            uniform_buffers_.push_back(std::move(ub));
            const auto& pipe = get_or_create_pipeline(
                "spatial_conv_gemm", wgsl_spatial_conv_gemm(use_f16),
                make_binding_types(4U, 1U));
            DispatchOp op;
            op.label = "spatial_conv#" + std::to_string(node.index);
            op.op_category = "spatial_conv";
            op.pipeline = pipe.pipeline.get();
            op.wg_x = div_up_u32(oc_conv / 4U, 16U);
            op.wg_y = div_up_u32(m_rows, 64U);
            op.wg_z = 1U;
            for (int p = 0; p < 2; ++p) {
                op.bind_groups[static_cast<std::size_t>(p)] = create_bind_group(
                    pipe.bgl.get(),
                    {ub_raw, resolve_buffer(in_id, p), w_buf, b_buf,
                     resolve_buffer(res_id, p), resolve_buffer(out_id, p)});
            }
            dispatches_.push_back(std::move(op));
            return;
        }

        // Generic conv fallback (arbitrary groups/channels/non-f16 spatial/dw).
        WGPUBuffer w_buf = upload_buffer(
            w_raw.data(), w_raw.size() * sizeof(Float16Storage),
            "gconv_w_" + std::to_string(node.index));
        WGPUBuffer b_buf = has_bias
            ? upload_buffer(b_raw.data(), b_raw.size() * sizeof(Float16Storage),
                            "gconv_b_" + std::to_string(node.index))
            : dummy_storage_buffer_.get();
        struct GenParams {
            std::uint32_t n, ic, ih, iw;
            std::uint32_t oc, oh, ow, groups;
            std::uint32_t kh, kw, sh, sw;
            std::uint32_t ph, pw, dh, dw;
            std::uint32_t has_bias, epilogue, p0, p1;
            float alpha, clamp_min, clamp_max, p2;
        } params{n, ic, ih, iw, oc_conv, oh, ow, groups,
                 kh, kw, sh, sw, ph, pw, dh, dw,
                 has_bias ? 1U : 0U, static_cast<std::uint32_t>(epi), 0U, 0U,
                 alpha, cmin, cmax, 0.0F};
        UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "gconv_u");
        WGPUBuffer ub_raw = ub.get();
        uniform_buffers_.push_back(std::move(ub));
        const auto& pipe = get_or_create_pipeline(
            "conv_generic", wgsl_conv_generic_nhwc(use_f16),
            make_binding_types(4U, 1U));
        const std::size_t total = static_cast<std::size_t>(n) * oh * ow * oc_conv;
        const std::size_t work = use_f16 ? total : ((total + 1U) / 2U);
        DispatchOp op;
        op.label = "conv_generic#" + std::to_string(node.index);
        op.op_category = (groups == ic) ? "depthwise_generic" : "conv_generic";
        op.pipeline = pipe.pipeline.get();
        op.wg_x = div_up_u32(work, 256U);
        op.wg_y = 1U;
        op.wg_z = 1U;
        for (int p = 0; p < 2; ++p) {
            op.bind_groups[static_cast<std::size_t>(p)] = create_bind_group(
                pipe.bgl.get(),
                {ub_raw, resolve_buffer(in_id, p), w_buf, b_buf,
                 resolve_buffer(res_id, p), resolve_buffer(out_id, p)});
        }
        dispatches_.push_back(std::move(op));
    };

    // -------------------------------------------------------------------------
    // 3. Lower each FusedStep into pre-bound GPU DispatchOps.
    // -------------------------------------------------------------------------
    for (std::size_t s = 0; s < n_steps; ++s) {
        const FusedStep& fs = schedule[s];
        switch (fs.kind) {
        case FusedStepKind::kConvWithEpilogue: {
            const aot::Node& node = graph.nodes[fs.first_node];
            emit_conv(node, fs.conv_input, fs.output_tensor, fs.epilogue_kind,
                      fs.alpha, fs.clamp_min, fs.clamp_max, fs.residual_input);
            break;
        }
        case FusedStepKind::kReglu: {
            materialize_init(fs.reglu_input);
            const TensorMeta& om = require_meta(fs.output_tensor);
            const std::uint32_t spatial = static_cast<std::uint32_t>(
                om.elem_count / fs.reglu_gate_channels);
            struct RegluParams {
                std::uint32_t spatial, gate_c, p0, p1;
                float cmin, cmax, p2, p3;
            } params{spatial, static_cast<std::uint32_t>(fs.reglu_gate_channels), 0U, 0U,
                     fs.clamp_min, fs.clamp_max, 0.0F, 0.0F};
            UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "reglu_u");
            WGPUBuffer ub_raw = ub.get();
            uniform_buffers_.push_back(std::move(ub));
            const auto& pipe = get_or_create_pipeline(
                "reglu_nhwc", wgsl_reglu_nhwc(use_f16), make_binding_types(1U, 1U));
            const std::size_t work = use_f16 ? om.elem_count : ((om.elem_count + 1U) / 2U);
            push_dispatch("reglu#" + std::to_string(fs.first_node), "reglu",
                          pipe, ub_raw, {fs.reglu_input, fs.output_tensor}, {},
                          div_up_u32(work, 256U), 1U, 1U);
            break;
        }
        case FusedStepKind::kFeatureUpdate:
        case FusedStepKind::kFeatureUpdateOutputs: {
            materialize_init(fs.fu_input);
            materialize_init(fs.fu_history);
            materialize_init(fs.fu_scale);
            const bool emit_sides = (fs.kind == FusedStepKind::kFeatureUpdateOutputs);
            const TensorMeta& hm = require_meta(fs.fu_history);
            const std::uint32_t spatial = static_cast<std::uint32_t>(
                hm.elem_count / fs.fu_channels);
            struct FuParams { std::uint32_t spatial, channels, p0, p1; } params{
                spatial, static_cast<std::uint32_t>(fs.fu_channels), 0U, 0U};
            UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "fu_u");
            WGPUBuffer ub_raw = ub.get();
            uniform_buffers_.push_back(std::move(ub));
            const std::string key = emit_sides ? "feature_update_outputs" : "feature_update";
            const auto& pipe = get_or_create_pipeline(
                key, wgsl_feature_update_nhwc(use_f16, emit_sides),
                make_binding_types(3U, emit_sides ? 3U : 1U));
            const std::size_t work = use_f16 ? hm.elem_count : (hm.elem_count / 2U);
            std::vector<aot::TensorId> tids = emit_sides
                ? std::vector<aot::TensorId>{fs.fu_input, fs.fu_history, fs.fu_scale,
                                             fs.fu_scaled_out, fs.fu_blended_out, fs.fu_feature_out}
                : std::vector<aot::TensorId>{fs.fu_input, fs.fu_history, fs.fu_scale,
                                             fs.fu_feature_out};
            push_dispatch(key + "#" + std::to_string(fs.first_node), "feature_update",
                          pipe, ub_raw, tids, {}, div_up_u32(work, 256U), 1U, 1U);
            break;
        }
        case FusedStepKind::kY0Tail: {
            materialize_init(fs.y_latent_or_norm);
            materialize_init(fs.y_prior);
            materialize_init(fs.y_quant_scale);
            materialize_init(fs.y_update_scale0);
            materialize_init(fs.y_update_scale1);
            const TensorMeta& rm = require_meta(fs.y_raw_io);
            const std::uint32_t spatial = static_cast<std::uint32_t>(
                rm.elem_count / fs.y_quant_channels);
            struct Y0Params {
                std::uint32_t spatial, q, p0, p1;
                float cmin, cmax, p2, p3;
            } params{spatial, static_cast<std::uint32_t>(fs.y_quant_channels), 0U, 0U,
                     fs.y_clip_min, fs.y_clip_max, 0.0F, 0.0F};
            UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "y0_tail_u");
            WGPUBuffer ub_raw = ub.get();
            uniform_buffers_.push_back(std::move(ub));
            const auto& pipe = get_or_create_pipeline(
                "y0_tail", wgsl_y0_tail_nhwc(use_f16), make_binding_types(5U, 5U));
            const std::size_t work = use_f16 ? rm.elem_count : (rm.elem_count / 2U);
            push_dispatch("y0_tail#" + std::to_string(fs.first_node), "quant_tail",
                          pipe, ub_raw,
                          {fs.y_latent_or_norm, fs.y_prior, fs.y_quant_scale,
                           fs.y_update_scale0, fs.y_update_scale1,
                           fs.y_normalized_out, fs.y_clipped_io, fs.y_raw_io,
                           fs.y_concat2_io, fs.y_concat3_or_out},
                          {}, div_up_u32(work, 256U), 1U, 1U);
            break;
        }
        case FusedStepKind::kY1Tail: {
            materialize_init(fs.y_latent_or_norm);
            materialize_init(fs.y_prior);
            materialize_init(fs.y_quant_scale);
            materialize_init(fs.y_update_scale0);
            materialize_init(fs.y_update_scale1);
            materialize_init(fs.y_concat2_io);
            materialize_init(fs.y_clipped_io);
            const TensorMeta& rm = require_meta(fs.y_raw_io);
            const std::uint32_t spatial = static_cast<std::uint32_t>(
                rm.elem_count / fs.y_quant_channels);
            struct Y1Params { std::uint32_t spatial, q, p0, p1; } params{
                spatial, static_cast<std::uint32_t>(fs.y_quant_channels), 0U, 0U};
            UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "y1_tail_u");
            WGPUBuffer ub_raw = ub.get();
            uniform_buffers_.push_back(std::move(ub));
            const auto& pipe = get_or_create_pipeline(
                "y1_tail", wgsl_y1_tail_nhwc(use_f16), make_binding_types(7U, 2U));
            const std::size_t work = use_f16 ? rm.elem_count : (rm.elem_count / 2U);
            push_dispatch("y1_tail#" + std::to_string(fs.first_node), "quant_tail",
                          pipe, ub_raw,
                          {fs.y_latent_or_norm, fs.y_prior, fs.y_quant_scale,
                           fs.y_update_scale0, fs.y_update_scale1,
                           fs.y_concat2_io, fs.y_clipped_io,
                           fs.y_raw_io, fs.y_concat3_or_out},
                          {}, div_up_u32(work, 256U), 1U, 1U);
            break;
        }
        case FusedStepKind::kDecY0Tail: {
            materialize_init(fs.y_prior);
            materialize_init(fs.y_raw_io);
            materialize_init(fs.y_update_scale0);
            materialize_init(fs.y_update_scale1);
            const TensorMeta& rm = require_meta(fs.y_raw_io);
            const std::uint32_t spatial = static_cast<std::uint32_t>(
                rm.elem_count / fs.y_quant_channels);
            struct DecY0Params {
                std::uint32_t spatial, q, p0, p1;
                float cmin, cmax, p2, p3;
            } params{spatial, static_cast<std::uint32_t>(fs.y_quant_channels), 0U, 0U,
                     fs.y_clip_min, fs.y_clip_max, 0.0F, 0.0F};
            UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "dec_y0_u");
            WGPUBuffer ub_raw = ub.get();
            uniform_buffers_.push_back(std::move(ub));
            const auto& pipe = get_or_create_pipeline(
                "dec_y0_tail", wgsl_dec_y0_tail_nhwc(use_f16), make_binding_types(4U, 3U));
            const std::size_t work = use_f16 ? rm.elem_count : (rm.elem_count / 2U);
            push_dispatch("dec_y0_tail#" + std::to_string(fs.first_node), "quant_tail",
                          pipe, ub_raw,
                          {fs.y_prior, fs.y_raw_io, fs.y_update_scale0, fs.y_update_scale1,
                           fs.y_clipped_io, fs.y_concat2_io, fs.y_concat3_or_out},
                          {}, div_up_u32(work, 256U), 1U, 1U);
            break;
        }
        case FusedStepKind::kDecY1Tail: {
            materialize_init(fs.y_prior);
            materialize_init(fs.y_raw_io);
            materialize_init(fs.y_update_scale0);
            materialize_init(fs.y_update_scale1);
            materialize_init(fs.y_concat2_io);
            materialize_init(fs.y_clipped_io);
            const TensorMeta& rm = require_meta(fs.y_raw_io);
            const std::uint32_t spatial = static_cast<std::uint32_t>(
                rm.elem_count / fs.y_quant_channels);
            struct DecY1Params { std::uint32_t spatial, q, p0, p1; } params{
                spatial, static_cast<std::uint32_t>(fs.y_quant_channels), 0U, 0U};
            UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "dec_y1_u");
            WGPUBuffer ub_raw = ub.get();
            uniform_buffers_.push_back(std::move(ub));
            const auto& pipe = get_or_create_pipeline(
                "dec_y1_tail", wgsl_dec_y1_tail_nhwc(use_f16), make_binding_types(6U, 1U));
            const std::size_t work = use_f16 ? rm.elem_count : (rm.elem_count / 2U);
            push_dispatch("dec_y1_tail#" + std::to_string(fs.first_node), "quant_tail",
                          pipe, ub_raw,
                          {fs.y_prior, fs.y_raw_io, fs.y_update_scale0, fs.y_update_scale1,
                           fs.y_concat2_io, fs.y_clipped_io, fs.y_concat3_or_out},
                          {}, div_up_u32(work, 256U), 1U, 1U);
            break;
        }
        case FusedStepKind::kSingleNode: {
            const aot::Node& node = graph.nodes[fs.first_node];
            switch (node.op) {
            case aot::Op::CONV:
                emit_conv(node, node.inputs[0], node.outputs[0],
                          EpilogueKind::kNone, 0.0F, 0.0F, 0.0F, 0);
                break;
            case aot::Op::LEAKY_RELU:
            case aot::Op::CLIP:
            case aot::Op::SIGMOID:
            case aot::Op::RECIPROCAL:
            case aot::Op::ROUND: {
                materialize_init(node.inputs[0]);
                const TensorMeta& om = require_meta(node.outputs[0]);
                std::uint32_t op_kind = 0U;
                float alpha = 0.0F;
                float cmin = -65504.0F;
                float cmax = 65504.0F;
                if (node.op == aot::Op::LEAKY_RELU) {
                    op_kind = 0U;
                    alpha = node.alpha_or_default();
                } else if (node.op == aot::Op::CLIP) {
                    op_kind = 1U;
                    if (node.inputs.size() > 1U && node.inputs[1] != 0)
                        cmin = read_fp16_scalar(node.inputs[1]);
                    if (node.inputs.size() > 2U && node.inputs[2] != 0)
                        cmax = read_fp16_scalar(node.inputs[2]);
                } else if (node.op == aot::Op::SIGMOID) {
                    op_kind = 2U;
                } else if (node.op == aot::Op::RECIPROCAL) {
                    op_kind = 3U;
                } else {
                    op_kind = 4U;
                }
                const bool use_vec4 = use_f16 && ((om.elem_count % 4U) == 0U);
                struct UnaryParams {
                    std::uint32_t count, op_kind, p0, p1;
                    float alpha, cmin, cmax, p2;
                } params{static_cast<std::uint32_t>(use_vec4 ? (om.elem_count / 4U) : om.elem_count),
                         op_kind, 0U, 0U, alpha, cmin, cmax, 0.0F};
                UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "unary_u");
                WGPUBuffer ub_raw = ub.get();
                uniform_buffers_.push_back(std::move(ub));
                const auto& pipe = use_vec4
                    ? get_or_create_pipeline("unary_vec4", wgsl_unary_vec4(), make_binding_types(1U, 1U))
                    : get_or_create_pipeline("unary", wgsl_unary(use_f16), make_binding_types(1U, 1U));
                const std::size_t work = use_vec4 ? (om.elem_count / 4U)
                    : (use_f16 ? om.elem_count : ((om.elem_count + 1U) / 2U));
                push_dispatch("unary#" + std::to_string(node.index), "elementwise",
                              pipe, ub_raw, {node.inputs[0], node.outputs[0]}, {},
                              div_up_u32(work, 256U), 1U, 1U);
                break;
            }
            case aot::Op::ADD:
            case aot::Op::SUB:
            case aot::Op::MUL: {
                materialize_init(node.inputs[0]);
                materialize_init(node.inputs[1]);
                const TensorMeta& m0 = require_meta(node.inputs[0]);
                const TensorMeta& m1 = require_meta(node.inputs[1]);
                const TensorMeta& mo = require_meta(node.outputs[0]);
                const std::uint32_t op_kind =
                    node.op == aot::Op::ADD ? 0U : (node.op == aot::Op::SUB ? 1U : 2U);
                std::uint32_t mode = 5U;
                std::uint32_t channels = 1U;
                if (m0.shape == mo.shape && m1.shape == mo.shape) {
                    mode = 0U;
                } else if (m1.elem_count == 1U) {
                    mode = 1U;
                } else if (m0.elem_count == 1U) {
                    mode = 2U;
                } else if (mo.shape.size() == 4U) {
                    const std::vector<std::int64_t> ch_shape{1, mo.shape[1], 1, 1};
                    if (m0.shape == mo.shape && m1.shape == ch_shape) {
                        mode = 3U;
                        channels = static_cast<std::uint32_t>(mo.shape[1]);
                    } else if (m1.shape == mo.shape && m0.shape == ch_shape) {
                        mode = 4U;
                        channels = static_cast<std::uint32_t>(mo.shape[1]);
                    }
                }
                const bool use_vec4 = use_f16 && ((mo.elem_count % 4U) == 0U) &&
                    (mode <= 2U || ((mode == 3U || mode == 4U) && ((channels % 4U) == 0U)));
                if (use_vec4) {
                    struct BinV4Params {
                        std::uint32_t total4, op_kind, mode, channels4;
                    } params{
                        static_cast<std::uint32_t>(mo.elem_count / 4U), op_kind, mode,
                        (mode == 3U || mode == 4U) ? (channels / 4U) : 1U};
                    UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "binary4_u");
                    WGPUBuffer ub_raw = ub.get();
                    uniform_buffers_.push_back(std::move(ub));
                    const auto& pipe = get_or_create_pipeline(
                        "binary_vec4", wgsl_binary_vec4(), make_binding_types(2U, 1U));
                    push_dispatch("binary#" + std::to_string(node.index), "elementwise",
                                  pipe, ub_raw, {node.inputs[0], node.inputs[1], node.outputs[0]}, {},
                                  div_up_u32(mo.elem_count / 4U, 256U), 1U, 1U);
                    break;
                }
                const auto od4 = aot::pad_dims_to_4d(mo.shape);
                const auto ad4 = aot::pad_dims_to_4d(m0.shape);
                const auto bd4 = aot::pad_dims_to_4d(m1.shape);
                struct BinParams {
                    std::uint32_t total, op_kind, mode, channels;
                    std::uint32_t on, oc, oh, ow;
                    std::uint32_t an, ac, ah, aw;
                    std::uint32_t bn, bc, bh, bw;
                } params{
                    static_cast<std::uint32_t>(mo.elem_count), op_kind, mode, channels,
                    static_cast<std::uint32_t>(od4[0]), static_cast<std::uint32_t>(od4[1]),
                    static_cast<std::uint32_t>(od4[2]), static_cast<std::uint32_t>(od4[3]),
                    static_cast<std::uint32_t>(ad4[0]), static_cast<std::uint32_t>(ad4[1]),
                    static_cast<std::uint32_t>(ad4[2]), static_cast<std::uint32_t>(ad4[3]),
                    static_cast<std::uint32_t>(bd4[0]), static_cast<std::uint32_t>(bd4[1]),
                    static_cast<std::uint32_t>(bd4[2]), static_cast<std::uint32_t>(bd4[3])};
                UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "binary_u");
                WGPUBuffer ub_raw = ub.get();
                uniform_buffers_.push_back(std::move(ub));
                const auto& pipe = get_or_create_pipeline(
                    "binary", wgsl_binary(use_f16), make_binding_types(2U, 1U));
                const std::size_t work = use_f16 ? mo.elem_count : ((mo.elem_count + 1U) / 2U);
                push_dispatch("binary#" + std::to_string(node.index), "elementwise",
                              pipe, ub_raw, {node.inputs[0], node.inputs[1], node.outputs[0]}, {},
                              div_up_u32(work, 256U), 1U, 1U);
                break;
            }
            case aot::Op::GATHER: {
                materialize_init(node.inputs[0]);
                materialize_init(node.inputs[1]);
                const TensorMeta& dm = require_meta(node.inputs[0]);
                const TensorMeta& im = require_meta(node.inputs[1]);
                const TensorMeta& om = require_meta(node.outputs[0]);
                const int rank = static_cast<int>(dm.shape.size());
                int norm_axis = node.axis_or_zero();
                if (norm_axis < 0)
                    norm_axis += rank;
                std::array<std::int64_t, 4> phys_shape{1, 1, 1, 1};
                int p_ax = norm_axis;
                if (rank == 4) {
                    phys_shape = {dm.shape[0], dm.shape[2], dm.shape[3], dm.shape[1]};
                    static constexpr int kMap[4] = {0, 3, 1, 2};
                    p_ax = kMap[norm_axis];
                } else {
                    for (int d = 0; d < rank; ++d)
                        phys_shape[static_cast<std::size_t>(d)] = dm.shape[static_cast<std::size_t>(d)];
                }
                std::uint32_t outer = 1U;
                for (int d = 0; d < p_ax; ++d)
                    outer *= static_cast<std::uint32_t>(phys_shape[static_cast<std::size_t>(d)]);
                const std::uint32_t axis_dim = static_cast<std::uint32_t>(phys_shape[static_cast<std::size_t>(p_ax)]);
                std::uint32_t inner = 1U;
                for (int d = p_ax + 1; d < rank; ++d)
                    inner *= static_cast<std::uint32_t>(phys_shape[static_cast<std::size_t>(d)]);
                struct GatherParams { std::uint32_t outer, axis_dim, idx_count, inner; } params{
                    outer, axis_dim, static_cast<std::uint32_t>(im.elem_count), inner};
                UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "gather_u");
                WGPUBuffer ub_raw = ub.get();
                uniform_buffers_.push_back(std::move(ub));
                const auto& pipe = get_or_create_pipeline(
                    "gather", wgsl_gather(use_f16), make_binding_types(2U, 1U));
                const std::size_t work = use_f16 ? om.elem_count : ((om.elem_count + 1U) / 2U);
                push_dispatch("gather#" + std::to_string(node.index), "tensor_ops",
                              pipe, ub_raw, {node.inputs[0], node.inputs[1], node.outputs[0]}, {},
                              div_up_u32(work, 256U), 1U, 1U);
                break;
            }
            case aot::Op::SLICE: {
                materialize_init(node.inputs[0]);
                const TensorMeta& im = require_meta(node.inputs[0]);
                const TensorMeta& om = require_meta(node.outputs[0]);
                const auto id4 = aot::dims_to_4d(im.shape);
                const auto od4 = aot::dims_to_4d(om.shape);
                std::array<std::uint32_t, 4> starts{0, 0, 0, 0};
                std::array<std::uint32_t, 4> steps{1, 1, 1, 1};
                const auto st_vals = read_int64_init(node.inputs[1]);
                std::vector<std::int64_t> ax_vals;
                if (node.inputs.size() > 3U && node.inputs[3] != 0) {
                    ax_vals = read_int64_init(node.inputs[3]);
                } else {
                    ax_vals.resize(st_vals.size());
                    std::iota(ax_vals.begin(), ax_vals.end(), 0);
                }
                std::vector<std::int64_t> step_vals(st_vals.size(), 1);
                if (node.inputs.size() > 4U && node.inputs[4] != 0)
                    step_vals = read_int64_init(node.inputs[4]);
                for (std::size_t a = 0; a < st_vals.size(); ++a) {
                    int ax = static_cast<int>(ax_vals[a]);
                    if (ax < 0)
                        ax += 4;
                    std::int64_t st = st_vals[a];
                    const std::int64_t dim = im.shape[static_cast<std::size_t>(ax)];
                    if (st < 0)
                        st += dim;
                    st = std::clamp<std::int64_t>(st, 0, dim);
                    starts[static_cast<std::size_t>(ax)] = static_cast<std::uint32_t>(st);
                    steps[static_cast<std::size_t>(ax)] = static_cast<std::uint32_t>(step_vals[a]);
                }
                struct SliceParams {
                    std::uint32_t in_n, in_c, in_h, in_w;
                    std::uint32_t on, oc, oh, ow;
                    std::uint32_t sn, sc, sh, sw;
                    std::uint32_t stn, stc, sth, stw;
                } params{
                    static_cast<std::uint32_t>(id4[0]), static_cast<std::uint32_t>(id4[1]),
                    static_cast<std::uint32_t>(id4[2]), static_cast<std::uint32_t>(id4[3]),
                    static_cast<std::uint32_t>(od4[0]), static_cast<std::uint32_t>(od4[1]),
                    static_cast<std::uint32_t>(od4[2]), static_cast<std::uint32_t>(od4[3]),
                    starts[0], starts[1], starts[2], starts[3],
                    steps[0], steps[1], steps[2], steps[3]};
                UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "slice_u");
                WGPUBuffer ub_raw = ub.get();
                uniform_buffers_.push_back(std::move(ub));
                const auto& pipe = get_or_create_pipeline(
                    "slice_nhwc", wgsl_slice_nhwc(use_f16), make_binding_types(1U, 1U));
                const std::size_t work = use_f16 ? om.elem_count : ((om.elem_count + 1U) / 2U);
                push_dispatch("slice#" + std::to_string(node.index), "tensor_ops",
                              pipe, ub_raw, {node.inputs[0], node.outputs[0]}, {},
                              div_up_u32(work, 256U), 1U, 1U);
                break;
            }
            case aot::Op::CONCAT: {
                int axis = node.axis_or_zero();
                if (axis < 0)
                    axis += 4;
                const TensorMeta& om = require_meta(node.outputs[0]);
                const auto od4 = aot::dims_to_4d(om.shape);
                std::array<std::uint32_t, 4> off{0, 0, 0, 0};
                const auto& pipe = get_or_create_pipeline(
                    "concat_nhwc", wgsl_concat_nhwc(use_f16), make_binding_types(1U, 1U));
                for (std::size_t k = 0; k < node.inputs.size(); ++k) {
                    const aot::TensorId cid = node.inputs[k];
                    materialize_init(cid);
                    const TensorMeta& im = require_meta(cid);
                    const auto id4 = aot::dims_to_4d(im.shape);
                    struct ConcatParams {
                        std::uint32_t in_n, in_c, in_h, in_w;
                        std::uint32_t on, oc, oh, ow;
                        std::uint32_t off_n, off_c, off_h, off_w;
                    } params{
                        static_cast<std::uint32_t>(id4[0]), static_cast<std::uint32_t>(id4[1]),
                        static_cast<std::uint32_t>(id4[2]), static_cast<std::uint32_t>(id4[3]),
                        static_cast<std::uint32_t>(od4[0]), static_cast<std::uint32_t>(od4[1]),
                        static_cast<std::uint32_t>(od4[2]), static_cast<std::uint32_t>(od4[3]),
                        off[0], off[1], off[2], off[3]};
                    UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "concat_u");
                    WGPUBuffer ub_raw = ub.get();
                    uniform_buffers_.push_back(std::move(ub));
                    const std::size_t work = use_f16 ? im.elem_count : (im.elem_count / 2U);
                    push_dispatch("concat#" + std::to_string(node.index) + "_" + std::to_string(k),
                                  "tensor_ops", pipe, ub_raw, {cid, node.outputs[0]}, {},
                                  div_up_u32(work, 256U), 1U, 1U);
                    off[static_cast<std::size_t>(axis)] +=
                        static_cast<std::uint32_t>(id4[static_cast<std::size_t>(axis)]);
                }
                break;
            }
            case aot::Op::DEPTH_TO_SPACE:
            case aot::Op::SPACE_TO_DEPTH: {
                materialize_init(node.inputs[0]);
                const TensorMeta& im = require_meta(node.inputs[0]);
                const TensorMeta& om = require_meta(node.outputs[0]);
                const auto id4 = aot::dims_to_4d(im.shape);
                const auto od4 = aot::dims_to_4d(om.shape);
                const bool is_d2s = (node.op == aot::Op::DEPTH_TO_SPACE);
                struct PixParams {
                    std::uint32_t n, in_c, in_h, in_w;
                    std::uint32_t block, out_c, out_h, out_w;
                } params{
                    static_cast<std::uint32_t>(id4[0]), static_cast<std::uint32_t>(id4[1]),
                    static_cast<std::uint32_t>(id4[2]), static_cast<std::uint32_t>(id4[3]),
                    static_cast<std::uint32_t>(node.block_size()),
                    static_cast<std::uint32_t>(od4[1]), static_cast<std::uint32_t>(od4[2]),
                    static_cast<std::uint32_t>(od4[3])};
                UniqueBuffer ub = create_uniform_buffer(&params, sizeof(params), "pix_u");
                WGPUBuffer ub_raw = ub.get();
                uniform_buffers_.push_back(std::move(ub));
                const std::string key = is_d2s ? "depth_to_space" : "space_to_depth";
                const auto& pipe = get_or_create_pipeline(
                    key,
                    is_d2s ? wgsl_depth_to_space_nhwc(use_f16)
                           : wgsl_space_to_depth_nhwc(use_f16),
                    make_binding_types(1U, 1U));
                const std::size_t work = use_f16 ? om.elem_count : ((om.elem_count + 1U) / 2U);
                push_dispatch(key + "#" + std::to_string(node.index), "tensor_ops",
                              pipe, ub_raw, {node.inputs[0], node.outputs[0]}, {},
                              div_up_u32(work, 256U), 1U, 1U);
                break;
            }
            default:
                throw std::runtime_error("webgpu: unsupported AOT operator");
            }
            break;
        }
        }
    }
}

void WebGpuExecutionPlan::encode_compute_schedule(
    WGPUCommandEncoder enc, int parity, bool skip_final_d2s8)
{
    if (pending_state_reset_) {
        for (const auto& pair : state_buffers_) {
            wgpuCommandEncoderClearBuffer(enc, pair[0].get(), 0U, WGPU_WHOLE_SIZE);
            wgpuCommandEncoderClearBuffer(enc, pair[1].get(), 0U, WGPU_WHOLE_SIZE);
        }
        pending_state_reset_ = false;
    }

    const std::size_t count = dispatches_.size();
    auto should_skip = [&](std::size_t i) -> bool {
        return skip_final_d2s8 &&
               (i == d2s8_step_index_ ||
                (d2s8_clip_step_index_ != kInvalidIndex && i == d2s8_clip_step_index_));
    };
    if (profile_enabled_ && profile_query_set_) {
        for (std::size_t i = 0; i < count; ++i) {
            if (should_skip(i))
                continue;
            const DispatchOp& op = dispatches_[i];
            WGPUPassTimestampWrites ts = WGPU_PASS_TIMESTAMP_WRITES_INIT;
            ts.querySet = profile_query_set_.get();
            ts.beginningOfPassWriteIndex = static_cast<std::uint32_t>(i * 2U);
            ts.endOfPassWriteIndex = static_cast<std::uint32_t>(i * 2U + 1U);
            WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
            cpd.label = to_wgpu_string(op.label);
            cpd.timestampWrites = &ts;
            UniqueComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(enc, &cpd));
            wgpuComputePassEncoderSetPipeline(pass.get(), op.pipeline);
            wgpuComputePassEncoderSetBindGroup(
                pass.get(), 0U, op.bind_groups[static_cast<std::size_t>(parity)].get(),
                0U, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass.get(), op.wg_x, op.wg_y, op.wg_z);
            wgpuComputePassEncoderEnd(pass.get());
        }
        const std::uint32_t qcount = static_cast<std::uint32_t>(count * 2U);
        wgpuCommandEncoderResolveQuerySet(
            enc, profile_query_set_.get(), 0U, qcount, profile_resolve_buf_.get(), 0U);
        wgpuCommandEncoderCopyBufferToBuffer(
            enc, profile_resolve_buf_.get(), 0U, profile_readback_buf_.get(), 0U,
            static_cast<std::uint64_t>(qcount) * sizeof(std::uint64_t));
        return;
    }

    // Fast path: encode ALL dispatches into ONE single ComputePassEncoder!
    WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    cpd.label = to_wgpu_string(model_name_);
    UniqueComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(enc, &cpd));
    WGPUComputePipeline last_pipe = nullptr;
    for (std::size_t i = 0; i < count; ++i) {
        if (should_skip(i))
            continue;
        const DispatchOp& op = dispatches_[i];
        if (op.pipeline != last_pipe) {
            wgpuComputePassEncoderSetPipeline(pass.get(), op.pipeline);
            last_pipe = op.pipeline;
        }
        wgpuComputePassEncoderSetBindGroup(
            pass.get(), 0U, op.bind_groups[static_cast<std::size_t>(parity)].get(),
            0U, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass.get(), op.wg_x, op.wg_y, op.wg_z);
    }
    wgpuComputePassEncoderEnd(pass.get());
}

void WebGpuExecutionPlan::resolve_and_log_profile()
{
#if !defined(__EMSCRIPTEN__)
    if (!profile_enabled_ || !profile_readback_buf_)
        return;
    ++profile_frame_count_;
    const std::size_t count = dispatches_.size();
    const std::size_t bytes = count * 2U * sizeof(std::uint64_t);
    std::vector<std::uint64_t> ts(count * 2U, 0U);

    struct MapState { WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error; } ms;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.userdata1 = &ms;
    cb.callback = [](WGPUMapAsyncStatus st, WGPUStringView, void* ud1, void*) {
        static_cast<MapState*>(ud1)->status = st;
    };
    WGPUFutureWaitInfo wi{};
    wi.future = wgpuBufferMapAsync(profile_readback_buf_.get(), WGPUMapMode_Read, 0U, bytes, cb);
    if (wgpuInstanceWaitAny(ctx_->instance(), 1U, &wi, UINT64_MAX) != WGPUWaitStatus_Success ||
        ms.status != WGPUMapAsyncStatus_Success) {
        return;
    }
    const void* mapped = wgpuBufferGetConstMappedRange(profile_readback_buf_.get(), 0U, bytes);
    if (mapped)
        std::memcpy(ts.data(), mapped, bytes);
    wgpuBufferUnmap(profile_readback_buf_.get());

    double total_ms = 0.0;
    std::unordered_map<std::string, std::pair<double, int>> by_cat;
    std::vector<std::pair<double, std::size_t>> ranked;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t t0 = ts[i * 2U];
        const std::uint64_t t1 = ts[i * 2U + 1U];
        const double ms_i = (t1 > t0) ? static_cast<double>(t1 - t0) * 1e-6 : 0.0;
        total_ms += ms_i;
        auto& entry = by_cat[profile_pass_cats_[i]];
        entry.first += ms_i;
        entry.second += 1;
        ranked.emplace_back(ms_i, i);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::fprintf(stderr, "[webgpu-profile] %s frame #%llu: total GPU = %.3f ms (%zu dispatches)\n",
                 model_name_.c_str(), static_cast<unsigned long long>(profile_frame_count_),
                 total_ms, count);
    for (const auto& [cat, info] : by_cat) {
        std::fprintf(stderr, "  %-20s : %7.3f ms (%3d dispatches, %5.1f%%)\n",
                     cat.c_str(), info.first, info.second,
                     total_ms > 0.0 ? (info.first * 100.0 / total_ms) : 0.0);
    }
    const std::size_t top_k = std::min<std::size_t>(8U, ranked.size());
    std::fprintf(stderr, "  top %zu slowest dispatches:\n", top_k);
    for (std::size_t k = 0; k < top_k; ++k) {
        std::fprintf(stderr, "    #%zu %-42s : %.3f ms\n",
                     k + 1U, profile_pass_labels_[ranked[k].second].c_str(), ranked[k].first);
    }
#endif
}

void WebGpuExecutionPlan::reset_state()
{
    state_parity_ = 0;
    pending_state_reset_ = true;
}

std::vector<Tensor> WebGpuExecutionPlan::run(const std::vector<Tensor>& inputs)
{
    if (inputs.size() != external_input_indices_.size()) {
        throw std::runtime_error(
            "webgpu: expected " + std::to_string(external_input_indices_.size()) +
            " inputs, got " + std::to_string(inputs.size()));
    }
    const bool use_f16 = uses_f16();
    const auto& nchw2nhwc = get_or_create_pipeline(
        "nchw_to_nhwc", wgsl_nchw_to_nhwc(use_f16), make_binding_types(1U, 1U));
    const auto& nhwc2nchw = get_or_create_pipeline(
        "nhwc_to_nchw", wgsl_nhwc_to_nchw(use_f16), make_binding_types(1U, 1U));

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    UniqueCommandEncoder enc(wgpuDeviceCreateCommandEncoder(ctx_->device(), &enc_desc));

    // Upload external inputs (and transpose NCHW -> NHWC on GPU when 4D FP16).
    std::vector<UniqueBuffer> tmp_uniforms;
    std::vector<UniqueBindGroup> tmp_bgs;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::size_t pos = external_input_indices_[i];
        const TensorMeta& m = require_meta(input_ids_[pos]);
        const Tensor& in_t = inputs[i];
        in_t.validate();
        if (in_t.shape != m.shape || in_t.data_type() != aot::to_public_dtype(m.dtype))
            throw std::runtime_error("webgpu: input shape or dtype mismatch for " + m.name);

        if (input_nchw_buffers_[pos]) {
            ctx_->write_buffer(input_nchw_buffers_[pos].get(), 0U, in_t.raw_data(), m.byte_size);
            struct LayoutParams { std::uint32_t n, c, hw, pad; } lp{
                static_cast<std::uint32_t>(m.shape[0]),
                static_cast<std::uint32_t>(m.shape[1]),
                static_cast<std::uint32_t>(m.shape[2] * m.shape[3]), 0U};
            tmp_uniforms.push_back(create_uniform_buffer(&lp, sizeof(lp), "in_t_u"));
            tmp_bgs.push_back(create_bind_group(
                nchw2nhwc.bgl.get(),
                {tmp_uniforms.back().get(), input_nchw_buffers_[pos].get(), input_buffers_[pos].get()}));
            WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
            UniqueComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(enc.get(), &cpd));
            wgpuComputePassEncoderSetPipeline(pass.get(), nchw2nhwc.pipeline.get());
            wgpuComputePassEncoderSetBindGroup(pass.get(), 0U, tmp_bgs.back().get(), 0U, nullptr);
            const std::size_t work = use_f16 ? m.elem_count : ((m.elem_count + 1U) / 2U);
            wgpuComputePassEncoderDispatchWorkgroups(pass.get(), div_up_u32(work, 256U), 1U, 1U);
            wgpuComputePassEncoderEnd(pass.get());
        } else {
            ctx_->write_buffer(input_buffers_[pos].get(), 0U, in_t.raw_data(), m.byte_size);
        }
    }

    const int run_parity = state_parity_;
    encode_compute_schedule(enc.get(), run_parity, /*skip_final_d2s8=*/false);

    // Transpose external outputs NHWC -> NCHW on GPU when needed.
    for (std::size_t i = 0; i < output_ids_.size(); ++i) {
        const TensorMeta& m = require_meta(output_ids_[i]);
        if (m.state_output_slot >= 0 || !output_nchw_buffers_[i])
            continue;
        struct LayoutParams { std::uint32_t n, c, hw, pad; } lp{
            static_cast<std::uint32_t>(m.shape[0]),
            static_cast<std::uint32_t>(m.shape[1]),
            static_cast<std::uint32_t>(m.shape[2] * m.shape[3]), 0U};
        tmp_uniforms.push_back(create_uniform_buffer(&lp, sizeof(lp), "out_t_u"));
        tmp_bgs.push_back(create_bind_group(
            nhwc2nchw.bgl.get(),
            {tmp_uniforms.back().get(), resolve_buffer(output_ids_[i], run_parity),
             output_nchw_buffers_[i].get()}));
        WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        UniqueComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(enc.get(), &cpd));
        wgpuComputePassEncoderSetPipeline(pass.get(), nhwc2nchw.pipeline.get());
        wgpuComputePassEncoderSetBindGroup(pass.get(), 0U, tmp_bgs.back().get(), 0U, nullptr);
        const std::size_t work = use_f16 ? m.elem_count : ((m.elem_count + 1U) / 2U);
        wgpuComputePassEncoderDispatchWorkgroups(pass.get(), div_up_u32(work, 256U), 1U, 1U);
        wgpuComputePassEncoderEnd(pass.get());
    }

    WGPUCommandBufferDescriptor cbd = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    UniqueCommandBuffer cmd(wgpuCommandEncoderFinish(enc.get(), &cbd));
    WGPUCommandBuffer cmd_raw = cmd.get();
    wgpuQueueSubmit(ctx_->queue(), 1U, &cmd_raw);
    if (!state_bindings_.empty())
        state_parity_ ^= 1;

    if (profile_enabled_)
        resolve_and_log_profile();

    std::vector<Tensor> outputs;
    for (std::size_t i = 0; i < output_ids_.size(); ++i) {
        const TensorMeta& m = require_meta(output_ids_[i]);
        if (m.state_output_slot >= 0)
            continue;
        WGPUBuffer src_buf = output_nchw_buffers_[i]
            ? output_nchw_buffers_[i].get()
            : resolve_buffer(output_ids_[i], run_parity);
        if (m.dtype == aot::DataType::FP16) {
            std::vector<Float16Storage> vals(m.elem_count);
            ctx_->read_buffer_sync(src_buf, 0U, vals.data(), m.byte_size);
            outputs.push_back(Tensor{m.name, m.shape, std::move(vals)});
        } else {
            std::vector<std::int32_t> vals(m.elem_count);
            ctx_->read_buffer_sync(src_buf, 0U, vals.data(), m.byte_size);
            outputs.push_back(Tensor{m.name, m.shape, std::move(vals)});
        }
    }
    ctx_->check_error("run");
    return outputs;
}

// =============================================================================
// BufferedCodecBackend (GPU YUV420 <-> NHWC + Double-Buffered Staging)
// =============================================================================

void WebGpuExecutionPlan::configure_codec_io(const CodecIoConfig& config)
{
    if (config.slots == 0U)
        throw std::runtime_error("webgpu: codec_io slots must be > 0");
    if (config.width <= 0 || config.height <= 0 || (config.width & 1) != 0 || (config.height & 1) != 0)
        throw std::runtime_error("webgpu: invalid YUV420 codec_io dimensions");

    codec_io_config_ = config;
    const std::size_t yuv_bytes = yuv420_frame_bytes(config.width, config.height);
    const bool use_f16 = uses_f16();

    codec_slots_.clear();
    codec_slots_.resize(config.slots);

    if (is_encoder_) {
        struct YuvToNhwcParams {
            std::uint32_t width, height, model_width, model_height;
            std::uint32_t rotated, p0, p1, p2;
            float scale, p3, p4, p5;
        } params{
            static_cast<std::uint32_t>(config.width),
            static_cast<std::uint32_t>(config.height),
            static_cast<std::uint32_t>(config.model_width),
            static_cast<std::uint32_t>(config.model_height),
            config.padding.rotated ? 1U : 0U, 0U, 0U, 0U,
            config.pixel_range / 255.0F, 0.0F, 0.0F, 0.0F};
        codec_io_uniform_ = create_uniform_buffer(&params, sizeof(params), "enc_yuv_u");
        get_or_create_pipeline("yuv420_to_nhwc", wgsl_yuv420_to_nhwc(use_f16),
                               make_binding_types(1U, 1U));
        get_or_create_pipeline("nhwc_to_nchw", wgsl_nhwc_to_nchw(use_f16),
                               make_binding_types(1U, 1U));

        for (std::size_t s = 0; s < config.slots; ++s) {
            auto& slot = codec_slots_[s];
            slot.yuv_bytes.assign(yuv_bytes, 0U);
            slot.gpu_yuv_buffer = ctx_->create_buffer(
                yuv_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                "enc_yuv_" + std::to_string(s));
            for (std::size_t oi = 0; oi < output_ids_.size(); ++oi) {
                const TensorMeta& m = require_meta(output_ids_[oi]);
                if (m.state_output_slot >= 0)
                    continue;
                slot.host_tensors.push_back(
                    Tensor{m.name, m.shape, std::vector<Float16Storage>(m.elem_count, 0)});
                slot.gpu_nchw_io_buffers.push_back(ctx_->create_buffer(
                    m.byte_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                    "enc_out_nchw_" + std::to_string(s) + "_" + m.name));
                slot.gpu_nchw_staging_buffers.push_back(ctx_->create_buffer(
                    m.byte_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
                    "enc_out_stg_" + std::to_string(s) + "_" + m.name));
            }
            for (const Tensor& t : slot.host_tensors)
                slot.host_output_views.push_back(tensor_view(t));
        }
    } else {
        struct NhwcToYuvParams {
            std::uint32_t width, height, model_width, model_height;
            std::uint32_t pad_left, pad_top, rotated, total_bytes;
            float inv_pixel_range, p0, p1, p2;
        } params{
            static_cast<std::uint32_t>(config.width),
            static_cast<std::uint32_t>(config.height),
            static_cast<std::uint32_t>(config.model_width),
            static_cast<std::uint32_t>(config.model_height),
            static_cast<std::uint32_t>(config.padding.left),
            static_cast<std::uint32_t>(config.padding.top),
            config.padding.rotated ? 1U : 0U,
            static_cast<std::uint32_t>(yuv_bytes),
            1.0F / config.pixel_range, 0.0F, 0.0F, 0.0F};
        codec_io_uniform_ = create_uniform_buffer(&params, sizeof(params), "dec_yuv_u");
        const std::string yuv_key = fused_d2s8_active() ? "d2s8_to_yuv420" : "nhwc_to_yuv420";
        get_or_create_pipeline(yuv_key, wgsl_nhwc_to_yuv420(use_f16, fused_d2s8_active()),
                               make_binding_types(1U, 1U));
        get_or_create_pipeline("nchw_to_nhwc", wgsl_nchw_to_nhwc(use_f16),
                               make_binding_types(1U, 1U));

        for (std::size_t s = 0; s < config.slots; ++s) {
            auto& slot = codec_slots_[s];
            slot.yuv_bytes.assign(yuv_bytes, 0U);
            slot.gpu_yuv_buffer = ctx_->create_buffer(
                yuv_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
                "dec_yuv_" + std::to_string(s));
            slot.gpu_yuv_staging = ctx_->create_buffer(
                yuv_bytes, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
                "dec_yuv_stg_" + std::to_string(s));
            for (const std::size_t pos : external_input_indices_) {
                if (pos == q_input_pos_)
                    continue;
                const TensorMeta& m = require_meta(input_ids_[pos]);
                slot.host_tensors.push_back(
                    Tensor{m.name, m.shape, std::vector<Float16Storage>(m.elem_count, 0)});
                slot.gpu_nchw_io_buffers.push_back(ctx_->create_buffer(
                    m.byte_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                    "dec_in_nchw_" + std::to_string(s) + "_" + m.name));
            }
            for (Tensor& t : slot.host_tensors)
                slot.host_input_views.push_back(mutable_tensor_view(t));
        }
    }

    // Pre-build layout uniforms for external NCHW <-> NHWC transposes.
    nchw_transpose_uniforms_.clear();
    if (is_encoder_) {
        for (std::size_t oi = 0; oi < output_ids_.size(); ++oi) {
            const TensorMeta& m = require_meta(output_ids_[oi]);
            if (m.state_output_slot >= 0)
                continue;
            struct Lp { std::uint32_t n, c, hw, pad; } lp{
                static_cast<std::uint32_t>(m.shape[0]),
                static_cast<std::uint32_t>(m.shape[1]),
                static_cast<std::uint32_t>(m.shape[2] * m.shape[3]), 0U};
            nchw_transpose_uniforms_.push_back(create_uniform_buffer(&lp, sizeof(lp), "enc_lo_u"));
        }
    } else {
        for (const std::size_t pos : external_input_indices_) {
            if (pos == q_input_pos_)
                continue;
            const TensorMeta& m = require_meta(input_ids_[pos]);
            struct Lp { std::uint32_t n, c, hw, pad; } lp{
                static_cast<std::uint32_t>(m.shape[0]),
                static_cast<std::uint32_t>(m.shape[1]),
                static_cast<std::uint32_t>(m.shape[2] * m.shape[3]), 0U};
            nchw_transpose_uniforms_.push_back(create_uniform_buffer(&lp, sizeof(lp), "dec_lo_u"));
        }
    }
    codec_io_configured_ = true;
}

MutableYuv420FrameView WebGpuExecutionPlan::encoder_input_yuv(std::size_t slot_idx)
{
    if (!codec_io_configured_ || !is_encoder_ || slot_idx >= codec_slots_.size())
        throw std::runtime_error("webgpu: invalid encoder_input_yuv slot");
    auto& buf = codec_slots_[slot_idx].yuv_bytes;
    const std::size_t w = static_cast<std::size_t>(codec_io_config_.width);
    const std::size_t h = static_cast<std::size_t>(codec_io_config_.height);
    const std::size_t y_size = w * h;
    const std::size_t uv_size = y_size / 4U;
    return MutableYuv420FrameView(
        codec_io_config_.width, codec_io_config_.height,
        std::span<std::uint8_t>(buf.data(), y_size), w,
        std::span<std::uint8_t>(buf.data() + y_size, uv_size), w / 2U,
        std::span<std::uint8_t>(buf.data() + y_size + uv_size, uv_size), w / 2U);
}

void WebGpuExecutionPlan::submit_encoder(std::size_t slot_idx, int shifted_q)
{
    if (!codec_io_configured_ || !is_encoder_ || slot_idx >= codec_slots_.size())
        throw std::runtime_error("webgpu: invalid submit_encoder call");
    auto& slot = codec_slots_[slot_idx];
    const bool use_f16 = uses_f16();
    ctx_->write_buffer(slot.gpu_yuv_buffer.get(), 0U, slot.yuv_bytes.data(), slot.yuv_bytes.size());
    const std::int32_t q_val = shifted_q;
    ctx_->write_buffer(input_buffers_[q_input_pos_].get(), 0U, &q_val, sizeof(q_val));

    const auto& yuv_pipe = pipeline_cache_.at("yuv420_to_nhwc");
    const auto& nhwc2nchw = pipeline_cache_.at("nhwc_to_nchw");

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    UniqueCommandEncoder enc(wgpuDeviceCreateCommandEncoder(ctx_->device(), &enc_desc));

    UniqueBindGroup yuv_bg = create_bind_group(
        yuv_pipe.bgl.get(),
        {codec_io_uniform_.get(), slot.gpu_yuv_buffer.get(), input_buffers_[x_input_pos_].get()});
    {
        WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        UniqueComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(enc.get(), &cpd));
        wgpuComputePassEncoderSetPipeline(pass.get(), yuv_pipe.pipeline.get());
        wgpuComputePassEncoderSetBindGroup(pass.get(), 0U, yuv_bg.get(), 0U, nullptr);
        const std::uint32_t gx =
            div_up_u32(static_cast<std::size_t>(codec_io_config_.model_width / 2), 16U);
        const std::uint32_t gy =
            div_up_u32(static_cast<std::size_t>(codec_io_config_.model_height), 16U);
        wgpuComputePassEncoderDispatchWorkgroups(pass.get(), gx, gy, 1U);
        wgpuComputePassEncoderEnd(pass.get());
    }

    const int run_parity = state_parity_;
    encode_compute_schedule(enc.get(), run_parity, /*skip_final_d2s8=*/false);

    std::vector<UniqueBindGroup> out_bgs;
    std::size_t out_idx = 0;
    for (std::size_t oi = 0; oi < output_ids_.size(); ++oi) {
        const TensorMeta& m = require_meta(output_ids_[oi]);
        if (m.state_output_slot >= 0)
            continue;
        WGPUBuffer src_nhwc = resolve_buffer(output_ids_[oi], run_parity);
        if (needs_nhwc_transpose(m.shape, m.dtype)) {
            out_bgs.push_back(create_bind_group(
                nhwc2nchw.bgl.get(),
                {nchw_transpose_uniforms_[out_idx].get(), src_nhwc,
                 slot.gpu_nchw_io_buffers[out_idx].get()}));
            WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
            UniqueComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(enc.get(), &cpd));
            wgpuComputePassEncoderSetPipeline(pass.get(), nhwc2nchw.pipeline.get());
            wgpuComputePassEncoderSetBindGroup(pass.get(), 0U, out_bgs.back().get(), 0U, nullptr);
            const std::size_t work = use_f16 ? m.elem_count : ((m.elem_count + 1U) / 2U);
            wgpuComputePassEncoderDispatchWorkgroups(pass.get(), div_up_u32(work, 256U), 1U, 1U);
            wgpuComputePassEncoderEnd(pass.get());
            wgpuCommandEncoderCopyBufferToBuffer(
                enc.get(), slot.gpu_nchw_io_buffers[out_idx].get(), 0U,
                slot.gpu_nchw_staging_buffers[out_idx].get(), 0U, align_up_u64(m.byte_size, 4U));
        } else {
            wgpuCommandEncoderCopyBufferToBuffer(
                enc.get(), src_nhwc, 0U,
                slot.gpu_nchw_staging_buffers[out_idx].get(), 0U, align_up_u64(m.byte_size, 4U));
        }
        ++out_idx;
    }

    WGPUCommandBufferDescriptor cbd = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    UniqueCommandBuffer cmd(wgpuCommandEncoderFinish(enc.get(), &cbd));
    WGPUCommandBuffer cmd_raw = cmd.get();
    wgpuQueueSubmit(ctx_->queue(), 1U, &cmd_raw);
    if (!state_bindings_.empty())
        state_parity_ ^= 1;
    slot.pending = true;
}

std::vector<TensorView> WebGpuExecutionPlan::encoder_outputs(std::size_t slot_idx) const
{
    if (!codec_io_configured_ || !is_encoder_ || slot_idx >= codec_slots_.size())
        throw std::runtime_error("webgpu: invalid encoder_outputs slot");
    return codec_slots_[slot_idx].host_output_views;
}

std::vector<MutableTensorView> WebGpuExecutionPlan::decoder_inputs(std::size_t slot_idx)
{
    if (!codec_io_configured_ || is_encoder_ || slot_idx >= codec_slots_.size())
        throw std::runtime_error("webgpu: invalid decoder_inputs slot");
    return codec_slots_[slot_idx].host_input_views;
}

void WebGpuExecutionPlan::submit_decoder(std::size_t slot_idx, int shifted_q)
{
    if (!codec_io_configured_ || is_encoder_ || slot_idx >= codec_slots_.size())
        throw std::runtime_error("webgpu: invalid submit_decoder call");
    auto& slot = codec_slots_[slot_idx];
    const bool use_f16 = uses_f16();

    const std::int32_t q_val = shifted_q;
    ctx_->write_buffer(input_buffers_[q_input_pos_].get(), 0U, &q_val, sizeof(q_val));

    const auto& nchw2nhwc = pipeline_cache_.at("nchw_to_nhwc");
    const std::string yuv_key = fused_d2s8_active() ? "d2s8_to_yuv420" : "nhwc_to_yuv420";
    const auto& yuv_pipe = pipeline_cache_.at(yuv_key);

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    UniqueCommandEncoder enc(wgpuDeviceCreateCommandEncoder(ctx_->device(), &enc_desc));

    std::vector<UniqueBindGroup> in_bgs;
    std::size_t in_idx = 0;
    for (const std::size_t pos : external_input_indices_) {
        if (pos == q_input_pos_)
            continue;
        const TensorMeta& m = require_meta(input_ids_[pos]);
        const Tensor& ht = slot.host_tensors[in_idx];
        if (needs_nhwc_transpose(m.shape, m.dtype)) {
            ctx_->write_buffer(slot.gpu_nchw_io_buffers[in_idx].get(), 0U, ht.raw_data(), m.byte_size);
            in_bgs.push_back(create_bind_group(
                nchw2nhwc.bgl.get(),
                {nchw_transpose_uniforms_[in_idx].get(),
                 slot.gpu_nchw_io_buffers[in_idx].get(),
                 input_buffers_[pos].get()}));
            WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
            UniqueComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(enc.get(), &cpd));
            wgpuComputePassEncoderSetPipeline(pass.get(), nchw2nhwc.pipeline.get());
            wgpuComputePassEncoderSetBindGroup(pass.get(), 0U, in_bgs.back().get(), 0U, nullptr);
            const std::size_t work = use_f16 ? m.elem_count : ((m.elem_count + 1U) / 2U);
            wgpuComputePassEncoderDispatchWorkgroups(pass.get(), div_up_u32(work, 256U), 1U, 1U);
            wgpuComputePassEncoderEnd(pass.get());
        } else {
            ctx_->write_buffer(input_buffers_[pos].get(), 0U, ht.raw_data(), m.byte_size);
        }
        ++in_idx;
    }

    const int run_parity = state_parity_;
    const bool skip_d2s8 = fused_d2s8_active();
    encode_compute_schedule(enc.get(), run_parity, skip_d2s8);

    const aot::TensorId src_tid = skip_d2s8 ? d2s8_input_id_ : output_ids_[x_hat_output_pos_];
    UniqueBindGroup yuv_bg = create_bind_group(
        yuv_pipe.bgl.get(),
        {codec_io_uniform_.get(), resolve_buffer(src_tid, run_parity), slot.gpu_yuv_buffer.get()});
    {
        WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        UniqueComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(enc.get(), &cpd));
        wgpuComputePassEncoderSetPipeline(pass.get(), yuv_pipe.pipeline.get());
        wgpuComputePassEncoderSetBindGroup(pass.get(), 0U, yuv_bg.get(), 0U, nullptr);
        const std::size_t words = (slot.yuv_bytes.size() + 3U) / 4U;
        wgpuComputePassEncoderDispatchWorkgroups(pass.get(), div_up_u32(words, 256U), 1U, 1U);
        wgpuComputePassEncoderEnd(pass.get());
    }
    wgpuCommandEncoderCopyBufferToBuffer(
        enc.get(), slot.gpu_yuv_buffer.get(), 0U, slot.gpu_yuv_staging.get(), 0U,
        align_up_u64(slot.yuv_bytes.size(), 4U));

    WGPUCommandBufferDescriptor cbd = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    UniqueCommandBuffer cmd(wgpuCommandEncoderFinish(enc.get(), &cbd));
    WGPUCommandBuffer cmd_raw = cmd.get();
    wgpuQueueSubmit(ctx_->queue(), 1U, &cmd_raw);
    if (!state_bindings_.empty())
        state_parity_ ^= 1;
    slot.pending = true;
}

Yuv420FrameView WebGpuExecutionPlan::decoder_output_yuv(std::size_t slot_idx) const
{
    if (!codec_io_configured_ || is_encoder_ || slot_idx >= codec_slots_.size())
        throw std::runtime_error("webgpu: invalid decoder_output_yuv slot");
    const auto& buf = codec_slots_[slot_idx].yuv_bytes;
    const std::size_t w = static_cast<std::size_t>(codec_io_config_.width);
    const std::size_t h = static_cast<std::size_t>(codec_io_config_.height);
    const std::size_t y_size = w * h;
    const std::size_t uv_size = y_size / 4U;
    return Yuv420FrameView(
        codec_io_config_.width, codec_io_config_.height,
        std::span<const std::uint8_t>(buf.data(), y_size), w,
        std::span<const std::uint8_t>(buf.data() + y_size, uv_size), w / 2U,
        std::span<const std::uint8_t>(buf.data() + y_size + uv_size, uv_size), w / 2U);
}

void WebGpuExecutionPlan::wait_codec_slot(std::size_t slot_idx)
{
    if (!codec_io_configured_ || slot_idx >= codec_slots_.size())
        throw std::runtime_error("webgpu: invalid wait_codec_slot");
    auto& slot = codec_slots_[slot_idx];
    if (!slot.pending)
        return;

#if defined(__EMSCRIPTEN__)
    slot.pending = false;
    throw std::runtime_error("webgpu: wait_codec_slot CPU readback is unsupported in browser WASM");
#else
    auto map_copy = [&](WGPUBuffer staging, void* dst, std::size_t bytes) {
        const std::size_t map_size = static_cast<std::size_t>(align_up_u64(bytes, 4U));
        struct MapState {
            WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
            std::string msg;
        } ms;
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_WaitAnyOnly;
        cb.userdata1 = &ms;
        cb.callback = [](WGPUMapAsyncStatus st, WGPUStringView msg, void* ud1, void*) {
            auto* s = static_cast<MapState*>(ud1);
            s->status = st;
            s->msg = from_wgpu_string(msg);
        };
        WGPUFutureWaitInfo wi{};
        wi.future = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0U, map_size, cb);
        if (wgpuInstanceWaitAny(ctx_->instance(), 1U, &wi, UINT64_MAX) != WGPUWaitStatus_Success ||
            ms.status != WGPUMapAsyncStatus_Success) {
            throw std::runtime_error("webgpu: slot mapAsync failed: " + ms.msg);
        }
        const void* mapped = wgpuBufferGetConstMappedRange(staging, 0U, map_size);
        if (!mapped) {
            wgpuBufferUnmap(staging);
            throw std::runtime_error("webgpu: mapped slot buffer is null");
        }
        std::memcpy(dst, mapped, bytes);
        wgpuBufferUnmap(staging);
    };

    if (is_encoder_) {
        for (std::size_t i = 0; i < slot.host_tensors.size(); ++i) {
            auto& vec = std::get<std::vector<Float16Storage>>(slot.host_tensors[i].data);
            map_copy(slot.gpu_nchw_staging_buffers[i].get(),
                     vec.data(), vec.size() * sizeof(Float16Storage));
        }
    } else {
        map_copy(slot.gpu_yuv_staging.get(), slot.yuv_bytes.data(), slot.yuv_bytes.size());
    }
    slot.pending = false;
    if (profile_enabled_)
        resolve_and_log_profile();
    ctx_->check_error("wait_codec_slot");
#endif
}

// =============================================================================
// Zero-Readback GPU Surface Render Path (Browser OffscreenCanvas / Native)
// =============================================================================

void WebGpuExecutionPlan::configure_surface_output(const SurfaceRenderConfig& config)
{
    if (is_encoder_)
        throw std::runtime_error("webgpu: configure_surface_output is decoder-only");
    surface_config_ = config;
    const bool use_f16 = uses_f16();
    const bool use_d2s8 = fused_d2s8_active();

    struct RenderParams {
        std::uint32_t width, height, model_width, model_height;
        std::uint32_t pad_left, pad_top, rotated, color_matrix;
        float inv_pixel_range, p0, p1, p2;
    } rp{
        static_cast<std::uint32_t>(config.width),
        static_cast<std::uint32_t>(config.height),
        static_cast<std::uint32_t>(config.model_width),
        static_cast<std::uint32_t>(config.model_height),
        static_cast<std::uint32_t>(config.padding.left),
        static_cast<std::uint32_t>(config.padding.top),
        config.padding.rotated ? 1U : 0U,
        config.color_matrix,
        1.0F / config.pixel_range, 0.0F, 0.0F, 0.0F};
    render_uniform_ = create_uniform_buffer(&rp, sizeof(rp), "render_yuv444_u");

    std::array<WGPUBindGroupLayoutEntry, 2> entries{};
    entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[0].binding = 0U;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;

    entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[1].binding = 1U;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = entries.size();
    bgl_desc.entries = entries.data();
    render_bgl_.reset(wgpuDeviceCreateBindGroupLayout(ctx_->device(), &bgl_desc));

    WGPUBindGroupLayout bgl_raw = render_bgl_.get();
    WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl_desc.bindGroupLayoutCount = 1U;
    pl_desc.bindGroupLayouts = &bgl_raw;
    render_layout_.reset(wgpuDeviceCreatePipelineLayout(ctx_->device(), &pl_desc));

    const std::string wgsl = wgsl_render_yuv444_to_rgba(use_f16, use_d2s8);
    UniqueShaderModule mod = ctx_->create_shader_module(wgsl, "render_yuv444_to_rgba");

    WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
    target.format = config.target_format;
    target.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState frag = WGPU_FRAGMENT_STATE_INIT;
    frag.module = mod.get();
    frag.entryPoint = to_wgpu_string("fs_main");
    frag.targetCount = 1U;
    frag.targets = &target;

    WGPURenderPipelineDescriptor rpd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    rpd.label = to_wgpu_string("mlvc_render_yuv444_to_rgba");
    rpd.layout = render_layout_.get();
    rpd.vertex.module = mod.get();
    rpd.vertex.entryPoint = to_wgpu_string("vs_main");
    rpd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    rpd.primitive.cullMode = WGPUCullMode_None;
    rpd.multisample.count = 1U;
    rpd.multisample.mask = 0xFFFFFFFFU;
    rpd.fragment = &frag;
    render_pipeline_.reset(wgpuDeviceCreateRenderPipeline(ctx_->device(), &rpd));
    if (!render_pipeline_)
        throw std::runtime_error("webgpu: failed to create render pipeline");

    const aot::TensorId src_tid = use_d2s8 ? d2s8_input_id_ : output_ids_[x_hat_output_pos_];
    for (int p = 0; p < 2; ++p) {
        render_bind_groups_[static_cast<std::size_t>(p)] = create_bind_group(
            render_bgl_.get(), {render_uniform_.get(), resolve_buffer(src_tid, p)});
    }

    // Ensure NCHW -> NHWC pipeline and uniforms are ready for latent upload.
    get_or_create_pipeline("nchw_to_nhwc", wgsl_nchw_to_nhwc(use_f16),
                           make_binding_types(1U, 1U));
    nchw_transpose_uniforms_.clear();
    for (const std::size_t pos : external_input_indices_) {
        if (pos == q_input_pos_)
            continue;
        const TensorMeta& m = require_meta(input_ids_[pos]);
        struct Lp { std::uint32_t n, c, hw, pad; } lp{
            static_cast<std::uint32_t>(m.shape[0]),
            static_cast<std::uint32_t>(m.shape[1]),
            static_cast<std::uint32_t>(m.shape[2] * m.shape[3]), 0U};
        nchw_transpose_uniforms_.push_back(create_uniform_buffer(&lp, sizeof(lp), "surf_lo_u"));
    }

    surface_configured_ = true;
    ctx_->check_error("configure_surface_output");
}

void WebGpuExecutionPlan::submit_decoder_to_texture(
    const std::vector<TensorView>& latents,
    int shifted_q,
    bool reset_state_before_run,
    WGPUTextureView target_view)
{
    if (!surface_configured_ || !target_view)
        throw std::runtime_error("webgpu: surface output not configured or null target_view");
    if (reset_state_before_run)
        reset_state();

    const bool use_f16 = uses_f16();
    const std::int32_t q_val = shifted_q;
    ctx_->write_buffer(input_buffers_[q_input_pos_].get(), 0U, &q_val, sizeof(q_val));

    const auto& nchw2nhwc = pipeline_cache_.at("nchw_to_nhwc");

    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    UniqueCommandEncoder enc(wgpuDeviceCreateCommandEncoder(ctx_->device(), &enc_desc));

    // Upload latents by matching name ("z_raw", "y_raw_0", "y_raw_1") + transpose on GPU.
    std::vector<UniqueBindGroup> tmp_bgs;
    std::size_t u_idx = 0;
    for (const std::size_t pos : external_input_indices_) {
        if (pos == q_input_pos_)
            continue;
        const TensorMeta& m = require_meta(input_ids_[pos]);
        const auto found = std::find_if(
            latents.begin(), latents.end(),
            [&](const TensorView& tv) { return tv.name() == m.name; });
        if (found == latents.end())
            throw std::runtime_error("webgpu: missing latent input " + m.name);

        if (input_nchw_buffers_[pos]) {
            ctx_->write_buffer(input_nchw_buffers_[pos].get(), 0U,
                               found->raw_data(), m.byte_size);
            tmp_bgs.push_back(create_bind_group(
                nchw2nhwc.bgl.get(),
                {nchw_transpose_uniforms_[u_idx].get(),
                 input_nchw_buffers_[pos].get(),
                 input_buffers_[pos].get()}));
            WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
            UniqueComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(enc.get(), &cpd));
            wgpuComputePassEncoderSetPipeline(pass.get(), nchw2nhwc.pipeline.get());
            wgpuComputePassEncoderSetBindGroup(pass.get(), 0U, tmp_bgs.back().get(), 0U, nullptr);
            const std::size_t work = use_f16 ? m.elem_count : ((m.elem_count + 1U) / 2U);
            wgpuComputePassEncoderDispatchWorkgroups(pass.get(), div_up_u32(work, 256U), 1U, 1U);
            wgpuComputePassEncoderEnd(pass.get());
        } else {
            ctx_->write_buffer(input_buffers_[pos].get(), 0U, found->raw_data(), m.byte_size);
        }
        ++u_idx;
    }

    const int run_parity = state_parity_;
    encode_compute_schedule(enc.get(), run_parity, /*skip_final_d2s8=*/fused_d2s8_active());

    // Zero-readback direct RenderPass to target texture (Canvas SwapChain)!
    WGPURenderPassColorAttachment color_att = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    color_att.view = target_view;
    color_att.loadOp = WGPULoadOp_Clear;
    color_att.storeOp = WGPUStoreOp_Store;
    color_att.clearValue = WGPUColor{0.0, 0.0, 0.0, 1.0};

    WGPURenderPassDescriptor rpd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    rpd.label = to_wgpu_string("mlvc_present");
    rpd.colorAttachmentCount = 1U;
    rpd.colorAttachments = &color_att;

    UniqueRenderPassEncoder rpass(wgpuCommandEncoderBeginRenderPass(enc.get(), &rpd));
    wgpuRenderPassEncoderSetPipeline(rpass.get(), render_pipeline_.get());
    wgpuRenderPassEncoderSetBindGroup(
        rpass.get(), 0U, render_bind_groups_[static_cast<std::size_t>(run_parity)].get(),
        0U, nullptr);
    wgpuRenderPassEncoderDraw(rpass.get(), 3U, 1U, 0U, 0U);
    wgpuRenderPassEncoderEnd(rpass.get());

    WGPUCommandBufferDescriptor cbd = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    UniqueCommandBuffer cmd(wgpuCommandEncoderFinish(enc.get(), &cbd));
    WGPUCommandBuffer cmd_raw = cmd.get();
    wgpuQueueSubmit(ctx_->queue(), 1U, &cmd_raw);
    if (!state_bindings_.empty())
        state_parity_ ^= 1;
}

}  // namespace mlvc::webgpu_backend
