#include "fusion_matcher.hpp"

#include "mlvc/core/half.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <unordered_map>
#include <unordered_set>

namespace mlvc::cpu_backend {
namespace {

using aot::DataType;
using aot::Node;
using aot::Op;
using aot::TensorDescriptor;
using aot::TensorId;

class GraphMatcher {
public:
    explicit GraphMatcher(const aot::LoadedGraph& loaded)
        : loaded_(loaded), graph_(loaded.descriptor)
    {
        for (const TensorId out_id : graph_.outputs)
            graph_outputs_.insert(out_id);
        for (const Node& node : graph_.nodes) {
            for (const TensorId in_id : node.inputs) {
                if (in_id != 0)
                    ++consumer_count_[in_id];
            }
        }
    }

    std::vector<FusedStep> build(bool enable_fusion)
    {
        std::vector<FusedStep> steps;
        const std::size_t n = graph_.nodes.size();
        steps.reserve(n);
        if (!enable_fusion) {
            for (std::size_t i = 0; i < n; ++i) {
                FusedStep s;
                s.kind = FusedStepKind::kSingleNode;
                s.first_node = i;
                s.end_node = i + 1;
                steps.push_back(s);
            }
            return steps;
        }

        std::vector<bool> elided(n, false);
        std::size_t i = 0;
        while (i < n) {
            if (elided[i]) {
                ++i;
                continue;
            }
            FusedStep step;
            if (try_y0_tail(i, step) ||
                try_y1_tail(i, step) ||
                try_feature_update(i, elided, step) ||
                try_pointwise_reglu(i, step) ||
                try_reglu(i, step) ||
                try_pointwise_epilogue(i, step)) {
                steps.push_back(step);
                i = step.end_node;
                continue;
            }
            step.kind = FusedStepKind::kSingleNode;
            step.first_node = i;
            step.end_node = i + 1;
            steps.push_back(step);
            ++i;
        }
        return steps;
    }

private:
    const TensorDescriptor* tensor(TensorId id) const noexcept
    {
        if (id == 0 || id >= graph_.tensors.size())
            return nullptr;
        const TensorDescriptor& t = graph_.tensors[id];
        return t.name.empty() ? nullptr : &t;
    }

    static bool is_init(const TensorDescriptor* t) noexcept
    {
        return t != nullptr && std::holds_alternative<aot::WeightLocation>(t->storage);
    }

    std::span<const std::byte> weight_bytes(const TensorDescriptor& t) const noexcept
    {
        const auto* wl = std::get_if<aot::WeightLocation>(&t.storage);
        if (!wl || wl->offset > loaded_.weights.size() ||
            wl->bytes > loaded_.weights.size() - wl->offset) {
            return {};
        }
        return std::span<const std::byte>(loaded_.weights.data() + wl->offset, wl->bytes);
    }

    static int axis_of(const Node& node) noexcept
    {
        const auto* ax = std::get_if<aot::AxisAttributes>(&node.attributes);
        return ax ? ax->axis : 0;
    }

    static float alpha_of(const Node& node) noexcept
    {
        const auto* act = std::get_if<aot::ActivationAttributes>(&node.attributes);
        return act ? act->alpha : 0.01F;
    }

    std::size_t consumers(TensorId id) const noexcept
    {
        const auto it = consumer_count_.find(id);
        return it == consumer_count_.end() ? 0U : it->second;
    }

    bool is_graph_output(TensorId id) const noexcept
    {
        return graph_outputs_.find(id) != graph_outputs_.end();
    }

    bool is_internal_only(TensorId id, std::size_t expected_consumers) const noexcept
    {
        return id != 0 && !is_graph_output(id) && consumers(id) == expected_consumers;
    }

    bool read_int64_init(TensorId id, std::vector<std::int64_t>& out) const
    {
        const TensorDescriptor* t = tensor(id);
        if (!is_init(t) || t->dtype != DataType::INT64)
            return false;
        const auto span = weight_bytes(*t);
        if (span.size() % sizeof(std::int64_t) != 0U)
            return false;
        out.resize(span.size() / sizeof(std::int64_t));
        if (!out.empty())
            std::memcpy(out.data(), span.data(), span.size());
        return true;
    }

    bool read_fp16_scalar_init(TensorId id, float& out) const noexcept
    {
        const TensorDescriptor* t = tensor(id);
        if (!is_init(t) || t->dtype != DataType::FP16)
            return false;
        const auto span = weight_bytes(*t);
        if (span.size() != sizeof(Float16Storage))
            return false;
        Float16Storage raw = 0;
        std::memcpy(&raw, span.data(), sizeof(raw));
        out = half_to_float(raw);
        return true;
    }

    bool match_clip_constants(const Node& node, float& min_v, float& max_v) const noexcept
    {
        if (node.op != Op::CLIP || node.inputs.size() < 2U || node.outputs.size() != 1U)
            return false;
        min_v = -std::numeric_limits<float>::infinity();
        max_v = std::numeric_limits<float>::infinity();
        if (node.inputs[1] != 0 && !read_fp16_scalar_init(node.inputs[1], min_v))
            return false;
        if (node.inputs.size() >= 3U && node.inputs[2] != 0 &&
            !read_fp16_scalar_init(node.inputs[2], max_v))
            return false;
        return true;
    }

    bool match_channel_slice(const Node& node, TensorId expected_input,
                             std::int64_t expected_start,
                             std::int64_t expected_end) const
    {
        if (node.op != Op::SLICE || node.inputs.size() < 3U || node.outputs.size() != 1U)
            return false;
        if (node.inputs[0] != expected_input)
            return false;
        std::vector<std::int64_t> starts;
        std::vector<std::int64_t> ends;
        if (!read_int64_init(node.inputs[1], starts) ||
            !read_int64_init(node.inputs[2], ends) ||
            starts.size() != 1U || ends.size() != 1U)
            return false;
        if (starts[0] != expected_start || ends[0] != expected_end)
            return false;
        if (node.inputs.size() >= 4U && node.inputs[3] != 0) {
            std::vector<std::int64_t> axes;
            if (!read_int64_init(node.inputs[3], axes) || axes.size() != 1U || axes[0] != 1)
                return false;
        } else {
            return false;
        }
        if (node.inputs.size() >= 5U && node.inputs[4] != 0) {
            std::vector<std::int64_t> steps;
            if (!read_int64_init(node.inputs[4], steps) || steps.size() != 1U || steps[0] != 1)
                return false;
        }
        const TensorDescriptor* out = tensor(node.outputs[0]);
        return out && out->dtype == DataType::FP16 && out->shape.size() == 4U &&
               out->shape[1] == (expected_end - expected_start);
    }

    static bool match_pair(const Node& node, Op op, TensorId x, TensorId y) noexcept
    {
        if (node.op != op || node.inputs.size() != 2U || node.outputs.size() != 1U)
            return false;
        return (node.inputs[0] == x && node.inputs[1] == y) ||
               (node.inputs[0] == y && node.inputs[1] == x);
    }

    static bool match_other_in_pair(const Node& node, Op op, TensorId x,
                                    TensorId& other_out) noexcept
    {
        if (node.op != op || node.inputs.size() != 2U || node.outputs.size() != 1U)
            return false;
        if (node.inputs[0] == x) {
            other_out = node.inputs[1];
            return true;
        }
        if (node.inputs[1] == x) {
            other_out = node.inputs[0];
            return true;
        }
        return false;
    }

    bool is_standard_conv(const Node& node, bool require_1x1) const noexcept
    {
        if (node.op != Op::CONV || node.inputs.size() < 2U || node.outputs.size() != 1U)
            return false;
        const auto& conv = node.conv();
        if (conv.groups != 1U || conv.strides.size() != 2U ||
            conv.pads.size() != 4U || conv.dilations.size() != 2U)
            return false;
        if (conv.dilations[0] != 1U || conv.dilations[1] != 1U)
            return false;
        const TensorDescriptor* w = tensor(node.inputs[1]);
        const TensorDescriptor* in = tensor(node.inputs[0]);
        const TensorDescriptor* out = tensor(node.outputs[0]);
        if (!is_init(w) || w->dtype != DataType::FP16 || w->shape.size() != 4U)
            return false;
        if (!in || in->dtype != DataType::FP16 || in->shape.size() != 4U ||
            !out || out->dtype != DataType::FP16 || out->shape.size() != 4U)
            return false;
        if (require_1x1) {
            if (w->shape[2] != 1 || w->shape[3] != 1 ||
                conv.strides[0] != 1U || conv.strides[1] != 1U ||
                conv.pads[0] != 0U || conv.pads[1] != 0U ||
                conv.pads[2] != 0U || conv.pads[3] != 0U)
                return false;
        }
        return true;
    }

    bool try_pointwise_reglu(std::size_t i, FusedStep& out) const
    {
        if (i + 5U > graph_.nodes.size())
            return false;
        const Node& conv = graph_.nodes[i];
        if (!is_standard_conv(conv, true))
            return false;
        const TensorId conv_out = conv.outputs[0];
        if (!is_internal_only(conv_out, 2U))
            return false;
        const TensorDescriptor* ct = tensor(conv_out);
        if (!ct || ct->shape[1] <= 0 || (ct->shape[1] % 2) != 0)
            return false;
        const std::int64_t g = ct->shape[1] / 2;

        if (!match_channel_slice(graph_.nodes[i + 1], conv_out, 0, g) ||
            !match_channel_slice(graph_.nodes[i + 2], conv_out, g, g * 2))
            return false;
        const TensorId gate = graph_.nodes[i + 1].outputs[0];
        const TensorId lin = graph_.nodes[i + 2].outputs[0];
        if (!is_internal_only(gate, 1U) || !is_internal_only(lin, 1U))
            return false;

        float min_v = 0.0F;
        float max_v = 0.0F;
        const Node& clip = graph_.nodes[i + 3];
        if (!match_clip_constants(clip, min_v, max_v) || clip.inputs[0] != gate)
            return false;
        const TensorId clipped = clip.outputs[0];
        if (!is_internal_only(clipped, 1U))
            return false;

        const Node& mul = graph_.nodes[i + 4];
        if (!match_pair(mul, Op::MUL, clipped, lin))
            return false;

        out = FusedStep{};
        out.kind = FusedStepKind::kConvWithEpilogue;
        out.first_node = i;
        out.end_node = i + 5U;
        out.epilogue_kind = GemmEpilogueKind::kReglu;
        out.clamp_min = min_v;
        out.clamp_max = max_v;
        out.conv_input = conv.inputs[0];
        out.output_tensor = mul.outputs[0];
        return true;
    }

    bool try_reglu(std::size_t i, FusedStep& out) const
    {
        if (i + 4U > graph_.nodes.size())
            return false;
        const Node& s0 = graph_.nodes[i];
        if (s0.op != Op::SLICE || s0.inputs.empty())
            return false;
        const TensorId in_id = s0.inputs[0];
        const TensorDescriptor* it = tensor(in_id);
        if (!it || it->dtype != DataType::FP16 || it->shape.size() != 4U ||
            it->shape[1] <= 0 || (it->shape[1] % 2) != 0)
            return false;
        const std::int64_t g = it->shape[1] / 2;
        if (!match_channel_slice(s0, in_id, 0, g) ||
            !match_channel_slice(graph_.nodes[i + 1], in_id, g, g * 2))
            return false;
        const TensorId gate = s0.outputs[0];
        const TensorId lin = graph_.nodes[i + 1].outputs[0];
        if (!is_internal_only(gate, 1U) || !is_internal_only(lin, 1U))
            return false;

        float min_v = 0.0F;
        float max_v = 0.0F;
        const Node& clip = graph_.nodes[i + 2];
        if (!match_clip_constants(clip, min_v, max_v) || clip.inputs[0] != gate)
            return false;
        const TensorId clipped = clip.outputs[0];
        if (!is_internal_only(clipped, 1U))
            return false;

        const Node& mul = graph_.nodes[i + 3];
        if (!match_pair(mul, Op::MUL, clipped, lin))
            return false;

        out = FusedStep{};
        out.kind = FusedStepKind::kReglu;
        out.first_node = i;
        out.end_node = i + 4U;
        out.clamp_min = min_v;
        out.clamp_max = max_v;
        out.reglu_input = in_id;
        out.reglu_gate_channels = static_cast<std::size_t>(g);
        out.output_tensor = mul.outputs[0];
        return true;
    }

    bool try_pointwise_epilogue(std::size_t i, FusedStep& out) const
    {
        if (i + 2U > graph_.nodes.size())
            return false;
        const Node& conv = graph_.nodes[i];
        if (!is_standard_conv(conv, false))
            return false;
        const TensorId conv_out = conv.outputs[0];
        if (!is_internal_only(conv_out, 1U))
            return false;
        const TensorDescriptor* ct = tensor(conv_out);
        if (!ct)
            return false;

        const Node& next = graph_.nodes[i + 1];
        if (next.op == Op::LEAKY_RELU && next.inputs.size() == 1U &&
            next.inputs[0] == conv_out && next.outputs.size() == 1U) {
            out = FusedStep{};
            out.kind = FusedStepKind::kConvWithEpilogue;
            out.first_node = i;
            out.end_node = i + 2U;
            out.epilogue_kind = GemmEpilogueKind::kLeakyRelu;
            out.alpha = alpha_of(next);
            out.conv_input = conv.inputs[0];
            out.output_tensor = next.outputs[0];
            return true;
        }
        float min_v = 0.0F;
        float max_v = 0.0F;
        if (match_clip_constants(next, min_v, max_v) && next.inputs[0] == conv_out) {
            out = FusedStep{};
            out.kind = FusedStepKind::kConvWithEpilogue;
            out.first_node = i;
            out.end_node = i + 2U;
            out.epilogue_kind = GemmEpilogueKind::kClamp;
            out.clamp_min = min_v;
            out.clamp_max = max_v;
            out.conv_input = conv.inputs[0];
            out.output_tensor = next.outputs[0];
            return true;
        }
        TensorId residual_id = 0;
        if (match_other_in_pair(next, Op::ADD, conv_out, residual_id) &&
            residual_id != conv_out) {
            const TensorDescriptor* rt = tensor(residual_id);
            if (rt && !is_init(rt) && rt->dtype == DataType::FP16 &&
                rt->shape == ct->shape) {
                out = FusedStep{};
                out.kind = FusedStepKind::kConvWithEpilogue;
                out.first_node = i;
                out.end_node = i + 2U;
                out.epilogue_kind = GemmEpilogueKind::kResidualAdd;
                out.conv_input = conv.inputs[0];
                out.residual_input = residual_id;
                out.output_tensor = next.outputs[0];
                return true;
            }
        }
        return false;
    }

    bool try_feature_update(std::size_t i, std::vector<bool>& elided, FusedStep& out) const
    {
        if (i + 11U > graph_.nodes.size())
            return false;
        const Node& s0 = graph_.nodes[i];
        if (s0.op != Op::SLICE || s0.inputs.empty())
            return false;
        const TensorId in_id = s0.inputs[0];
        const TensorDescriptor* it = tensor(in_id);
        if (!it || it->dtype != DataType::FP16 || it->shape.size() != 4U ||
            it->shape[1] <= 0 || (it->shape[1] % 3) != 0)
            return false;
        const std::int64_t ch = it->shape[1] / 3;

        if (!match_channel_slice(graph_.nodes[i + 0], in_id, 0, ch) ||
            !match_channel_slice(graph_.nodes[i + 1], in_id, ch, ch * 2) ||
            !match_channel_slice(graph_.nodes[i + 2], in_id, ch * 2, ch * 3))
            return false;
        const TensorId a = graph_.nodes[i + 0].outputs[0];
        const TensorId b = graph_.nodes[i + 1].outputs[0];
        const TensorId cg = graph_.nodes[i + 2].outputs[0];
        if (!is_internal_only(a, 1U) || !is_internal_only(b, 1U) || !is_internal_only(cg, 1U))
            return false;

        const Node& sig0 = graph_.nodes[i + 3];
        const Node& sig1 = graph_.nodes[i + 4];
        if (sig0.op != Op::SIGMOID || sig0.inputs.size() != 1U || sig0.inputs[0] != b || sig0.outputs.size() != 1U ||
            sig1.op != Op::SIGMOID || sig1.inputs.size() != 1U || sig1.inputs[0] != cg || sig1.outputs.size() != 1U)
            return false;
        const TensorId sig_b = sig0.outputs[0];
        const TensorId sig_c = sig1.outputs[0];
        if (!is_internal_only(sig_b, 2U) || !is_internal_only(sig_c, 1U))
            return false;

        TensorId history = 0;
        if (!match_other_in_pair(graph_.nodes[i + 5], Op::MUL, sig_b, history))
            return false;
        const TensorDescriptor* ht = tensor(history);
        const TensorDescriptor* at = tensor(a);
        if (!ht || is_init(ht) || ht->dtype != DataType::FP16 || ht->shape != at->shape)
            return false;
        const TensorId w_b = graph_.nodes[i + 5].outputs[0];
        if (!is_internal_only(w_b, 1U))
            return false;

        const Node& sub = graph_.nodes[i + 6];
        if (sub.op != Op::SUB || sub.inputs.size() != 2U || sub.outputs.size() != 1U ||
            sub.inputs[1] != sig_b)
            return false;
        float one_val = 0.0F;
        if (!read_fp16_scalar_init(sub.inputs[0], one_val) || std::fabs(one_val - 1.0F) > 1e-4F)
            return false;
        const TensorId inv_gate = sub.outputs[0];
        if (!is_internal_only(inv_gate, 1U))
            return false;

        if (!match_pair(graph_.nodes[i + 7], Op::MUL, inv_gate, a))
            return false;
        const TensorId w_a = graph_.nodes[i + 7].outputs[0];
        if (!is_internal_only(w_a, 1U))
            return false;

        if (!match_pair(graph_.nodes[i + 8], Op::ADD, w_b, w_a))
            return false;
        const TensorId blended = graph_.nodes[i + 8].outputs[0];

        if (!match_pair(graph_.nodes[i + 9], Op::MUL, sig_c, blended))
            return false;
        const TensorId gated = graph_.nodes[i + 9].outputs[0];
        if (!is_internal_only(gated, 1U))
            return false;

        TensorId scale = 0;
        if (!match_other_in_pair(graph_.nodes[i + 10], Op::MUL, gated, scale))
            return false;
        const TensorDescriptor* st = tensor(scale);
        if (!st || st->dtype != DataType::FP16 ||
            st->shape != std::vector<std::int64_t>{1, ch, 1, 1})
            return false;
        const TensorId scaled = graph_.nodes[i + 10].outputs[0];

        // Case A: CONCAT immediately at i + 11 -> single kFeatureUpdate step.
        if (i + 12U <= graph_.nodes.size() &&
            is_internal_only(scaled, 1U) && is_internal_only(blended, 2U)) {
            const Node& concat = graph_.nodes[i + 11];
            if (concat.op == Op::CONCAT && axis_of(concat) == 1 &&
                concat.inputs.size() == 2U && concat.inputs[0] == scaled &&
                concat.inputs[1] == blended && concat.outputs.size() == 1U) {
                out = FusedStep{};
                out.kind = FusedStepKind::kFeatureUpdate;
                out.first_node = i;
                out.end_node = i + 12U;
                out.fu_input = in_id;
                out.fu_history = history;
                out.fu_scale = scale;
                out.fu_feature_out = concat.outputs[0];
                out.output_tensor = concat.outputs[0];
                out.fu_channels = static_cast<std::size_t>(ch);
                return true;
            }
        }

        // Case B: deferred CONCAT downstream, with `blended` used only in that CONCAT.
        if (!is_graph_output(scaled) && is_internal_only(blended, 2U)) {
            for (std::size_t j = i + 11U; j < graph_.nodes.size(); ++j) {
                const Node& concat = graph_.nodes[j];
                if (concat.op == Op::CONCAT && axis_of(concat) == 1 &&
                    concat.inputs.size() == 2U && concat.inputs[0] == scaled &&
                    concat.inputs[1] == blended && concat.outputs.size() == 1U) {
                    elided[j] = true;
                    out = FusedStep{};
                    out.kind = FusedStepKind::kFeatureUpdateOutputs;
                    out.first_node = i;
                    out.end_node = i + 11U;
                    out.fu_input = in_id;
                    out.fu_history = history;
                    out.fu_scale = scale;
                    out.fu_scaled_out = scaled;
                    out.fu_blended_out = blended;
                    out.fu_feature_out = concat.outputs[0];
                    out.output_tensor = concat.outputs[0];
                    out.fu_channels = static_cast<std::size_t>(ch);
                    out.deferred_concat_node = j;
                    return true;
                }
            }
        }
        return false;
    }

    bool try_y0_tail(std::size_t i, FusedStep& out) const
    {
        if (i + 19U > graph_.nodes.size())
            return false;
        const Node& n0 = graph_.nodes[i];
        if (n0.op != Op::SLICE || n0.inputs.empty())
            return false;
        const TensorId prior = n0.inputs[0];
        const TensorDescriptor* pt = tensor(prior);
        if (!pt || pt->dtype != DataType::FP16 || pt->shape.size() != 4U ||
            pt->shape[1] <= 0 || (pt->shape[1] % 4) != 0)
            return false;
        const std::int64_t q = pt->shape[1] / 4;
        const std::int64_t norm_c = q * 2;
        const std::int64_t prior_c = q * 4;

        if (!match_channel_slice(n0, prior, 0, norm_c) ||
            !match_channel_slice(graph_.nodes[i + 1], prior, norm_c, prior_c))
            return false;
        const TensorId p_head = n0.outputs[0];
        const TensorId p_tail = graph_.nodes[i + 1].outputs[0];
        if (!is_internal_only(p_head, 1U) || !is_internal_only(p_tail, 3U))
            return false;

        float clip_min = 0.0F;
        float clip_max = 0.0F;
        if (!match_clip_constants(graph_.nodes[i + 2], clip_min, clip_max) ||
            graph_.nodes[i + 2].inputs[0] != p_head || clip_max < 65504.0F)
            return false;
        const TensorId clipped = graph_.nodes[i + 2].outputs[0];
        if (is_graph_output(clipped))
            return false;

        const Node& rec = graph_.nodes[i + 3];
        if (rec.op != Op::RECIPROCAL || rec.inputs.size() != 1U ||
            rec.inputs[0] != clipped || rec.outputs.size() != 1U)
            return false;
        TensorId inv = rec.outputs[0];
        if (!is_internal_only(inv, 1U))
            return false;

        std::size_t cursor = i + 4U;
        if (cursor + 15U <= graph_.nodes.size() && graph_.nodes[cursor].op == Op::MUL &&
            graph_.nodes[cursor].inputs.size() == 2U && graph_.nodes[cursor].outputs.size() == 1U) {
            TensorId other = 0;
            float one_val = 0.0F;
            if (match_other_in_pair(graph_.nodes[cursor], Op::MUL, inv, other) &&
                read_fp16_scalar_init(other, one_val) && std::fabs(one_val - 1.0F) <= 1e-4F) {
                inv = graph_.nodes[cursor].outputs[0];
                if (!is_internal_only(inv, 1U))
                    return false;
                ++cursor;
            }
        }
        if (cursor + 15U > graph_.nodes.size())
            return false;

        TensorId latent = 0;
        if (!match_other_in_pair(graph_.nodes[cursor + 0], Op::MUL, inv, latent))
            return false;
        const TensorDescriptor* lt = tensor(latent);
        const std::vector<std::int64_t> norm_shape = {pt->shape[0], norm_c, pt->shape[2], pt->shape[3]};
        const std::vector<std::int64_t> q_shape = {pt->shape[0], q, pt->shape[2], pt->shape[3]};
        if (!lt || is_init(lt) || lt->dtype != DataType::FP16 || lt->shape != norm_shape)
            return false;
        const TensorId normalized = graph_.nodes[cursor + 0].outputs[0];

        const Node& sub = graph_.nodes[cursor + 1];
        if (sub.op != Op::SUB || sub.inputs.size() != 2U || sub.outputs.size() != 1U ||
            sub.inputs[0] != normalized || sub.inputs[1] != p_tail)
            return false;
        const TensorId diff = sub.outputs[0];
        if (!is_internal_only(diff, 1U))
            return false;

        TensorId quant_scale = 0;
        if (!match_other_in_pair(graph_.nodes[cursor + 2], Op::MUL, diff, quant_scale))
            return false;
        const TensorDescriptor* qst = tensor(quant_scale);
        if (!qst || qst->dtype != DataType::FP16 || qst->shape != norm_shape)
            return false;
        const TensorId scaled = graph_.nodes[cursor + 2].outputs[0];
        if (!is_internal_only(scaled, 2U))
            return false;

        if (!match_channel_slice(graph_.nodes[cursor + 3], scaled, 0, q) ||
            !match_channel_slice(graph_.nodes[cursor + 4], scaled, q, norm_c))
            return false;
        const TensorId s0 = graph_.nodes[cursor + 3].outputs[0];
        const TensorId s1 = graph_.nodes[cursor + 4].outputs[0];
        if (!is_internal_only(s0, 1U) || !is_internal_only(s1, 1U))
            return false;

        if (!match_pair(graph_.nodes[cursor + 5], Op::ADD, s0, s1))
            return false;
        const TensorId sum_s = graph_.nodes[cursor + 5].outputs[0];
        if (!is_internal_only(sum_s, 1U))
            return false;

        const Node& rnd = graph_.nodes[cursor + 6];
        if (rnd.op != Op::ROUND || rnd.inputs.size() != 1U || rnd.inputs[0] != sum_s ||
            rnd.outputs.size() != 1U)
            return false;
        const TensorId y_raw = rnd.outputs[0];

        if (!match_channel_slice(graph_.nodes[cursor + 7], p_tail, 0, q) ||
            !match_channel_slice(graph_.nodes[cursor + 8], p_tail, q, norm_c))
            return false;
        const TensorId pt0 = graph_.nodes[cursor + 7].outputs[0];
        const TensorId pt1 = graph_.nodes[cursor + 8].outputs[0];
        if (!is_internal_only(pt0, 1U) || !is_internal_only(pt1, 1U))
            return false;

        if (!match_pair(graph_.nodes[cursor + 9], Op::ADD, y_raw, pt0))
            return false;
        const TensorId a0 = graph_.nodes[cursor + 9].outputs[0];
        if (!is_internal_only(a0, 1U))
            return false;
        TensorId us0 = 0;
        if (!match_other_in_pair(graph_.nodes[cursor + 10], Op::MUL, a0, us0))
            return false;
        const TensorDescriptor* ust0 = tensor(us0);
        if (!ust0 || ust0->dtype != DataType::FP16 || ust0->shape != q_shape)
            return false;
        const TensorId u0 = graph_.nodes[cursor + 10].outputs[0];
        if (!is_internal_only(u0, 1U))
            return false;

        if (!match_pair(graph_.nodes[cursor + 11], Op::ADD, y_raw, pt1))
            return false;
        const TensorId a1 = graph_.nodes[cursor + 11].outputs[0];
        if (!is_internal_only(a1, 1U))
            return false;
        TensorId us1 = 0;
        if (!match_other_in_pair(graph_.nodes[cursor + 12], Op::MUL, a1, us1))
            return false;
        const TensorDescriptor* ust1 = tensor(us1);
        if (!ust1 || ust1->dtype != DataType::FP16 || ust1->shape != q_shape)
            return false;
        const TensorId u1 = graph_.nodes[cursor + 12].outputs[0];
        if (!is_internal_only(u1, 1U))
            return false;

        const Node& c2 = graph_.nodes[cursor + 13];
        if (c2.op != Op::CONCAT || axis_of(c2) != 1 || c2.inputs.size() != 2U ||
            c2.inputs[0] != u0 || c2.inputs[1] != u1 || c2.outputs.size() != 1U)
            return false;
        const TensorId concat2 = c2.outputs[0];

        const Node& c3 = graph_.nodes[cursor + 14];
        if (c3.op != Op::CONCAT || axis_of(c3) != 1 || c3.inputs.size() != 2U ||
            c3.inputs[0] != concat2 || c3.inputs[1] != prior || c3.outputs.size() != 1U)
            return false;

        out = FusedStep{};
        out.kind = FusedStepKind::kY0Tail;
        out.first_node = i;
        out.end_node = cursor + 15U;
        out.y_latent_or_norm = latent;
        out.y_prior = prior;
        out.y_quant_scale = quant_scale;
        out.y_update_scale0 = us0;
        out.y_update_scale1 = us1;
        out.y_normalized_out = normalized;
        out.y_clipped_io = clipped;
        out.y_raw_out = y_raw;
        out.y_concat2_io = concat2;
        out.y_concat3_or_out = c3.outputs[0];
        out.output_tensor = c3.outputs[0];
        out.y_clip_min = clip_min;
        out.y_quant_channels = static_cast<std::size_t>(q);
        return true;
    }

    bool try_y1_tail(std::size_t i, FusedStep& out) const
    {
        if (i + 15U > graph_.nodes.size())
            return false;
        const Node& sub = graph_.nodes[i];
        if (sub.op != Op::SUB || sub.inputs.size() != 2U || sub.outputs.size() != 1U)
            return false;
        const TensorId normalized = sub.inputs[0];
        const TensorId prior = sub.inputs[1];
        const TensorDescriptor* nt = tensor(normalized);
        const TensorDescriptor* pt = tensor(prior);
        if (!nt || !pt || nt->dtype != DataType::FP16 || pt->dtype != DataType::FP16 ||
            nt->shape.size() != 4U || nt->shape != pt->shape ||
            nt->shape[1] <= 0 || (nt->shape[1] % 2) != 0)
            return false;
        const std::int64_t norm_c = nt->shape[1];
        const std::int64_t q = norm_c / 2;
        const std::vector<std::int64_t> q_shape = {nt->shape[0], q, nt->shape[2], nt->shape[3]};

        const TensorId diff = sub.outputs[0];
        if (!is_internal_only(diff, 1U))
            return false;

        TensorId quant_scale = 0;
        if (!match_other_in_pair(graph_.nodes[i + 1], Op::MUL, diff, quant_scale))
            return false;
        const TensorDescriptor* qst = tensor(quant_scale);
        if (!qst || qst->dtype != DataType::FP16 || qst->shape != nt->shape)
            return false;
        const TensorId scaled = graph_.nodes[i + 1].outputs[0];
        if (!is_internal_only(scaled, 2U))
            return false;

        if (!match_channel_slice(graph_.nodes[i + 2], scaled, 0, q) ||
            !match_channel_slice(graph_.nodes[i + 3], scaled, q, norm_c))
            return false;
        const TensorId s0 = graph_.nodes[i + 2].outputs[0];
        const TensorId s1 = graph_.nodes[i + 3].outputs[0];
        if (!is_internal_only(s0, 1U) || !is_internal_only(s1, 1U))
            return false;

        if (!match_pair(graph_.nodes[i + 4], Op::ADD, s0, s1))
            return false;
        const TensorId sum_s = graph_.nodes[i + 4].outputs[0];
        if (!is_internal_only(sum_s, 1U))
            return false;

        const Node& rnd = graph_.nodes[i + 5];
        if (rnd.op != Op::ROUND || rnd.inputs.size() != 1U || rnd.inputs[0] != sum_s ||
            rnd.outputs.size() != 1U)
            return false;
        const TensorId y_raw = rnd.outputs[0];

        if (!match_channel_slice(graph_.nodes[i + 6], prior, 0, q) ||
            !match_channel_slice(graph_.nodes[i + 7], prior, q, norm_c))
            return false;
        const TensorId pt0 = graph_.nodes[i + 6].outputs[0];
        const TensorId pt1 = graph_.nodes[i + 7].outputs[0];
        if (!is_internal_only(pt0, 1U) || !is_internal_only(pt1, 1U))
            return false;

        if (!match_pair(graph_.nodes[i + 8], Op::ADD, y_raw, pt0))
            return false;
        const TensorId a0 = graph_.nodes[i + 8].outputs[0];
        if (!is_internal_only(a0, 1U))
            return false;
        TensorId us0 = 0;
        if (!match_other_in_pair(graph_.nodes[i + 9], Op::MUL, a0, us0))
            return false;
        const TensorDescriptor* ust0 = tensor(us0);
        if (!ust0 || ust0->dtype != DataType::FP16 || ust0->shape != q_shape)
            return false;
        const TensorId u0 = graph_.nodes[i + 9].outputs[0];
        if (!is_internal_only(u0, 1U))
            return false;

        if (!match_pair(graph_.nodes[i + 10], Op::ADD, y_raw, pt1))
            return false;
        const TensorId a1 = graph_.nodes[i + 10].outputs[0];
        if (!is_internal_only(a1, 1U))
            return false;
        TensorId us1 = 0;
        if (!match_other_in_pair(graph_.nodes[i + 11], Op::MUL, a1, us1))
            return false;
        const TensorDescriptor* ust1 = tensor(us1);
        if (!ust1 || ust1->dtype != DataType::FP16 || ust1->shape != q_shape)
            return false;
        const TensorId u1 = graph_.nodes[i + 11].outputs[0];
        if (!is_internal_only(u1, 1U))
            return false;

        const Node& c2_node = graph_.nodes[i + 12];
        if (c2_node.op != Op::CONCAT || axis_of(c2_node) != 1 ||
            c2_node.inputs.size() != 2U || c2_node.inputs[0] != u0 ||
            c2_node.inputs[1] != u1 || c2_node.outputs.size() != 1U)
            return false;
        const TensorId concat_out = c2_node.outputs[0];
        if (!is_internal_only(concat_out, 1U))
            return false;

        TensorId concat2 = 0;
        if (!match_other_in_pair(graph_.nodes[i + 13], Op::ADD, concat_out, concat2))
            return false;
        const TensorDescriptor* c2t = tensor(concat2);
        if (!c2t || is_init(c2t) || c2t->dtype != DataType::FP16 || c2t->shape != nt->shape)
            return false;
        const TensorId added = graph_.nodes[i + 13].outputs[0];
        if (!is_internal_only(added, 1U))
            return false;

        TensorId clipped = 0;
        if (!match_other_in_pair(graph_.nodes[i + 14], Op::MUL, added, clipped))
            return false;
        const TensorDescriptor* clt = tensor(clipped);
        if (!clt || is_init(clt) || clt->dtype != DataType::FP16 || clt->shape != nt->shape)
            return false;

        out = FusedStep{};
        out.kind = FusedStepKind::kY1Tail;
        out.first_node = i;
        out.end_node = i + 15U;
        out.y_latent_or_norm = normalized;
        out.y_prior = prior;
        out.y_quant_scale = quant_scale;
        out.y_update_scale0 = us0;
        out.y_update_scale1 = us1;
        out.y_clipped_io = clipped;
        out.y_raw_out = y_raw;
        out.y_concat2_io = concat2;
        out.y_concat3_or_out = graph_.nodes[i + 14].outputs[0];
        out.output_tensor = graph_.nodes[i + 14].outputs[0];
        out.y_quant_channels = static_cast<std::size_t>(q);
        return true;
    }

    const aot::LoadedGraph& loaded_;
    const aot::GraphDescriptor& graph_;
    std::unordered_map<TensorId, std::size_t> consumer_count_;
    std::unordered_set<TensorId> graph_outputs_;
};

}  // namespace

std::vector<FusedStep> build_fused_schedule(const aot::LoadedGraph& loaded,
                                            bool enable_fusion)
{
    return GraphMatcher(loaded).build(enable_fusion);
}

}  // namespace mlvc::cpu_backend
