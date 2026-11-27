#pragma once

#include "runtime/aot_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlvc::webgpu_backend {

enum class EpilogueKind {
    kNone = 0,
    kLeakyRelu = 1,
    kClamp = 2,
    kResidualAdd = 3,
    kReglu = 4,
};

enum class FusedStepKind {
    kSingleNode,
    kConvWithEpilogue,
    kReglu,
    kFeatureUpdate,
    kFeatureUpdateOutputs,
    kY0Tail,
    kY1Tail,
    kDecY0Tail,
    kDecY1Tail,
};

struct FusedStep {
    FusedStepKind kind = FusedStepKind::kSingleNode;
    std::size_t first_node = 0;
    std::size_t end_node = 0;

    // kConvWithEpilogue (conv node index is first_node):
    EpilogueKind epilogue_kind = EpilogueKind::kNone;
    float alpha = 0.0F;
    float clamp_min = 0.0F;
    float clamp_max = 0.0F;
    aot::TensorId conv_input = 0;
    aot::TensorId residual_input = 0;
    aot::TensorId output_tensor = 0;

    // kReglu:
    aot::TensorId reglu_input = 0;
    std::size_t reglu_gate_channels = 0;

    // kFeatureUpdate / kFeatureUpdateOutputs:
    aot::TensorId fu_input = 0;
    aot::TensorId fu_history = 0;
    aot::TensorId fu_scale = 0;
    aot::TensorId fu_scaled_out = 0;
    aot::TensorId fu_blended_out = 0;
    aot::TensorId fu_feature_out = 0;
    std::size_t fu_channels = 0;
    std::size_t deferred_concat_node = static_cast<std::size_t>(-1);

    // kY0Tail / kY1Tail / kDecY0Tail / kDecY1Tail:
    aot::TensorId y_latent_or_norm = 0;
    aot::TensorId y_prior = 0;
    aot::TensorId y_quant_scale = 0;
    aot::TensorId y_update_scale0 = 0;
    aot::TensorId y_update_scale1 = 0;
    aot::TensorId y_normalized_out = 0;
    aot::TensorId y_clipped_io = 0;
    aot::TensorId y_raw_io = 0;          // output on enc, input on dec
    aot::TensorId y_concat2_io = 0;
    aot::TensorId y_concat3_or_out = 0;
    float y_clip_min = 0.0F;
    float y_clip_max = 65504.0F;
    std::size_t y_quant_channels = 0;
};

// Matches all high-value multi-node subgraphs in topological order while
// respecting live consumer counts, graph outputs, and device binding limits.
std::vector<FusedStep> build_fused_schedule(
    const aot::LoadedGraph& loaded,
    bool enable_fusion,
    std::uint32_t max_storage_buffers_per_stage = 12U);

}  // namespace mlvc::webgpu_backend
