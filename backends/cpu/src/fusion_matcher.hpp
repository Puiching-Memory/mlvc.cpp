#pragma once

#include "kernels_common.hpp"
#include "mlvc/runtime/aot_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlvc::cpu_backend {

enum class FusedStepKind {
    kSingleNode,
    kConvWithEpilogue,
    kReglu,
    kFeatureUpdate,
    kFeatureUpdateOutputs,
    kY0Tail,
    kY1Tail,
};

struct FusedStep {
    FusedStepKind kind = FusedStepKind::kSingleNode;
    std::size_t first_node = 0;
    std::size_t end_node = 0;

    // Kernels & tensor IDs populated per FusedStepKind.
    // kConvWithEpilogue: conv node index is first_node.
    GemmEpilogueKind epilogue_kind = GemmEpilogueKind::kNone;
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

    // kY0Tail / kY1Tail:
    aot::TensorId y_latent_or_norm = 0;
    aot::TensorId y_prior = 0;
    aot::TensorId y_quant_scale = 0;
    aot::TensorId y_update_scale0 = 0;
    aot::TensorId y_update_scale1 = 0;
    aot::TensorId y_normalized_out = 0;  // kY0Tail output / unused on kY1Tail
    aot::TensorId y_clipped_io = 0;      // kY0Tail output / kY1Tail input
    aot::TensorId y_raw_out = 0;
    aot::TensorId y_concat2_io = 0;      // kY0Tail output / kY1Tail input
    aot::TensorId y_concat3_or_out = 0;  // kY0Tail concat3 / kY1Tail final output
    float y_clip_min = 0.0F;
    std::size_t y_quant_channels = 0;
};

// Matches high-value multi-node patterns inside `loaded.graph` in topological
// order while respecting live consumer counts and graph outputs. When
// `enable_fusion` is false, emits 1:1 `kSingleNode` steps preserving original
// node order for unfused equivalence validation.
std::vector<FusedStep> build_fused_schedule(const aot::LoadedGraph& loaded,
                                            bool enable_fusion);

}  // namespace mlvc::cpu_backend
