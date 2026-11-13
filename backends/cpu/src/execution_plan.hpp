#pragma once

#include "cpu_features.hpp"
#include "fusion_matcher.hpp"
#include "kernels_common.hpp"
#include "thread_pool.hpp"

#include "mlvc/core/assets.hpp"
#include "mlvc/core/tensor.hpp"
#include "mlvc/runtime/aot_graph.hpp"
#include "mlvc/runtime/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mlvc::cpu_backend {

class CpuExecutionPlan final {
public:
    CpuExecutionPlan(const ModelAssetReader& assets,
                     const std::string& model_name,
                     const ModelExecutionConfig& config,
                     int num_threads);

    std::vector<Tensor> run(const std::vector<Tensor>& inputs);
    void reset_state();

    GemmMode gemm_mode() const noexcept { return kernels_->mode; }
    std::size_t thread_count() noexcept { return pool_.thread_count(); }
    std::size_t step_count() const noexcept { return steps_.size(); }
    std::size_t arena_bytes() const noexcept { return arena_.size(); }

private:
    struct TensorMeta {
        aot::TensorId id = 0;
        std::string name;
        aot::DataType dtype = aot::DataType::FP16;
        std::vector<std::int64_t> shape;
        std::size_t elem_count = 0;
        std::size_t byte_size = 0;
        bool is_initializer = false;
        bool is_graph_input = false;
        bool is_graph_output = false;
        int state_input_slot = -1;
        int state_output_slot = -1;
        std::size_t graph_input_pos = 0;
        std::size_t arena_offset = 0;
        std::vector<Float16Storage> const_fp16;
        std::vector<std::int32_t> const_int32;
        std::vector<std::int64_t> const_int64;
    };

    struct ConvExecInfo {
        bool is_depthwise = false;
        bool is_1x1 = false;
        bool is_dw_3x3_s1p1 = false;
        int n = 0;
        int ic = 0;
        int ih = 0;
        int iw = 0;
        int oc = 0;
        int oh = 0;
        int ow = 0;
        int kh = 0;
        int kw = 0;
        int sh = 1;
        int sw = 1;
        int ph = 0;
        int pw = 0;
        int groups = 1;
        std::vector<Float16Storage> packed_weights;
    };

    struct SingleOpInfo {
        aot::Op op = aot::Op::ADD;
        std::vector<aot::TensorId> inputs;
        aot::TensorId output = 0;

        // Op-specific predecoded metadata:
        float alpha = 0.0F;
        float clamp_min = 0.0F;
        float clamp_max = 0.0F;
        int axis = 0;
        int block_size = 1;
        BinaryKind binary_kind = BinaryKind::kAdd;
        bool scalar_or_chan_bcast = false;
        bool is_scalar_bcast = false;
        bool is_chan_bcast = false;
        bool bcast_on_lhs = false;
        std::size_t bcast_spatial = 0;
        std::size_t bcast_channels = 0;

        // Predecoded Slice NCHW metadata:
        int slice_id[4] = {};
        int slice_od[4] = {};
        int slice_starts[4] = {};
        int slice_steps[4] = {};

        // Predecoded Concat dims:
        std::vector<std::vector<int>> concat_in_dims;
        int concat_od[4] = {};

        // Conv metadata (for unfused CONV nodes):
        ConvExecInfo conv;
    };

    struct CompiledStep {
        FusedStep fused;
        ConvExecInfo fused_conv;
        SingleOpInfo single;
    };

    struct ResolvedStateBinding {
        std::size_t input_index = 0;
        std::size_t output_index = 0;
        aot::TensorId input_id = 0;
        aot::TensorId output_id = 0;
        std::size_t byte_size = 0;
    };

    const TensorMeta* find_meta(aot::TensorId id) const noexcept;
    TensorMeta& require_meta(aot::TensorId id);
    const TensorMeta& require_meta(aot::TensorId id) const;
    void* resolve_mut_ptr(aot::TensorId id);
    const void* resolve_const_ptr(aot::TensorId id) const;
    Float16Storage* fp16_mut(aot::TensorId id);
    const Float16Storage* fp16_const(aot::TensorId id) const;

    void compile_steps(const aot::LoadedGraph& loaded, const ModelExecutionConfig& config);
    void allocate_arena();
    void execute_step(const CompiledStep& step);

    const KernelSet* kernels_ = nullptr;
    CpuThreadPool pool_;
    std::mutex exec_mutex_;

    std::vector<TensorMeta> metas_;
    std::vector<int> id_to_meta_;
    std::vector<aot::TensorId> input_ids_;
    std::vector<aot::TensorId> output_ids_;
    std::vector<std::size_t> external_input_indices_;
    std::vector<ResolvedStateBinding> state_bindings_;
    bool state_initialized_ = false;

    std::vector<std::vector<std::byte>> input_buffers_;
    std::vector<std::vector<std::byte>> state_output_buffers_;
    std::vector<std::byte> arena_;
    std::vector<CompiledStep> steps_;
};

}  // namespace mlvc::cpu_backend
