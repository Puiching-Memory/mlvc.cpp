#pragma once

#include "aligned_buffer.hpp"
#include "cpu_features.hpp"
#include "fusion_matcher.hpp"
#include "kernels_common.hpp"
#include "thread_pool.hpp"
#include "weight_cache.hpp"

#include "core/assets.hpp"
#include "core/half.hpp"
#include "core/tensor.hpp"
#include "core/yuv.hpp"
#include "runtime/aot_graph.hpp"
#include "runtime/backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mlvc::cpu_backend {

class CpuExecutionPlan final {
public:
    CpuExecutionPlan(const ModelAssetReader& assets,
                     const std::string& model_name,
                     const ModelExecutionConfig& config,
                     int num_threads,
                     bool share_thread_pool = false);

    std::vector<Tensor> run(const std::vector<Tensor>& inputs);
    void reset_state();

    void configure_codec_io(const CodecIoConfig& config);
    std::size_t codec_slot_count() const noexcept;
    MutableYuv420FrameView encoder_input_yuv(std::size_t slot);
    void submit_encoder(std::size_t slot, int shifted_q);
    std::vector<TensorView> encoder_outputs(std::size_t slot) const;
    std::vector<MutableTensorView> decoder_inputs(std::size_t slot);
    void submit_decoder(std::size_t slot, int shifted_q);
    Yuv420FrameView decoder_output_yuv(std::size_t slot) const;
    void wait_codec_slot(std::size_t slot);

    GemmMode gemm_mode() const noexcept { return kernels_->mode; }
    std::size_t thread_count() const noexcept { return pool_->thread_count(); }
    std::size_t step_count() const noexcept { return steps_.size(); }
    std::size_t arena_bytes() const noexcept { return arena_.size(); }
    const void* arena_data() const noexcept { return arena_.data(); }
    const void* weight_bank_identity() const noexcept { return weight_bank_.get(); }

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
        const Float16Storage* const_fp16 = nullptr;
        const std::int32_t* const_int32 = nullptr;
        const std::int64_t* const_int64 = nullptr;
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
        const Float16Storage* packed_weights = nullptr;
    };

    struct SingleOpInfo {
        aot::Op op = aot::Op::ADD;
        std::vector<aot::TensorId> inputs;
        aot::TensorId output = 0;

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

        int slice_id[4] = {};
        int slice_od[4] = {};
        int slice_starts[4] = {};
        int slice_steps[4] = {};

        std::vector<std::vector<int>> concat_in_dims;
        int concat_od[4] = {};

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

    struct CodecSlot {
        std::vector<std::uint8_t> yuv;
        std::vector<std::vector<std::byte>> inputs;
        std::vector<std::vector<std::byte>> outputs;
    };

    static bool needs_nhwc_transpose(const std::vector<std::int64_t>& shape,
                                     aot::DataType dt) noexcept;

    const TensorMeta* find_meta(aot::TensorId id) const noexcept;
    TensorMeta& require_meta(aot::TensorId id);
    const TensorMeta& require_meta(aot::TensorId id) const;
    void* resolve_mut_ptr(aot::TensorId id);
    const void* resolve_const_ptr(aot::TensorId id) const;
    Float16Storage* fp16_mut(aot::TensorId id);
    const Float16Storage* fp16_const(aot::TensorId id) const;

    void write_graph_input_bytes(const TensorMeta& meta, const void* src_raw,
                                 std::byte* dst_raw);
    void write_graph_output_bytes(const TensorMeta& meta, const void* src_raw,
                                  std::byte* dst_raw);

    void compile_steps(const aot::LoadedGraph& loaded,
                       const ModelExecutionConfig& config);
    void allocate_arena();
    void execute_step(const CompiledStep& step);
    void execute_all_steps_and_swap_states();

    const KernelSet* kernels_ = nullptr;
    std::shared_ptr<CpuThreadPool> pool_;
    std::shared_ptr<const SharedWeightBank> weight_bank_;
    std::string model_name_;
    std::mutex exec_mutex_;

    std::vector<TensorMeta> metas_;
    std::vector<int> id_to_meta_;
    std::vector<aot::TensorId> input_ids_;
    std::vector<aot::TensorId> output_ids_;
    std::vector<std::size_t> external_input_indices_;
    std::vector<ResolvedStateBinding> state_bindings_;
    bool state_initialized_ = false;

    std::vector<AlignedByteBuffer> input_buffers_;
    std::vector<AlignedByteBuffer> state_output_buffers_;
    AlignedByteBuffer arena_;
    std::vector<CompiledStep> steps_;

    CodecIoConfig codec_io_{};
    bool codec_io_configured_ = false;
    std::array<Float16Storage, 256> codec_byte_lut_{};
    std::size_t x_input_graph_idx_ = 0;
    std::size_t q_input_graph_idx_ = 0;
    aot::TensorId x_hat_output_id_ = 0;
    std::vector<CodecSlot> codec_slots_;
};

}  // namespace mlvc::cpu_backend
