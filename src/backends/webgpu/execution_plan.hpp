#pragma once

#include "fusion_matcher.hpp"
#include "wgpu_context.hpp"

#include "core/assets.hpp"
#include "core/tensor.hpp"
#include "core/yuv.hpp"
#include "runtime/aot_graph.hpp"
#include "runtime/backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlvc::webgpu_backend {

struct SurfaceRenderConfig {
    int width = 0;
    int height = 0;
    int model_width = 0;
    int model_height = 0;
    float pixel_range = 1.0F;
    FramePadding padding;
    WGPUTextureFormat target_format = WGPUTextureFormat_BGRA8Unorm;
    std::uint32_t color_matrix = 0U;  // 0=BT.709 full, 1=BT.601 full, 2=BT.709 limited
};

class WebGpuExecutionPlan final {
public:
    WebGpuExecutionPlan(std::shared_ptr<WgpuContext> ctx,
                        const ModelAssetReader& assets,
                        const std::string& model_name,
                        const ModelExecutionConfig& config);
    ~WebGpuExecutionPlan();

    WebGpuExecutionPlan(const WebGpuExecutionPlan&) = delete;
    WebGpuExecutionPlan& operator=(const WebGpuExecutionPlan&) = delete;

    std::vector<Tensor> run(const std::vector<Tensor>& inputs);
    void reset_state();

    // BufferedCodecBackend support.
    void configure_codec_io(const CodecIoConfig& config);
    std::size_t codec_slot_count() const noexcept { return codec_slots_.size(); }
    MutableYuv420FrameView encoder_input_yuv(std::size_t slot);
    void submit_encoder(std::size_t slot, int shifted_q);
    std::vector<TensorView> encoder_outputs(std::size_t slot) const;
    std::vector<MutableTensorView> decoder_inputs(std::size_t slot);
    void submit_decoder(std::size_t slot, int shifted_q);
    Yuv420FrameView decoder_output_yuv(std::size_t slot) const;
    void wait_codec_slot(std::size_t slot);

    // Zero-readback GPU texture render path (for browser OffscreenCanvas / native surfaces).
    void configure_surface_output(const SurfaceRenderConfig& config);
    void submit_decoder_to_texture(const std::vector<TensorView>& latents,
                                   int shifted_q,
                                   bool reset_state_before_run,
                                   WGPUTextureView target_view);

    std::size_t step_count() const noexcept { return dispatches_.size(); }
    std::size_t pool_buffer_count() const noexcept { return pool_buffers_.size(); }
    bool uses_f16() const noexcept { return ctx_->caps().has_shader_f16; }
    bool fused_d2s8_active() const noexcept { return fused_d2s8_output_ && d2s8_step_index_ != kInvalidIndex; }

private:
    static constexpr std::size_t kInvalidIndex = static_cast<std::size_t>(-1);

    struct TensorMeta {
        aot::TensorId id = 0;
        std::string name;
        aot::DataType dtype = aot::DataType::DATA_TYPE_UNSPECIFIED;
        std::vector<std::int64_t> shape;
        std::size_t elem_count = 0;
        std::size_t byte_size = 0;
        bool is_initializer = false;
        bool is_graph_input = false;
        bool is_graph_output = false;
        std::size_t graph_input_pos = 0;
        int state_input_slot = -1;
        int state_output_slot = -1;
        int pool_id = -1;
        WGPUBuffer const_buffer = nullptr;
    };

    struct ResolvedStateBinding {
        std::size_t input_index = 0;
        std::size_t output_index = 0;
        aot::TensorId input_id = 0;
        aot::TensorId output_id = 0;
        std::size_t byte_size = 0;
    };

    struct DispatchOp {
        std::string label;
        std::string op_category;
        WGPUComputePipeline pipeline = nullptr;
        std::array<UniqueBindGroup, 2> bind_groups;  // parity 0 & 1
        std::uint32_t wg_x = 1U;
        std::uint32_t wg_y = 1U;
        std::uint32_t wg_z = 1U;
    };

    struct CodecSlot {
        std::vector<std::uint8_t> yuv_bytes;
        std::vector<Tensor> host_tensors;
        std::vector<TensorView> host_output_views;
        std::vector<MutableTensorView> host_input_views;
        UniqueBuffer gpu_yuv_buffer;
        UniqueBuffer gpu_yuv_staging;  // MAP_READ for decoder
        std::vector<UniqueBuffer> gpu_nchw_io_buffers;
        std::vector<UniqueBuffer> gpu_nchw_staging_buffers;
        bool pending = false;
    };

    struct PipelineCacheEntry {
        UniqueBindGroupLayout bgl;
        UniquePipelineLayout layout;
        UniqueComputePipeline pipeline;
    };

    static bool needs_nhwc_transpose(const std::vector<std::int64_t>& shape,
                                     aot::DataType dt) noexcept;

    const TensorMeta& require_meta(aot::TensorId id) const;
    TensorMeta& require_meta(aot::TensorId id);
    WGPUBuffer resolve_buffer(aot::TensorId id, int parity) const;

    const PipelineCacheEntry& get_or_create_pipeline(
        const std::string& key,
        const std::string& wgsl_source,
        const std::vector<WGPUBufferBindingType>& binding_types);

    UniqueBuffer create_uniform_buffer(const void* data, std::size_t size,
                                       std::string_view label);
    UniqueBindGroup create_bind_group(WGPUBindGroupLayout bgl,
                                      const std::vector<WGPUBuffer>& buffers);

    void compile_graph(const aot::LoadedGraph& loaded,
                       const ModelExecutionConfig& config);
    void build_dispatches(const aot::LoadedGraph& loaded,
                          const std::vector<FusedStep>& schedule);

    void encode_compute_schedule(WGPUCommandEncoder enc, int parity,
                                 bool skip_final_d2s8);
    void resolve_and_log_profile();

    std::shared_ptr<WgpuContext> ctx_;
    std::string model_name_;
    bool is_encoder_ = false;
    bool profile_enabled_ = false;
    [[maybe_unused]] std::uint64_t profile_frame_count_ = 0U;

    std::vector<TensorMeta> metas_;
    std::vector<int> id_to_meta_;
    std::vector<aot::TensorId> input_ids_;
    std::vector<aot::TensorId> output_ids_;
    std::vector<std::size_t> external_input_indices_;
    std::vector<ResolvedStateBinding> state_bindings_;

    // GPU resource ownership
    UniqueBuffer dummy_storage_buffer_;
    std::vector<UniqueBuffer> input_buffers_;          // internal NHWC (or int32)
    std::vector<UniqueBuffer> input_nchw_buffers_;     // external NCHW staging for run()
    std::vector<UniqueBuffer> output_nchw_buffers_;    // external NCHW staging for run()
    std::vector<std::array<UniqueBuffer, 2>> state_buffers_;
    std::vector<UniqueBuffer> pool_buffers_;
    std::vector<UniqueBuffer> weight_buffers_;
    std::vector<UniqueBuffer> uniform_buffers_;

    std::unordered_map<std::string, PipelineCacheEntry> pipeline_cache_;
    std::vector<DispatchOp> dispatches_;

    // Optional pre-D2S(8) (+ Clip[0,1]) output fusion for MLVCDecoder
    bool fused_d2s8_output_ = false;
    std::size_t d2s8_step_index_ = kInvalidIndex;
    std::size_t d2s8_clip_step_index_ = kInvalidIndex;
    aot::TensorId d2s8_input_id_ = 0;

    // Ping-pong state management
    int state_parity_ = 0;
    bool pending_state_reset_ = true;

    // Codec I/O fast path
    CodecIoConfig codec_io_config_{};
    bool codec_io_configured_ = false;
    std::vector<CodecSlot> codec_slots_;
    UniqueBuffer codec_io_uniform_;
    std::vector<UniqueBuffer> nchw_transpose_uniforms_;
    std::size_t q_input_pos_ = kInvalidIndex;
    std::size_t x_input_pos_ = kInvalidIndex;
    std::size_t x_hat_output_pos_ = kInvalidIndex;

    // Zero-readback surface render pass
    SurfaceRenderConfig surface_config_{};
    bool surface_configured_ = false;
    UniqueBindGroupLayout render_bgl_;
    UniquePipelineLayout render_layout_;
    UniqueRenderPipeline render_pipeline_;
    UniqueBuffer render_uniform_;
    std::array<UniqueBindGroup, 2> render_bind_groups_;

    // Optional GPU TimestampQuery profiler
    UniqueQuerySet profile_query_set_;
    UniqueBuffer profile_resolve_buf_;
    UniqueBuffer profile_readback_buf_;
    std::vector<std::string> profile_pass_labels_;
    std::vector<std::string> profile_pass_cats_;
};

}  // namespace mlvc::webgpu_backend
