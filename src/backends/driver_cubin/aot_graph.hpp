#pragma once

#include "core/assets.hpp"
#include "core/half.hpp"
#include "backends/driver_cubin/cutlass.hpp"
#include "backends/driver_cubin/driver.hpp"
#include "runtime/aot_graph.hpp"
#include "runtime/backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlvc::driver_cubin_backend {

using namespace mlvc::aot;

using DeviceAddress = driver_cubin::abi::DeviceAddress;

std::size_t element_count(const std::vector<int64_t>& shape);
std::size_t dtype_bytes(DataType dtype);
TensorDataType public_dtype(DataType dtype);
std::array<int, 4> dims4(const std::vector<int64_t>& shape);
unsigned int divide_up(std::size_t value, unsigned int divisor);

struct Value {
    std::string name;
    DataType dtype;
    std::vector<int64_t> shape;
    DeviceAddress address = 0;
    std::size_t host_offset = 0;
    int nhwc_stride = 0;
    bool initializer = false;
};

class AotGraph final {
public:
    AotGraph(const ModelAssetReader& assets,
             const std::string& model_name,
             const ModelExecutionConfig& execution_config,
             driver_cubin::Driver& driver,
             const driver_cubin::Module& module);
    ~AotGraph();

    AotGraph(const AotGraph&) = delete;
    AotGraph& operator=(const AotGraph&) = delete;

    void reset_state();
    std::vector<Tensor> run(const std::vector<Tensor>& inputs);
    void configure_codec_io(const CodecIoConfig& config);
    std::size_t codec_slot_count() const noexcept;
    MutableYuv420FrameView encoder_input_yuv(std::size_t slot);
    void submit_encoder(std::size_t slot, int shifted_q);
    std::vector<TensorView> encoder_outputs(std::size_t slot) const;
    std::vector<MutableTensorView> decoder_inputs(std::size_t slot);
    void submit_decoder(std::size_t slot, int shifted_q);
    Yuv420FrameView decoder_output_yuv(std::size_t slot) const;
    void wait_codec_slot(std::size_t slot);

private:
    struct StateBinding {
        std::size_t input_index;
        std::size_t output_index;
    };

    struct HostSlot {
        std::vector<driver_cubin::PinnedHostBuffer> inputs;
        std::vector<driver_cubin::PinnedHostBuffer> outputs;
        driver_cubin::PinnedHostBuffer yuv;
        driver_cubin::DeviceBuffer yuv_device;
        driver_cubin::DeviceBuffer q_device;
        std::vector<driver_cubin::DeviceBuffer> input_staging;
        std::vector<driver_cubin::DeviceBuffer> output_staging;
        driver_cubin::ExecutableGraph codec_graph;
        driver_cubin::Event htod_done;
        driver_cubin::Event compute_done;
        driver_cubin::Event completion;
        bool pending = false;
    };

    struct ExternalOutput {
        std::size_t output_index;
        TensorId id;
        DeviceAddress address;
        std::size_t bytes;
    };

    struct PendingCutlassInit {
        std::size_t node_index;
        std::size_t offset;
        bool is_spatial;
    };

    void load_model(const ModelAssetReader& assets,
                    const std::string& model_name);
    void collect_graph_outputs();

    void plan_input_slice_aliases();
    void plan_epilogue_buffers();
    void plan_spatial_slice_aliases();
    void plan_reglu_buffer();
    void plan_spatial_buffers();
    void plan_direct_concat_buffers();

    void configure_state(const ModelExecutionConfig& config);
    void warmup_and_capture();
    void cache_runtime_metadata();
    bool is_state_input(std::size_t index) const;
    bool is_state_output(std::size_t index) const;
    bool is_channel_slice(const Node& node, int start, int end) const;
    static bool ranges_overlap(const Value& lhs, const Value& rhs);
    static int value_stride(const Value& v) noexcept
    {
        if (v.nhwc_stride > 0)
            return v.nhwc_stride;
        return v.shape.size() == 4 ? static_cast<int>(v.shape[1]) : 0;
    }
    static bool is_packed_nhwc(const Value& v) noexcept
    {
        return v.shape.size() != 4 ||
               value_stride(v) == static_cast<int>(v.shape[1]);
    }
    bool supports_strided_nhwc_input(const Node& consumer) const;
    bool supports_strided_nhwc_output(const Node& producer) const;
    bool is_nchw_external_tensor(const Value& v) const noexcept;
    bool needs_nhwc_boundary_transpose(const Value& v) const noexcept;
    void ensure_boundary_transpose_scratch(std::size_t bytes);
    void launch_transpose_2d(DeviceAddress src, DeviceAddress dst, int rows,
                             int cols);
    DeviceAddress get_or_pack_dwconv_weight(std::size_t node_index,
                                            DeviceAddress weight,
                                            int channels);

    void register_kernels();
    void ensure_host_slots(std::size_t count);
    HostSlot& host_slot(std::size_t slot);
    const HostSlot& host_slot(std::size_t slot) const;
    void enqueue_model();
    void enqueue_outputs(HostSlot& slot);
    void set_q_index(HostSlot& slot, int shifted_q);
    void launch_yuv420_to_nchw(DeviceAddress yuv_device);
    void launch_nchw_to_yuv420(DeviceAddress yuv_device);
    void capture_codec_graph(HostSlot& slot);
    void finish_submit(HostSlot& slot);
    void wait_codec_slot_locked(std::size_t slot);
    bool can_use_cutlass_pointwise(
        int batch_count, bool has_bias, int in_channels, int spatial_count,
        int out_channels, int epilogue, float epilogue_alpha,
        bool has_residual, int in_stride = 0, int res_stride = 0,
        int out_stride = 0) const noexcept;

    const Value& value(TensorId name) const;
    std::size_t tensor_consumer_count(TensorId id) const noexcept
    {
        return id < tensor_consumer_counts_.size()
            ? tensor_consumer_counts_[id] : 0U;
    }

    template <typename T>
    std::vector<T> initializer_values(TensorId name) const
    {
        const Value& item = value(name);
        if (!item.initializer ||
            item.dtype != (sizeof(T) == 8 ? DataType::INT64 : DataType::FP16)) {
            throw std::runtime_error(
                "driver-cubin: unexpected initializer type " + std::to_string(name));
        }
        const auto found = host_initializers_.find(name);
        if (found == host_initializers_.end()) {
            throw std::runtime_error(
                "driver-cubin: host initializer not retained for tensor " +
                std::to_string(name));
        }
        const std::size_t count = element_count(item.shape);
        if (found->second.size() != count * sizeof(T)) {
            throw std::runtime_error(
                "driver-cubin: host initializer size mismatch for tensor " +
                std::to_string(name));
        }
        std::vector<T> result(count);
        std::memcpy(result.data(), found->second.data(), found->second.size());
        return result;
    }

    float scalar_fp16(TensorId name) const;

    void execute_schedule();
    std::size_t try_execute_y0_tail(const std::vector<Node>& nodes, std::size_t index);
    bool try_execute_y1_tail(const std::vector<Node>& nodes, std::size_t index);
    bool try_execute_dec_y0_tail(const std::vector<Node>& nodes, std::size_t index);
    bool try_execute_dec_y1_tail(const std::vector<Node>& nodes, std::size_t index);
    bool try_execute_feature_update_outputs(const std::vector<Node>& nodes,
                                            std::size_t index);
    bool try_execute_feature_update(const std::vector<Node>& nodes, std::size_t index);
    bool try_execute_pointwise_epilogue(const std::vector<Node>& nodes, std::size_t index);
    bool try_execute_reglu(const std::vector<Node>& nodes, std::size_t index);
    bool try_execute_pointwise_reglu(const std::vector<Node>& nodes, std::size_t index);

    bool launch_cutlass_spatial(
        const Node& node, DeviceAddress input, DeviceAddress weight,
        DeviceAddress bias, DeviceAddress output, int batch_count,
        int in_channels, int input_height, int input_width, int out_channels,
        int output_height, int output_width, int kernel_height,
        int kernel_width, int stride_height, int stride_width,
        int pad_height, int pad_width);
    bool launch_cutlass_pointwise(
        const Node& node, DeviceAddress input, DeviceAddress weight,
        DeviceAddress bias, DeviceAddress residual, DeviceAddress output,
        int batch_count, int in_channels, int spatial_count, int out_channels,
        int epilogue, float epilogue_alpha, int in_stride = 0,
        int res_stride = 0, int out_stride = 0);
    bool launch_cutlass_pointwise_reglu(
        const Node& node, DeviceAddress input, DeviceAddress weight,
        DeviceAddress bias, DeviceAddress output, int batch_count,
        int in_channels, int spatial_count, int gate_channels, float minimum,
        float maximum, int in_stride = 0, int out_stride = 0);
    void launch_linear(driver_cubin::abi::Function function, std::size_t count,
                       std::span<void*> parameters);
    void execute(const Node& node, const Value* output_override = nullptr,
                 DeviceAddress epilogue_input = 0, int epilogue = 0,
                 float epilogue_alpha = 0.0F, int epilogue_stride = 0);

    driver_cubin::Driver& driver_;
    const driver_cubin::Module& module_;
    std::string model_name_;
    mutable std::mutex exec_mutex_;
    GraphDescriptor graph_;
    std::unordered_map<TensorId, std::vector<std::byte>> host_initializers_;
    driver_cubin::DeviceBuffer weights_device_;
    driver_cubin::DeviceBuffer arena_device_;
    driver_cubin::DeviceBuffer reglu_buffer_;
    driver_cubin::DeviceBuffer spatial_input_buffer_;
    driver_cubin::DeviceBuffer spatial_output_buffer_;
    driver_cubin::DeviceBuffer cutlass_init_scratch_;
    std::vector<PendingCutlassInit> pending_cutlass_inits_;
    bool batch_initializing_cutlass_ = false;
    driver_cubin::DeviceBuffer codec_byte_lut_device_;
    driver_cubin::DeviceBuffer boundary_transpose_scratch_;
    std::vector<driver_cubin::DeviceBuffer> epilogue_buffers_;
    std::vector<driver_cubin::DeviceBuffer> alias_buffers_;
    std::vector<driver_cubin::DeviceBuffer> concat_buffers_;
    std::unordered_map<std::size_t, driver_cubin::DeviceBuffer>
        dwconv_weight_buffers_;
    std::unordered_map<std::size_t, driver_cubin::DeviceBuffer>
        feature_update_output_buffers_;
    std::unordered_map<
        std::size_t,
        std::array<std::byte,
                   driver_cubin::kCutlassPointwiseParamsStorageBytes>>
        cutlass_host_parameters_;
    std::unordered_map<std::size_t, driver_cubin::DeviceBuffer>
        cutlass_reglu_weight_buffers_;
    std::unordered_map<std::size_t, driver_cubin::DeviceBuffer>
        cutlass_reglu_bias_buffers_;
    std::unordered_map<std::size_t, driver_cubin::DeviceBuffer>
        cutlass_spatial_weight_buffers_;
    std::unordered_map<
        std::size_t,
        std::array<std::byte,
                   driver_cubin::kCutlassPointwiseParamsStorageBytes>>
        cutlass_spatial_host_parameters_;
    std::vector<driver_cubin::DeviceBuffer> input_buffers_;
    std::vector<TensorId> input_ids_;
    std::vector<std::size_t> aliased_input_slices_;
    std::vector<std::size_t> elided_schedule_nodes_;
    std::vector<std::size_t> codec_elided_nodes_;
    std::vector<TensorId> direct_concat_values_;
    std::vector<TensorId> aliased_spatial_values_;
    std::vector<TensorId> output_ids_;
    std::vector<StateBinding> state_bindings_;
    std::vector<ExternalOutput> external_outputs_;
    std::vector<std::size_t> decoder_latent_inputs_;
    std::optional<std::size_t> q_input_index_;
    DeviceAddress codec_image_address_ = 0;
    DeviceAddress active_codec_yuv_device_ = 0;
    std::optional<std::size_t> codec_fused_s2d8_node_;
    DeviceAddress codec_s2d8_output_ = 0;
    std::optional<std::size_t> codec_fused_d2s8_node_;
    std::optional<std::size_t> codec_fused_d2s8_clip_node_;
    DeviceAddress codec_d2s8_input_ = 0;
    bool codec_d2s8_has_clip_ = false;
    float codec_d2s8_clip_min_ = 0.0F;
    float codec_d2s8_clip_max_ = 1.0F;
    std::size_t codec_yuv_bytes_ = 0;
    std::vector<HostSlot> host_slots_;
    std::vector<Value> values_;
    std::vector<std::size_t> tensor_consumer_counts_;
    TensorId tensor_id(std::string_view name) const;
    const Value& value(std::string_view name) const { return value(tensor_id(name)); }
    driver_cubin::abi::Function binary_ = nullptr;
    driver_cubin::abi::Function binary_contiguous_ = nullptr;
    driver_cubin::abi::Function binary_channel_broadcast_ = nullptr;
    driver_cubin::abi::Function unary_ = nullptr;
    driver_cubin::abi::Function feature_update_ = nullptr;
    driver_cubin::abi::Function feature_update_outputs_ = nullptr;
    driver_cubin::abi::Function y0_tail_ = nullptr;
    driver_cubin::abi::Function y1_tail_ = nullptr;
    driver_cubin::abi::Function dec_y0_tail_ = nullptr;
    driver_cubin::abi::Function dec_y1_tail_ = nullptr;
    driver_cubin::abi::Function reglu_ = nullptr;
    driver_cubin::abi::Function reglu_vec8_ = nullptr;
    driver_cubin::abi::Function convolution_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_wide_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_balanced_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_mma_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_mma_small_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_mma_tiny_ = nullptr;
    driver_cubin::abi::Function pointwise_reglu_mma_ = nullptr;
    driver_cubin::abi::Function pointwise_reglu_mma_small_ = nullptr;
    driver_cubin::abi::Function pointwise_reglu_mma_tiny_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_leaky_relu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_leaky_relu_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_residual_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_residual_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_stage4_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_stage4_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_leaky_relu_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_leaky_relu_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_residual_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_residual_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_spatial_wide_residual_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_spatial_wide_residual_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_spatial_wide_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_spatial_wide_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_spatial_wide_leaky_relu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_spatial_wide_leaky_relu_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n128_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n128_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n128_leaky_relu_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n128_leaky_relu_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n128_residual_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n128_residual_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n64_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n64_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n64_leaky_relu_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n64_leaky_relu_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n64_residual_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n64_residual_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_swizzle_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_leaky_relu_swizzle_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_residual_swizzle_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_swizzle_init_ =
        nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_medium_leaky_relu_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_medium_residual_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_spatial_wide_residual_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_spatial_wide_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_spatial_wide_leaky_relu_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_m64n128_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_m64n128_leaky_relu_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_m64n128_residual_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_m64n64_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_m64n64_leaky_relu_swizzle_init_ = nullptr;
    driver_cubin::abi::Function
        cutlass_pointwise_m64n64_residual_swizzle_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_reglu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_reglu_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_reglu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_reglu_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_stage4_reglu_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_medium_stage4_reglu_ =
        nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n64_reglu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_pointwise_m64n64_reglu_ = nullptr;
    int cutlass_log_tile_ = 0;
    driver_cubin::abi::Function cutlass_spatial_convolution_init_ = nullptr;
    driver_cubin::abi::Function cutlass_spatial_convolution_ = nullptr;
    driver_cubin::abi::Function cutlass_spatial_convolution_small_init_ = nullptr;
    driver_cubin::abi::Function cutlass_spatial_convolution_small_ = nullptr;
    driver_cubin::abi::Function transpose_ = nullptr;
    driver_cubin::abi::Function oihw_to_krsc_ = nullptr;
    driver_cubin::abi::Function interleave_reglu_rows_ = nullptr;
    driver_cubin::abi::Function spatial_convolution_mma_ = nullptr;
    driver_cubin::abi::Function spatial_convolution_mma_wide_ = nullptr;
    driver_cubin::abi::Function spatial_convolution_ = nullptr;
    driver_cubin::abi::Function depthwise_convolution_ = nullptr;
    driver_cubin::abi::Function depthwise_convolution_pair_ = nullptr;
    driver_cubin::abi::Function depthwise_convolution_quad_ = nullptr;
    driver_cubin::abi::Function depthwise_convolution_oct_ = nullptr;
    driver_cubin::abi::Function depthwise_convolution_oct2_ = nullptr;
    driver_cubin::abi::Function depthwise_convolution_3x3_small_ = nullptr;
    driver_cubin::abi::Function gather_ = nullptr;
    driver_cubin::abi::Function slice_ = nullptr;
    driver_cubin::abi::Function concat_ = nullptr;
    driver_cubin::abi::Function copy_ = nullptr;
    driver_cubin::abi::Function depth_to_space_ = nullptr;
    driver_cubin::abi::Function space_to_depth_ = nullptr;
    driver_cubin::abi::Function depth_to_space_3d_ = nullptr;
    driver_cubin::abi::Function space_to_depth_3d_ = nullptr;
    driver_cubin::abi::Function yuv420_to_nchw_ = nullptr;
    driver_cubin::abi::Function nchw_to_yuv420_ = nullptr;
    // Global NHWC function handles (SM80+)
    driver_cubin::abi::Function binary_strided_nhwc_ = nullptr;
    driver_cubin::abi::Function binary_channel_broadcast_nhwc_ = nullptr;
    driver_cubin::abi::Function unary_strided_nhwc_ = nullptr;
    driver_cubin::abi::Function reglu_nhwc_ = nullptr;
    driver_cubin::abi::Function feature_update_nhwc_ = nullptr;
    driver_cubin::abi::Function feature_update_outputs_nhwc_ = nullptr;
    driver_cubin::abi::Function y0_tail_nhwc_ = nullptr;
    driver_cubin::abi::Function y1_tail_nhwc_ = nullptr;
    driver_cubin::abi::Function dec_y0_tail_nhwc_ = nullptr;
    driver_cubin::abi::Function dec_y1_tail_nhwc_ = nullptr;
    driver_cubin::abi::Function yuv420_to_nhwc_ = nullptr;
    driver_cubin::abi::Function yuv420_to_s2d8_nhwc_ = nullptr;
    driver_cubin::abi::Function nhwc_to_yuv420_ = nullptr;
    driver_cubin::abi::Function d2s8_nhwc_to_yuv420_ = nullptr;
    driver_cubin::abi::Function copy_2d_nhwc_ = nullptr;
    driver_cubin::abi::Function slice_nhwc_ = nullptr;
    driver_cubin::abi::Function concat_copy_nhwc_ = nullptr;
    driver_cubin::abi::Function depth_to_space_nhwc_ = nullptr;
    driver_cubin::abi::Function space_to_depth_nhwc_ = nullptr;
    driver_cubin::abi::Function round_nhwc_to_nchw_ = nullptr;
    driver_cubin::abi::Function interleave_reglu_pairs_ = nullptr;
    driver_cubin::abi::Function dwconv_weight_oihw_to_k9c_ = nullptr;
    driver_cubin::abi::Function depthwise_conv_3x3_nhwc_h4_u2_ = nullptr;
    driver_cubin::abi::Function depthwise_conv_3x3_nhwc_h2_u2_ = nullptr;
    driver_cubin::abi::Function pointwise_conv_mma_tiny_nhwc_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n64_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n64_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n64_leaky_relu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n64_leaky_relu_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n64_residual_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n64_residual_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_stage4_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_stage4_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_leaky_relu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_leaky_relu_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_residual_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_residual_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_reglu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_reglu_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_stage4_reglu_init_ =
        nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m128n128_stage4_reglu_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n128_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n128_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n128_leaky_relu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n128_leaky_relu_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n128_residual_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n128_residual_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n128_reglu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n128_reglu_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n64_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n64_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n64_leaky_relu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n64_leaky_relu_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n64_residual_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n64_residual_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n64_reglu_init_ = nullptr;
    driver_cubin::abi::Function cutlass_nhwc_m64n64_reglu_ = nullptr;
    bool use_nhwc_ = false;
    CodecIoConfig codec_io_;
    bool codec_io_configured_ = false;
    bool cutlass_parameters_ready_ = false;
    bool state_initialized_ = false;
    driver_cubin::ExecutableGraph executable_graph_;
};

}  // namespace mlvc::driver_cubin_backend
