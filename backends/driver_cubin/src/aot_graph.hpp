#pragma once

#include "mlvc/core/half.hpp"
#include "mlvc/driver_cubin/cutlass.hpp"
#include "mlvc/driver_cubin/driver.hpp"
#include "mlvc/runtime/backend.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlvc::driver_cubin_backend {

using json = nlohmann::json;
using DeviceAddress = driver_cubin::abi::DeviceAddress;

std::size_t element_count(const std::vector<int64_t>& shape);
std::size_t dtype_bytes(const std::string& dtype);
TensorDataType public_dtype(const std::string& dtype);
std::array<int, 4> dims4(const std::vector<int64_t>& shape);
unsigned int divide_up(std::size_t value, unsigned int divisor);

struct Value {
    std::string dtype;
    std::vector<int64_t> shape;
    DeviceAddress address = 0;
    std::size_t host_offset = 0;
    bool initializer = false;
};

class AotGraph final {
public:
    AotGraph(const std::filesystem::path& model_dir,
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
        driver_cubin::Event completion;
        bool pending = false;
    };

    void load_model(const std::filesystem::path& model_dir,
                    const std::string& model_name);
    void collect_graph_outputs();

    void plan_input_slice_aliases();
    void plan_epilogue_buffers();
    void plan_reglu_buffer();
    void plan_spatial_buffers();
    void plan_direct_concat_buffers();

    void configure_state(const ModelExecutionConfig& config);
    bool is_state_input(std::size_t index) const;
    bool is_state_output(std::size_t index) const;
    bool is_channel_slice(const json& node, int start, int end) const;
    static bool ranges_overlap(const Value& lhs, const Value& rhs);

    void register_kernels();
    void ensure_host_slots(std::size_t count);
    HostSlot& host_slot(std::size_t slot);
    const HostSlot& host_slot(std::size_t slot) const;
    void enqueue_model();
    void enqueue_outputs(HostSlot& slot);
    void set_q_index(HostSlot& slot, int shifted_q);
    void finish_submit(HostSlot& slot);

    const Value& value(const std::string& name) const;

    template <typename T>
    std::vector<T> initializer_values(const std::string& name) const
    {
        const Value& item = value(name);
        if (!item.initializer ||
            item.dtype != (sizeof(T) == 8 ? "int64" : "fp16")) {
            throw std::runtime_error(
                "driver-cubin: unexpected initializer type " + name);
        }
        const std::size_t count = element_count(item.shape);
        std::vector<T> result(count);
        std::memcpy(result.data(), weights_host_.data() + item.host_offset,
                    result.size() * sizeof(T));
        return result;
    }

    float scalar_fp16(const std::string& name) const;

    void execute_schedule();
    bool try_execute_y0_tail(const json& nodes, std::size_t index);
    bool try_execute_y1_tail(const json& nodes, std::size_t index);
    bool try_execute_feature_update_outputs(const json& nodes,
                                            std::size_t index);
    bool try_execute_feature_update(const json& nodes, std::size_t index);
    bool try_execute_pointwise_epilogue(const json& nodes, std::size_t index);
    bool try_execute_reglu(const json& nodes, std::size_t index);
    bool try_execute_pointwise_reglu(const json& nodes, std::size_t index);

    bool launch_cutlass_spatial(
        const json& node, DeviceAddress input, DeviceAddress weight,
        DeviceAddress bias, DeviceAddress output, int batch_count,
        int in_channels, int input_height, int input_width, int out_channels,
        int output_height, int output_width, int kernel_height,
        int kernel_width, int stride_height, int stride_width,
        int pad_height, int pad_width);
    bool launch_cutlass_pointwise(
        const json& node, DeviceAddress input, DeviceAddress weight,
        DeviceAddress bias, DeviceAddress residual, DeviceAddress output,
        int batch_count, int in_channels, int spatial_count, int out_channels,
        int epilogue, float epilogue_alpha);
    void launch_linear(driver_cubin::abi::Function function, std::size_t count,
                       std::span<void*> parameters);
    void execute(const json& node, const Value* output_override = nullptr,
                 DeviceAddress epilogue_input = 0, int epilogue = 0,
                 float epilogue_alpha = 0.0F);

    driver_cubin::Driver& driver_;
    const driver_cubin::Module& module_;
    std::string model_name_;
    json manifest_;
    std::vector<std::byte> weights_host_;
    driver_cubin::DeviceBuffer weights_device_;
    driver_cubin::DeviceBuffer arena_device_;
    driver_cubin::DeviceBuffer reglu_buffer_;
    driver_cubin::DeviceBuffer spatial_input_buffer_;
    driver_cubin::DeviceBuffer spatial_output_buffer_;
    driver_cubin::DeviceBuffer codec_yuv_device_;
    driver_cubin::DeviceBuffer codec_byte_lut_device_;
    std::vector<driver_cubin::DeviceBuffer> epilogue_buffers_;
    std::vector<driver_cubin::DeviceBuffer> concat_buffers_;
    std::unordered_map<std::size_t, driver_cubin::DeviceBuffer>
        feature_update_output_buffers_;
    std::unordered_map<std::size_t, driver_cubin::DeviceBuffer>
        cutlass_parameter_buffers_;
    std::unordered_map<
        std::size_t,
        std::array<std::byte,
                   driver_cubin::kCutlassPointwiseParamsStorageBytes>>
        cutlass_host_parameters_;
    std::unordered_map<std::size_t, driver_cubin::DeviceBuffer>
        cutlass_spatial_parameter_buffers_;
    std::unordered_map<std::size_t, driver_cubin::DeviceBuffer>
        cutlass_spatial_weight_buffers_;
    std::unordered_map<
        std::size_t,
        std::array<std::byte,
                   driver_cubin::kCutlassPointwiseParamsStorageBytes>>
        cutlass_spatial_host_parameters_;
    std::vector<driver_cubin::DeviceBuffer> input_buffers_;
    std::vector<std::string> input_names_;
    std::vector<std::size_t> aliased_input_slices_;
    std::vector<std::size_t> elided_schedule_nodes_;
    std::vector<std::string> direct_concat_values_;
    std::vector<std::string> output_names_;
    std::vector<StateBinding> state_bindings_;
    std::vector<HostSlot> host_slots_;
    std::unordered_map<std::string, Value> values_;
    driver_cubin::abi::Function binary_ = nullptr;
    driver_cubin::abi::Function binary_contiguous_ = nullptr;
    driver_cubin::abi::Function unary_ = nullptr;
    driver_cubin::abi::Function feature_update_ = nullptr;
    driver_cubin::abi::Function feature_update_outputs_ = nullptr;
    driver_cubin::abi::Function y0_tail_ = nullptr;
    driver_cubin::abi::Function y1_tail_ = nullptr;
    driver_cubin::abi::Function reglu_ = nullptr;
    driver_cubin::abi::Function reglu_vec8_ = nullptr;
    driver_cubin::abi::Function convolution_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_wide_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_balanced_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_mma_ = nullptr;
    driver_cubin::abi::Function pointwise_convolution_mma_small_ = nullptr;
    driver_cubin::abi::Function pointwise_reglu_mma_ = nullptr;
    driver_cubin::abi::Function pointwise_reglu_mma_small_ = nullptr;
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
    driver_cubin::abi::Function cutlass_spatial_convolution_init_ = nullptr;
    driver_cubin::abi::Function cutlass_spatial_convolution_ = nullptr;
    driver_cubin::abi::Function transpose_ = nullptr;
    driver_cubin::abi::Function oihw_to_krsc_ = nullptr;
    driver_cubin::abi::Function spatial_convolution_mma_ = nullptr;
    driver_cubin::abi::Function spatial_convolution_mma_wide_ = nullptr;
    driver_cubin::abi::Function spatial_convolution_ = nullptr;
    driver_cubin::abi::Function depthwise_convolution_ = nullptr;
    driver_cubin::abi::Function depthwise_convolution_pair_ = nullptr;
    driver_cubin::abi::Function depthwise_convolution_quad_ = nullptr;
    driver_cubin::abi::Function gather_ = nullptr;
    driver_cubin::abi::Function slice_ = nullptr;
    driver_cubin::abi::Function concat_ = nullptr;
    driver_cubin::abi::Function depth_to_space_ = nullptr;
    driver_cubin::abi::Function space_to_depth_ = nullptr;
    driver_cubin::abi::Function yuv420_to_nchw_ = nullptr;
    driver_cubin::abi::Function nchw_to_yuv420_ = nullptr;
    CodecIoConfig codec_io_;
    bool codec_io_configured_ = false;
    bool cutlass_parameters_ready_ = false;
    bool state_initialized_ = false;
    driver_cubin::ExecutableGraph executable_graph_;
};

}  // namespace mlvc::driver_cubin_backend
