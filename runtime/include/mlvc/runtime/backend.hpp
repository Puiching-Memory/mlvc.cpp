#pragma once
// Inference backend abstraction.
//
// The MLVC neural parts (MLVCEncoder / MLVCDecoder graphs, see docs/design.md)
// are packaged in four separate NVIDIA GPU builds:
//
//   - onnxruntime: ONNX graphs via ONNX Runtime CUDA EP
//   - libtorch:    TorchScript exports via libtorch (requires the converter to
//                  export TorchScript in addition to ONNX)
//   - tensorrt:    ONNX graphs parsed and built into TensorRT engines (NVIDIA)
//   - driver_cubin: fixed-shape AOT graph dispatched through the CUDA Driver API
//
// Each release compiles exactly one backend. Floating-point model tensors are
// always fp16. Int32 remains available for control inputs such as
// q_index_shifted.

#include "mlvc/core/tensor.hpp"
#include "mlvc/core/yuv.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mlvc {

struct BackendOptions {
    std::string model_dir;
    int device_id = 0;
    std::size_t workspace_size = std::size_t{4} << 30;
    std::string engine_cache_dir;
};

// Feeds one model output directly into an input of the next invocation. The
// bound input and output are removed from run()'s host-visible tensor lists;
// reset_state() initializes the device-resident value to zero.
struct StateTensorBinding {
    std::size_t input_index = 0;
    std::size_t output_index = 0;
    TensorDataType data_type = TensorDataType::kFloat16;
    std::vector<int64_t> shape;
};

struct ModelExecutionConfig {
    std::vector<StateTensorBinding> state_bindings;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual std::string_view name() const noexcept = 0;

    // Loads one split-model graph and fixes its host/device tensor routing.
    virtual void load(const std::string& model_name,
                      const ModelExecutionConfig& config) = 0;

    // Runs the loaded model. Inputs and outputs keep graph order after bound
    // state tensors have been removed.
    virtual std::vector<Tensor> run(const std::vector<Tensor>& inputs) = 0;

    virtual void reset_state() = 0;
};

// Optional fast path used by fixed-shape backends that can keep persistent
// pinned host buffers and insert codec conversion kernels around inference.
// Requests are submitted in stream order; a slot must be waited before its
// host buffers are reused.
struct CodecIoConfig {
    int width = 0;
    int height = 0;
    int model_width = 0;
    int model_height = 0;
    float pixel_range = 1.0F;
    FramePadding padding;
    std::size_t slots = 2;
};

class BufferedCodecBackend {
public:
    virtual ~BufferedCodecBackend() = default;

    virtual void configure_codec_io(const CodecIoConfig& config) = 0;
    virtual std::size_t codec_slot_count() const noexcept = 0;

    virtual MutableYuv420FrameView encoder_input_yuv(std::size_t slot) = 0;
    virtual void submit_encoder(std::size_t slot, int shifted_q) = 0;
    virtual std::vector<TensorView> encoder_outputs(std::size_t slot) const = 0;

    virtual std::vector<MutableTensorView> decoder_inputs(
        std::size_t slot) = 0;
    virtual void submit_decoder(std::size_t slot, int shifted_q) = 0;
    virtual Yuv420FrameView decoder_output_yuv(std::size_t slot) const = 0;

    virtual void wait_codec_slot(std::size_t slot) = 0;
};

// Exactly one implementation of these symbols is linked into each release.
const char* compiled_backend_name_c_str() noexcept;
inline std::string_view compiled_backend_name() noexcept
{
    return compiled_backend_name_c_str();
}
std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options);

}  // namespace mlvc
