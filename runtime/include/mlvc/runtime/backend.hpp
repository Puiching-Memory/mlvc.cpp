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

// Exactly one implementation of these symbols is linked into each release.
std::string_view compiled_backend_name() noexcept;
std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options);

}  // namespace mlvc
