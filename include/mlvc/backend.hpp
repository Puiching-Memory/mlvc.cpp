#pragma once
// Inference backend abstraction.
//
// The MLVC neural parts (MLVCEncoder / MLVCDecoder graphs, see docs/design.md)
// can be executed by any of three interchangeable backends:
//
//   - onnxruntime: ONNX graphs via ONNX Runtime (CPU / CUDA EP)
//   - libtorch:    TorchScript exports via libtorch (requires the converter to
//                  export TorchScript in addition to ONNX)
//   - tensorrt:    ONNX graphs parsed and built into TensorRT engines (NVIDIA)
//
// Backends are compiled in per build (see MLVC_WITH_* CMake options) and
// selected at runtime. Host-side tensor buffers are fp32; each backend
// converts to the model's native dtype (fp16 for MLVC exports) internally.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlvc {

enum class BackendKind { kOnnxRuntime, kLibTorch, kTensorRt };

inline const char* to_string(BackendKind kind) noexcept
{
    switch (kind) {
    case BackendKind::kOnnxRuntime: return "onnxruntime";
    case BackendKind::kLibTorch:    return "libtorch";
    case BackendKind::kTensorRt:    return "tensorrt";
    }
    return "unknown";
}

// Single dense fp32 tensor in host memory, row-major.
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct BackendOptions {
    std::string model_dir;
    std::string device = "cpu";  // "cpu" or "cuda"
    int intra_op_threads = 0;    // 0 = library default
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual BackendKind kind() const noexcept = 0;

    // Loads one split-model graph by name, e.g. "MLVCEncoder" or "MLVCDecoder".
    virtual void load(const std::string& model_name) = 0;

    // Runs the loaded model; inputs/outputs keep the graph's declared order.
    virtual std::vector<Tensor> run(const std::vector<Tensor>& inputs) = 0;
};

// Backends compiled into this binary.
std::vector<BackendKind> available_backends();

// Creates a backend; throws std::runtime_error if the kind was not compiled in.
std::unique_ptr<InferenceBackend> create_backend(BackendKind kind, const BackendOptions& options);
std::unique_ptr<InferenceBackend> create_backend(const std::string& name, const BackendOptions& options);

}  // namespace mlvc
