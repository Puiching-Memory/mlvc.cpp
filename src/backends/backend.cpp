#include "mlvc/backend.hpp"

#include <stdexcept>

namespace mlvc {

// Defined by the per-backend translation units; each throws std::runtime_error
// when its MLVC_WITH_* build option was off.
std::unique_ptr<InferenceBackend> create_onnxruntime_backend(const BackendOptions&);
std::unique_ptr<InferenceBackend> create_libtorch_backend(const BackendOptions&);
std::unique_ptr<InferenceBackend> create_tensorrt_backend(const BackendOptions&);

std::vector<BackendKind> available_backends()
{
    std::vector<BackendKind> kinds;
#ifdef MLVC_WITH_ONNXRUNTIME
    kinds.push_back(BackendKind::kOnnxRuntime);
#endif
#ifdef MLVC_WITH_LIBTORCH
    kinds.push_back(BackendKind::kLibTorch);
#endif
#ifdef MLVC_WITH_TENSORRT
    kinds.push_back(BackendKind::kTensorRt);
#endif
    return kinds;
}

std::unique_ptr<InferenceBackend> create_backend(BackendKind kind, const BackendOptions& options)
{
    switch (kind) {
    case BackendKind::kOnnxRuntime: return create_onnxruntime_backend(options);
    case BackendKind::kLibTorch:    return create_libtorch_backend(options);
    case BackendKind::kTensorRt:    return create_tensorrt_backend(options);
    }
    throw std::runtime_error("unknown backend kind");
}

std::unique_ptr<InferenceBackend> create_backend(const std::string& name, const BackendOptions& options)
{
    if (name == "onnxruntime") return create_backend(BackendKind::kOnnxRuntime, options);
    if (name == "libtorch")    return create_backend(BackendKind::kLibTorch, options);
    if (name == "tensorrt")    return create_backend(BackendKind::kTensorRt, options);
    throw std::runtime_error("unknown backend: " + name);
}

}  // namespace mlvc
