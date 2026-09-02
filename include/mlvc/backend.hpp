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
// Each release compiles exactly one backend (see MLVC_BACKEND in CMake).
// Floating-point model tensors are always fp16. Int32 remains available for
// control inputs such as q_index_shifted.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mlvc {

enum class TensorDataType { kFloat16, kInt32 };

// IEEE-754 binary16 is stored as its raw 16-bit representation. Backends pass
// these bits directly to the selected backend.
using Float16Storage = std::uint16_t;
using TensorStorage = std::variant<
    std::vector<Float16Storage>, std::vector<std::int32_t>>;

// Single dense tensor in host memory, row-major.
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    TensorStorage data;

    TensorDataType data_type() const noexcept
    {
        if (std::holds_alternative<std::vector<Float16Storage>>(data))
            return TensorDataType::kFloat16;
        return TensorDataType::kInt32;
    }

    std::size_t element_count() const noexcept
    {
        return std::visit([](const auto& values) { return values.size(); }, data);
    }

    std::size_t byte_size() const noexcept
    {
        return std::visit([](const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            return values.size() * sizeof(Value);
        }, data);
    }

    const void* raw_data() const noexcept
    {
        return std::visit([](const auto& values) -> const void* {
            return values.data();
        }, data);
    }
};

struct BackendOptions {
    std::string model_dir;
    int device_id = 0;
    std::size_t workspace_size = std::size_t{4} << 30;
    std::string engine_cache_dir;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual std::string_view name() const noexcept = 0;

    // Loads one split-model graph by name, e.g. "MLVCEncoder" or "MLVCDecoder".
    virtual void load(const std::string& model_name) = 0;

    // Runs the loaded model; inputs/outputs keep the graph's declared order.
    virtual std::vector<Tensor> run(const std::vector<Tensor>& inputs) = 0;
};

// Exactly one implementation of these symbols is linked into each release.
std::string_view compiled_backend_name() noexcept;
std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options);

}  // namespace mlvc
