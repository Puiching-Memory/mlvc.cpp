// ONNX Runtime backend: runs the exported MLVC ONNX graphs via ORT.

#include "mlvc/backend.hpp"

#ifdef MLVC_WITH_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>

#include <stdexcept>
#include <utility>

namespace mlvc {
namespace {

class OnnxRuntimeBackend final : public InferenceBackend {
public:
    explicit OnnxRuntimeBackend(BackendOptions options) : options_(std::move(options))
    {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (options_.intra_op_threads > 0)
            session_options_.SetIntraOpNumThreads(options_.intra_op_threads);
        if (options_.device == "cuda") {
            OrtCUDAProviderOptions cuda_options{};
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
        } else if (options_.device != "cpu") {
            throw std::runtime_error("onnxruntime: unsupported device " + options_.device);
        }
    }

    BackendKind kind() const noexcept override { return BackendKind::kOnnxRuntime; }

    void load(const std::string& model_name) override
    {
        const std::string path = options_.model_dir + "/" + model_name + ".onnx";
        session_ = Ort::Session(env_, path.c_str(), session_options_);

        Ort::AllocatorWithDefaultOptions allocator;
        input_names_.clear();
        output_names_.clear();
        for (size_t i = 0; i < session_.GetInputCount(); ++i)
            input_names_.push_back(session_.GetInputNameAllocated(i, allocator));
        for (size_t i = 0; i < session_.GetOutputCount(); ++i)
            output_names_.push_back(session_.GetOutputNameAllocated(i, allocator));
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (!session_)
            throw std::runtime_error("onnxruntime: load() must be called first");
        if (inputs.size() != input_names_.size())
            throw std::runtime_error("onnxruntime: input count mismatch");

        std::vector<Ort::Value> values;
        values.reserve(inputs.size());
        for (const Tensor& t : inputs) {
            values.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(t.data.data()), t.data.size(),
                t.shape.data(), t.shape.size()));
        }

        std::vector<const char*> input_names, output_names;
        for (const auto& n : input_names_) input_names.push_back(n.get());
        for (const auto& n : output_names_) output_names.push_back(n.get());

        std::vector<Ort::Value> outputs = session_.Run(
            Ort::RunOptions{}, input_names.data(), values.data(), values.size(),
            output_names.data(), output_names.size());

        std::vector<Tensor> result;
        result.reserve(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error("onnxruntime: non-fp32 output (dtype conversion pending)");
            Tensor t;
            t.name = output_names_[i].get();
            t.shape = info.GetShape();
            const size_t count = static_cast<size_t>(info.GetElementCount());
            const float* src = outputs[i].GetTensorData<float>();
            t.data.assign(src, src + count);
            result.push_back(std::move(t));
        }
        return result;
    }

private:
    BackendOptions options_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "mlvc"};
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_ =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::AllocatedStringPtr> input_names_;
    std::vector<Ort::AllocatedStringPtr> output_names_;
};

}  // namespace

std::unique_ptr<InferenceBackend> create_onnxruntime_backend(const BackendOptions& options)
{
    return std::make_unique<OnnxRuntimeBackend>(options);
}

}  // namespace mlvc

#else  // MLVC_WITH_ONNXRUNTIME

#include <stdexcept>

namespace mlvc {

std::unique_ptr<InferenceBackend> create_onnxruntime_backend(const BackendOptions&)
{
    throw std::runtime_error("onnxruntime backend not compiled in (MLVC_WITH_ONNXRUNTIME=OFF)");
}

}  // namespace mlvc

#endif  // MLVC_WITH_ONNXRUNTIME
