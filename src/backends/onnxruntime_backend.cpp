// ONNX Runtime backend: runs the exported MLVC ONNX graphs via ORT.

#include "mlvc/backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <stdexcept>
#include <utility>

namespace mlvc {
namespace {

ONNXTensorElementDataType ort_data_type(TensorDataType type)
{
    switch (type) {
    case TensorDataType::kFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case TensorDataType::kInt32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
    throw std::runtime_error("onnxruntime: unsupported tensor dtype");
}

TensorStorage allocate_storage(ONNXTensorElementDataType type, std::size_t count)
{
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return std::vector<Float16Storage>(count);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return std::vector<std::int32_t>(count);
    default:
        throw std::runtime_error("onnxruntime: unsupported output dtype");
    }
}

class OnnxRuntimeBackend final : public InferenceBackend {
public:
    explicit OnnxRuntimeBackend(BackendOptions options) : options_(std::move(options))
    {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = options_.device_id;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.gpu_mem_limit = SIZE_MAX;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.do_copy_in_default_stream = 1;
        cuda_options.tunable_op_enable = true;
        cuda_options.tunable_op_tuning_enable = true;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
    }

    std::string_view name() const noexcept override { return "onnxruntime"; }

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

        for (size_t i = 0; i < session_.GetInputCount(); ++i) {
            validate_model_type(session_.GetInputTypeInfo(i), input_names_[i].get(),
                                "input");
        }
        for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
            validate_model_type(session_.GetOutputTypeInfo(i), output_names_[i].get(),
                                "output");
        }
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
            values.push_back(Ort::Value::CreateTensor(
                memory_info_, const_cast<void*>(t.raw_data()), t.byte_size(),
                t.shape.data(), t.shape.size(), ort_data_type(t.data_type())));
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
            Tensor t;
            t.name = output_names_[i].get();
            t.shape = info.GetShape();
            const size_t count = static_cast<size_t>(info.GetElementCount());
            t.data = allocate_storage(info.GetElementType(), count);
            std::visit([&](auto& values) {
                std::memcpy(values.data(), outputs[i].GetTensorRawData(),
                            values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type));
            }, t.data);
            result.push_back(std::move(t));
        }
        return result;
    }

private:
    static void validate_model_type(const Ort::TypeInfo& type_info,
                                    const char* tensor_name, const char* direction)
    {
        const auto type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
            type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            throw std::runtime_error(
                std::string("onnxruntime: ") + direction + " " + tensor_name +
                " is not fp16/int32; mlvc.cpp accepts FP16 models only");
        }
    }

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

std::string_view compiled_backend_name() noexcept
{
    return "onnxruntime";
}

std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options)
{
    return std::make_unique<OnnxRuntimeBackend>(options);
}

}  // namespace mlvc
