// ONNX Runtime backend: runs the exported MLVC ONNX graphs via ORT.

#include "mlvc/runtime/backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
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
    explicit OnnxRuntimeBackend(BackendOptions options)
        : options_(std::move(options)),
          cuda_memory_info_("Cuda", OrtDeviceAllocator, options_.device_id,
                            OrtMemTypeDefault)
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

    void load(const std::string& model_name,
              const ModelExecutionConfig& config) override
    {
        const std::string path = options_.model_dir + "/" + model_name + ".onnx";
        state_bindings_.clear();
        io_binding_ = Ort::IoBinding{nullptr};
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
        io_binding_ = Ort::IoBinding(session_);
        configure_state(config);
    }

    void reset_state() override
    {
        if (!session_)
            throw std::runtime_error("onnxruntime: load() must be called first");
        for (StateBinding& binding : state_bindings_) {
            std::visit([](auto& values) {
                std::fill(values.begin(), values.end(), 0);
            }, binding.zero.data);
            binding.value = Ort::Value::CreateTensor(
                memory_info_, const_cast<void*>(binding.zero.raw_data()),
                binding.zero.byte_size(), binding.zero.shape.data(),
                binding.zero.shape.size(),
                ort_data_type(binding.zero.data_type()));
        }
        state_initialized_ = true;
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (!session_)
            throw std::runtime_error("onnxruntime: load() must be called first");
        if (!state_bindings_.empty() && !state_initialized_)
            throw std::runtime_error(
                "onnxruntime: reset_state() must be called before run()");
        if (inputs.size() + state_bindings_.size() != input_names_.size())
            throw std::runtime_error("onnxruntime: input count mismatch");

        io_binding_.ClearBoundInputs();
        io_binding_.ClearBoundOutputs();
        std::vector<Ort::Value> external_values;
        external_values.reserve(inputs.size());
        std::size_t external_index = 0;
        for (std::size_t graph_index = 0; graph_index < input_names_.size();
             ++graph_index) {
            StateBinding* state = state_input(graph_index);
            if (state) {
                io_binding_.BindInput(input_names_[graph_index].get(),
                                      state->value);
                continue;
            }
            const Tensor& tensor = inputs.at(external_index++);
            external_values.push_back(Ort::Value::CreateTensor(
                memory_info_, const_cast<void*>(tensor.raw_data()),
                tensor.byte_size(), tensor.shape.data(), tensor.shape.size(),
                ort_data_type(tensor.data_type())));
            io_binding_.BindInput(input_names_[graph_index].get(),
                                  external_values.back());
        }

        for (std::size_t index = 0; index < output_names_.size(); ++index) {
            io_binding_.BindOutput(
                output_names_[index].get(),
                is_state_output(index) ? cuda_memory_info_ : memory_info_);
        }
        session_.Run(Ort::RunOptions{}, io_binding_);
        io_binding_.SynchronizeOutputs();
        std::vector<Ort::Value> outputs = io_binding_.GetOutputValues();
        if (outputs.size() != output_names_.size())
            throw std::runtime_error("onnxruntime: output count mismatch");

        std::vector<Tensor> result;
        result.reserve(outputs.size() - state_bindings_.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            StateBinding* state = state_output(i);
            if (state) {
                if (info.GetElementType() !=
                        ort_data_type(state->spec.data_type) ||
                    info.GetShape() != state->spec.shape) {
                    throw std::runtime_error(
                        "onnxruntime: state tensor shape or dtype mismatch");
                }
                state->value = std::move(outputs[i]);
                continue;
            }
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
    struct StateBinding {
        StateTensorBinding spec;
        Tensor zero;
        Ort::Value value{nullptr};
    };

    void configure_state(const ModelExecutionConfig& config)
    {
        state_bindings_.clear();
        state_initialized_ = false;
        for (const StateTensorBinding& requested : config.state_bindings) {
            if (requested.input_index >= session_.GetInputCount() ||
                requested.output_index >= session_.GetOutputCount()) {
                throw std::runtime_error(
                    "onnxruntime: state tensor index is out of range");
            }
            if (state_input(requested.input_index) ||
                state_output(requested.output_index)) {
                throw std::runtime_error(
                    "onnxruntime: duplicate state tensor binding");
            }
            const Ort::TypeInfo input_type = session_.GetInputTypeInfo(
                requested.input_index);
            const Ort::TypeInfo output_type = session_.GetOutputTypeInfo(
                requested.output_index);
            const auto input_info = input_type.GetTensorTypeAndShapeInfo();
            const auto output_info = output_type.GetTensorTypeAndShapeInfo();
            if (input_info.GetElementType() != ort_data_type(requested.data_type)) {
                throw std::runtime_error(
                    std::string("onnxruntime: state input dtype mismatch for ") +
                    input_names_[requested.input_index].get() + " (actual=" +
                    std::to_string(static_cast<int>(input_info.GetElementType())) +
                    ", expected=" +
                    std::to_string(static_cast<int>(ort_data_type(
                        requested.data_type))) + ")");
            }
            if (output_info.GetElementType() != ort_data_type(requested.data_type)) {
                throw std::runtime_error(
                    std::string("onnxruntime: state output dtype mismatch for ") +
                    output_names_[requested.output_index].get());
            }
            if (input_info.GetShape() != requested.shape) {
                throw std::runtime_error(
                    std::string("onnxruntime: state input shape mismatch for ") +
                    input_names_[requested.input_index].get());
            }
            if (output_info.GetShape() != requested.shape) {
                throw std::runtime_error(
                    std::string("onnxruntime: state output shape mismatch for ") +
                    output_names_[requested.output_index].get());
            }

            Tensor zero;
            zero.name = input_names_[requested.input_index].get();
            zero.shape = requested.shape;
            const std::size_t count = static_cast<std::size_t>(
                input_info.GetElementCount());
            zero.data = allocate_storage(input_info.GetElementType(), count);
            state_bindings_.push_back(
                {requested, std::move(zero), Ort::Value{nullptr}});
        }
    }

    StateBinding* state_input(std::size_t index)
    {
        const auto found = std::find_if(
            state_bindings_.begin(), state_bindings_.end(),
            [&](const StateBinding& binding) {
                return binding.spec.input_index == index;
            });
        return found == state_bindings_.end() ? nullptr : &*found;
    }

    StateBinding* state_output(std::size_t index)
    {
        const auto found = std::find_if(
            state_bindings_.begin(), state_bindings_.end(),
            [&](const StateBinding& binding) {
                return binding.spec.output_index == index;
            });
        return found == state_bindings_.end() ? nullptr : &*found;
    }

    bool is_state_output(std::size_t index) const
    {
        return std::any_of(
            state_bindings_.begin(), state_bindings_.end(),
            [&](const StateBinding& binding) {
                return binding.spec.output_index == index;
            });
    }

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
    Ort::IoBinding io_binding_{nullptr};
    Ort::MemoryInfo memory_info_ =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo cuda_memory_info_;
    std::vector<Ort::AllocatedStringPtr> input_names_;
    std::vector<Ort::AllocatedStringPtr> output_names_;
    std::vector<StateBinding> state_bindings_;
    bool state_initialized_ = false;
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
