// libtorch backend: runs TorchScript exports of the MLVC graphs.
//
// The official MLVC converter exports ONNX; the TorchScript artifacts
// (<model_dir>/<name>.ts) must be produced separately (e.g. via
// torch.jit.trace on the reference modules).

#include "mlvc/runtime/backend.hpp"

#include <torch/script.h>
#include <torch/cuda.h>
#include <ATen/Parallel.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace mlvc {
namespace {

at::ScalarType torch_data_type(TensorDataType type)
{
    switch (type) {
    case TensorDataType::kFloat16: return at::kHalf;
    case TensorDataType::kInt32:   return at::kInt;
    }
    throw std::runtime_error("libtorch: unsupported tensor dtype");
}

TensorStorage copy_from_torch(const at::Tensor& host)
{
    const auto count = static_cast<std::size_t>(host.numel());
    switch (host.scalar_type()) {
    case at::kHalf: {
        std::vector<Float16Storage> values(count);
        std::memcpy(values.data(), host.data_ptr(),
                    values.size() * sizeof(Float16Storage));
        return values;
    }
    case at::kInt: {
        std::vector<std::int32_t> values(count);
        std::memcpy(values.data(), host.data_ptr(),
                    values.size() * sizeof(std::int32_t));
        return values;
    }
    default:
        throw std::runtime_error("libtorch: unsupported output dtype");
    }
}

class LibTorchBackend final : public InferenceBackend {
public:
    explicit LibTorchBackend(BackendOptions options) : options_(std::move(options))
    {
        if (!torch::cuda::is_available())
            throw std::runtime_error("libtorch: CUDA is not available");
        device_ = torch::Device(torch::kCUDA, options_.device_id);
        at::globalContext().setBenchmarkCuDNN(true);
        at::globalContext().setAllowTF32CuDNN(false);
        at::globalContext().setAllowTF32CuBLAS(false);
    }

    std::string_view name() const noexcept override { return "libtorch"; }

    void load(const std::string& model_name,
              const ModelExecutionConfig& config) override
    {
        const std::string path = options_.model_dir + "/" + model_name + ".ts";
        try {
            module_ = torch::jit::load(path, device_);
        } catch (const c10::Error& e) {
            throw std::runtime_error("libtorch: failed to load " + path + ": " + e.what());
        }
        module_->to(device_, at::kHalf);
        module_->eval();
        state_bindings_.clear();
        state_initialized_ = false;
        for (const StateTensorBinding& requested : config.state_bindings) {
            const auto duplicate = std::find_if(
                state_bindings_.begin(), state_bindings_.end(),
                [&](const StateBinding& binding) {
                    return binding.spec.input_index == requested.input_index ||
                           binding.spec.output_index == requested.output_index;
                });
            if (duplicate != state_bindings_.end())
                throw std::runtime_error("libtorch: duplicate state tensor binding");
            if (requested.shape.empty() ||
                std::any_of(requested.shape.begin(), requested.shape.end(),
                            [](int64_t extent) { return extent <= 0; })) {
                throw std::runtime_error("libtorch: invalid state tensor shape");
            }
            StateBinding binding;
            binding.spec = requested;
            binding.tensor = torch::zeros(
                requested.shape,
                torch::TensorOptions()
                    .dtype(torch_data_type(requested.data_type))
                    .device(device_));
            state_bindings_.push_back(std::move(binding));
        }
    }

    void reset_state() override
    {
        if (!module_)
            throw std::runtime_error("libtorch: load() must be called first");
        c10::InferenceMode inference_mode;
        for (StateBinding& binding : state_bindings_)
            binding.tensor.zero_();
        state_initialized_ = true;
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (!module_)
            throw std::runtime_error("libtorch: load() must be called first");
        if (!state_bindings_.empty() && !state_initialized_)
            throw std::runtime_error(
                "libtorch: reset_state() must be called before run()");

        std::vector<at::Tensor> keep_alive;
        keep_alive.reserve(inputs.size());
        std::vector<torch::IValue> ivalue_inputs;
        const std::size_t graph_input_count =
            inputs.size() + state_bindings_.size();
        if (std::any_of(
                state_bindings_.begin(), state_bindings_.end(),
                [&](const StateBinding& binding) {
                    return binding.spec.input_index >= graph_input_count;
                })) {
            throw std::runtime_error("libtorch: state input index is out of range");
        }
        ivalue_inputs.reserve(graph_input_count);
        std::size_t external_index = 0;
        for (std::size_t graph_index = 0; graph_index < graph_input_count;
             ++graph_index) {
            StateBinding* state = state_input(graph_index);
            if (state) {
                ivalue_inputs.emplace_back(state->tensor);
                continue;
            }
            const Tensor& t = inputs.at(external_index++);
            at::Tensor at = at::from_blob(
                                const_cast<void*>(t.raw_data()), t.shape,
                                at::TensorOptions().dtype(torch_data_type(t.data_type())))
                                .to(device_);
            keep_alive.push_back(at);
            ivalue_inputs.emplace_back(at);
        }

        c10::InferenceMode inference_mode;
        torch::IValue output = module_->forward(std::move(ivalue_inputs));

        std::vector<at::Tensor> at_outputs;
        if (output.isTensor()) {
            at_outputs.push_back(output.toTensor());
        } else if (output.isTuple()) {
            for (const torch::IValue& item : output.toTuple()->elements())
                at_outputs.push_back(item.toTensor());
        } else if (output.isTensorList()) {
            at_outputs = output.toTensorVector();
        } else {
            throw std::runtime_error("libtorch: unsupported output kind");
        }

        for (StateBinding& binding : state_bindings_) {
            if (binding.spec.output_index >= at_outputs.size())
                throw std::runtime_error("libtorch: state output index is out of range");
            at::Tensor& output = at_outputs[binding.spec.output_index];
            if (output.scalar_type() !=
                    torch_data_type(binding.spec.data_type) ||
                output.sizes() != at::IntArrayRef(binding.spec.shape)) {
                throw std::runtime_error(
                    "libtorch: state tensor shape or dtype mismatch");
            }
            binding.tensor = output;
        }

        std::vector<Tensor> result;
        result.reserve(at_outputs.size() - state_bindings_.size());
        for (std::size_t index = 0; index < at_outputs.size(); ++index) {
            if (is_state_output(index))
                continue;
            at::Tensor& at = at_outputs[index];
            at::Tensor host = at.to(torch::kCPU).contiguous();
            Tensor t;
            t.shape.assign(host.sizes().begin(), host.sizes().end());
            t.data = copy_from_torch(host);
            result.push_back(std::move(t));
        }
        return result;
    }

private:
    struct StateBinding {
        StateTensorBinding spec;
        at::Tensor tensor;
    };

    StateBinding* state_input(std::size_t index)
    {
        const auto found = std::find_if(
            state_bindings_.begin(), state_bindings_.end(),
            [&](const StateBinding& binding) {
                return binding.spec.input_index == index;
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

    BackendOptions options_;
    torch::Device device_{torch::kCPU};
    std::optional<torch::jit::Module> module_;
    std::vector<StateBinding> state_bindings_;
    bool state_initialized_ = false;
};

}  // namespace

std::string_view compiled_backend_name() noexcept
{
    return "libtorch";
}

std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options)
{
    return std::make_unique<LibTorchBackend>(options);
}

}  // namespace mlvc
