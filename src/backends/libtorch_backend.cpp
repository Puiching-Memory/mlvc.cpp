// libtorch backend: runs TorchScript exports of the MLVC graphs.
//
// The official MLVC converter exports ONNX; the TorchScript artifacts
// (<model_dir>/<name>.ts) must be produced separately (e.g. via
// torch.jit.trace on the reference modules).

#include "mlvc/backend.hpp"

#include <torch/script.h>
#include <torch/cuda.h>
#include <ATen/Parallel.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>

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

    void load(const std::string& model_name) override
    {
        const std::string path = options_.model_dir + "/" + model_name + ".ts";
        try {
            module_ = torch::jit::load(path, device_);
        } catch (const c10::Error& e) {
            throw std::runtime_error("libtorch: failed to load " + path + ": " + e.what());
        }
        module_->to(device_, at::kHalf);
        module_->eval();
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (!module_)
            throw std::runtime_error("libtorch: load() must be called first");

        std::vector<at::Tensor> keep_alive;
        keep_alive.reserve(inputs.size());
        std::vector<torch::IValue> ivalue_inputs;
        ivalue_inputs.reserve(inputs.size());
        for (const Tensor& t : inputs) {
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

        std::vector<Tensor> result;
        result.reserve(at_outputs.size());
        for (at::Tensor& at : at_outputs) {
            at::Tensor host = at.to(torch::kCPU).contiguous();
            Tensor t;
            t.shape.assign(host.sizes().begin(), host.sizes().end());
            t.data = copy_from_torch(host);
            result.push_back(std::move(t));
        }
        return result;
    }

private:
    BackendOptions options_;
    torch::Device device_{torch::kCPU};
    std::optional<torch::jit::Module> module_;
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
