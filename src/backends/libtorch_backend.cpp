// libtorch backend: runs TorchScript exports of the MLVC graphs.
//
// The official MLVC converter exports ONNX; the TorchScript artifacts
// (<model_dir>/<name>.ts) must be produced separately (e.g. via
// torch.jit.trace on the reference modules).

#include "mlvc/backend.hpp"

#ifdef MLVC_WITH_LIBTORCH

#include <torch/script.h>
#include <torch/cuda.h>
#include <ATen/Parallel.h>

#include <stdexcept>
#include <utility>

namespace mlvc {
namespace {

class LibTorchBackend final : public InferenceBackend {
public:
    explicit LibTorchBackend(BackendOptions options) : options_(std::move(options))
    {
        if (options_.device == "cuda") {
            if (!torch::cuda::is_available())
                throw std::runtime_error("libtorch: CUDA requested but not available");
            device_ = torch::Device(torch::kCUDA);
        } else if (options_.device == "cpu") {
            device_ = torch::Device(torch::kCPU);
        } else {
            throw std::runtime_error("libtorch: unsupported device " + options_.device);
        }
        if (options_.intra_op_threads > 0)
            at::set_num_threads(options_.intra_op_threads);
    }

    BackendKind kind() const noexcept override { return BackendKind::kLibTorch; }

    void load(const std::string& model_name) override
    {
        const std::string path = options_.model_dir + "/" + model_name + ".ts";
        try {
            module_ = torch::jit::load(path, device_);
        } catch (const c10::Error& e) {
            throw std::runtime_error("libtorch: failed to load " + path + ": " + e.what());
        }
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
                                const_cast<float*>(t.data.data()), t.shape,
                                at::TensorOptions().dtype(at::kFloat))
                                .to(device_);
            keep_alive.push_back(at);
            ivalue_inputs.emplace_back(at);
        }

        torch::NoGradGuard no_grad;
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
            at::Tensor host = at.to(torch::kCPU, at::kFloat).contiguous();
            Tensor t;
            t.shape.assign(host.sizes().begin(), host.sizes().end());
            const float* src = host.data_ptr<float>();
            t.data.assign(src, src + host.numel());
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

std::unique_ptr<InferenceBackend> create_libtorch_backend(const BackendOptions& options)
{
    return std::make_unique<LibTorchBackend>(options);
}

}  // namespace mlvc

#else  // MLVC_WITH_LIBTORCH

#include <stdexcept>

namespace mlvc {

std::unique_ptr<InferenceBackend> create_libtorch_backend(const BackendOptions&)
{
    throw std::runtime_error("libtorch backend not compiled in (MLVC_WITH_LIBTORCH=OFF)");
}

}  // namespace mlvc

#endif  // MLVC_WITH_LIBTORCH
