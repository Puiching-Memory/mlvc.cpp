#include "mlvc/codec/pipeline.hpp"
#include "mlvc/core/assets.hpp"
#include "mlvc/core/half.hpp"
#include "mlvc/core/model.hpp"
#include "mlvc/core/yuv.hpp"
#include "mlvc/runtime/aot_graph.hpp"
#include "mlvc/runtime/backend.hpp"
#include "mlvc_model.pb.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace mlvc;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::string canonical_path_key(const std::filesystem::path& path)
{
    return path.lexically_normal().generic_string();
}

std::vector<std::byte> serialize_message(const google::protobuf::MessageLite& message)
{
    const std::string encoded = message.SerializeAsString();
    const auto span = std::as_bytes(std::span<const char>{encoded.data(), encoded.size()});
    return {span.begin(), span.end()};
}

class InMemoryAssetReader final : public ModelAssetReader {
public:
    explicit InMemoryAssetReader(std::string label) : label_(std::move(label)) {}

    void put(const std::filesystem::path& relative_path, std::vector<std::byte> bytes)
    {
        files_[canonical_path_key(relative_path)] = std::move(bytes);
    }

    std::string location_label() const override { return label_; }

    bool exists(const std::filesystem::path& relative_path) const override
    {
        return files_.contains(canonical_path_key(relative_path));
    }

    std::vector<std::byte> read_binary(
        const std::filesystem::path& relative_path) const override
    {
        const auto found = files_.find(canonical_path_key(relative_path));
        if (found == files_.end()) {
            throw std::runtime_error(
                "missing in-memory asset: " + relative_path.generic_string());
        }
        return found->second;
    }

private:
    std::string label_;
    std::map<std::string, std::vector<std::byte>> files_;
};

wire::Graph make_dummy_graph(const std::string& checkpoint_sha,
                             std::size_t weights_bytes)
{
    wire::Graph graph;
    graph.set_schema_version(3);
    graph.set_checkpoint_sha256(checkpoint_sha);
    graph.set_weights_sha256(std::string(32, 'w'));
    graph.set_weights_bytes(weights_bytes);
    graph.set_arena_bytes(256);
    graph.set_arena_alignment(256);
    for (std::uint32_t id = 1; id <= 3; ++id) {
        auto* tensor = graph.add_tensors();
        tensor->set_id(id);
        tensor->set_name("t" + std::to_string(id));
        tensor->set_dtype(wire::FP16);
        tensor->add_shape(id == 1 ? static_cast<std::int64_t>(weights_bytes / 2) : 1);
    }
    graph.mutable_tensors(0)->mutable_weight()->set_size_bytes(weights_bytes);
    auto* arena = graph.mutable_tensors(2)->mutable_arena();
    arena->set_size_bytes(256);
    arena->set_death(1);
    graph.add_inputs(2);
    graph.add_outputs(3);
    auto* node = graph.add_nodes();
    node->set_op(wire::ADD);
    node->add_inputs(1);
    node->add_inputs(2);
    node->add_outputs(3);
    return graph;
}

std::shared_ptr<InMemoryAssetReader> make_synthetic_bundle()
{
    const std::string checkpoint_sha(32, 'c');
    const std::string entropy_sha(32, 'e');

    wire::ModelMetadata metadata;
    metadata.set_schema_version(1);
    metadata.set_profile("synthetic-v1");
    metadata.set_model_version("v1");
    metadata.set_width(64);
    metadata.set_height(64);
    metadata.set_pixel_range(1.0F);
    metadata.set_qp_num(4);
    metadata.set_total_qp_num(8);
    metadata.add_frame_index_map(0);
    metadata.add_frame_index_map(1);
    metadata.add_qp_shift(0);
    metadata.add_qp_shift(2);
    metadata.set_feature_channels(2);
    metadata.set_latent_channels(4);
    metadata.set_hyperprior_channels(2);
    metadata.set_downsample_feature(16);
    metadata.set_downsample_latent(16);
    metadata.set_downsample_hyperprior(2);
    metadata.set_y_scale_repeat(2);
    metadata.set_iframe_period(2);
    metadata.set_disable_feature_reset(true);
    metadata.set_checkpoint_sha256(checkpoint_sha);

    wire::ModelBundle bundle;
    bundle.set_schema_version(2);
    bundle.set_profile("synthetic-v1");
    bundle.set_model_version("v1");
    bundle.set_width(64);
    bundle.set_height(64);
    bundle.set_checkpoint_sha256(checkpoint_sha);
    bundle.set_entropy_model_sha256(entropy_sha);
    bundle.mutable_contract()->set_container("mlvc-frame-le-v1");
    bundle.mutable_contract()->set_entropy("canonical-pmf-v1");

    // 4 scale levels, each accepting symbols in {-1, 0, 1}.
    wire::GaussianPmf gaussian;
    gaussian.set_schema_version(1);
    gaussian.set_scale_levels(4);
    gaussian.set_index_space(true);
    for (int row = 0; row < 4; ++row) {
        gaussian.mutable_table()->add_lengths(3);
        gaussian.mutable_table()->add_offsets(-1);
        gaussian.mutable_table()->add_values(16384);
        gaussian.mutable_table()->add_values(32768);
        gaussian.mutable_table()->add_values(16384);
    }

    // total_qp_num=8, hyperprior_channels=2 -> 16 rows.
    wire::BitEstimatorPmf bit_estimator;
    bit_estimator.set_schema_version(1);
    bit_estimator.set_qp_num(8);
    bit_estimator.set_channels(2);
    for (int row = 0; row < 16; ++row) {
        bit_estimator.mutable_table()->add_lengths(3);
        bit_estimator.mutable_table()->add_offsets(-1);
        bit_estimator.mutable_table()->add_values(16384);
        bit_estimator.mutable_table()->add_values(32768);
        bit_estimator.mutable_table()->add_values(16384);
    }

    auto reader = std::make_shared<InMemoryAssetReader>("memory://synthetic-v1");
    reader->put("metadata.pb", serialize_message(metadata));
    reader->put("model_bundle.pb", serialize_message(bundle));
    reader->put("gaussian_pmf.pb", serialize_message(gaussian));
    reader->put("bit_estimator_pmf.pb", serialize_message(bit_estimator));
    reader->put("aot/MLVCEncoder/graph.pb",
                serialize_message(make_dummy_graph(checkpoint_sha, 4)));
    reader->put("aot/MLVCEncoder/weights.bin",
                {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}});
    reader->put("aot/MLVCDecoder/graph.pb",
                serialize_message(make_dummy_graph(checkpoint_sha, 2)));
    reader->put("aot/MLVCDecoder/weights.bin",
                {std::byte{5}, std::byte{6}});
    return reader;
}

struct MockTelemetry {
    int resets = 0;
    int encoder_runs = 0;
    int decoder_runs = 0;
    int internal_state = 0;
    std::vector<int> shifted_qs;
};

class MockCodecBackend final : public InferenceBackend {
public:
    MockCodecBackend(std::string backend_name, BackendOptions options,
                     std::shared_ptr<MockTelemetry> telemetry)
        : name_(std::move(backend_name)),
          options_(std::move(options)),
          telemetry_(std::move(telemetry))
    {
    }

    std::string_view name() const noexcept override { return name_; }

    void load(const std::string& model_name,
              const ModelExecutionConfig& execution) override
    {
        require(options_.asset_reader != nullptr,
                "backend should receive resolved ModelAssetReader");
        loaded_ = aot::load_aot_graph(*options_.asset_reader, model_name);
        config_ = load_model_config(*options_.asset_reader);
        model_name_ = model_name;
        device_state_ = !execution.state_bindings.empty();
    }

    void reset_state() override
    {
        ++telemetry_->resets;
        telemetry_->internal_state = 0;
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (model_name_ == "MLVCEncoder") {
            require(device_state_ && inputs.size() == 2,
                    "EncoderSession with device state should pass x and q_index_shifted");
            ++telemetry_->encoder_runs;
            const auto& q_values = std::get<std::vector<std::int32_t>>(inputs[1].data);
            telemetry_->shifted_qs.push_back(q_values.at(0));

            const float luma0 = half_to_float(
                std::get<std::vector<Float16Storage>>(inputs[0].data).at(0));
            const Float16Storage token =
                float_to_half(luma0 >= 0.5F ? 1.0F : -1.0F);
            const Float16Storage state_token =
                float_to_half(telemetry_->internal_state > 0 ? 1.0F : 0.0F);
            telemetry_->internal_state += 1;

            Tensor z{"z_raw",
                {1, config_.hyperprior_channels,
                 config_.hyperprior_height(), config_.hyperprior_width()},
                std::vector<Float16Storage>(
                    static_cast<std::size_t>(config_.hyperprior_channels) *
                    config_.hyperprior_height() * config_.hyperprior_width(),
                    float_to_half(0.0F))};
            const std::vector<std::int64_t> y_shape{
                1, config_.latent_channels / 2,
                config_.latent_height(), config_.latent_width()};
            const std::size_t y_count = checked_tensor_element_count(y_shape);
            Tensor y0{"y_raw_0", y_shape,
                std::vector<Float16Storage>(y_count, token)};
            Tensor y1{"y_raw_1", y_shape,
                std::vector<Float16Storage>(y_count, state_token)};
            return {std::move(z), std::move(y0), std::move(y1)};
        }

        if (model_name_ == "MLVCDecoder") {
            require(device_state_ && inputs.size() == 4,
                    "DecoderSession with device state should pass latents and q_index_shifted");
            ++telemetry_->decoder_runs;
            const auto& q_values = std::get<std::vector<std::int32_t>>(inputs[3].data);
            telemetry_->shifted_qs.push_back(q_values.at(0));

            const float y0 = half_to_float(
                std::get<std::vector<Float16Storage>>(inputs[1].data).at(0));
            const float y1 = half_to_float(
                std::get<std::vector<Float16Storage>>(inputs[2].data).at(0));
            const float luma = y0 > 0.0F ? 0.75F : 0.25F;
            const float offset = y1 > 0.0F ? 0.125F : 0.0F;
            const std::size_t plane =
                static_cast<std::size_t>(config_.model_height) * config_.model_width;
            std::vector<Float16Storage> rgb(plane * 3U);
            const Float16Storage fill = float_to_half(luma + offset);
            std::fill(rgb.begin(), rgb.end(), fill);
            return {Tensor{"x_hat", {1, 3, config_.model_height, config_.model_width},
                           std::move(rgb)}};
        }
        throw std::runtime_error("unexpected model name: " + model_name_);
    }

private:
    std::string name_;
    BackendOptions options_;
    std::shared_ptr<MockTelemetry> telemetry_;
    aot::LoadedGraph loaded_;
    ModelConfig config_;
    std::string model_name_;
    bool device_state_ = false;
};

void test_backend_registry_and_aot_assets()
{
    const auto bundle = make_synthetic_bundle();
    const ModelConfig config = load_model_config(*bundle);
    require(config.name == "synthetic-v1" && config.model_width == 64,
            "ModelConfig loaded via ModelAssetReader");

    const aot::LoadedGraph encoder_graph = aot::load_aot_graph(*bundle, "MLVCEncoder");
    require(encoder_graph.weights.size() == 4, "AOT graph weights loaded via asset reader");

    auto mismatched = make_synthetic_bundle();
    wire::Graph bad_graph = make_dummy_graph(std::string(32, 'x'), 4);
    mismatched->put("aot/MLVCEncoder/graph.pb", serialize_message(bad_graph));
    bool threw = false;
    try {
        (void)aot::load_aot_graph(*mismatched, "MLVCEncoder");
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "mismatched AOT checkpoint SHA must throw");

    auto telemetry = std::make_shared<MockTelemetry>();
    register_backend(
        "mock-alt",
        [telemetry](const BackendOptions& opts) {
            return std::make_unique<MockCodecBackend>("mock-alt", opts, telemetry);
        },
        false);
    register_backend(
        "mock-primary",
        [telemetry](const BackendOptions& opts) {
            return std::make_unique<MockCodecBackend>("mock-primary", opts, telemetry);
        },
        true);

    const auto names = available_backends();
    require(std::find(names.begin(), names.end(), "mock-alt") != names.end() &&
            std::find(names.begin(), names.end(), "mock-primary") != names.end(),
            "registered backends must appear in available_backends()");
    require(default_backend_name() == "mock-primary",
            "make_default=true should update default_backend_name()");

    BackendOptions opts;
    opts.backend_name = "mock-alt";
    opts.asset_reader = bundle;
    auto created = create_backend(opts);
    require(created->name() == "mock-alt", "explicit backend selection must pick mock-alt");

    opts.backend_name = "nonexistent-backend";
    threw = false;
    try {
        (void)create_backend(opts);
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "unknown backend name must throw");
}

void test_sessions_without_disk_io()
{
    const auto bundle = make_synthetic_bundle();
    auto enc_telemetry = std::make_shared<MockTelemetry>();
    auto dec_telemetry = std::make_shared<MockTelemetry>();
    register_backend(
        "mock-enc",
        [enc_telemetry](const BackendOptions& opts) {
            return std::make_unique<MockCodecBackend>("mock-enc", opts, enc_telemetry);
        });
    register_backend(
        "mock-dec",
        [dec_telemetry](const BackendOptions& opts) {
            return std::make_unique<MockCodecBackend>("mock-dec", opts, dec_telemetry);
        });

    // Exercise odd non-model dimensions cropped/padded inside Session.
    SessionOptions enc_opts;
    enc_opts.width = 50;
    enc_opts.height = 38;
    enc_opts.q_index = 1;
    enc_opts.backend_name = "mock-enc";
    enc_opts.asset_reader = bundle;
    EncoderSession encoder(enc_opts);

    SessionOptions dec_opts;
    dec_opts.width = 50;
    dec_opts.height = 38;
    dec_opts.backend_name = "mock-dec";
    dec_opts.asset_reader = bundle;
    DecoderSession decoder(dec_opts);

    auto make_solid_frame = [](std::uint8_t y_val) {
        Yuv420Frame f;
        f.width = 50;
        f.height = 38;
        f.y.assign(50U * 38U, y_val);
        f.u.assign(25U * 19U, 128);
        f.v.assign(25U * 19U, 128);
        return f;
    };

    const auto f0 = make_solid_frame(220);
    const auto f1 = make_solid_frame(30);
    const auto f2 = make_solid_frame(220);

    // Frame 0 (gop 0 -> reset), frame 1 (gop 1 -> no reset), frame 2 (gop 0 due to iframe_period=2 -> reset).
    const EncodedFrame p0 = encoder.encode_frame(yuv420_view(f0));
    const EncodedFrame p1 = encoder.encode_frame(yuv420_view(f1));
    const EncodedFrame p2 = encoder.encode_frame(yuv420_view(f2), 2);
    require(encoder.frame_index() == 3, "encoder frame_index must track encoded frames");
    require(enc_telemetry->encoder_runs == 3 && enc_telemetry->resets == 2,
            "GOP boundary should trigger backend reset_state() on I-frames");
    require(enc_telemetry->shifted_qs == (std::vector<int>{3, 1, 4}),
            "QP shifts across GOP frames should match metadata schedule");

    const Yuv420Frame d0 = decoder.decode_frame(p0);
    Yuv420Frame reusable = make_solid_frame(0);
    decoder.decode_frame_into(p1, mutable_yuv420_view(reusable));
    const Yuv420Frame d2 = decoder.decode_frame(p2);
    require(decoder.frame_index() == 3, "decoder frame_index must track decoded frames");
    require(dec_telemetry->decoder_runs == 3 && dec_telemetry->resets == 2,
            "decoder GOP boundary should reset state identically");
    // Frame 0 had state_token=0 => filler 0.75 * 255 = ~191; Frame 1 had state_token=1 => filler (0.25 + 0.125) * 255 = ~96.
    // Frame 2 reset state => back to state_token=0!
    require(d0.y.at(0) > 150 && reusable.y.at(0) < 120 && d2.y.at(0) == d0.y.at(0),
            "decoded YUV frames should reflect entropy-coded latents and GOP resets");

    encoder.reset();
    decoder.reset();
    require(encoder.frame_index() == 0 && decoder.frame_index() == 0,
            "explicit reset() should rewind frame counters");
}

}  // namespace

int main()
{
    try {
        test_backend_registry_and_aot_assets();
        test_sessions_without_disk_io();
        std::cout << "Runtime decoupling tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
