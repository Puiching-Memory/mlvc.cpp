#include "test_support.hpp"

#include "codec/pipeline.hpp"
#include "core/assets.hpp"
#include "core/bitstream.hpp"
#include "core/half.hpp"
#include "core/model.hpp"
#include "core/sha256.hpp"
#include "core/yuv.hpp"
#include "runtime/aot_graph.hpp"
#include "runtime/backend.hpp"
#include "mlvc_model.pb.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using namespace mlvc;
using mlvc::test::InMemoryAssetReader;

std::vector<std::byte> serialize_message(const google::protobuf::MessageLite& message)
{
    return mlvc::test::to_bytes(message.SerializeAsString());
}

std::string sha256_bytes_str(std::span<const std::byte> data)
{
    const Sha256Digest d = sha256_digest(data);
    return std::string(reinterpret_cast<const char*>(d.data()), d.size());
}

wire::Graph make_dummy_graph(const std::string& checkpoint_sha,
                             const std::vector<std::byte>& weights)
{
    wire::Graph graph;
    graph.set_schema_version(3);
    graph.set_checkpoint_sha256(checkpoint_sha);
    graph.set_weights_sha256(sha256_bytes_str(weights));
    graph.set_weights_bytes(weights.size());
    graph.set_arena_bytes(256);
    graph.set_arena_alignment(256);
    for (std::uint32_t id = 1; id <= 3; ++id) {
        auto* tensor = graph.add_tensors();
        tensor->set_id(id);
        tensor->set_name("t" + std::to_string(id));
        tensor->set_dtype(wire::FP16);
        tensor->add_shape(id == 1 ? static_cast<std::int64_t>(weights.size() / 2) : 1);
    }
    graph.mutable_tensors(0)->mutable_weight()->set_size_bytes(weights.size());
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

    const auto gaussian_bytes = serialize_message(gaussian);
    const auto bit_est_bytes = serialize_message(bit_estimator);
    const Sha256Digest g_hash = sha256_digest(gaussian_bytes);
    const Sha256Digest b_hash = sha256_digest(bit_est_bytes);
    Sha256 entropy_hasher;
    entropy_hasher.update(g_hash);
    entropy_hasher.update(b_hash);
    const Sha256Digest entropy_digest = entropy_hasher.finalize();
    const std::string entropy_sha(
        reinterpret_cast<const char*>(entropy_digest.data()), entropy_digest.size());

    wire::ModelBundle bundle;
    bundle.set_schema_version(2);
    bundle.set_profile("synthetic-v1");
    bundle.set_model_version("v1");
    bundle.set_width(64);
    bundle.set_height(64);
    bundle.set_checkpoint_sha256(checkpoint_sha);
    bundle.set_entropy_model_sha256(entropy_sha);
    bundle.mutable_contract()->set_container("mlvc-nalu-v0");
    bundle.mutable_contract()->set_entropy("canonical-pmf-v1");

    const std::vector<std::byte> enc_weights{
        std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    const std::vector<std::byte> dec_weights{
        std::byte{5}, std::byte{6}};

    auto reader = std::make_shared<InMemoryAssetReader>("memory://synthetic-v1");
    reader->put("metadata.pb", serialize_message(metadata));
    reader->put("model_bundle.pb", serialize_message(bundle));
    reader->put("gaussian_pmf.pb", gaussian_bytes);
    reader->put("bit_estimator_pmf.pb", bit_est_bytes);
    reader->put("aot/MLVCEncoder/graph.pb",
                serialize_message(make_dummy_graph(checkpoint_sha, enc_weights)));
    reader->put("aot/MLVCEncoder/weights.bin", enc_weights);
    reader->put("aot/MLVCDecoder/graph.pb",
                serialize_message(make_dummy_graph(checkpoint_sha, dec_weights)));
    reader->put("aot/MLVCDecoder/weights.bin", dec_weights);
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
        MLVC_REQUIRE(options_.asset_reader != nullptr,
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
            MLVC_REQUIRE(device_state_ && inputs.size() == 2,
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
            MLVC_REQUIRE(device_state_ && inputs.size() == 4,
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
    MLVC_REQUIRE(config.name == "synthetic-v1" && config.model_width == 64,
            "ModelConfig loaded via ModelAssetReader");

    const aot::LoadedGraph encoder_graph = aot::load_aot_graph(*bundle, "MLVCEncoder");
    MLVC_REQUIRE(encoder_graph.weights.size() == 4, "AOT graph weights loaded via asset reader");

    auto mismatched = make_synthetic_bundle();
    const std::vector<std::byte> enc_w{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    wire::Graph bad_graph = make_dummy_graph(std::string(32, 'x'), enc_w);
    mismatched->put("aot/MLVCEncoder/graph.pb", serialize_message(bad_graph));
    bool threw = false;
    try {
        (void)aot::load_aot_graph(*mismatched, "MLVCEncoder");
    } catch (const std::exception&) {
        threw = true;
    }
    MLVC_REQUIRE(threw, "mismatched AOT checkpoint SHA must throw");

    // Verify tampered weights.bin fails real SHA-256 verification.
    auto tampered = make_synthetic_bundle();
    tampered->put("aot/MLVCEncoder/weights.bin",
                  {std::byte{9}, std::byte{2}, std::byte{3}, std::byte{4}});
    threw = false;
    try {
        (void)aot::load_aot_graph(*tampered, "MLVCEncoder");
    } catch (const std::exception&) {
        threw = true;
    }
    MLVC_REQUIRE(threw, "tampered AOT weights.bin must fail SHA-256 verification");

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
    MLVC_REQUIRE(std::find(names.begin(), names.end(), "mock-alt") != names.end() &&
            std::find(names.begin(), names.end(), "mock-primary") != names.end(),
            "registered backends must appear in available_backends()");
    MLVC_REQUIRE(default_backend_name() == "mock-primary",
            "make_default=true should update default_backend_name()");

    BackendOptions opts;
    opts.backend_name = "mock-alt";
    opts.asset_reader = bundle;
    auto created = create_backend(opts);
    MLVC_REQUIRE(created->name() == "mock-alt", "explicit backend selection must pick mock-alt");

    opts.backend_name = "nonexistent-backend";
    threw = false;
    try {
        (void)create_backend(opts);
    } catch (const std::exception&) {
        threw = true;
    }
    MLVC_REQUIRE(threw, "unknown backend name must throw");
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
    MLVC_REQUIRE(p0.frame_type == FrameType::kIFrame &&
            p1.frame_type == FrameType::kPFrame &&
            p2.frame_type == FrameType::kIFrame &&
            !p0.bitstream.empty() && !p1.bitstream.empty() &&
            !p2.bitstream.empty(),
            "encoded frames must produce valid NALU access units");
    MLVC_REQUIRE(encoder.frame_index() == 3, "encoder frame_index must track encoded frames");
    MLVC_REQUIRE(enc_telemetry->encoder_runs == 3 && enc_telemetry->resets == 2,
            "GOP boundary should trigger backend reset_state() on I-frames");
    MLVC_REQUIRE(enc_telemetry->shifted_qs == (std::vector<int>{3, 1, 4}),
            "QP shifts across GOP frames should match metadata schedule");

    const Yuv420Frame d0 = decoder.decode_frame(p0);
    Yuv420Frame reusable = make_solid_frame(0);
    decoder.decode_frame_into(p1, mutable_yuv420_view(reusable));
    const Yuv420Frame d2 = decoder.decode_frame(p2);
    MLVC_REQUIRE(decoder.frame_index() == 3, "decoder frame_index must track decoded frames");
    MLVC_REQUIRE(dec_telemetry->decoder_runs == 3 && dec_telemetry->resets == 2,
            "decoder GOP boundary should reset state identically");
    // Frame 0 had state_token=0 => filler 0.75 * 255 = ~191; Frame 1 had state_token=1 => filler (0.25 + 0.125) * 255 = ~96.
    // Frame 2 reset state => back to state_token=0!
    MLVC_REQUIRE(d0.y.at(0) > 150 && reusable.y.at(0) < 120 && d2.y.at(0) == d0.y.at(0),
            "decoded YUV frames should reflect entropy-coded latents and GOP resets");

    encoder.reset();
    decoder.reset();
    MLVC_REQUIRE(encoder.frame_index() == 0 && decoder.frame_index() == 0,
            "explicit reset() should rewind frame counters");

    // Verify force_idr triggers an immediate IDR even mid-GOP.
    const EncodedFrame idr0 = encoder.encode_frame(yuv420_view(f0));
    const EncodedFrame forced_idr = encoder.encode_frame(yuv420_view(f1), 1, true);
    MLVC_REQUIRE(idr0.frame_type == FrameType::kIFrame &&
                forced_idr.frame_type == FrameType::kIFrame,
            "force_idr must produce an IDR frame mid-GOP");

    // Verify DecoderSession rejects a P-frame before IDR and rejects frame loss.
    decoder.reset();
    MLVC_REQUIRE_THROWS(static_cast<void>(decoder.decode_frame(p1)),
                        "decoding P-frame before IDR must throw");

    // Verify decode_video() also rejects missing-IDR P-frame on disk streams.
    {
        mlvc::test::ScopedTempDir temp("mlvc-pkt-loss");
        const auto bad_stream_path = temp.path() / "bad.mlvc";
        const auto out_yuv_path = temp.path() / "out.yuv";
        {
            std::ofstream out(bad_stream_path, std::ios::binary);
            mlvc::write_access_unit(out, p0.bitstream);
            // Feed p1 twice: second p1 expects ref_frame_idx=0 while last decoded is 1.
            mlvc::write_access_unit(out, p1.bitstream);
            mlvc::write_access_unit(out, p1.bitstream);
        }
        CodecOptions bad_dec_opts;
        bad_dec_opts.width = 50;
        bad_dec_opts.height = 38;
        bad_dec_opts.input_path = bad_stream_path.string();
        bad_dec_opts.output_path = out_yuv_path.string();
        bad_dec_opts.backend_name = "mock-dec";
        bad_dec_opts.asset_reader = bundle;
        MLVC_REQUIRE_THROWS(static_cast<void>(decode_video(bad_dec_opts)),
                            "decode_video must reject mismatched ref_frame_idx");
    }
}

void test_execution_preference_and_concurrent_registry()
{
    const auto bundle = make_synthetic_bundle();
    auto telemetry = std::make_shared<MockTelemetry>();

    // Register failing default and NPU backends and verify automatic fallback to
    // "cpu" when allowed (works identically even when a real GPU is installed).
    register_backend(
        "mock-npu-failing",
        [](const BackendOptions&) -> std::unique_ptr<InferenceBackend> {
            throw std::runtime_error("simulated accelerator driver unavailable");
        },
        true);
    // Ensure a "cpu" backend is available for fallback check if compiled with MLVC_BACKEND=none.
    const auto existing = available_backends();
    if (std::find(existing.begin(), existing.end(), "cpu") == existing.end()) {
        register_backend(
            "cpu",
            [telemetry](const BackendOptions& opts) {
                return std::make_unique<MockCodecBackend>("cpu", opts, telemetry);
            },
            false);
    }

    BackendOptions fallback_opts;
    fallback_opts.execution_preference = ExecutionPreference::kNpu;
    fallback_opts.allow_cpu_fallback = true;
    fallback_opts.asset_reader = bundle;
    auto resolved = create_backend(fallback_opts);
    MLVC_REQUIRE(resolved->name() == "cpu",
            "accelerator failure with allow_cpu_fallback=true must fall back to cpu");

    BackendOptions auto_fallback_opts;
    auto_fallback_opts.execution_preference = ExecutionPreference::kAuto;
    auto_fallback_opts.allow_cpu_fallback = true;
    auto_fallback_opts.asset_reader = bundle;
    auto resolved_auto = create_backend(auto_fallback_opts);
    MLVC_REQUIRE(resolved_auto->name() == "cpu",
            "default backend failure under kAuto with allow_cpu_fallback=true must fall back to cpu");

    BackendOptions strict_opts = fallback_opts;
    strict_opts.allow_cpu_fallback = false;
    MLVC_REQUIRE_THROWS(static_cast<void>(create_backend(strict_opts)),
            "accelerator failure with allow_cpu_fallback=false must throw");

    // Concurrent registry mutation & default_backend_name() reads must never dangle or race.
    std::atomic<bool> stop{false};
    std::thread reader([&]() {
        while (!stop.load(std::memory_order_acquire)) {
            const std::string_view name = default_backend_name();
            const char* cstr = compiled_backend_name_c_str();
            if (name.empty() || !cstr || cstr[0] == '\0')
                std::abort();
        }
    });
    for (int i = 0; i < 64; ++i) {
        const std::string bname = "mock-concurrent-" + std::to_string(i);
        register_backend(
            bname,
            [bname, telemetry](const BackendOptions& opts) {
                return std::make_unique<MockCodecBackend>(bname, opts, telemetry);
            },
            true);
    }
    stop.store(true, std::memory_order_release);
    reader.join();
}

class CountingAssetReader final : public ModelAssetReader {
public:
    explicit CountingAssetReader(std::shared_ptr<const ModelAssetReader> inner)
        : inner_(std::move(inner))
    {
    }

    std::string location_label() const override { return inner_->location_label(); }
    bool exists(const std::filesystem::path& relative_path) const override
    {
        return inner_->exists(relative_path);
    }
    std::vector<std::byte> read_binary(
        const std::filesystem::path& relative_path) const override
    {
        ++read_counts_[relative_path.generic_string()];
        return inner_->read_binary(relative_path);
    }
    AssetView read_view(const std::filesystem::path& relative_path) const override
    {
        ++read_counts_[relative_path.generic_string()];
        return inner_->read_view(relative_path);
    }

    int count(const std::string& path) const
    {
        const auto it = read_counts_.find(path);
        return it == read_counts_.end() ? 0 : it->second;
    }

private:
    std::shared_ptr<const ModelAssetReader> inner_;
    mutable std::unordered_map<std::string, int> read_counts_;
};

std::shared_ptr<InMemoryAssetReader> make_synthetic_bundle_with_artifacts()
{
    auto reader = make_synthetic_bundle();
    const AssetView bundle_bytes = reader->read_view("model_bundle.pb");
    wire::ModelBundle bundle;
    MLVC_REQUIRE(bundle.ParseFromArray(bundle_bytes.data(),
                                       static_cast<int>(bundle_bytes.size())),
                 "parse synthetic model_bundle.pb");
    for (const char* path : {
             "metadata.pb", "gaussian_pmf.pb", "bit_estimator_pmf.pb",
             "aot/MLVCEncoder/graph.pb", "aot/MLVCEncoder/weights.bin",
             "aot/MLVCDecoder/graph.pb", "aot/MLVCDecoder/weights.bin"}) {
        const AssetView data = reader->read_view(path);
        auto* art = bundle.add_artifacts();
        art->set_path(path);
        art->set_size_bytes(data.size());
        art->set_sha256(sha256_bytes_str(data.span()));
    }
    reader->put("model_bundle.pb", serialize_message(bundle));
    return reader;
}

void test_scoped_bundle_verification_avoids_duplicate_hashing()
{
    const auto base = make_synthetic_bundle_with_artifacts();

    // 1. Scoped encoder load must skip decoder artifacts, read PMF files once,
    //    and hash MLVCEncoder/weights.bin only once (inside load_aot_graph).
    CountingAssetReader enc_counter(base);
    const ModelConfig enc_cfg = load_model_config(enc_counter, "MLVCEncoder");
    MLVC_REQUIRE(enc_cfg.name == "synthetic-v1", "scoped encoder config loaded");
    MLVC_REQUIRE(enc_counter.count("aot/MLVCEncoder/weights.bin") == 0,
                 "load_model_config(MLVCEncoder) must defer weights.bin scan");
    MLVC_REQUIRE(enc_counter.count("aot/MLVCDecoder/weights.bin") == 0 &&
                 enc_counter.count("aot/MLVCDecoder/graph.pb") == 0,
                 "load_model_config(MLVCEncoder) must skip sibling decoder artifacts");
    MLVC_REQUIRE(enc_counter.count("gaussian_pmf.pb") == 1 &&
                 enc_counter.count("bit_estimator_pmf.pb") == 1,
                 "PMF artifacts must be hashed only once");
    const aot::LoadedGraph enc_graph = aot::load_aot_graph(enc_counter, "MLVCEncoder");
    MLVC_REQUIRE(enc_graph.weights.size() == 4 &&
                 enc_counter.count("aot/MLVCEncoder/weights.bin") == 1,
                 "weights.bin must be read & hashed exactly once across config + graph load");

    // 2. Full (unscoped) load_model_config must still verify every artifact while
    //    avoiding duplicate PMF hashing.
    CountingAssetReader full_counter(base);
    static_cast<void>(load_model_config(full_counter));
    MLVC_REQUIRE(full_counter.count("aot/MLVCEncoder/weights.bin") == 1 &&
                 full_counter.count("aot/MLVCDecoder/weights.bin") == 1 &&
                 full_counter.count("gaussian_pmf.pb") == 1 &&
                 full_counter.count("bit_estimator_pmf.pb") == 1,
                 "unscoped load_model_config must verify all artifacts once");

    // 3. Tampering decoder weights does not break encoder scoped load, but
    //    still fails unscoped load and decoder load.
    auto dec_tampered = make_synthetic_bundle_with_artifacts();
    dec_tampered->put("aot/MLVCDecoder/weights.bin", {std::byte{9}, std::byte{9}});
    static_cast<void>(load_model_config(*dec_tampered, "MLVCEncoder"));
    MLVC_REQUIRE_THROWS(static_cast<void>(load_model_config(*dec_tampered)),
                        "unscoped load must catch tampered decoder weights");
    MLVC_REQUIRE_THROWS(
        static_cast<void>(aot::load_aot_graph(*dec_tampered, "MLVCDecoder")),
        "decoder graph load must catch tampered decoder weights");

    // 4. Tampering encoder weights must still fail aot::load_aot_graph.
    auto enc_tampered = make_synthetic_bundle_with_artifacts();
    enc_tampered->put("aot/MLVCEncoder/weights.bin",
                      {std::byte{9}, std::byte{2}, std::byte{3}, std::byte{4}});
    MLVC_REQUIRE_THROWS(
        static_cast<void>(aot::load_aot_graph(*enc_tampered, "MLVCEncoder")),
        "encoder graph load must catch tampered encoder weights");

    // 5. Mismatch between model_bundle.pb weights SHA and graph.pb weights SHA
    //    must fail load_model_config(MLVCEncoder) immediately.
    auto bundle_mismatch = make_synthetic_bundle_with_artifacts();
    wire::ModelBundle bad_bundle;
    const AssetView bb = bundle_mismatch->read_view("model_bundle.pb");
    MLVC_REQUIRE(bad_bundle.ParseFromArray(bb.data(), static_cast<int>(bb.size())),
                 "parse bundle for mismatch test");
    for (int i = 0; i < bad_bundle.artifacts_size(); ++i) {
        if (bad_bundle.artifacts(i).path() == "aot/MLVCEncoder/weights.bin")
            bad_bundle.mutable_artifacts(i)->set_sha256(std::string(32, 'z'));
    }
    bundle_mismatch->put("model_bundle.pb", serialize_message(bad_bundle));
    MLVC_REQUIRE_THROWS(
        static_cast<void>(load_model_config(*bundle_mismatch, "MLVCEncoder")),
        "mismatched weights SHA between model_bundle.pb and graph.pb must throw");

    // 6. Invalid model_part path traversal must throw.
    MLVC_REQUIRE_THROWS(
        static_cast<void>(load_model_config(*base, "../evil")),
        "invalid model_part must throw");
}

}  // namespace

int main()
{
    try {
        test_backend_registry_and_aot_assets();
        test_sessions_without_disk_io();
        test_execution_preference_and_concurrent_registry();
        test_scoped_bundle_verification_avoids_duplicate_hashing();
        std::cout << "Runtime decoupling tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
