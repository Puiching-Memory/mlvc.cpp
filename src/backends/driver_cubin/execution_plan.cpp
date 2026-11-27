#include "aot_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlvc::driver_cubin_backend {

void AotGraph::execute_schedule()
{
    const auto& nodes = graph_.nodes;
    for (std::size_t index = 0; index < nodes.size();) {
        if (active_codec_yuv_device_ != 0 && use_nhwc_) {
            if (codec_fused_s2d8_node_.has_value() &&
                index == *codec_fused_s2d8_node_) {
                launch_yuv420_to_nchw(active_codec_yuv_device_);
                ++index;
                continue;
            }
            if (codec_fused_d2s8_node_.has_value() &&
                index == *codec_fused_d2s8_node_) {
                launch_nchw_to_yuv420(active_codec_yuv_device_);
                ++index;
                continue;
            }
        }
        if (std::find(elided_schedule_nodes_.begin(),
                      elided_schedule_nodes_.end(), index) !=
                elided_schedule_nodes_.end() ||
            std::find(codec_elided_nodes_.begin(),
                      codec_elided_nodes_.end(), index) !=
                codec_elided_nodes_.end()) {
            ++index;
        } else if (const std::size_t consumed =
                       try_execute_y0_tail(nodes, index);
                   consumed > 0) {
            index += consumed;
        } else if (try_execute_y1_tail(nodes, index))
            index += 15;
        else if (try_execute_dec_y0_tail(nodes, index))
            index += 11;
        else if (try_execute_dec_y1_tail(nodes, index))
            index += 9;
        else if (try_execute_feature_update_outputs(nodes, index))
            index += 11;
        else if (try_execute_feature_update(nodes, index))
            index += 12;
        else if (try_execute_pointwise_reglu(nodes, index))
            index += 5;
        else if (try_execute_pointwise_epilogue(nodes, index))
            index += 2;
        else if (try_execute_reglu(nodes, index))
            index += 4;
        else
            execute(nodes.at(index++));
    }
}

bool AotGraph::launch_cutlass_spatial(
    const Node& node, DeviceAddress input, DeviceAddress weight,
    DeviceAddress bias, DeviceAddress output, int batch_count,
    int in_channels, int input_height, int input_width, int out_channels,
    int output_height, int output_width, int kernel_height,
    int kernel_width, int stride_height, int stride_width,
    int pad_height, int pad_width)
{
    if (driver_.device_info().compute_major < 8 || batch_count != 1 ||
        bias == 0 ||
        (!use_nhwc_ && (!spatial_input_buffer_ || !spatial_output_buffer_)) ||
        in_channels % 8 != 0 || out_channels % 8 != 0 ||
        kernel_height <= 1 || kernel_width <= 1) {
        return false;
    }

    const int output_spatial = output_height * output_width;
    const unsigned int large_blocks =
        divide_up(output_spatial, 128) * divide_up(out_channels, 256);
    const bool use_small_spatial =
        large_blocks < static_cast<unsigned int>(
                           std::max(1, driver_.device_info().multiprocessor_count / 2));

    const std::size_t node_index = node.index;
    auto [host_params, inserted] =
        cutlass_spatial_host_parameters_.try_emplace(node_index);

    DeviceAddress input_nhwc =
        use_nhwc_ ? input : spatial_input_buffer_.address();
    DeviceAddress output_nhwc =
        use_nhwc_ ? output : spatial_output_buffer_.address();
    if (inserted) {
        try {
            const std::size_t weight_elements =
                static_cast<std::size_t>(out_channels) * in_channels *
                kernel_height * kernel_width;
            auto [transformed_weight, weight_inserted] =
                cutlass_spatial_weight_buffers_.try_emplace(node_index);
            if (!weight_inserted)
                throw std::runtime_error(
                    "driver-cubin: inconsistent spatial CUTLASS state");
            transformed_weight->second =
                driver_.allocate(weight_elements * sizeof(Float16Storage));
            DeviceAddress weight_krsc = transformed_weight->second.address();
            void* transform_parameters[] = {
                &weight, &weight_krsc, &out_channels, &in_channels,
                &kernel_height, &kernel_width};
            driver_.launch(
                oihw_to_krsc_, {divide_up(weight_elements, 256), 1, 1},
                {256, 1, 1}, 0, transform_parameters);

            driver_cubin::DeviceBuffer single_params_buffer;
            DeviceAddress params_storage = 0;
            if (batch_initializing_cutlass_) {
                constexpr std::size_t kSlotStride =
                    driver_cubin::kCutlassPointwiseParamsStorageBytes;
                const std::size_t offset =
                    pending_cutlass_inits_.size() * kSlotStride;
                params_storage = cutlass_init_scratch_.address() + offset;
                pending_cutlass_inits_.push_back({node_index, offset, true});
            } else {
                single_params_buffer = driver_.allocate(
                    driver_cubin::kCutlassPointwiseParamsStorageBytes);
                params_storage = single_params_buffer.address();
            }
            void* init_parameters[] = {
                &params_storage, &input_nhwc, &weight_krsc, &bias, &output_nhwc,
                &batch_count, &in_channels, &input_height, &input_width,
                &out_channels, &output_height, &output_width, &kernel_height,
                &kernel_width, &stride_height, &stride_width, &pad_height,
                &pad_width};
            driver_.launch(
                use_small_spatial ? cutlass_spatial_convolution_small_init_
                                  : cutlass_spatial_convolution_init_,
                {1, 1, 1}, {1, 1, 1}, 0, init_parameters);
            if (batch_initializing_cutlass_)
                return true;
            driver_.download(
                host_params->second.data(), single_params_buffer.address(),
                host_params->second.size());
        } catch (...) {
            cutlass_spatial_host_parameters_.erase(node_index);
            cutlass_spatial_weight_buffers_.erase(node_index);
            throw;
        }
    } else if (batch_initializing_cutlass_) {
        return true;
    }

    if (!use_nhwc_) {
        int input_rows = in_channels;
        int input_columns = input_height * input_width;
        void* input_transform_parameters[] = {
            &input, &input_nhwc, &input_rows, &input_columns};
        driver_.launch(
            transpose_,
            {divide_up(input_columns, 32), divide_up(input_rows, 32), 1},
            {32, 8, 1}, 0, input_transform_parameters);
    }

    void* convolution_parameters[] = {host_params->second.data()};
    if (use_small_spatial) {
        driver_.launch(
            cutlass_spatial_convolution_small_,
            {divide_up(output_spatial, 64), divide_up(out_channels, 64), 1},
            {128, 1, 1}, 32768U, convolution_parameters);
    } else {
        driver_.launch(
            cutlass_spatial_convolution_,
            {divide_up(output_spatial, 128), divide_up(out_channels, 256), 1},
            {256, 1, 1}, 73728U, convolution_parameters);
    }

    if (!use_nhwc_) {
        int output_rows = output_spatial;
        int output_columns = out_channels;
        void* output_transform_parameters[] = {
            &output_nhwc, &output, &output_rows, &output_columns};
        driver_.launch(
            transpose_,
            {divide_up(output_columns, 32), divide_up(output_rows, 32), 1},
            {32, 8, 1}, 0, output_transform_parameters);
    }
    return true;
}

bool AotGraph::can_use_cutlass_pointwise(
    int batch_count, bool has_bias, int in_channels, int spatial_count,
    int out_channels, int epilogue, float epilogue_alpha,
    bool has_residual, int in_stride, int res_stride,
    int out_stride) const noexcept
{
    if (use_nhwc_) {
        const int eff_in = in_stride > 0 ? in_stride : in_channels;
        const int eff_res = res_stride > 0 ? res_stride : out_channels;
        const int eff_out = out_stride > 0 ? out_stride : out_channels;
        // For tiny sp <= 64 with 16-aligned channels, Warp Split-K tiny_nhwc
        // beats CUTLASS m64n64 (4.71 us vs 5.65 us at sp=15).
        if (spatial_count <= 64 && (in_channels % 16) == 0 &&
            (out_channels % 16) == 0) {
            return false;
        }
        return driver_.device_info().compute_major >= 8 && batch_count == 1 &&
               has_bias && spatial_count >= 1 && (in_channels % 8) == 0 &&
               (out_channels % 8) == 0 && (eff_in % 8) == 0 &&
               (eff_out % 8) == 0 && epilogue >= 0 && epilogue <= 2 &&
               (epilogue != 1 || std::abs(epilogue_alpha - 0.01F) <= 1.0e-7F) &&
               (epilogue != 2 || (has_residual && (eff_res % 8) == 0));
    }
    return driver_.device_info().compute_major >= 8 && batch_count == 1 &&
           has_bias && spatial_count >= 920 && (spatial_count % 8) == 0 &&
           (in_channels % 8) == 0 && (out_channels % 8) == 0 &&
           epilogue >= 0 && epilogue <= 2 &&
           (epilogue != 1 || std::abs(epilogue_alpha - 0.01F) <= 1.0e-7F) &&
           (epilogue != 2 || has_residual);
}

bool AotGraph::launch_cutlass_pointwise(
    const Node& node, DeviceAddress input, DeviceAddress weight,
    DeviceAddress bias, DeviceAddress residual, DeviceAddress output,
    int batch_count, int in_channels, int spatial_count, int out_channels,
    int epilogue, float epilogue_alpha, int in_stride, int res_stride,
    int out_stride)
{
    if (!can_use_cutlass_pointwise(batch_count, bias != 0, in_channels,
                                   spatial_count, out_channels, epilogue,
                                   epilogue_alpha, residual != 0,
                                   in_stride, res_stride, out_stride)) {
        return false;
    }

    if (use_nhwc_) {
        int eff_in = in_stride > 0 ? in_stride : in_channels;
        int eff_res = res_stride > 0 ? res_stride : out_channels;
        int eff_out = out_stride > 0 ? out_stride : out_channels;

        // Empirical A30 NHWC tile dispatch (M = spatial_count, N = out_channels):
        // - sp >= 3000: m128n128 (3-stage) wins on wide ep0 (384x3680x256:
        //   15.31 us vs 16.60 us NCHW); m128n64 wins on 256x3680x256 ep1 and
        //   non-128-aligned channels (192); m64n128 wins on 256 ep2.
        // - 920 <= sp < 3000: m64n128 wins across 256/384 (-9% to -10% vs NCHW).
        // - sp < 920 (e.g. 240): m64n64 maximizes wave occupancy.
        enum NhwcTile { kM128N64, kM128N128, kM64N128, kM64N64 };
        NhwcTile tile = kM64N64;
        if (spatial_count >= 3000) {
            if (epilogue == 0 && out_channels >= 384 &&
                (out_channels % 128) == 0) {
                tile = kM128N128;
            } else if ((out_channels % 128) == 0) {
                tile = (out_channels == 256 && in_channels == 256 && epilogue == 1)
                    ? kM128N64 : kM64N128;
            } else {
                tile = kM128N64;
            }
        } else if (spatial_count >= 920) {
            tile = (out_channels >= 256 && (out_channels % 128) == 0)
                ? kM64N128 : kM64N64;
        } else {
            tile = kM64N64;
        }

        driver_cubin::abi::Function init = nullptr;
        driver_cubin::abi::Function function = nullptr;
        unsigned int tile_m = 64U, tile_n = 64U, shared_bytes = 24576U;
        if (tile == kM128N64) {
            tile_m = 128U; tile_n = 64U; shared_bytes = 36864U;
            init = epilogue == 0 ? cutlass_nhwc_m128n64_init_
                 : epilogue == 1 ? cutlass_nhwc_m128n64_leaky_relu_init_
                                 : cutlass_nhwc_m128n64_residual_init_;
            function = epilogue == 0 ? cutlass_nhwc_m128n64_
                     : epilogue == 1 ? cutlass_nhwc_m128n64_leaky_relu_
                                     : cutlass_nhwc_m128n64_residual_;
        } else if (tile == kM128N128) {
            tile_m = 128U; tile_n = 128U; shared_bytes = 49152U;
            init = epilogue == 0 ? cutlass_nhwc_m128n128_init_
                 : epilogue == 1 ? cutlass_nhwc_m128n128_leaky_relu_init_
                                 : cutlass_nhwc_m128n128_residual_init_;
            function = epilogue == 0 ? cutlass_nhwc_m128n128_
                     : epilogue == 1 ? cutlass_nhwc_m128n128_leaky_relu_
                                     : cutlass_nhwc_m128n128_residual_;
        } else if (tile == kM64N128) {
            tile_m = 64U; tile_n = 128U; shared_bytes = 36864U;
            init = epilogue == 0 ? cutlass_nhwc_m64n128_init_
                 : epilogue == 1 ? cutlass_nhwc_m64n128_leaky_relu_init_
                                 : cutlass_nhwc_m64n128_residual_init_;
            function = epilogue == 0 ? cutlass_nhwc_m64n128_
                     : epilogue == 1 ? cutlass_nhwc_m64n128_leaky_relu_
                                     : cutlass_nhwc_m64n128_residual_;
        } else {
            tile_m = 64U; tile_n = 64U; shared_bytes = 24576U;
            init = epilogue == 0 ? cutlass_nhwc_m64n64_init_
                 : epilogue == 1 ? cutlass_nhwc_m64n64_leaky_relu_init_
                                 : cutlass_nhwc_m64n64_residual_init_;
            function = epilogue == 0 ? cutlass_nhwc_m64n64_
                     : epilogue == 1 ? cutlass_nhwc_m64n64_leaky_relu_
                                     : cutlass_nhwc_m64n64_residual_;
        }

        int log_tile = cutlass_log_tile_;
        const std::size_t node_index = node.index;
        auto [host_params, inserted] =
            cutlass_host_parameters_.try_emplace(node_index);
        if (inserted) {
            try {
                driver_cubin::DeviceBuffer single_params_buffer;
                DeviceAddress params_storage = 0;
                constexpr std::size_t kSlotStride =
                    driver_cubin::kCutlassPointwiseParamsStorageBytes;
                if (batch_initializing_cutlass_) {
                    const std::size_t offset =
                        pending_cutlass_inits_.size() * kSlotStride;
                    params_storage = cutlass_init_scratch_.address() + offset;
                    pending_cutlass_inits_.push_back({node_index, offset, false});
                } else {
                    single_params_buffer = driver_.allocate(kSlotStride);
                    params_storage = single_params_buffer.address();
                }
                void* init_parameters[] = {
                    &params_storage, &input, &weight, &bias, &residual, &output,
                    &spatial_count, &out_channels, &in_channels,
                    &eff_in, &eff_res, &eff_out, &log_tile};
                driver_.launch(init, {1, 1, 1}, {1, 1, 1}, 0, init_parameters);
                if (batch_initializing_cutlass_)
                    return true;
                driver_.download(host_params->second.data(),
                                 single_params_buffer.address(),
                                 host_params->second.size());
            } catch (...) {
                cutlass_host_parameters_.erase(node_index);
                throw;
            }
        } else if (batch_initializing_cutlass_) {
            return true;
        }

        void* parameters[] = {host_params->second.data()};
        unsigned int grid_m = divide_up(spatial_count, tile_m);
        unsigned int grid_n = divide_up(out_channels, tile_n);
        if (log_tile > 0) {
            const unsigned int group = 1U << log_tile;
            grid_m *= group;
            grid_n = divide_up(grid_n, group);
        }
        driver_.launch(function, {grid_m, grid_n, 1}, {128, 1, 1},
                       shared_bytes, parameters);
        return true;
    }

    // Guard high-smem kernels (>48KB) so sm_86/sm_89 (100KB smem/SM) do not
    // drop to 1 CTA/SM where a 3-stage 48KB/36KB kernel fits 2 CTAs/SM.
    const bool large_smem_sm =
        driver_.device_info().max_shared_memory_per_sm >= 131072;
    const unsigned int sm_count = static_cast<unsigned int>(
        std::max(1, driver_.device_info().multiprocessor_count));

    // Spatial-wide 64x256 needs >= 2 full SM waves (e.g. 1080p+) to avoid
    // 45/60-CTA wave quantization on 56-SM GPUs; at 360p (sp=3680), 64x128
    // launches 87-116 CTAs at 36KB smem/CTA and beats both 128x128 (58 CTAs)
    // and 64x256 (45-60 CTAs) across every oc <= 256 shape.
    const unsigned int sw_blocks =
        divide_up(out_channels, 64) * divide_up(spatial_count, 256);
    const bool use_spatial_wide_tile = large_smem_sm &&
        spatial_count >= 3000 && sw_blocks >= sm_count * 2U &&
        ((in_channels >= 512 && out_channels <= 256) ||
         (in_channels >= 384 && out_channels <= 192));
    const bool use_small_m128_tile = !use_spatial_wide_tile &&
        spatial_count >= 3000 && out_channels <= 256;
    const bool use_medium_tile = !use_spatial_wide_tile &&
        !use_small_m128_tile && in_channels <= 384 &&
        spatial_count >= 3000 && out_channels > 256;
    // Small 64x64 boosts SM occupancy & cuts M-tile waste on compact sp < 3000
    // for oc <= 256 or non-128-aligned channels; oc >= 384 (128-aligned) at
    // sp = 920 uses base 128x64 (45-60 CTAs vs only 24 CTAs for 128x128).
    const bool use_small_m64n64_tile = !use_medium_tile &&
        !use_small_m128_tile && spatial_count < 3000 &&
        (out_channels <= 256 || (out_channels % 128 != 0));
    // medium_stage4 has no swizzle init variant; swizzle mode uses the
    // 3-stage medium tile instead.
    const bool use_medium_stage4_tile = large_smem_sm &&
        epilogue == 0 && use_medium_tile && spatial_count >= 3000 &&
        cutlass_log_tile_ == 0;

    enum PointwiseVariant {
        kBias,
        kLeakyRelu,
        kResidual,
        kMediumBias,
        kMediumStage4Bias,
        kMediumLeakyRelu,
        kMediumResidual,
        kSpatialWideBias,
        kSpatialWideLeakyRelu,
        kSpatialWideResidual,
        kSmallM128Bias,
        kSmallM128LeakyRelu,
        kSmallM128Residual,
        kSmallM64N64Bias,
        kSmallM64N64LeakyRelu,
        kSmallM64N64Residual,
        kVariantCount
    };
    PointwiseVariant variant = use_spatial_wide_tile ? kSpatialWideBias
        : use_medium_stage4_tile                     ? kMediumStage4Bias
        : use_medium_tile                            ? kMediumBias
        : use_small_m128_tile                        ? kSmallM128Bias
        : use_small_m64n64_tile                      ? kSmallM64N64Bias
                                                     : kBias;
    if (epilogue == 1) {
        variant = use_spatial_wide_tile ? kSpatialWideLeakyRelu
            : use_medium_tile           ? kMediumLeakyRelu
            : use_small_m128_tile       ? kSmallM128LeakyRelu
            : use_small_m64n64_tile     ? kSmallM64N64LeakyRelu
                                        : kLeakyRelu;
    } else if (epilogue == 2) {
        variant = use_spatial_wide_tile ? kSpatialWideResidual
            : use_medium_tile           ? kMediumResidual
            : use_small_m128_tile       ? kSmallM128Residual
            : use_small_m64n64_tile     ? kSmallM64N64Residual
                                        : kResidual;
    }

    const std::array<driver_cubin::abi::Function, kVariantCount> base_init{
        cutlass_pointwise_init_,
        cutlass_pointwise_leaky_relu_init_,
        cutlass_pointwise_residual_init_,
        cutlass_pointwise_medium_init_,
        cutlass_pointwise_medium_stage4_init_,
        cutlass_pointwise_medium_leaky_relu_init_,
        cutlass_pointwise_medium_residual_init_,
        cutlass_pointwise_spatial_wide_init_,
        cutlass_pointwise_spatial_wide_leaky_relu_init_,
        cutlass_pointwise_spatial_wide_residual_init_,
        cutlass_pointwise_m64n128_init_,
        cutlass_pointwise_m64n128_leaky_relu_init_,
        cutlass_pointwise_m64n128_residual_init_,
        cutlass_pointwise_m64n64_init_,
        cutlass_pointwise_m64n64_leaky_relu_init_,
        cutlass_pointwise_m64n64_residual_init_};
    const std::array<driver_cubin::abi::Function, kVariantCount> base_main{
        cutlass_pointwise_,
        cutlass_pointwise_leaky_relu_,
        cutlass_pointwise_residual_,
        cutlass_pointwise_medium_,
        cutlass_pointwise_medium_stage4_,
        cutlass_pointwise_medium_leaky_relu_,
        cutlass_pointwise_medium_residual_,
        cutlass_pointwise_spatial_wide_,
        cutlass_pointwise_spatial_wide_leaky_relu_,
        cutlass_pointwise_spatial_wide_residual_,
        cutlass_pointwise_m64n128_,
        cutlass_pointwise_m64n128_leaky_relu_,
        cutlass_pointwise_m64n128_residual_,
        cutlass_pointwise_m64n64_,
        cutlass_pointwise_m64n64_leaky_relu_,
        cutlass_pointwise_m64n64_residual_};
    const std::array<driver_cubin::abi::Function, kVariantCount> swizzle_init{
        cutlass_pointwise_swizzle_init_,
        cutlass_pointwise_leaky_relu_swizzle_init_,
        cutlass_pointwise_residual_swizzle_init_,
        cutlass_pointwise_medium_swizzle_init_,
        nullptr,
        cutlass_pointwise_medium_leaky_relu_swizzle_init_,
        cutlass_pointwise_medium_residual_swizzle_init_,
        cutlass_pointwise_spatial_wide_swizzle_init_,
        cutlass_pointwise_spatial_wide_leaky_relu_swizzle_init_,
        cutlass_pointwise_spatial_wide_residual_swizzle_init_,
        cutlass_pointwise_m64n128_swizzle_init_,
        cutlass_pointwise_m64n128_leaky_relu_swizzle_init_,
        cutlass_pointwise_m64n128_residual_swizzle_init_,
        cutlass_pointwise_m64n64_swizzle_init_,
        cutlass_pointwise_m64n64_leaky_relu_swizzle_init_,
        cutlass_pointwise_m64n64_residual_swizzle_init_};
    constexpr std::array<unsigned int, kVariantCount> variant_tile_rows{
        128U, 128U, 128U, 128U, 128U, 128U, 128U,
        64U, 64U, 64U, 64U, 64U, 64U, 64U, 64U, 64U};
    constexpr std::array<unsigned int, kVariantCount> variant_tile_columns{
        64U, 64U, 64U, 128U, 128U, 128U, 128U,
        256U, 256U, 256U, 128U, 128U, 128U, 64U, 64U, 64U};
    constexpr std::array<unsigned int, kVariantCount> variant_shared_bytes{
        36864U, 36864U, 36864U, 49152U, 65536U, 49152U, 49152U,
        61440U, 61440U, 61440U, 36864U, 36864U, 36864U,
        24576U, 24576U, 24576U};

    driver_cubin::abi::Function init = nullptr;
    driver_cubin::abi::Function function = nullptr;
    int log_tile = 0;
    if (cutlass_log_tile_ > 0) {
        init = swizzle_init[variant];
        function = base_main[variant];
        log_tile = cutlass_log_tile_;
    } else {
        init = base_init[variant];
        function = base_main[variant];
    }

    const std::size_t node_index = node.index;
    auto [host_params, inserted] =
        cutlass_host_parameters_.try_emplace(node_index);
    if (inserted) {
        try {
            driver_cubin::DeviceBuffer single_params_buffer;
            DeviceAddress params_storage = 0;
            constexpr std::size_t kSlotStride =
                driver_cubin::kCutlassPointwiseParamsStorageBytes;
            if (batch_initializing_cutlass_) {
                const std::size_t offset =
                    pending_cutlass_inits_.size() * kSlotStride;
                params_storage = cutlass_init_scratch_.address() + offset;
                pending_cutlass_inits_.push_back({node_index, offset, false});
            } else {
                single_params_buffer = driver_.allocate(kSlotStride);
                params_storage = single_params_buffer.address();
            }
            void* init_parameters[] = {
                &params_storage, &input, &weight, &bias, &residual, &output,
                &out_channels, &spatial_count, &in_channels, &log_tile};
            driver_.launch(
                init, {1, 1, 1}, {1, 1, 1}, 0,
                std::span<void*>(init_parameters,
                                 cutlass_log_tile_ > 0 ? 10 : 9));
            if (batch_initializing_cutlass_)
                return true;
            driver_.download(
                host_params->second.data(), single_params_buffer.address(),
                host_params->second.size());
        } catch (...) {
            cutlass_host_parameters_.erase(node_index);
            throw;
        }
    } else if (batch_initializing_cutlass_) {
        return true;
    }

    void* parameters[] = {host_params->second.data()};
    const unsigned int tile_rows = variant_tile_rows[variant];
    const unsigned int tile_columns = variant_tile_columns[variant];
    unsigned int grid_rows = divide_up(out_channels, tile_rows);
    unsigned int grid_columns = divide_up(spatial_count, tile_columns);
    if (log_tile > 0) {
        const unsigned int group = 1U << log_tile;
        grid_rows *= group;
        grid_columns = divide_up(grid_columns, group);
    }
    driver_.launch(
        function, {grid_rows, grid_columns, 1}, {128, 1, 1},
        variant_shared_bytes[variant], parameters);
    return true;
}

bool AotGraph::launch_cutlass_pointwise_reglu(
    const Node& node, DeviceAddress input, DeviceAddress weight,
    DeviceAddress bias, DeviceAddress output, int batch_count,
    int in_channels, int spatial_count, int gate_channels, float minimum,
    float maximum, int in_stride, int out_stride)
{
    if (use_nhwc_) {
        int eff_in = in_stride > 0 ? in_stride : in_channels;
        int eff_out = out_stride > 0 ? out_stride : gate_channels;
        if (driver_.device_info().compute_major < 8 || batch_count != 1 ||
            bias == 0 || spatial_count < 1 || (in_channels % 8) != 0 ||
            (gate_channels % 8) != 0 || (eff_in % 8) != 0 ||
            (eff_out % 8) != 0) {
            return false;
        }
        int out_channels = gate_channels * 2;
        const bool large_smem_sm =
            driver_.device_info().max_shared_memory_per_sm >= 131072;
        driver_cubin::abi::Function init = cutlass_nhwc_m64n64_reglu_init_;
        driver_cubin::abi::Function function = cutlass_nhwc_m64n64_reglu_;
        unsigned int tile_m = 64U, tile_n = 64U, shared_bytes = 24576U;
        if (spatial_count >= 3000) {
            if (large_smem_sm && out_channels >= 1024) {
                init = cutlass_nhwc_m128n128_stage4_reglu_init_;
                function = cutlass_nhwc_m128n128_stage4_reglu_;
                tile_m = 128U; tile_n = 128U; shared_bytes = 65536U;
            } else {
                init = cutlass_nhwc_m128n128_reglu_init_;
                function = cutlass_nhwc_m128n128_reglu_;
                tile_m = 128U; tile_n = 128U; shared_bytes = 49152U;
            }
        } else if (spatial_count >= 920) {
            if (out_channels >= 1024 && out_channels < 1536) {
                init = cutlass_nhwc_m64n128_reglu_init_;
                function = cutlass_nhwc_m64n128_reglu_;
                tile_m = 64U; tile_n = 128U; shared_bytes = 36864U;
            } else {
                init = cutlass_nhwc_m128n128_reglu_init_;
                function = cutlass_nhwc_m128n128_reglu_;
                tile_m = 128U; tile_n = 128U; shared_bytes = 49152U;
            }
        }
        int log_tile = cutlass_log_tile_;
        const std::size_t node_index = node.index;
        auto [host_params, inserted] =
            cutlass_host_parameters_.try_emplace(node_index);
        if (inserted) {
            try {
                const std::size_t weight_bytes =
                    static_cast<std::size_t>(out_channels) * in_channels *
                    sizeof(Float16Storage);
                const std::size_t bias_bytes =
                    static_cast<std::size_t>(out_channels) *
                    sizeof(Float16Storage);
                auto [w_buf, w_ins] =
                    cutlass_reglu_weight_buffers_.try_emplace(node_index);
                auto [b_buf, b_ins] =
                    cutlass_reglu_bias_buffers_.try_emplace(node_index);
                if (!w_ins || !b_ins) {
                    throw std::runtime_error(
                        "driver-cubin: inconsistent NHWC ReGLU CUTLASS state");
                }
                w_buf->second = driver_.allocate(weight_bytes);
                b_buf->second = driver_.allocate(bias_bytes);
                DeviceAddress interleaved_w = w_buf->second.address();
                DeviceAddress interleaved_b = b_buf->second.address();
                void* w_interleave_params[] = {
                    &weight, &interleaved_w, &gate_channels, &in_channels};
                driver_.launch(
                    interleave_reglu_pairs_,
                    {divide_up(static_cast<std::size_t>(in_channels), 128),
                     static_cast<unsigned int>(out_channels), 1},
                    {128, 1, 1}, 0, w_interleave_params);
                int bias_width = 1;
                void* b_interleave_params[] = {
                    &bias, &interleaved_b, &gate_channels, &bias_width};
                driver_.launch(
                    interleave_reglu_pairs_,
                    {1, static_cast<unsigned int>(out_channels), 1},
                    {32, 1, 1}, 0, b_interleave_params);

                driver_cubin::DeviceBuffer single_params_buffer;
                DeviceAddress params_storage = 0;
                constexpr std::size_t kSlotStride =
                    driver_cubin::kCutlassPointwiseParamsStorageBytes;
                if (batch_initializing_cutlass_) {
                    const std::size_t offset =
                        pending_cutlass_inits_.size() * kSlotStride;
                    params_storage = cutlass_init_scratch_.address() + offset;
                    pending_cutlass_inits_.push_back({node_index, offset, false});
                } else {
                    single_params_buffer = driver_.allocate(kSlotStride);
                    params_storage = single_params_buffer.address();
                }
                void* init_parameters[] = {
                    &params_storage, &input, &interleaved_w, &interleaved_b,
                    &output, &spatial_count, &out_channels, &in_channels,
                    &eff_in, &eff_out, &minimum, &maximum, &log_tile};
                driver_.launch(init, {1, 1, 1}, {1, 1, 1}, 0, init_parameters);
                if (batch_initializing_cutlass_)
                    return true;
                driver_.download(host_params->second.data(),
                                 single_params_buffer.address(),
                                 host_params->second.size());
            } catch (...) {
                cutlass_host_parameters_.erase(node_index);
                cutlass_reglu_weight_buffers_.erase(node_index);
                cutlass_reglu_bias_buffers_.erase(node_index);
                throw;
            }
        } else if (batch_initializing_cutlass_) {
            return true;
        }
        void* parameters[] = {host_params->second.data()};
        unsigned int grid_m = divide_up(spatial_count, tile_m);
        unsigned int grid_n = divide_up(out_channels, tile_n);
        if (log_tile > 0) {
            const unsigned int group = 1U << log_tile;
            grid_m *= group;
            grid_n = divide_up(grid_n, group);
        }
        driver_.launch(function, {grid_m, grid_n, 1}, {128, 1, 1},
                       shared_bytes, parameters);
        return true;
    }

    if (driver_.device_info().compute_major < 8 || batch_count != 1 ||
        bias == 0 || spatial_count < 920 || (spatial_count % 8) != 0 ||
        (in_channels % 8) != 0 || (gate_channels % 8) != 0) {
        return false;
    }

    int out_channels = gate_channels * 2;
    const bool large_smem_sm =
        driver_.device_info().max_shared_memory_per_sm >= 131072;
    const bool use_medium_stage4 = large_smem_sm && spatial_count >= 3000 &&
        out_channels >= 512 && cutlass_log_tile_ == 0;
    const bool use_medium = !use_medium_stage4 &&
        ((spatial_count >= 3000 && out_channels >= 512) ||
         (spatial_count < 3000 && out_channels >= 1024 && in_channels >= 256));
    const bool use_m128n64 = !use_medium_stage4 && !use_medium &&
        (out_channels % 128 == 0) && out_channels >= 512;

    driver_cubin::abi::Function init = cutlass_pointwise_m64n64_reglu_init_;
    driver_cubin::abi::Function function = cutlass_pointwise_m64n64_reglu_;
    unsigned int tile_rows = 64U;
    unsigned int tile_columns = 64U;
    unsigned int shared_bytes = 24576U;
    if (use_medium_stage4) {
        init = cutlass_pointwise_medium_stage4_reglu_init_;
        function = cutlass_pointwise_medium_stage4_reglu_;
        tile_rows = 128U;
        tile_columns = 128U;
        shared_bytes = 65536U;
    } else if (use_medium) {
        init = cutlass_pointwise_medium_reglu_init_;
        function = cutlass_pointwise_medium_reglu_;
        tile_rows = 128U;
        tile_columns = 128U;
        shared_bytes = 49152U;
    } else if (use_m128n64) {
        init = cutlass_pointwise_reglu_init_;
        function = cutlass_pointwise_reglu_;
        tile_rows = 128U;
        tile_columns = 64U;
        shared_bytes = 36864U;
    }

    int log_tile = use_medium_stage4 ? 0 : cutlass_log_tile_;
    const std::size_t node_index = node.index;
    auto [host_params, inserted] =
        cutlass_host_parameters_.try_emplace(node_index);
    if (inserted) {
        try {
            const std::size_t weight_bytes =
                static_cast<std::size_t>(out_channels) * in_channels *
                sizeof(Float16Storage);
            const std::size_t bias_bytes =
                static_cast<std::size_t>(out_channels) * sizeof(Float16Storage);
            auto [w_buf, w_ins] =
                cutlass_reglu_weight_buffers_.try_emplace(node_index);
            auto [b_buf, b_ins] =
                cutlass_reglu_bias_buffers_.try_emplace(node_index);
            if (!w_ins || !b_ins) {
                throw std::runtime_error(
                    "driver-cubin: inconsistent ReGLU CUTLASS state");
            }
            w_buf->second = driver_.allocate(weight_bytes);
            b_buf->second = driver_.allocate(bias_bytes);
            DeviceAddress interleaved_w = w_buf->second.address();
            DeviceAddress interleaved_b = b_buf->second.address();
            void* w_interleave_params[] = {
                &weight, &interleaved_w, &gate_channels, &in_channels};
            driver_.launch(
                interleave_reglu_rows_,
                {divide_up(static_cast<std::size_t>(in_channels), 128),
                 static_cast<unsigned int>(out_channels), 1},
                {128, 1, 1}, 0, w_interleave_params);
            int bias_width = 1;
            void* b_interleave_params[] = {
                &bias, &interleaved_b, &gate_channels, &bias_width};
            driver_.launch(
                interleave_reglu_rows_,
                {1, static_cast<unsigned int>(out_channels), 1},
                {32, 1, 1}, 0, b_interleave_params);

            driver_cubin::DeviceBuffer single_params_buffer;
            DeviceAddress params_storage = 0;
            constexpr std::size_t kSlotStride =
                driver_cubin::kCutlassPointwiseParamsStorageBytes;
            if (batch_initializing_cutlass_) {
                const std::size_t offset =
                    pending_cutlass_inits_.size() * kSlotStride;
                params_storage = cutlass_init_scratch_.address() + offset;
                pending_cutlass_inits_.push_back({node_index, offset, false});
            } else {
                single_params_buffer = driver_.allocate(kSlotStride);
                params_storage = single_params_buffer.address();
            }
            void* init_parameters[] = {
                &params_storage, &input, &interleaved_w, &interleaved_b,
                &output, &out_channels, &spatial_count, &in_channels,
                &minimum, &maximum, &log_tile};
            driver_.launch(init, {1, 1, 1}, {1, 1, 1}, 0, init_parameters);
            if (batch_initializing_cutlass_)
                return true;
            driver_.download(
                host_params->second.data(), single_params_buffer.address(),
                host_params->second.size());
        } catch (...) {
            cutlass_host_parameters_.erase(node_index);
            cutlass_reglu_weight_buffers_.erase(node_index);
            cutlass_reglu_bias_buffers_.erase(node_index);
            throw;
        }
    } else if (batch_initializing_cutlass_) {
        return true;
    }

    void* parameters[] = {host_params->second.data()};
    unsigned int grid_rows = divide_up(out_channels, tile_rows);
    unsigned int grid_columns = divide_up(spatial_count, tile_columns);
    if (log_tile > 0) {
        const unsigned int group = 1U << log_tile;
        grid_rows *= group;
        grid_columns = divide_up(grid_columns, group);
    }
    driver_.launch(function, {grid_rows, grid_columns, 1}, {128, 1, 1},
                   shared_bytes, parameters);
    return true;
}

void AotGraph::launch_linear(driver_cubin::abi::Function function, std::size_t count,
                   std::span<void*> parameters)
{
    constexpr unsigned int block = 256;
    driver_.launch(function, {divide_up(count, block), 1, 1}, {block, 1, 1},
                   0, parameters);
}

void AotGraph::execute(const Node& node, const Value* output_override,
             DeviceAddress epilogue_input, int epilogue,
             float epilogue_alpha, int epilogue_stride)
{
    const Op op = node.op;
    const auto inputs = node.inputs;
    const auto outputs = node.outputs;
    if (outputs.size() != 1)
        throw std::runtime_error("driver-cubin: only single-output nodes are supported");
    const Value& output = output_override ? *output_override : value(outputs[0]);
    const std::size_t count_size = element_count(output.shape);
    if (count_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("driver-cubin: tensor is too large");
    int count = static_cast<int>(count_size);

    if (op == Op::ADD || op == Op::MUL || op == Op::SUB) {
        if (batch_initializing_cutlass_)
            return;
        const Value& lhs_val = value(inputs.at(0));
        const Value& rhs_val = value(inputs.at(1));
        DeviceAddress lhs = lhs_val.address;
        DeviceAddress rhs = rhs_val.address;
        DeviceAddress result = output.address;
        int operation = op == Op::ADD ? 0 : (op == Op::MUL ? 1 : 2);
        const auto od = dims4(output.shape);
        const auto ld = dims4(lhs_val.shape);
        const auto rd = dims4(rhs_val.shape);
        if (use_nhwc_) {
            if (lhs_val.shape == output.shape && rhs_val.shape == output.shape) {
                if (is_packed_nhwc(lhs_val) && is_packed_nhwc(rhs_val) &&
                    is_packed_nhwc(output)) {
                    void* contiguous_parameters[] = {
                        &lhs, &rhs, &result, &count, &operation};
                    launch_linear(binary_contiguous_, (count_size + 1) / 2,
                                  contiguous_parameters);
                } else {
                    int pixels = od[0] * od[2] * od[3];
                    int channels = od[1];
                    int ls = value_stride(lhs_val);
                    int rs = value_stride(rhs_val);
                    int os = value_stride(output);
                    const bool vec8 = ((channels & 7) == 0) &&
                        ((ls & 7) == 0) && ((rs & 7) == 0) && ((os & 7) == 0) &&
                        ((lhs & 15U) == 0) && ((rhs & 15U) == 0) &&
                        ((result & 15U) == 0);
                    const std::size_t work = static_cast<std::size_t>(pixels) *
                        static_cast<std::size_t>(vec8 ? (channels / 8) : channels);
                    void* sp[] = {&lhs, &rhs, &result, &pixels, &channels,
                                  &ls, &rs, &os, &operation};
                    launch_linear(binary_strided_nhwc_, work, sp);
                }
                return;
            }
            if (ld == od && rd[0] == 1 && rd[1] == od[1] &&
                rd[2] == 1 && rd[3] == 1) {
                int pixels = od[0] * od[2] * od[3], channels = od[1];
                int in_s = value_stride(lhs_val), out_s = value_stride(output);
                int swapped = 0;
                const bool vec8 = ((channels & 7) == 0) && ((in_s & 7) == 0) &&
                    ((out_s & 7) == 0) && ((lhs & 15U) == 0) &&
                    ((rhs & 15U) == 0) && ((result & 15U) == 0);
                const std::size_t work = static_cast<std::size_t>(pixels) *
                    static_cast<std::size_t>(vec8 ? (channels / 8) : channels);
                void* bp[] = {&lhs, &rhs, &result, &pixels, &channels,
                              &in_s, &out_s, &operation, &swapped};
                launch_linear(binary_channel_broadcast_nhwc_, work, bp);
                return;
            }
            if (rd == od && ld[0] == 1 && ld[1] == od[1] &&
                ld[2] == 1 && ld[3] == 1) {
                int pixels = od[0] * od[2] * od[3], channels = od[1];
                int in_s = value_stride(rhs_val), out_s = value_stride(output);
                int swapped = 1;
                const bool vec8 = ((channels & 7) == 0) && ((in_s & 7) == 0) &&
                    ((out_s & 7) == 0) && ((rhs & 15U) == 0) &&
                    ((lhs & 15U) == 0) && ((result & 15U) == 0);
                const std::size_t work = static_cast<std::size_t>(pixels) *
                    static_cast<std::size_t>(vec8 ? (channels / 8) : channels);
                void* bp[] = {&rhs, &lhs, &result, &pixels, &channels,
                              &in_s, &out_s, &operation, &swapped};
                launch_linear(binary_channel_broadcast_nhwc_, work, bp);
                return;
            }
            // General broadcast in packed NHWC: permute logical (N,C,H,W) -> (N,H,W,C).
            std::array<int, 4> nh_od{od[0], od[2], od[3], od[1]};
            std::array<int, 4> nh_ld{ld[0], ld[2], ld[3], ld[1]};
            std::array<int, 4> nh_rd{rd[0], rd[2], rd[3], rd[1]};
            void* nh_params[] = {
                &lhs, &rhs, &result, &count, &operation,
                &nh_od[0], &nh_od[1], &nh_od[2], &nh_od[3],
                &nh_ld[0], &nh_ld[1], &nh_ld[2], &nh_ld[3],
                &nh_rd[0], &nh_rd[1], &nh_rd[2], &nh_rd[3]};
            launch_linear(binary_, count_size, nh_params);
            return;
        }
        void* parameters[] = {
            &lhs, &rhs, &result, &count, &operation,
            const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
            const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
            const_cast<int*>(&ld[0]), const_cast<int*>(&ld[1]),
            const_cast<int*>(&ld[2]), const_cast<int*>(&ld[3]),
            const_cast<int*>(&rd[0]), const_cast<int*>(&rd[1]),
            const_cast<int*>(&rd[2]), const_cast<int*>(&rd[3]),
        };
        if (lhs_val.shape == output.shape &&
            rhs_val.shape == output.shape) {
            void* contiguous_parameters[] = {
                &lhs, &rhs, &result, &count, &operation};
            launch_linear(binary_contiguous_, (count_size + 1) / 2,
                          contiguous_parameters);
        } else if (ld == od && rd[0] == 1 && rd[1] == od[1] &&
                   rd[2] == 1 && rd[3] == 1) {
            int batch = od[0], channels = od[1], spatial = od[2] * od[3];
            int swapped = 0;
            void* bcast_parameters[] = {
                &lhs, &rhs, &result, &batch, &channels, &spatial,
                &operation, &swapped};
            driver_.launch(
                binary_channel_broadcast_,
                {divide_up(static_cast<std::size_t>(spatial + 1) / 2, 128),
                 static_cast<unsigned int>(channels),
                 static_cast<unsigned int>(batch)},
                {128, 1, 1}, 0, bcast_parameters);
        } else if (rd == od && ld[0] == 1 && ld[1] == od[1] &&
                   ld[2] == 1 && ld[3] == 1) {
            int batch = od[0], channels = od[1], spatial = od[2] * od[3];
            int swapped = 1;
            void* bcast_parameters[] = {
                &rhs, &lhs, &result, &batch, &channels, &spatial,
                &operation, &swapped};
            driver_.launch(
                binary_channel_broadcast_,
                {divide_up(static_cast<std::size_t>(spatial + 1) / 2, 128),
                 static_cast<unsigned int>(channels),
                 static_cast<unsigned int>(batch)},
                {128, 1, 1}, 0, bcast_parameters);
        } else {
            launch_linear(binary_, count_size, parameters);
        }
        return;
    }

    if (op == Op::LEAKY_RELU || op == Op::CLIP || op == Op::SIGMOID ||
        op == Op::RECIPROCAL || op == Op::ROUND) {
        if (batch_initializing_cutlass_)
            return;
        const Value& in_val = value(inputs.at(0));
        DeviceAddress input = in_val.address;
        DeviceAddress result = output.address;
        int operation = op == Op::LEAKY_RELU ? 0 : op == Op::CLIP ? 1 :
                        op == Op::SIGMOID ? 2 : op == Op::RECIPROCAL ? 3 : 4;
        float alpha = node.alpha_or_default();
        float minimum = -std::numeric_limits<float>::infinity();
        float maximum = std::numeric_limits<float>::infinity();
        if (op == Op::CLIP) {
            if (inputs.size() > 1 && !(inputs[1] == 0)) minimum = scalar_fp16(inputs[1]);
            if (inputs.size() > 2 && !(inputs[2] == 0)) maximum = scalar_fp16(inputs[2]);
        }
        if (use_nhwc_ && (!is_packed_nhwc(in_val) || !is_packed_nhwc(output))) {
            const auto od = dims4(output.shape);
            int pixels = od[0] * od[2] * od[3];
            int channels = od[1];
            int in_s = value_stride(in_val);
            int out_s = value_stride(output);
            const bool vec8 = ((channels & 7) == 0) && ((in_s & 7) == 0) &&
                ((out_s & 7) == 0) && ((input & 15U) == 0) &&
                ((result & 15U) == 0);
            const std::size_t work = static_cast<std::size_t>(pixels) *
                static_cast<std::size_t>(vec8 ? (channels / 8) : channels);
            void* up[] = {&input, &result, &pixels, &channels, &in_s, &out_s,
                          &operation, &alpha, &minimum, &maximum};
            launch_linear(unary_strided_nhwc_, work, up);
            return;
        }
        void* parameters[] = {
            &input, &result, &count, &operation, &alpha, &minimum, &maximum};
        launch_linear(unary_, count_size, parameters);
        return;
    }

    if (op == Op::CONV) {
        const Value& input_value = value(inputs.at(0));
        const Value& weight_value = value(inputs.at(1));
        const auto& attributes = node.conv();
        const auto strides = attributes.strides;
        const auto pads = attributes.pads;
        int groups = attributes.groups;
        const auto in = dims4(input_value.shape);
        const auto out = dims4(output.shape);
        const auto kernel = weight_value.shape;
        DeviceAddress input = input_value.address;
        DeviceAddress weight = weight_value.address;
        DeviceAddress bias = inputs.size() > 2 && !(inputs[2] == 0)
            ? value(inputs[2]).address : 0;
        DeviceAddress result = output.address;
        int n = in[0], ic = in[1], ih = in[2], iw = in[3];
        int oc = out[1], oh = out[2], ow = out[3];
        int kh = static_cast<int>(kernel.at(2));
        int kw = static_cast<int>(kernel.at(3));
        int sh = strides.at(0), sw = strides.at(1);
        int ph = pads.at(0), pw = pads.at(1);
        int spatial = oh * ow;
        const bool is_pointwise_1x1 =
            groups == 1 && kh == 1 && kw == 1 &&
            sh == 1 && sw == 1 && ph == 0 && pw == 0 &&
            ih == oh && iw == ow;
        if (is_pointwise_1x1) {
            int in_s = use_nhwc_ ? value_stride(input_value) : 0;
            int out_s = use_nhwc_ ? value_stride(output) : 0;
            int res_s = use_nhwc_
                ? (epilogue_stride > 0 ? epilogue_stride : oc) : 0;
            if (launch_cutlass_pointwise(
                    node, input, weight, bias, epilogue_input, result,
                    n, ic, spatial, oc, epilogue, epilogue_alpha,
                    in_s, res_s, out_s)) {
                return;
            }
            if (batch_initializing_cutlass_)
                return;
            if (use_nhwc_) {
                if (ic % 16 == 0 && oc % 16 == 0 && in_s % 8 == 0) {
                    void* mma_parameters[] = {
                        &input, &weight, &bias, &result, &n, &ic, &spatial,
                        &oc, &in_s, &res_s, &out_s, &epilogue_input, &epilogue,
                        &epilogue_alpha};
                    driver_.launch(
                        pointwise_conv_mma_tiny_nhwc_,
                        {divide_up(spatial, 16), divide_up(oc, 16),
                         static_cast<unsigned int>(n)},
                        {32, 4, 1}, 0, mma_parameters);
                    return;
                }
                throw std::runtime_error(
                    "driver-cubin: unsupported NHWC 1x1 conv shape");
            }
            if (driver_.device_info().compute_major >= 8 && spatial < 920 &&
                ic % 16 == 0 && oc % 16 == 0) {
                void* mma_parameters[] = {
                    &input, &weight, &bias, &result, &n, &ic, &spatial,
                    &oc, &epilogue_input, &epilogue, &epilogue_alpha};
                driver_.launch(
                    pointwise_convolution_mma_tiny_,
                    {divide_up(spatial, 16), divide_up(oc, 16),
                     static_cast<unsigned int>(n)},
                    {32, 4, 1}, 0, mma_parameters);
                return;
            }
            if (ic % 16 == 0 && oc % 64 == 0) {
                if (oc % 128 == 0) {
                    const bool use_mma = driver_.device_info().compute_major >= 8;
                    if (use_mma) {
                        void* mma_parameters[] = {
                            &input, &weight, &bias, &result, &n, &ic, &spatial,
                            &oc, &epilogue_input, &epilogue, &epilogue_alpha};
                        const unsigned int large_tile_blocks =
                            divide_up(spatial, 64) * divide_up(oc, 128) * n;
                        const bool use_small_tile = large_tile_blocks <
                            static_cast<unsigned int>(
                                driver_.device_info().multiprocessor_count * 2);
                        driver_.launch(
                            use_small_tile ? pointwise_convolution_mma_small_
                                           : pointwise_convolution_mma_,
                            {divide_up(spatial, 64),
                             divide_up(oc, use_small_tile ? 64 : 128),
                             static_cast<unsigned int>(n)},
                            {32, 8, 1}, 0, mma_parameters);
                    } else {
                        if (epilogue != 0) {
                            throw std::runtime_error(
                                "driver-cubin: epilogue fusion requires Ampere+ kernel");
                        }
                        void* pointwise_parameters[] = {
                            &input, &weight, &bias, &result, &n, &ic, &spatial,
                            &oc};
                        driver_.launch(pointwise_convolution_balanced_,
                            {divide_up(spatial, 64), divide_up(oc, 128),
                             static_cast<unsigned int>(n)},
                            {32, 8, 1}, 0, pointwise_parameters);
                    }
                } else {
                    if (epilogue != 0) {
                        throw std::runtime_error(
                            "driver-cubin: epilogue fusion requires Ampere+ kernel");
                    }
                    void* pointwise_parameters[] = {
                        &input, &weight, &bias, &result, &n, &ic, &spatial, &oc};
                    driver_.launch(pointwise_convolution_,
                        {divide_up(spatial, 16), divide_up(oc, 64),
                         static_cast<unsigned int>(n)},
                        {32, 4, 1}, 0, pointwise_parameters);
                }
                return;
            }
            if (epilogue != 0) {
                throw std::runtime_error(
                    "driver-cubin: pointwise epilogue failed to dispatch");
            }
        }
        if (groups == 1 && kh > 1 && kw > 1) {
            if ((!use_nhwc_ ||
                 (is_packed_nhwc(input_value) && is_packed_nhwc(output))) &&
                launch_cutlass_spatial(
                    node, input, weight, bias, result, n, ic, ih, iw, oc,
                    oh, ow, kh, kw, sh, sw, ph, pw)) {
                return;
            }
            if (use_nhwc_) {
                throw std::runtime_error(
                    "driver-cubin: unsupported NHWC spatial conv shape");
            }
        }
        const bool use_depthwise = groups == ic && oc == ic &&
                                   weight_value.shape.at(1) == 1;
        if (use_nhwc_ && use_depthwise) {
            const bool is_3x3_same = kh == 3 && kw == 3 &&
                sh == 1 && sw == 1 && ph == 1 && pw == 1 &&
                ih == oh && iw == ow;
            int in_s = value_stride(input_value);
            int out_s = value_stride(output);
            if (is_3x3_same && (ic % 4) == 0 && (in_s % 4) == 0 &&
                (out_s % 4) == 0) {
                DeviceAddress w_k9c =
                    get_or_pack_dwconv_weight(node.index, weight, ic);
                if (batch_initializing_cutlass_)
                    return;
                const unsigned int cg = static_cast<unsigned int>(ic / 4);
                const bool compact_map = (oh * ow <= 240);
                unsigned int chunk_threads = 32U;
                unsigned int tpb = 1U;
                if (compact_map && (cg % 16U) == 0U && ow <= 10) {
                    // Split channels into 64-ch (128B cache-line) chunks on tiny
                    // 6x10 / 3x5 maps to double SM parallelism.
                    chunk_threads = 16U;
                    tpb = 1U;
                } else if ((cg % 32U) == 0U) {
                    chunk_threads = 32U;
                    tpb = 1U;
                } else if ((cg % 16U) == 0U) {
                    chunk_threads = 16U;
                    tpb = 2U;
                } else {
                    chunk_threads = std::min(cg, 32U);
                    tpb = 1U;
                }
                int c_chunks = static_cast<int>(divide_up(cg, chunk_threads));
                void* dw_params[] = {
                    &input, &w_k9c, &bias, &result, &n, &ic, &oh, &ow,
                    &in_s, &out_s, &c_chunks};
                const unsigned int grid_z =
                    static_cast<unsigned int>(n * c_chunks);
                const unsigned int rows_y =
                    divide_up(static_cast<std::size_t>(oh), 2);
                if (compact_map) {
                    const unsigned int strips_x =
                        divide_up(static_cast<std::size_t>(ow), 2);
                    driver_.launch(
                        depthwise_conv_3x3_nhwc_h2_u2_,
                        {divide_up(strips_x, tpb), rows_y, grid_z},
                        {chunk_threads, tpb, 1}, 0, dw_params);
                } else {
                    const unsigned int strips_x =
                        divide_up(static_cast<std::size_t>(ow), 4);
                    driver_.launch(
                        depthwise_conv_3x3_nhwc_h4_u2_,
                        {divide_up(strips_x, tpb), rows_y, grid_z},
                        {chunk_threads, tpb, 1}, 0, dw_params);
                }
                return;
            }
            throw std::runtime_error(
                "driver-cubin: unsupported NHWC depthwise conv shape");
        }
        if (batch_initializing_cutlass_)
            return;
        if (use_nhwc_) {
            throw std::runtime_error(
                "driver-cubin: unsupported NHWC conv operator");
        }
        const int reduction = ic * kh * kw;
        const bool use_spatial = groups == 1 && reduction % 16 == 0 &&
                                 oc % 64 == 0;
        if (use_spatial) {
            void* spatial_parameters[] = {
                &input, &weight, &bias, &result, &n, &ic, &ih, &iw,
                &oc, &oh, &ow, &kh, &kw, &sh, &sw, &ph, &pw};
            const unsigned int mma_blocks =
                divide_up(spatial, 16) * divide_up(oc, 128) * n;
            const bool use_mma =
                driver_.device_info().compute_major >= 8 && mma_blocks >= 1;
            const bool use_wide_mma = use_mma && oc >= 512;
            driver_.launch(
                use_wide_mma ? spatial_convolution_mma_wide_
                             : use_mma ? spatial_convolution_mma_
                                       : spatial_convolution_,
                {divide_up(spatial, use_wide_mma ? 32 : 16),
                 divide_up(oc, use_mma ? 128 : 64),
                 static_cast<unsigned int>(n)},
                {32, use_mma ? 8U : 4U, 1}, 0, spatial_parameters);
            return;
        }
        if (use_depthwise) {
            const bool is_3x3_same = kh == 3 && kw == 3 &&
                sh == 1 && sw == 1 && ph == 1 && pw == 1 &&
                ih == oh && iw == ow;
            if (is_3x3_same && ow % 8 == 0) {
                void* oct_parameters[] = {
                    &input, &weight, &bias, &result, &n, &ic,
                    &ih, &iw, &oh, &ow};
                const std::size_t total_pairs =
                    static_cast<std::size_t>((oh + 1) / 2) * (ow / 8);
                const unsigned int block = total_pairs <= 64 ? 64U : 128U;
                driver_.launch(
                    depthwise_convolution_oct2_,
                    {divide_up(total_pairs, block),
                     static_cast<unsigned int>(oc),
                     static_cast<unsigned int>(n)},
                    {block, 1, 1}, 0, oct_parameters);
                return;
            }
            if (is_3x3_same && oh <= 16 && ow <= 32) {
                void* small_parameters[] = {
                    &input, &weight, &bias, &result, &n, &ic, &oh, &ow};
                driver_.launch(
                    depthwise_convolution_3x3_small_,
                    {divide_up(static_cast<std::size_t>(oc), 2),
                     static_cast<unsigned int>(n), 1},
                    {32, 2, 1}, 0, small_parameters);
                return;
            }
            void* depthwise_parameters[] = {
                &input, &weight, &bias, &result, &n, &ic, &ih, &iw,
                &oh, &ow, &kh, &kw, &sh, &sw, &ph, &pw};
            const bool use_quads = ow % 4 == 0 && kh == 3 && kw == 3 &&
                sh == 1 && sw == 1 && ph == 1 && pw == 1;
            const bool use_pairs = !use_quads && ow % 2 == 0 && sw == 1;
            const int outputs_per_thread = use_quads ? 4 : use_pairs ? 2 : 1;
            const std::size_t depthwise_count =
                static_cast<std::size_t>(n) * oc * oh *
                (ow / outputs_per_thread);
            driver_.launch(use_quads ? depthwise_convolution_quad_
                                     : use_pairs ? depthwise_convolution_pair_
                                                 : depthwise_convolution_,
                {divide_up(depthwise_count, 256), 1, 1}, {256, 1, 1},
                0, depthwise_parameters);
            return;
        }
        void* parameters[] = {
            &input, &weight, &bias, &result, &n, &ic, &ih, &iw, &oc, &oh, &ow,
            &kh, &kw, &sh, &sw, &ph, &pw, &groups};
        driver_.launch(convolution_,
            {divide_up(static_cast<std::size_t>(n) * oh * ow, 32),
             divide_up(oc, 4), 1},
            {32, 4, 1}, 0, parameters);
        return;
    }

    if (batch_initializing_cutlass_)
        return;

    if (op == Op::GATHER) {
        const Value& data = value(inputs.at(0));
        if (node.axis_or_zero() != 0)
            throw std::runtime_error("driver-cubin: only Gather axis 0 is supported");
        DeviceAddress input = data.address;
        DeviceAddress index = value(inputs.at(1)).address;
        DeviceAddress result = output.address;
        int rows = static_cast<int>(data.shape.at(0));
        int row_elements = static_cast<int>(element_count(data.shape) / rows);
        void* parameters[] = {&input, &index, &result, &row_elements, &rows};
        launch_linear(gather_, row_elements, parameters);
        return;
    }

    if (op == Op::SLICE) {
        const std::size_t node_index = node.index;
        if (std::find(aliased_input_slices_.begin(),
                      aliased_input_slices_.end(), node_index) !=
            aliased_input_slices_.end()) {
            return;
        }
        const Value& input_value = value(inputs.at(0));
        const auto starts_values = initializer_values<std::int64_t>(inputs.at(1));
        const auto axes_values = inputs.size() > 3 && !(inputs[3] == 0)
            ? initializer_values<std::int64_t>(inputs[3])
            : std::vector<std::int64_t>{};
        const auto steps_values = inputs.size() > 4 && !(inputs[4] == 0)
            ? initializer_values<std::int64_t>(inputs[4])
            : std::vector<std::int64_t>{};
        auto id = dims4(input_value.shape);
        const auto od = dims4(output.shape);
        std::array<int, 4> starts{0, 0, 0, 0};
        std::array<int, 4> steps{1, 1, 1, 1};
        for (std::size_t i = 0; i < starts_values.size(); ++i) {
            int axis = axes_values.empty() ? static_cast<int>(i) :
                       static_cast<int>(axes_values.at(i));
            if (axis < 0) axis += static_cast<int>(input_value.shape.size());
            axis += 4 - static_cast<int>(input_value.shape.size());
            int start = static_cast<int>(starts_values[i]);
            if (start < 0) start += id.at(static_cast<std::size_t>(axis));
            starts.at(static_cast<std::size_t>(axis)) = start;
            if (!steps_values.empty())
                steps.at(static_cast<std::size_t>(axis)) =
                    static_cast<int>(steps_values.at(i));
        }
        DeviceAddress input = input_value.address;
        DeviceAddress result = output.address;
        if (use_nhwc_) {
            int in_s = value_stride(input_value);
            int out_s = value_stride(output);
            const bool contiguous_h_slice =
                od[0] == 1 && id[0] == 1 &&
                starts[0] == 0 && starts[1] == 0 && starts[3] == 0 &&
                steps[0] == 1 && steps[1] == 1 && steps[2] == 1 && steps[3] == 1 &&
                od[1] == id[1] && od[3] == id[3] &&
                is_packed_nhwc(input_value) && is_packed_nhwc(output) &&
                output.dtype == DataType::FP16;
            if (contiguous_h_slice && count_size > 0) {
                DeviceAddress src = input +
                    static_cast<std::size_t>(starts[2]) * id[3] * id[1] *
                    sizeof(Float16Storage);
                void* cp[] = {&src, &result, &count};
                launch_linear(copy_, divide_up(count_size, 8), cp);
                return;
            }
            const bool contiguous_c_slice =
                od[0] == 1 && id[0] == 1 &&
                starts[0] == 0 && starts[2] == 0 && starts[3] == 0 &&
                steps[0] == 1 && steps[1] == 1 && steps[2] == 1 && steps[3] == 1 &&
                od[2] == id[2] && od[3] == id[3] &&
                output.dtype == DataType::FP16;
            if (contiguous_c_slice && count_size > 0) {
                DeviceAddress src = input +
                    static_cast<std::size_t>(starts[1]) * sizeof(Float16Storage);
                int pixels = od[2] * od[3];
                int ch = od[1];
                const bool vec8 = ((ch & 7) == 0) && ((in_s & 7) == 0) &&
                    ((out_s & 7) == 0) && ((src & 15U) == 0) &&
                    ((result & 15U) == 0);
                void* cp[] = {&src, &result, &pixels, &ch, &in_s, &out_s};
                driver_.launch(
                    copy_2d_nhwc_,
                    {divide_up(static_cast<std::size_t>(vec8 ? (ch / 8) : ch), 32),
                     divide_up(static_cast<std::size_t>(pixels), 8), 1},
                    {32, 8, 1}, 0, cp);
                return;
            }
            void* nh_params[] = {
                &input, &result, &count,
                const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
                const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
                &id[0], &id[1], &id[2], &id[3],
                &starts[0], &starts[1], &starts[2], &starts[3],
                &steps[0], &steps[1], &steps[2], &steps[3],
                &in_s, &out_s};
            launch_linear(slice_nhwc_, count_size, nh_params);
            return;
        }
        const bool contiguous_channel_slice =
            od[0] == 1 && id[0] == 1 &&
            starts[0] == 0 && starts[2] == 0 && starts[3] == 0 &&
            steps[0] == 1 && steps[1] == 1 && steps[2] == 1 && steps[3] == 1 &&
            od[2] == id[2] && od[3] == id[3] &&
            output.dtype == DataType::FP16;
        if (contiguous_channel_slice && count_size > 0) {
            DeviceAddress src = input +
                static_cast<std::size_t>(starts[1]) *
                static_cast<std::size_t>(id[2]) * id[3] *
                sizeof(Float16Storage);
            void* copy_parameters[] = {&src, &result, &count};
            launch_linear(copy_, divide_up(count_size, 8), copy_parameters);
            return;
        }
        void* parameters[] = {
            &input, &result, &count,
            const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
            const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
            &id[0], &id[1], &id[2], &id[3],
            &starts[0], &starts[1], &starts[2], &starts[3],
            &steps[0], &steps[1], &steps[2], &steps[3],
        };
        launch_linear(slice_, count_size, parameters);
        return;
    }

    if (op == Op::CONCAT) {
        int axis = node.axis();
        if (axis < 0) axis += static_cast<int>(output.shape.size());
        axis += 4 - static_cast<int>(output.shape.size());
        const auto od = dims4(output.shape);
        int axis_offset = 0;
        for (const TensorId input_name : inputs) {
            const Value& input_value = value(input_name);
            const auto id = dims4(input_value.shape);
            const std::size_t input_count = element_count(input_value.shape);
            int input_count_parameter = static_cast<int>(input_count);
            DeviceAddress input = input_value.address;
            DeviceAddress result = output.address;
            if (use_nhwc_) {
                int in_s = value_stride(input_value);
                int out_s = value_stride(output);
                const DeviceAddress direct_c_addr = result +
                    static_cast<std::size_t>(axis_offset) *
                    dtype_bytes(output.dtype);
                if (axis == 1 && od[0] == 1 && input == direct_c_addr &&
                    in_s == out_s) {
                    // Already written directly into the destination NHWC slice.
                } else if (axis == 1 && od[0] == 1 && id[0] == 1 &&
                           id[2] == od[2] && id[3] == od[3] &&
                           output.dtype == DataType::FP16 && input_count > 0) {
                    DeviceAddress dst = direct_c_addr;
                    int pixels = od[2] * od[3];
                    int ch = id[1];
                    const bool vec8 = ((ch & 7) == 0) && ((in_s & 7) == 0) &&
                        ((out_s & 7) == 0) && ((input & 15U) == 0) &&
                        ((dst & 15U) == 0);
                    void* cp[] = {&input, &dst, &pixels, &ch, &in_s, &out_s};
                    driver_.launch(
                        copy_2d_nhwc_,
                        {divide_up(
                             static_cast<std::size_t>(vec8 ? (ch / 8) : ch), 32),
                         divide_up(static_cast<std::size_t>(pixels), 8), 1},
                        {32, 8, 1}, 0, cp);
                } else if (axis == 2 && od[0] == 1 && id[0] == 1 &&
                           id[1] == od[1] && id[3] == od[3] &&
                           is_packed_nhwc(input_value) && is_packed_nhwc(output) &&
                           output.dtype == DataType::FP16 && input_count > 0) {
                    DeviceAddress dst = result +
                        static_cast<std::size_t>(axis_offset) * od[3] * od[1] *
                        sizeof(Float16Storage);
                    if (input != dst) {
                        void* cp[] = {&input, &dst, &input_count_parameter};
                        launch_linear(copy_, divide_up(input_count, 8), cp);
                    }
                } else {
                    void* nh_params[] = {
                        &input, &result, &input_count_parameter,
                        const_cast<int*>(&id[0]), const_cast<int*>(&id[1]),
                        const_cast<int*>(&id[2]), const_cast<int*>(&id[3]),
                        const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
                        const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
                        &axis, &axis_offset, &in_s, &out_s};
                    launch_linear(concat_copy_nhwc_, input_count, nh_params);
                }
                axis_offset += id.at(static_cast<std::size_t>(axis));
                continue;
            }
            const std::size_t channel_bytes =
                static_cast<std::size_t>(od[2]) * od[3] *
                dtype_bytes(output.dtype);
            const DeviceAddress direct_address = result +
                static_cast<std::size_t>(axis_offset) * channel_bytes;
            if (axis == 1 && od[0] == 1 && input == direct_address) {
                // Already written directly into the destination slice.
            } else if (axis == 1 && od[0] == 1 && id[0] == 1 &&
                       id[2] == od[2] && id[3] == od[3] &&
                       output.dtype == DataType::FP16 && input_count > 0) {
                DeviceAddress dst = direct_address;
                void* copy_parameters[] = {
                    &input, &dst, &input_count_parameter};
                launch_linear(copy_, divide_up(input_count, 8), copy_parameters);
            } else {
                void* parameters[] = {
                    &input, &result, &input_count_parameter,
                    const_cast<int*>(&id[0]), const_cast<int*>(&id[1]),
                    const_cast<int*>(&id[2]), const_cast<int*>(&id[3]),
                    const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
                    const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
                    &axis, &axis_offset,
                };
                launch_linear(concat_, input_count, parameters);
            }
            axis_offset += id.at(static_cast<std::size_t>(axis));
        }
        return;
    }

    if (op == Op::DEPTH_TO_SPACE) {
        const Value& input_value = value(inputs.at(0));
        const auto in = dims4(input_value.shape);
        DeviceAddress input = input_value.address;
        DeviceAddress result = output.address;
        int batch = in[0], channels = in[1], height = in[2], width = in[3];
        int block = node.block_size();
        int mode = 1;
        if (use_nhwc_) {
            int in_s = value_stride(input_value);
            int out_s = value_stride(output);
            void* nh_params[] = {
                &input, &result, &batch, &channels, &height, &width,
                &block, &mode, &in_s, &out_s};
            driver_.launch(
                depth_to_space_nhwc_,
                {static_cast<unsigned int>(width),
                 static_cast<unsigned int>(height),
                 static_cast<unsigned int>(batch)},
                {128, 1, 1}, 0, nh_params);
            return;
        }
        const int out_channels = channels / (block * block);
        const int out_height = height * block;
        const int out_width = width * block;
        void* parameters[] = {
            &input, &result, &batch, &channels, &height, &width, &block, &mode};
        driver_.launch(
            depth_to_space_3d_,
            {divide_up(static_cast<std::size_t>(out_width), 32),
             divide_up(static_cast<std::size_t>(out_height), 8),
             static_cast<unsigned int>(batch * out_channels)},
            {32, 8, 1}, 0, parameters);
        return;
    }

    if (op == Op::SPACE_TO_DEPTH) {
        const Value& input_value = value(inputs.at(0));
        const auto in = dims4(input_value.shape);
        DeviceAddress input = input_value.address;
        DeviceAddress result = output.address;
        int batch = in[0], channels = in[1], height = in[2], width = in[3];
        int block = node.block_size();
        const int out_channels = channels * block * block;
        const int out_height = height / block;
        const int out_width = width / block;
        if (use_nhwc_) {
            int in_s = value_stride(input_value);
            int out_s = value_stride(output);
            void* nh_params[] = {
                &input, &result, &batch, &channels, &height, &width,
                &block, &in_s, &out_s};
            driver_.launch(
                space_to_depth_nhwc_,
                {static_cast<unsigned int>(out_width),
                 static_cast<unsigned int>(out_height),
                 static_cast<unsigned int>(batch)},
                {128, 1, 1}, 0, nh_params);
            return;
        }
        void* parameters[] = {
            &input, &result, &batch, &channels, &height, &width, &block};
        driver_.launch(
            space_to_depth_3d_,
            {divide_up(static_cast<std::size_t>(out_width), 32),
             divide_up(static_cast<std::size_t>(out_height), 8),
             static_cast<unsigned int>(batch * out_channels)},
            {32, 8, 1}, 0, parameters);
        return;
    }

    throw std::runtime_error("driver-cubin: unsupported operator " + std::to_string(static_cast<int>(op)));
}

}  // namespace mlvc::driver_cubin_backend
