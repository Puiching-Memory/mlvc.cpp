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
    const auto& nodes = manifest_.at("nodes");
    for (std::size_t index = 0; index < nodes.size();) {
        if (try_execute_y0_tail(nodes, index))
            index += 19;
        else if (try_execute_y1_tail(nodes, index))
            index += 15;
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
    const json& node, DeviceAddress input, DeviceAddress weight,
    DeviceAddress bias, DeviceAddress output, int batch_count,
    int in_channels, int input_height, int input_width, int out_channels,
    int output_height, int output_width, int kernel_height,
    int kernel_width, int stride_height, int stride_width,
    int pad_height, int pad_width)
{
    if (driver_.device_info().compute_major < 8 || batch_count != 1 ||
        bias == 0 || !spatial_input_buffer_ || !spatial_output_buffer_ ||
        in_channels % 8 != 0 || out_channels % 8 != 0 ||
        kernel_height <= 1 || kernel_width <= 1) {
        return false;
    }

    const std::size_t node_index = node.at("index").get<std::size_t>();
    auto [params, params_inserted] =
        cutlass_spatial_parameter_buffers_.try_emplace(node_index);
    auto [transformed_weight, weight_inserted] =
        cutlass_spatial_weight_buffers_.try_emplace(node_index);
    if (params_inserted != weight_inserted)
        throw std::runtime_error("driver-cubin: inconsistent spatial CUTLASS state");

    DeviceAddress input_nhwc = spatial_input_buffer_.address();
    DeviceAddress output_nhwc = spatial_output_buffer_.address();
    if (params_inserted) {
        const std::size_t weight_elements =
            static_cast<std::size_t>(out_channels) * in_channels *
            kernel_height * kernel_width;
        params->second = driver_.allocate(
            driver_cubin::kCutlassPointwiseParamsStorageBytes);
        transformed_weight->second =
            driver_.allocate(weight_elements * sizeof(Float16Storage));
        DeviceAddress weight_krsc = transformed_weight->second.address();
        void* transform_parameters[] = {
            &weight, &weight_krsc, &out_channels, &in_channels,
            &kernel_height, &kernel_width};
        driver_.launch(
            oihw_to_krsc_, {divide_up(weight_elements, 256), 1, 1},
            {256, 1, 1}, 0, transform_parameters);

        DeviceAddress params_storage = params->second.address();
        void* init_parameters[] = {
            &params_storage, &input_nhwc, &weight_krsc, &bias, &output_nhwc,
            &batch_count, &in_channels, &input_height, &input_width,
            &out_channels, &output_height, &output_width, &kernel_height,
            &kernel_width, &stride_height, &stride_width, &pad_height,
            &pad_width};
        driver_.launch(
            cutlass_spatial_convolution_init_, {1, 1, 1}, {1, 1, 1}, 0,
            init_parameters);
        auto [host_params, host_params_inserted] =
            cutlass_spatial_host_parameters_.try_emplace(node_index);
        if (!host_params_inserted)
            throw std::runtime_error(
                "driver-cubin: duplicate spatial CUTLASS host state");
        driver_.download(
            host_params->second.data(), params->second.address(),
            host_params->second.size());
    }

    int input_rows = in_channels;
    int input_columns = input_height * input_width;
    void* input_transform_parameters[] = {
        &input, &input_nhwc, &input_rows, &input_columns};
    driver_.launch(
        transpose_, {divide_up(input_columns, 32), divide_up(input_rows, 32), 1},
        {32, 8, 1}, 0, input_transform_parameters);

    auto host_params = cutlass_spatial_host_parameters_.find(node_index);
    if (host_params == cutlass_spatial_host_parameters_.end())
        throw std::runtime_error(
            "driver-cubin: missing spatial CUTLASS host parameters");
    void* convolution_parameters[] = {host_params->second.data()};
    const int output_spatial = output_height * output_width;
    driver_.launch(
        cutlass_spatial_convolution_,
        {divide_up(output_spatial, 128), divide_up(out_channels, 256), 1},
        {256, 1, 1}, 73728U, convolution_parameters);

    int output_rows = output_spatial;
    int output_columns = out_channels;
    void* output_transform_parameters[] = {
        &output_nhwc, &output, &output_rows, &output_columns};
    driver_.launch(
        transpose_,
        {divide_up(output_columns, 32), divide_up(output_rows, 32), 1},
        {32, 8, 1}, 0, output_transform_parameters);
    return true;
}

bool AotGraph::launch_cutlass_pointwise(
    const json& node, DeviceAddress input, DeviceAddress weight,
    DeviceAddress bias, DeviceAddress residual, DeviceAddress output,
    int batch_count, int in_channels, int spatial_count, int out_channels,
    int epilogue, float epilogue_alpha)
{
    if (driver_.device_info().compute_major < 8 || batch_count != 1 ||
        bias == 0 ||
        spatial_count < 920 || (spatial_count % 8) != 0 ||
        (in_channels % 8) != 0 || (out_channels % 8) != 0 ||
        epilogue < 0 || epilogue > 2 ||
        (epilogue == 1 &&
         std::abs(epilogue_alpha - 0.01F) > 1.0e-7F) ||
        (epilogue == 2 && residual == 0)) {
        return false;
    }

    const bool use_medium_tile = in_channels <= 384 &&
        ((spatial_count >= 3000 && out_channels >= 256) ||
         (spatial_count < 3000 && out_channels >= 384 &&
          in_channels >= 256));
    const bool use_spatial_wide_tile =
        epilogue == 2 && spatial_count >= 3000 && in_channels >= 512;
    const bool use_medium_stage4_tile =
        epilogue == 0 && use_medium_tile && spatial_count >= 3000;
    driver_cubin::abi::Function init = use_medium_stage4_tile
        ? cutlass_pointwise_medium_stage4_init_
        : use_medium_tile ? cutlass_pointwise_medium_init_
                          : cutlass_pointwise_init_;
    driver_cubin::abi::Function function = use_medium_stage4_tile
        ? cutlass_pointwise_medium_stage4_
        : use_medium_tile ? cutlass_pointwise_medium_
                          : cutlass_pointwise_;
    if (epilogue == 1) {
        init = use_medium_tile
            ? cutlass_pointwise_medium_leaky_relu_init_
            : cutlass_pointwise_leaky_relu_init_;
        function = use_medium_tile
            ? cutlass_pointwise_medium_leaky_relu_
            : cutlass_pointwise_leaky_relu_;
    } else if (epilogue == 2) {
        if (use_spatial_wide_tile) {
            init = cutlass_pointwise_spatial_wide_residual_init_;
            function = cutlass_pointwise_spatial_wide_residual_;
        } else {
            init = use_medium_tile
                ? cutlass_pointwise_medium_residual_init_
                : cutlass_pointwise_residual_init_;
            function = use_medium_tile
                ? cutlass_pointwise_medium_residual_
                : cutlass_pointwise_residual_;
        }
    }

    const std::size_t node_index = node.at("index").get<std::size_t>();
    auto [found, inserted] = cutlass_parameter_buffers_.try_emplace(node_index);
    if (inserted) {
        found->second = driver_.allocate(
            driver_cubin::kCutlassPointwiseParamsStorageBytes);
        DeviceAddress params_storage = found->second.address();
        void* init_parameters[] = {
            &params_storage, &input, &weight, &bias, &residual, &output,
            &out_channels, &spatial_count, &in_channels};
        driver_.launch(init, {1, 1, 1}, {1, 1, 1}, 0, init_parameters);
        auto [host_params, host_params_inserted] =
            cutlass_host_parameters_.try_emplace(node_index);
        if (!host_params_inserted)
            throw std::runtime_error(
                "driver-cubin: duplicate CUTLASS host state");
        driver_.download(
            host_params->second.data(), found->second.address(),
            host_params->second.size());
    }

    auto host_params = cutlass_host_parameters_.find(node_index);
    if (host_params == cutlass_host_parameters_.end())
        throw std::runtime_error(
            "driver-cubin: missing CUTLASS host parameters");
    void* parameters[] = {host_params->second.data()};
    driver_.launch(
        function,
        {divide_up(out_channels, use_spatial_wide_tile ? 64 : 128),
         divide_up(spatial_count,
                   use_spatial_wide_tile ? 256 : use_medium_tile ? 128 : 64),
         1},
        {128, 1, 1},
        use_spatial_wide_tile ? 61440U
            : use_medium_stage4_tile ? 65536U
            : use_medium_tile ? 49152U : 36864U,
        parameters);
    return true;
}

void AotGraph::launch_linear(driver_cubin::abi::Function function, std::size_t count,
                   std::span<void*> parameters)
{
    constexpr unsigned int block = 256;
    driver_.launch(function, {divide_up(count, block), 1, 1}, {block, 1, 1},
                   0, parameters);
}

void AotGraph::execute(const json& node, const Value* output_override,
             DeviceAddress epilogue_input, int epilogue,
             float epilogue_alpha)
{
    const std::string op = node.at("op").get<std::string>();
    const auto inputs = node.at("inputs").get<std::vector<std::string>>();
    const auto outputs = node.at("outputs").get<std::vector<std::string>>();
    if (outputs.size() != 1)
        throw std::runtime_error("driver-cubin: only single-output nodes are supported");
    const Value& output = output_override ? *output_override : value(outputs[0]);
    const std::size_t count_size = element_count(output.shape);
    if (count_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("driver-cubin: tensor is too large");
    int count = static_cast<int>(count_size);

    if (op == "Add" || op == "Mul" || op == "Sub") {
        DeviceAddress lhs = value(inputs.at(0)).address;
        DeviceAddress rhs = value(inputs.at(1)).address;
        DeviceAddress result = output.address;
        int operation = op == "Add" ? 0 : (op == "Mul" ? 1 : 2);
        const auto od = dims4(output.shape);
        const auto ld = dims4(value(inputs[0]).shape);
        const auto rd = dims4(value(inputs[1]).shape);
        void* parameters[] = {
            &lhs, &rhs, &result, &count, &operation,
            const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
            const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
            const_cast<int*>(&ld[0]), const_cast<int*>(&ld[1]),
            const_cast<int*>(&ld[2]), const_cast<int*>(&ld[3]),
            const_cast<int*>(&rd[0]), const_cast<int*>(&rd[1]),
            const_cast<int*>(&rd[2]), const_cast<int*>(&rd[3]),
        };
        if (value(inputs[0]).shape == output.shape &&
            value(inputs[1]).shape == output.shape) {
            void* contiguous_parameters[] = {
                &lhs, &rhs, &result, &count, &operation};
            launch_linear(binary_contiguous_, (count_size + 1) / 2,
                          contiguous_parameters);
        } else {
            launch_linear(binary_, count_size, parameters);
        }
        return;
    }

    if (op == "LeakyRelu" || op == "Clip" || op == "Sigmoid" ||
        op == "Reciprocal" || op == "Round") {
        DeviceAddress input = value(inputs.at(0)).address;
        DeviceAddress result = output.address;
        int operation = op == "LeakyRelu" ? 0 : op == "Clip" ? 1 :
                        op == "Sigmoid" ? 2 : op == "Reciprocal" ? 3 : 4;
        float alpha = node.at("attributes").value("alpha", 0.01F);
        float minimum = -std::numeric_limits<float>::infinity();
        float maximum = std::numeric_limits<float>::infinity();
        if (op == "Clip") {
            if (inputs.size() > 1 && !inputs[1].empty()) minimum = scalar_fp16(inputs[1]);
            if (inputs.size() > 2 && !inputs[2].empty()) maximum = scalar_fp16(inputs[2]);
        }
        void* parameters[] = {
            &input, &result, &count, &operation, &alpha, &minimum, &maximum};
        launch_linear(unary_, count_size, parameters);
        return;
    }

    if (op == "Conv") {
        const Value& input_value = value(inputs.at(0));
        const Value& weight_value = value(inputs.at(1));
        const auto& attributes = node.at("attributes");
        const auto strides = attributes.at("strides").get<std::vector<int>>();
        const auto pads = attributes.at("pads").get<std::vector<int>>();
        int groups = attributes.value("group", 1);
        const auto in = dims4(input_value.shape);
        const auto out = dims4(output.shape);
        const auto kernel = weight_value.shape;
        DeviceAddress input = input_value.address;
        DeviceAddress weight = weight_value.address;
        DeviceAddress bias = inputs.size() > 2 && !inputs[2].empty()
            ? value(inputs[2]).address : 0;
        DeviceAddress result = output.address;
        int n = in[0], ic = in[1], ih = in[2], iw = in[3];
        int oc = out[1], oh = out[2], ow = out[3];
        int kh = static_cast<int>(kernel.at(2));
        int kw = static_cast<int>(kernel.at(3));
        int sh = strides.at(0), sw = strides.at(1);
        int ph = pads.at(0), pw = pads.at(1);
        int spatial = oh * ow;
        const bool use_pointwise =
            groups == 1 && kh == 1 && kw == 1 &&
            sh == 1 && sw == 1 && ph == 0 && pw == 0 &&
            ih == oh && iw == ow && ic % 16 == 0 && oc % 64 == 0;
        if (use_pointwise) {
            if (launch_cutlass_pointwise(
                    node, input, weight, bias, epilogue_input, result,
                    n, ic, spatial, oc, epilogue, epilogue_alpha)) {
                return;
            }
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
                    void* pointwise_parameters[] = {
                        &input, &weight, &bias, &result, &n, &ic, &spatial,
                        &oc};
                    driver_.launch(pointwise_convolution_balanced_,
                        {divide_up(spatial, 64), divide_up(oc, 128),
                         static_cast<unsigned int>(n)},
                        {32, 8, 1}, 0, pointwise_parameters);
                }
            } else {
                void* pointwise_parameters[] = {
                    &input, &weight, &bias, &result, &n, &ic, &spatial, &oc};
                driver_.launch(pointwise_convolution_,
                    {divide_up(spatial, 16), divide_up(oc, 64),
                     static_cast<unsigned int>(n)},
                    {32, 4, 1}, 0, pointwise_parameters);
            }
            return;
        }
        const int reduction = ic * kh * kw;
        const bool use_spatial = groups == 1 && reduction % 16 == 0 &&
                                 oc % 64 == 0;
        if (use_spatial) {
            if (launch_cutlass_spatial(
                    node, input, weight, bias, result, n, ic, ih, iw, oc,
                    oh, ow, kh, kw, sh, sw, ph, pw)) {
                return;
            }
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
        const bool use_depthwise = groups == ic && oc == ic &&
                                   weight_value.shape.at(1) == 1;
        if (use_depthwise) {
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

    if (op == "Gather") {
        const Value& data = value(inputs.at(0));
        if (node.at("attributes").value("axis", 0) != 0)
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

    if (op == "Slice") {
        const std::size_t node_index = node.at("index").get<std::size_t>();
        if (std::find(aliased_input_slices_.begin(),
                      aliased_input_slices_.end(), node_index) !=
            aliased_input_slices_.end()) {
            return;
        }
        const Value& input_value = value(inputs.at(0));
        const auto starts_values = initializer_values<std::int64_t>(inputs.at(1));
        const auto axes_values = inputs.size() > 3 && !inputs[3].empty()
            ? initializer_values<std::int64_t>(inputs[3])
            : std::vector<std::int64_t>{};
        const auto steps_values = inputs.size() > 4 && !inputs[4].empty()
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

    if (op == "Concat") {
        int axis = node.at("attributes").at("axis").get<int>();
        if (axis < 0) axis += static_cast<int>(output.shape.size());
        axis += 4 - static_cast<int>(output.shape.size());
        const auto od = dims4(output.shape);
        int axis_offset = 0;
        for (const std::string& input_name : inputs) {
            const Value& input_value = value(input_name);
            const auto id = dims4(input_value.shape);
            const std::size_t input_count = element_count(input_value.shape);
            int input_count_parameter = static_cast<int>(input_count);
            DeviceAddress input = input_value.address;
            DeviceAddress result = output.address;
            const std::size_t channel_bytes =
                static_cast<std::size_t>(od[2]) * od[3] *
                dtype_bytes(output.dtype);
            const DeviceAddress direct_address = result +
                static_cast<std::size_t>(axis_offset) * channel_bytes;
            if (axis != 1 || od[0] != 1 || input != direct_address) {
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

    if (op == "DepthToSpace") {
        const Value& input_value = value(inputs.at(0));
        const auto in = dims4(input_value.shape);
        DeviceAddress input = input_value.address;
        DeviceAddress result = output.address;
        int channels = in[1], height = in[2], width = in[3];
        int block = node.at("attributes").at("blocksize").get<int>();
        int mode = node.at("attributes").value("mode", std::string("DCR")) == "DCR"
            ? 0 : 1;
        void* parameters[] = {
            &input, &result, &count, &channels, &height, &width, &block, &mode};
        launch_linear(depth_to_space_, count_size, parameters);
        return;
    }

    if (op == "SpaceToDepth") {
        const Value& input_value = value(inputs.at(0));
        const auto in = dims4(input_value.shape);
        DeviceAddress input = input_value.address;
        DeviceAddress result = output.address;
        int channels = in[1], height = in[2], width = in[3];
        int block = node.at("attributes").at("blocksize").get<int>();
        void* parameters[] = {
            &input, &result, &count, &channels, &height, &width, &block};
        launch_linear(space_to_depth_, count_size, parameters);
        return;
    }

    throw std::runtime_error("driver-cubin: unsupported operator " + op);
}

}  // namespace mlvc::driver_cubin_backend

