extern "C" __global__ void mlvc_depthwise_conv_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int channels, int input_height,
    int input_width, int output_height, int output_width, int kernel_height,
    int kernel_width, int stride_height, int stride_width, int pad_height,
    int pad_width)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int output_plane = output_height * output_width;
    const int count = batch_count * channels * output_plane;
    if (index >= count)
        return;

    int linear = index;
    const int output_x = linear % output_width;
    linear /= output_width;
    const int output_y = linear % output_height;
    linear /= output_height;
    const int channel = linear % channels;
    const int batch = linear / channels;
    const int kernel_plane = kernel_height * kernel_width;
    float accumulator = bias ? __half2float(bias[channel]) : 0.0F;

    for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
        const int input_y =
            output_y * stride_height + kernel_y - pad_height;
        if (input_y < 0 || input_y >= input_height)
            continue;
        for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
            const int input_x =
                output_x * stride_width + kernel_x - pad_width;
            if (input_x < 0 || input_x >= input_width)
                continue;
            const int input_index =
                ((batch * channels + channel) * input_height + input_y) *
                    input_width + input_x;
            const int weight_index =
                channel * kernel_plane + kernel_y * kernel_width + kernel_x;
            accumulator = fmaf(__half2float(input[input_index]),
                               __half2float(weight[weight_index]), accumulator);
        }
    }
    output[index] = __float2half_rn(accumulator);
}

extern "C" __global__ void mlvc_depthwise_conv_pair_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int channels, int input_height,
    int input_width, int output_height, int output_width, int kernel_height,
    int kernel_width, int stride_height, int stride_width, int pad_height,
    int pad_width)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int output_pairs_per_row = output_width / 2;
    const int output_pairs = output_height * output_pairs_per_row;
    const int count = batch_count * channels * output_pairs;
    if (index >= count)
        return;

    int linear = index;
    const int output_pair_x = linear % output_pairs_per_row;
    linear /= output_pairs_per_row;
    const int output_y = linear % output_height;
    linear /= output_height;
    const int channel = linear % channels;
    const int batch = linear / channels;
    const int output_x = output_pair_x * 2;
    const int kernel_plane = kernel_height * kernel_width;
    const float initial = bias ? __half2float(bias[channel]) : 0.0F;
    float low_accumulator = initial;
    float high_accumulator = initial;

    for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
        const int input_y =
            output_y * stride_height + kernel_y - pad_height;
        if (input_y < 0 || input_y >= input_height)
            continue;
        for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
            const int input_x =
                output_x * stride_width + kernel_x - pad_width;
            const float weight_value = __half2float(
                weight[channel * kernel_plane +
                       kernel_y * kernel_width + kernel_x]);
            const int input_row =
                ((batch * channels + channel) * input_height + input_y) *
                input_width;
            if (input_x >= 0 && input_x < input_width) {
                low_accumulator = fmaf(
                    __half2float(input[input_row + input_x]), weight_value,
                    low_accumulator);
            }
            const int high_input_x = input_x + stride_width;
            if (high_input_x >= 0 && high_input_x < input_width) {
                high_accumulator = fmaf(
                    __half2float(input[input_row + high_input_x]), weight_value,
                    high_accumulator);
            }
        }
    }

    const int output_index =
        ((batch * channels + channel) * output_height + output_y) *
        output_width + output_x;
    reinterpret_cast<__half2*>(output + output_index)[0] =
        __floats2half2_rn(low_accumulator, high_accumulator);
}

extern "C" __global__ void mlvc_depthwise_conv_quad_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int channels, int input_height,
    int input_width, int output_height, int output_width, int kernel_height,
    int kernel_width, int stride_height, int stride_width, int pad_height,
    int pad_width)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int output_quads_per_row = output_width / 4;
    const int output_quads = output_height * output_quads_per_row;
    const int count = batch_count * channels * output_quads;
    if (index >= count)
        return;

    int linear = index;
    const int output_quad_x = linear % output_quads_per_row;
    linear /= output_quads_per_row;
    const int output_y = linear % output_height;
    linear /= output_height;
    const int channel = linear % channels;
    const int batch = linear / channels;
    const int output_x = output_quad_x * 4;
    const float initial = bias ? __half2float(bias[channel]) : 0.0F;
    float accumulators[4] = {initial, initial, initial, initial};

#pragma unroll
    for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
        const int input_y =
            output_y * stride_height + kernel_y - pad_height;
        if (input_y < 0 || input_y >= input_height)
            continue;
        const int input_x_base = output_x * stride_width - pad_width;
        const int input_row =
            ((batch * channels + channel) * input_height + input_y) *
            input_width;
        float input_values[6];
#pragma unroll
        for (int item = 0; item < 6; ++item) {
            const int input_x = input_x_base + item;
            input_values[item] = input_x >= 0 && input_x < input_width
                ? __half2float(input[input_row + input_x]) : 0.0F;
        }
#pragma unroll
        for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
            const float weight_value = __half2float(
                weight[channel * 9 + kernel_y * 3 + kernel_x]);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                accumulators[item] = fmaf(
                    input_values[item + kernel_x], weight_value,
                    accumulators[item]);
            }
        }
    }

    const int output_index =
        ((batch * channels + channel) * output_height + output_y) *
        output_width + output_x;
    reinterpret_cast<__half2*>(output + output_index)[0] =
        __floats2half2_rn(accumulators[0], accumulators[1]);
    reinterpret_cast<__half2*>(output + output_index)[1] =
        __floats2half2_rn(accumulators[2], accumulators[3]);

    (void)kernel_height;
    (void)kernel_width;
}

