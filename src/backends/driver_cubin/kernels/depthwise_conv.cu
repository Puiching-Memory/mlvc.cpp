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

extern "C" __global__ void mlvc_depthwise_conv_oct_fp16(
    const __half* __restrict__ input, const __half* __restrict__ weight,
    const __half* __restrict__ bias, __half* __restrict__ output,
    int batch_count, int channels, int input_height, int input_width,
    int output_height, int output_width)
{
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int octs_per_row = output_width >> 3;
    const int total_octs = output_height * octs_per_row;
    if (tid >= total_octs || channel >= channels || batch >= batch_count)
        return;

    int output_y = 0;
    if (octs_per_row == 10) {
        output_y = static_cast<int>(
            __umulhi(static_cast<unsigned int>(tid), 0xCCCCCCCDU) >> 3);
    } else if (octs_per_row == 5) {
        output_y = static_cast<int>(
            __umulhi(static_cast<unsigned int>(tid), 0xCCCCCCCDU) >> 2);
    } else {
        output_y = tid / octs_per_row;
    }
    const int output_oct_x = tid - output_y * octs_per_row;
    const int output_x = output_oct_x << 3;

    const float initial = bias ? __half2float(__ldg(bias + channel)) : 0.0F;
    float acc[8] = {initial, initial, initial, initial,
                    initial, initial, initial, initial};
    const __half* w_ptr = weight + channel * 9;
    float w[9];
#pragma unroll
    for (int i = 0; i < 9; ++i)
        w[i] = __half2float(__ldg(w_ptr + i));

    const std::size_t plane_base =
        static_cast<std::size_t>(batch * channels + channel) * input_height *
        input_width;
    const __half* plane_in = input + plane_base;
    const bool aligned16 =
        (reinterpret_cast<std::uintptr_t>(plane_in) & 15U) == 0 &&
        (input_width & 7) == 0;

#pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
        const int iy = output_y + ky - 1;
        if (iy < 0 || iy >= input_height)
            continue;
        const __half* row_ptr = plane_in + static_cast<std::size_t>(iy) * input_width;
        float vals[10];
        vals[0] = output_x > 0
            ? __half2float(__ldg(row_ptr + output_x - 1)) : 0.0F;
        if (aligned16) {
            const uint4 packed =
                __ldg(reinterpret_cast<const uint4*>(row_ptr + output_x));
            const __half2 h0 = *reinterpret_cast<const __half2*>(&packed.x);
            const __half2 h1 = *reinterpret_cast<const __half2*>(&packed.y);
            const __half2 h2 = *reinterpret_cast<const __half2*>(&packed.z);
            const __half2 h3 = *reinterpret_cast<const __half2*>(&packed.w);
            const float2 f0 = __half22float2(h0);
            const float2 f1 = __half22float2(h1);
            const float2 f2 = __half22float2(h2);
            const float2 f3 = __half22float2(h3);
            vals[1] = f0.x; vals[2] = f0.y;
            vals[3] = f1.x; vals[4] = f1.y;
            vals[5] = f2.x; vals[6] = f2.y;
            vals[7] = f3.x; vals[8] = f3.y;
        } else {
#pragma unroll
            for (int item = 0; item < 8; ++item)
                vals[item + 1] = __half2float(__ldg(row_ptr + output_x + item));
        }
        vals[9] = (output_x + 8) < input_width
            ? __half2float(__ldg(row_ptr + output_x + 8)) : 0.0F;

        const float w0 = w[ky * 3 + 0];
        const float w1 = w[ky * 3 + 1];
        const float w2 = w[ky * 3 + 2];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            acc[item] = fmaf(vals[item + 0], w0, acc[item]);
            acc[item] = fmaf(vals[item + 1], w1, acc[item]);
            acc[item] = fmaf(vals[item + 2], w2, acc[item]);
        }
    }

    __half* out_ptr = output +
        (static_cast<std::size_t>(batch * channels + channel) * output_height +
         output_y) * output_width + output_x;
    const __half2 o0 = __floats2half2_rn(acc[0], acc[1]);
    const __half2 o1 = __floats2half2_rn(acc[2], acc[3]);
    const __half2 o2 = __floats2half2_rn(acc[4], acc[5]);
    const __half2 o3 = __floats2half2_rn(acc[6], acc[7]);
    if ((reinterpret_cast<std::uintptr_t>(out_ptr) & 15U) == 0) {
        uint4 packed_out;
        packed_out.x = *reinterpret_cast<const unsigned int*>(&o0);
        packed_out.y = *reinterpret_cast<const unsigned int*>(&o1);
        packed_out.z = *reinterpret_cast<const unsigned int*>(&o2);
        packed_out.w = *reinterpret_cast<const unsigned int*>(&o3);
        *reinterpret_cast<uint4*>(out_ptr) = packed_out;
    } else {
        auto* out2 = reinterpret_cast<__half2*>(out_ptr);
        out2[0] = o0;
        out2[1] = o1;
        out2[2] = o2;
        out2[3] = o3;
    }
}

