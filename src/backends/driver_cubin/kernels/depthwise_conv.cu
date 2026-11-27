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

template <int kOctsPerRow>
__device__ __forceinline__ void load_oct_row(
    const __half* __restrict__ plane_in, int iy, int input_height,
    int input_width, int output_x, bool aligned16, float (&vals)[10])
{
    if (iy < 0 || iy >= input_height) {
#pragma unroll
        for (int i = 0; i < 10; ++i)
            vals[i] = 0.0F;
        return;
    }
    const __half* row_ptr =
        plane_in + static_cast<std::size_t>(iy) * input_width;
    vals[0] = output_x > 0
        ? __half2float(__ldg(row_ptr + output_x - 1)) : 0.0F;
    if (aligned16) {
        const uint4 packed =
            __ldg(reinterpret_cast<const uint4*>(row_ptr + output_x));
        const float2 f0 =
            __half22float2(*reinterpret_cast<const __half2*>(&packed.x));
        const float2 f1 =
            __half22float2(*reinterpret_cast<const __half2*>(&packed.y));
        const float2 f2 =
            __half22float2(*reinterpret_cast<const __half2*>(&packed.z));
        const float2 f3 =
            __half22float2(*reinterpret_cast<const __half2*>(&packed.w));
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
    (void)kOctsPerRow;
}

__device__ __forceinline__ void store_oct_row(
    __half* __restrict__ out_ptr, const float (&acc)[8], bool aligned16)
{
    const __half2 o0 = __floats2half2_rn(acc[0], acc[1]);
    const __half2 o1 = __floats2half2_rn(acc[2], acc[3]);
    const __half2 o2 = __floats2half2_rn(acc[4], acc[5]);
    const __half2 o3 = __floats2half2_rn(acc[6], acc[7]);
    if (aligned16) {
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

// Output-stationary 2-row register sliding-window 3x3 DWConv (zero smem, zero
// barriers). Reuses the two middle input rows across y0 and y0+1, cutting
// global loads and FP16->FP32 conversions by 33% and weight loads by 50%.
extern "C" __global__ void mlvc_depthwise_conv_oct2_fp16(
    const __half* __restrict__ input, const __half* __restrict__ weight,
    const __half* __restrict__ bias, __half* __restrict__ output,
    int batch_count, int channels, int input_height, int input_width,
    int output_height, int output_width)
{
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int octs_per_row = output_width >> 3;
    const int row_pairs = (output_height + 1) >> 1;
    const int total_work = row_pairs * octs_per_row;
    if (tid >= total_work || channel >= channels || batch >= batch_count)
        return;

    int y_pair = 0;
    if (octs_per_row == 10) {
        y_pair = static_cast<int>(
            __umulhi(static_cast<unsigned int>(tid), 0xCCCCCCCDU) >> 3);
    } else if (octs_per_row == 5) {
        y_pair = static_cast<int>(
            __umulhi(static_cast<unsigned int>(tid), 0xCCCCCCCDU) >> 2);
    } else {
        y_pair = tid / octs_per_row;
    }
    const int output_oct_x = tid - y_pair * octs_per_row;
    const int output_x = output_oct_x << 3;
    const int y0 = y_pair << 1;
    const int y1 = y0 + 1;
    const bool has_y1 = y1 < output_height;

    const float initial = bias ? __half2float(__ldg(bias + channel)) : 0.0F;
    float acc0[8] = {initial, initial, initial, initial,
                     initial, initial, initial, initial};
    float acc1[8] = {initial, initial, initial, initial,
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
    __half* plane_out = output +
        static_cast<std::size_t>(batch * channels + channel) * output_height *
        output_width;
    const bool in_aligned16 =
        (reinterpret_cast<std::uintptr_t>(plane_in) & 15U) == 0 &&
        (input_width & 7) == 0;
    const bool out_aligned16 =
        (reinterpret_cast<std::uintptr_t>(plane_out) & 15U) == 0 &&
        (output_width & 7) == 0;

    float vals[10];
    // Row y0 - 1 -> acc0 with w[0..2]
    if (y0 > 0) {
        load_oct_row<0>(plane_in, y0 - 1, input_height, input_width, output_x,
                        in_aligned16, vals);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = fmaf(vals[i + 0], w[0], acc0[i]);
            acc0[i] = fmaf(vals[i + 1], w[1], acc0[i]);
            acc0[i] = fmaf(vals[i + 2], w[2], acc0[i]);
        }
    }
    // Row y0 -> acc0 with w[3..5], acc1 with w[0..2]
    load_oct_row<0>(plane_in, y0, input_height, input_width, output_x,
                    in_aligned16, vals);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc0[i] = fmaf(vals[i + 0], w[3], acc0[i]);
        acc0[i] = fmaf(vals[i + 1], w[4], acc0[i]);
        acc0[i] = fmaf(vals[i + 2], w[5], acc0[i]);
        acc1[i] = fmaf(vals[i + 0], w[0], acc1[i]);
        acc1[i] = fmaf(vals[i + 1], w[1], acc1[i]);
        acc1[i] = fmaf(vals[i + 2], w[2], acc1[i]);
    }
    // Row y0 + 1 -> acc0 with w[6..8], acc1 with w[3..5]
    if (y0 + 1 < input_height) {
        load_oct_row<0>(plane_in, y0 + 1, input_height, input_width, output_x,
                        in_aligned16, vals);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = fmaf(vals[i + 0], w[6], acc0[i]);
            acc0[i] = fmaf(vals[i + 1], w[7], acc0[i]);
            acc0[i] = fmaf(vals[i + 2], w[8], acc0[i]);
            acc1[i] = fmaf(vals[i + 0], w[3], acc1[i]);
            acc1[i] = fmaf(vals[i + 1], w[4], acc1[i]);
            acc1[i] = fmaf(vals[i + 2], w[5], acc1[i]);
        }
    }
    store_oct_row(
        plane_out + static_cast<std::size_t>(y0) * output_width + output_x,
        acc0, out_aligned16);

    if (has_y1) {
        // Row y0 + 2 -> acc1 with w[6..8]
        if (y0 + 2 < input_height) {
            load_oct_row<0>(plane_in, y0 + 2, input_height, input_width,
                            output_x, in_aligned16, vals);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc1[i] = fmaf(vals[i + 0], w[6], acc1[i]);
                acc1[i] = fmaf(vals[i + 1], w[7], acc1[i]);
                acc1[i] = fmaf(vals[i + 2], w[8], acc1[i]);
            }
        }
        store_oct_row(
            plane_out + static_cast<std::size_t>(y1) * output_width + output_x,
            acc1, out_aligned16);
    }
}

// Zero-division warp-per-channel 3x3 DWConv for compact maps (oh <= 16,
// ow <= 32, e.g. 12x20, 6x10, 3x5). Loads each plane once into smem.
extern "C" __global__ void mlvc_depthwise_conv_3x3_small_fp16(
    const __half* __restrict__ input, const __half* __restrict__ weight,
    const __half* __restrict__ bias, __half* __restrict__ output,
    int batch_count, int channels, int height, int width)
{
    constexpr int kWarpsPerBlock = 2;
    __shared__ __half smem[kWarpsPerBlock][18][34];

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int channel = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int batch = static_cast<int>(blockIdx.y);
    if (batch >= batch_count || channel >= channels)
        return;

    const std::size_t plane_offset =
        static_cast<std::size_t>(batch * channels + channel) * height * width;
    const __half* plane_in = input + plane_offset;
    __half* plane_out = output + plane_offset;

    for (int x = lane; x < width + 2; x += 32) {
        smem[warp][0][x] = __float2half(0.0F);
        smem[warp][height + 1][x] = __float2half(0.0F);
    }
    if (lane < height) {
        smem[warp][lane + 1][0] = __float2half(0.0F);
        smem[warp][lane + 1][width + 1] = __float2half(0.0F);
    }
    if (lane < width) {
        for (int y = 0; y < height; ++y)
            smem[warp][y + 1][lane + 1] = __ldg(plane_in + y * width + lane);
    }
    __syncwarp();

    if (lane >= width)
        return;
    const float initial = bias ? __half2float(__ldg(bias + channel)) : 0.0F;
    const __half* w_ptr = weight + channel * 9;
    float w[9];
#pragma unroll
    for (int i = 0; i < 9; ++i)
        w[i] = __half2float(__ldg(w_ptr + i));

    float r0[3] = {__half2float(smem[warp][0][lane]),
                   __half2float(smem[warp][0][lane + 1]),
                   __half2float(smem[warp][0][lane + 2])};
    float r1[3] = {__half2float(smem[warp][1][lane]),
                   __half2float(smem[warp][1][lane + 1]),
                   __half2float(smem[warp][1][lane + 2])};
    for (int y = 0; y < height; ++y) {
        const float r2[3] = {__half2float(smem[warp][y + 2][lane]),
                             __half2float(smem[warp][y + 2][lane + 1]),
                             __half2float(smem[warp][y + 2][lane + 2])};
        float v = initial;
        v = fmaf(r0[0], w[0], v);
        v = fmaf(r0[1], w[1], v);
        v = fmaf(r0[2], w[2], v);
        v = fmaf(r1[0], w[3], v);
        v = fmaf(r1[1], w[4], v);
        v = fmaf(r1[2], w[5], v);
        v = fmaf(r2[0], w[6], v);
        v = fmaf(r2[1], w[7], v);
        v = fmaf(r2[2], w[8], v);
        plane_out[y * width + lane] = __float2half_rn(v);
        r0[0] = r1[0]; r0[1] = r1[1]; r0[2] = r1[2];
        r1[0] = r2[0]; r1[1] = r2[1]; r1[2] = r2[2];
    }
}

