__device__ __forceinline__ __half reglu_value(
    __half gate, __half linear, float minimum, float maximum)
{
    const float clipped = fminf(fmaxf(__half2float(gate), minimum), maximum);
    const __half rounded_clip = __float2half_rn(clipped);
    return __float2half_rn(__half2float(rounded_clip) * __half2float(linear));
}

extern "C" __global__ void mlvc_reglu_fp16(
    const __half* input, __half* output, int batch_count, int channels,
    int spatial_count, float minimum, float maximum)
{
    const bool vectorized = (spatial_count & 1) == 0;
    const int work_count = vectorized ? spatial_count / 2 : spatial_count;
    const int spatial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    if (spatial >= work_count || channel >= channels || batch >= batch_count)
        return;

    const std::size_t plane = static_cast<std::size_t>(spatial_count);
    const std::size_t gate_base =
        (static_cast<std::size_t>(batch) * channels * 2 + channel) * plane;
    const std::size_t linear_base = gate_base + channels * plane;
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * channels + channel) * plane;

    if (vectorized) {
        const __half2 gates = reinterpret_cast<const __half2*>(
            input + gate_base)[spatial];
        const __half2 linears = reinterpret_cast<const __half2*>(
            input + linear_base)[spatial];
        reinterpret_cast<__half2*>(output + output_base)[spatial] =
            __halves2half2(
                reglu_value(__low2half(gates), __low2half(linears),
                            minimum, maximum),
                reglu_value(__high2half(gates), __high2half(linears),
                            minimum, maximum));
    } else {
        output[output_base + spatial] =
            reglu_value(input[gate_base + spatial],
                        input[linear_base + spatial], minimum, maximum);
    }
}

struct __align__(16) RegLuHalf8 {
    __half2 values[4];
};

extern "C" __global__ void mlvc_reglu_vec8_fp16(
    const __half* input, __half* output, int batch_count, int channels,
    int spatial_count, float minimum, float maximum)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int chunks_per_channel = spatial_count / 8;
    const int count = batch_count * channels * chunks_per_channel;
    if (index >= count)
        return;

    int linear = index;
    const int chunk = linear % chunks_per_channel;
    linear /= chunks_per_channel;
    const int channel = linear % channels;
    const int batch = linear / channels;
    const std::size_t plane = static_cast<std::size_t>(spatial_count);
    const std::size_t gate_base =
        (static_cast<std::size_t>(batch) * channels * 2 + channel) * plane;
    const std::size_t linear_base = gate_base + channels * plane;
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * channels + channel) * plane;
    const RegLuHalf8 gates = *reinterpret_cast<const RegLuHalf8*>(
        input + gate_base + chunk * 8);
    const RegLuHalf8 linears = *reinterpret_cast<const RegLuHalf8*>(
        input + linear_base + chunk * 8);
    RegLuHalf8 result;

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        result.values[item] = __halves2half2(
            reglu_value(__low2half(gates.values[item]),
                        __low2half(linears.values[item]), minimum, maximum),
            reglu_value(__high2half(gates.values[item]),
                        __high2half(linears.values[item]), minimum, maximum));
    }
    *reinterpret_cast<RegLuHalf8*>(output + output_base + chunk * 8) = result;
}

extern "C" __global__ void mlvc_reglu_bias_fp16(
    const __half* input, const __half* bias, __half* output,
    int batch_count, int channels, int spatial_count,
    float minimum, float maximum)
{
    const bool vectorized = (spatial_count & 1) == 0;
    const int work_count = vectorized ? spatial_count / 2 : spatial_count;
    const int spatial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    if (spatial >= work_count || channel >= channels || batch >= batch_count)
        return;

    const std::size_t plane = static_cast<std::size_t>(spatial_count);
    const std::size_t gate_base =
        (static_cast<std::size_t>(batch) * channels * 2 + channel) * plane;
    const std::size_t linear_base = gate_base + channels * plane;
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * channels + channel) * plane;
    const float gate_bias = bias ? __half2float(bias[channel]) : 0.0F;
    const float linear_bias =
        bias ? __half2float(bias[channel + channels]) : 0.0F;

    auto apply = [&](const __half gate, const __half linear) {
        const __half biased_gate =
            __float2half_rn(__half2float(gate) + gate_bias);
        const __half biased_linear =
            __float2half_rn(__half2float(linear) + linear_bias);
        return reglu_value(biased_gate, biased_linear, minimum, maximum);
    };

    if (vectorized) {
        const __half2 gates = reinterpret_cast<const __half2*>(
            input + gate_base)[spatial];
        const __half2 linears = reinterpret_cast<const __half2*>(
            input + linear_base)[spatial];
        reinterpret_cast<__half2*>(output + output_base)[spatial] =
            __halves2half2(
                apply(__low2half(gates), __low2half(linears)),
                apply(__high2half(gates), __high2half(linears)));
    } else {
        output[output_base + spatial] =
            apply(input[gate_base + spatial], input[linear_base + spatial]);
    }
}

