extern "C" __global__ void mlvc_binary_fp16(
    const __half* lhs, const __half* rhs, __half* output, int count, int op,
    int od0, int od1, int od2, int od3,
    int ld0, int ld1, int ld2, int ld3,
    int rd0, int rd1, int rd2, int rd3)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int lhs_index = offset4(
        index, od1, od2, od3,
        ld0 == 1 ? 0 : ld1 * ld2 * ld3,
        ld1 == 1 ? 0 : ld2 * ld3,
        ld2 == 1 ? 0 : ld3,
        ld3 == 1 ? 0 : 1);
    const int rhs_index = offset4(
        index, od1, od2, od3,
        rd0 == 1 ? 0 : rd1 * rd2 * rd3,
        rd1 == 1 ? 0 : rd2 * rd3,
        rd2 == 1 ? 0 : rd3,
        rd3 == 1 ? 0 : 1);
    const float a = __half2float(lhs[lhs_index]);
    const float b = __half2float(rhs[rhs_index]);
    float value = 0.0F;
    if (op == 0)
        value = a + b;
    else if (op == 1)
        value = a * b;
    else
        value = a - b;
    output[index] = __float2half_rn(value);
}

__device__ __forceinline__ __half binary_value(__half lhs, __half rhs, int op)
{
    const float a = __half2float(lhs);
    const float b = __half2float(rhs);
    return __float2half_rn(op == 0 ? a + b : (op == 1 ? a * b : a - b));
}

extern "C" __global__ void mlvc_binary_contiguous_fp16(
    const __half* lhs, const __half* rhs, __half* output, int count, int op)
{
    const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int index = pair * 2;
    if (index >= count)
        return;
    if (index + 1 < count) {
        const __half2 a = reinterpret_cast<const __half2*>(lhs)[pair];
        const __half2 b = reinterpret_cast<const __half2*>(rhs)[pair];
        reinterpret_cast<__half2*>(output)[pair] = __halves2half2(
            binary_value(__low2half(a), __low2half(b), op),
            binary_value(__high2half(a), __high2half(b), op));
    } else {
        output[index] = binary_value(lhs[index], rhs[index], op);
    }
}

extern "C" __global__ void mlvc_binary_channel_broadcast_fp16(
    const __half* __restrict__ tensor, const __half* __restrict__ channel_vec,
    __half* __restrict__ output, int batch_count, int channels,
    int spatial_count, int op, int swapped)
{
    const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int spatial = pair * 2;
    if (spatial >= spatial_count || channel >= channels || batch >= batch_count)
        return;

    const __half cval = __ldg(channel_vec + channel);
    const std::size_t base =
        (static_cast<std::size_t>(batch) * channels + channel) * spatial_count +
        spatial;
    auto apply = [&](__half tval) {
        return swapped ? binary_value(cval, tval, op)
                       : binary_value(tval, cval, op);
    };
    if (spatial + 1 < spatial_count &&
        ((reinterpret_cast<std::uintptr_t>(tensor + base) & 3U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(output + base) & 3U) == 0)) {
        const __half2 tv = *reinterpret_cast<const __half2*>(tensor + base);
        *reinterpret_cast<__half2*>(output + base) = __halves2half2(
            apply(__low2half(tv)), apply(__high2half(tv)));
    } else {
        output[base] = apply(tensor[base]);
        if (spatial + 1 < spatial_count)
            output[base + 1] = apply(tensor[base + 1]);
    }
}

extern "C" __global__ void mlvc_unary_fp16(
    const __half* input, __half* output, int count, int op,
    float alpha, float minimum, float maximum)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const float x = __half2float(input[index]);
    float value = x;
    if (op == 0)
        value = x >= 0.0F ? x : alpha * x;
    else if (op == 1)
        value = fminf(fmaxf(x, minimum), maximum);
    else if (op == 2)
        value = 1.0F / (1.0F + expf(-x));
    else if (op == 3)
        value = 1.0F / x;
    else if (op == 4)
        value = nearbyintf(x);
    output[index] = __float2half_rn(value);
}

__device__ __forceinline__ unsigned int binary_half2_bits(
    unsigned int a_bits, unsigned int b_bits, int op)
{
    const __half2 a = *reinterpret_cast<const __half2*>(&a_bits);
    const __half2 b = *reinterpret_cast<const __half2*>(&b_bits);
    const __half2 r = __halves2half2(
        binary_value(__low2half(a), __low2half(b), op),
        binary_value(__high2half(a), __high2half(b), op));
    return *reinterpret_cast<const unsigned int*>(&r);
}

__device__ __forceinline__ uint4 binary_uint4(uint4 a, uint4 b, int op)
{
    uint4 out;
    out.x = binary_half2_bits(a.x, b.x, op);
    out.y = binary_half2_bits(a.y, b.y, op);
    out.z = binary_half2_bits(a.z, b.z, op);
    out.w = binary_half2_bits(a.w, b.w, op);
    return out;
}

__device__ __forceinline__ __half unary_value(
    __half in, int op, float alpha, float minimum, float maximum)
{
    const float x = __half2float(in);
    float value = x;
    if (op == 0)
        value = x >= 0.0F ? x : alpha * x;
    else if (op == 1)
        value = fminf(fmaxf(x, minimum), maximum);
    else if (op == 2)
        value = 1.0F / (1.0F + expf(-x));
    else if (op == 3)
        value = 1.0F / x;
    else if (op == 4)
        value = nearbyintf(x);
    return __float2half_rn(value);
}

__device__ __forceinline__ unsigned int unary_half2_bits(
    unsigned int a_bits, int op, float alpha, float minimum, float maximum)
{
    const __half2 a = *reinterpret_cast<const __half2*>(&a_bits);
    const __half2 r = __halves2half2(
        unary_value(__low2half(a), op, alpha, minimum, maximum),
        unary_value(__high2half(a), op, alpha, minimum, maximum));
    return *reinterpret_cast<const unsigned int*>(&r);
}

// Strided NHWC binary elementwise op (128-bit uint4 when 16B aligned).
// Launch: 1D linear over pixels * (vec8 ? channels / 8 : channels).
extern "C" __global__ void mlvc_binary_strided_nhwc_fp16(
    const __half* __restrict__ lhs, const __half* __restrict__ rhs,
    __half* __restrict__ output, int pixels, int channels,
    int lhs_stride, int rhs_stride, int out_stride, int op)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool vec8 =
        ((channels & 7) == 0) && ((lhs_stride & 7) == 0) &&
        ((rhs_stride & 7) == 0) && ((out_stride & 7) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(lhs) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(rhs) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(output) & 15U) == 0);
    if (vec8) {
        const std::size_t vec_cols = static_cast<std::size_t>(channels >> 3);
        const std::size_t total = static_cast<std::size_t>(pixels) * vec_cols;
        if (idx >= total)
            return;
        const std::size_t p = idx / vec_cols;
        const std::size_t c = (idx - p * vec_cols) << 3;
        const uint4 a = __ldg(reinterpret_cast<const uint4*>(
            lhs + p * static_cast<std::size_t>(lhs_stride) + c));
        const uint4 b = __ldg(reinterpret_cast<const uint4*>(
            rhs + p * static_cast<std::size_t>(rhs_stride) + c));
        *reinterpret_cast<uint4*>(
            output + p * static_cast<std::size_t>(out_stride) + c) =
            binary_uint4(a, b, op);
        return;
    }
    const std::size_t total =
        static_cast<std::size_t>(pixels) * static_cast<std::size_t>(channels);
    if (idx >= total)
        return;
    const std::size_t p = idx / static_cast<std::size_t>(channels);
    const std::size_t c = idx - p * static_cast<std::size_t>(channels);
    const __half a = __ldg(lhs + p * static_cast<std::size_t>(lhs_stride) + c);
    const __half b = __ldg(rhs + p * static_cast<std::size_t>(rhs_stride) + c);
    output[p * static_cast<std::size_t>(out_stride) + c] =
        binary_value(a, b, op);
}

// NHWC per-channel broadcast binary op [N, H, W, C] op [1, 1, 1, C] (128-bit uint4).
// Launch: 1D linear over pixels * (vec8 ? channels / 8 : channels).
extern "C" __global__ void mlvc_binary_channel_broadcast_nhwc_fp16(
    const __half* __restrict__ tensor, const __half* __restrict__ channel_vec,
    __half* __restrict__ output, int pixels, int channels,
    int in_stride, int out_stride, int op, int swapped)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool vec8 =
        ((channels & 7) == 0) && ((in_stride & 7) == 0) &&
        ((out_stride & 7) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(tensor) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(channel_vec) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(output) & 15U) == 0);
    if (vec8) {
        const std::size_t vec_cols = static_cast<std::size_t>(channels >> 3);
        const std::size_t total = static_cast<std::size_t>(pixels) * vec_cols;
        if (idx >= total)
            return;
        const std::size_t p = idx / vec_cols;
        const std::size_t c = (idx - p * vec_cols) << 3;
        const uint4 tv = __ldg(reinterpret_cast<const uint4*>(
            tensor + p * static_cast<std::size_t>(in_stride) + c));
        const uint4 cv = __ldg(reinterpret_cast<const uint4*>(channel_vec + c));
        *reinterpret_cast<uint4*>(
            output + p * static_cast<std::size_t>(out_stride) + c) =
            swapped ? binary_uint4(cv, tv, op) : binary_uint4(tv, cv, op);
        return;
    }
    const std::size_t total =
        static_cast<std::size_t>(pixels) * static_cast<std::size_t>(channels);
    if (idx >= total)
        return;
    const std::size_t p = idx / static_cast<std::size_t>(channels);
    const std::size_t c = idx - p * static_cast<std::size_t>(channels);
    const __half tval =
        __ldg(tensor + p * static_cast<std::size_t>(in_stride) + c);
    const __half cval = __ldg(channel_vec + c);
    output[p * static_cast<std::size_t>(out_stride) + c] = swapped
        ? binary_value(cval, tval, op)
        : binary_value(tval, cval, op);
}

// Strided NHWC unary elementwise op (128-bit uint4 when 16B aligned).
// Launch: 1D linear over pixels * (vec8 ? channels / 8 : channels).
extern "C" __global__ void mlvc_unary_strided_nhwc_fp16(
    const __half* __restrict__ input, __half* __restrict__ output,
    int pixels, int channels, int in_stride, int out_stride, int op,
    float alpha, float minimum, float maximum)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool vec8 =
        ((channels & 7) == 0) && ((in_stride & 7) == 0) &&
        ((out_stride & 7) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(input) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(output) & 15U) == 0);
    if (vec8) {
        const std::size_t vec_cols = static_cast<std::size_t>(channels >> 3);
        const std::size_t total = static_cast<std::size_t>(pixels) * vec_cols;
        if (idx >= total)
            return;
        const std::size_t p = idx / vec_cols;
        const std::size_t c = (idx - p * vec_cols) << 3;
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(
            input + p * static_cast<std::size_t>(in_stride) + c));
        uint4 out;
        out.x = unary_half2_bits(v.x, op, alpha, minimum, maximum);
        out.y = unary_half2_bits(v.y, op, alpha, minimum, maximum);
        out.z = unary_half2_bits(v.z, op, alpha, minimum, maximum);
        out.w = unary_half2_bits(v.w, op, alpha, minimum, maximum);
        *reinterpret_cast<uint4*>(
            output + p * static_cast<std::size_t>(out_stride) + c) = out;
        return;
    }
    const std::size_t total =
        static_cast<std::size_t>(pixels) * static_cast<std::size_t>(channels);
    if (idx >= total)
        return;
    const std::size_t p = idx / static_cast<std::size_t>(channels);
    const std::size_t c = idx - p * static_cast<std::size_t>(channels);
    output[p * static_cast<std::size_t>(out_stride) + c] = unary_value(
        __ldg(input + p * static_cast<std::size_t>(in_stride) + c),
        op, alpha, minimum, maximum);
}

