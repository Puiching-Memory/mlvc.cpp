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

