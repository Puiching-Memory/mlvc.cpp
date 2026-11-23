#include <cstdint>

extern "C" __global__ void mlvc_copy_fp16_bits(
    const std::uint16_t* input, std::uint16_t* output, std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = input[index];
}
