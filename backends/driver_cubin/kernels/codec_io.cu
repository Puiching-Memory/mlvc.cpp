namespace {

__device__ __forceinline__ std::uint8_t quantize_yuv(float value)
{
    value = fminf(fmaxf(value, 0.0F), 1.0F);
    return static_cast<std::uint8_t>(__float2int_rn(value * 255.0F));
}

__device__ __forceinline__ float model_sample(
    const __half* input, int channel, int y, int x, int width, int height,
    int model_width, int model_height, int pad_left, int pad_top, int rotated,
    float inverse_pixel_range)
{
    int model_y = rotated ? x : y;
    int model_x = rotated ? y : x;
    model_y += pad_top;
    model_x += pad_left;
    if (model_y < 0 || model_y >= model_height ||
        model_x < 0 || model_x >= model_width ||
        y < 0 || y >= height || x < 0 || x >= width) {
        return 0.0F;
    }
    const std::size_t index =
        (static_cast<std::size_t>(channel) * model_height + model_y) *
            model_width + model_x;
    return __half2float(input[index]) * inverse_pixel_range;
}

}  // namespace

extern "C" __global__ void mlvc_yuv420_to_nchw_fp16(
    const std::uint8_t* input, const __half* byte_lut, __half* output,
    int width, int height, int model_width, int model_height, int rotated)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane_size = model_width * model_height;
    if (index >= plane_size * 3)
        return;

    const int channel = index / plane_size;
    const int spatial = index - channel * plane_size;
    const int y = spatial / model_width;
    const int x = spatial - y * model_width;
    const int active_width = rotated ? height : width;
    const int active_height = rotated ? width : height;
    const int edge_y = y < active_height ? y : active_height - 1;
    const int edge_x = x < active_width ? x : active_width - 1;
    const int source_y = rotated ? edge_x : edge_y;
    const int source_x = rotated ? edge_y : edge_x;
    const std::size_t y_size = static_cast<std::size_t>(width) * height;
    const std::size_t uv_size = y_size / 4;

    std::uint8_t sample = 0;
    if (channel == 0) {
        sample = input[static_cast<std::size_t>(source_y) * width + source_x];
    } else {
        const std::size_t plane_offset =
            y_size + static_cast<std::size_t>(channel - 1) * uv_size;
        sample = input[plane_offset +
            static_cast<std::size_t>(source_y / 2) * (width / 2) +
            source_x / 2];
    }
    output[index] = byte_lut[sample];
}

extern "C" __global__ void mlvc_nchw_to_yuv420(
    const __half* input, std::uint8_t* output, int width, int height,
    int model_width, int model_height, int pad_left, int pad_top, int rotated,
    float inverse_pixel_range)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y_count = width * height;
    const int uv_width = width / 2;
    const int uv_count = y_count / 4;
    if (index >= y_count)
        return;

    const int y = index / width;
    const int x = index - y * width;
    output[index] = quantize_yuv(model_sample(
        input, 0, y, x, width, height, model_width, model_height,
        pad_left, pad_top, rotated, inverse_pixel_range));

    if (index >= uv_count)
        return;
    const int uv_y = index / uv_width;
    const int uv_x = index - uv_y * uv_width;
    float u = 0.0F;
    float v = 0.0F;
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            u = __fadd_rn(u, model_sample(
                input, 1, uv_y * 2 + dy, uv_x * 2 + dx,
                width, height, model_width, model_height,
                pad_left, pad_top, rotated, inverse_pixel_range));
            v = __fadd_rn(v, model_sample(
                input, 2, uv_y * 2 + dy, uv_x * 2 + dx,
                width, height, model_width, model_height,
                pad_left, pad_top, rotated, inverse_pixel_range));
        }
    }
    output[y_count + index] = quantize_yuv(__fmul_rn(u, 0.25F));
    output[y_count + uv_count + index] =
        quantize_yuv(__fmul_rn(v, 0.25F));
}
