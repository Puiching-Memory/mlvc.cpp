struct __align__(32) PointwiseOperandTile {
    __half weight[128 * 16];
    __half input[16 * 128];
};

struct __align__(32) PointwiseOperandTile64 {
    __half weight[128 * 16];
    __half input[16 * 72];
};

struct __align__(32) PointwiseOperandSwizzled {
    __half weight[128 * 16];
    __half input[16 * 64];
};

struct __align__(32) PointwiseOperandSwizzled64 {
    __half weight[64 * 16];
    __half input[16 * 64];
};

struct __align__(32) SpatialOperandK32 {
    PointwiseOperandSwizzled halves[2];
};

__device__ __forceinline__ void copy_async_16(
    __half* destination, const __half* source, int valid_halves)
{
#if __CUDA_ARCH__ >= 800
    const auto source_address = reinterpret_cast<unsigned long long>(source);
    if (valid_halves == 8 && (source_address & 15ULL) == 0) {
        const unsigned int shared_address =
            static_cast<unsigned int>(__cvta_generic_to_shared(destination));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     : : "r"(shared_address), "l"(source) : "memory");
    } else {
#pragma unroll
        for (int index = 0; index < 8; ++index) {
            destination[index] = index < valid_halves
                ? source[index] : __float2half(0.0F);
        }
    }
#else
#pragma unroll
    for (int index = 0; index < 8; ++index) {
        destination[index] = index < valid_halves
            ? source[index] : __float2half(0.0F);
    }
#endif
}

__device__ __forceinline__ void commit_async_copies()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" : : : "memory");
#endif
}

__device__ __forceinline__ void wait_for_async_copies()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" : : : "memory");
#endif
}

__device__ __forceinline__ void load_pointwise_tile(
    PointwiseOperandTile& tile, const __half* input, const __half* weight,
    int batch, int block_channel_base, int channel, int in_channels,
    int spatial_base, int spatial_count, int linear_thread)
{
    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    copy_async_16(
        tile.weight + weight_row * 16 + weight_column,
        weight + (block_channel_base + weight_row) * in_channels + channel +
            weight_column,
        8);

    const int input_row = linear_thread / 16;
    const int input_column = (linear_thread % 16) * 8;
    const int remaining = spatial_count - spatial_base - input_column;
    const int valid_halves = remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
    copy_async_16(
        tile.input + input_row * 128 + input_column,
        input + (batch * in_channels + channel + input_row) * spatial_count +
            spatial_base + input_column,
        valid_halves);
}

__device__ __forceinline__ void load_pointwise_tile64(
    PointwiseOperandTile64& tile, const __half* input, const __half* weight,
    int batch, int block_channel_base, int channel, int in_channels,
    int spatial_base, int spatial_count, int linear_thread)
{
    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    copy_async_16(
        tile.weight + weight_row * 16 + weight_column,
        weight + (block_channel_base + weight_row) * in_channels + channel +
            weight_column,
        8);

    if (linear_thread < 128) {
        const int input_row = linear_thread / 8;
        const int input_column = (linear_thread % 8) * 8;
        const int remaining = spatial_count - spatial_base - input_column;
        const int valid_halves =
            remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
        copy_async_16(
            tile.input + input_row * 72 + input_column,
            input + (batch * in_channels + channel + input_row) * spatial_count +
                spatial_base + input_column,
            valid_halves);
    }
}

__device__ __forceinline__ int swizzle_input_offset(int row, int column)
{
    const int chunk = column / 8;
    return row * 64 + ((chunk ^ (row & 7)) * 8) + (column & 7);
}

__device__ __forceinline__ int swizzle_weight_offset(int row, int column)
{
    return row * 16 + (column ^ ((row & 4) * 2));
}

__device__ __forceinline__ void load_pointwise_tile_swizzled(
    PointwiseOperandSwizzled& tile, const __half* input, const __half* weight,
    int batch, int block_channel_base, int channel, int in_channels,
    int spatial_base, int spatial_count, int linear_thread)
{
    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    copy_async_16(
        tile.weight + swizzle_weight_offset(weight_row, weight_column),
        weight + (block_channel_base + weight_row) * in_channels + channel +
            weight_column,
        8);

    if (linear_thread < 128) {
        const int input_row = linear_thread / 8;
        const int input_column = (linear_thread % 8) * 8;
        const int remaining = spatial_count - spatial_base - input_column;
        const int valid_halves =
            remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
        copy_async_16(
            tile.input + swizzle_input_offset(input_row, input_column),
            input + (batch * in_channels + channel + input_row) * spatial_count +
                spatial_base + input_column,
            valid_halves);
    }
}

__device__ __forceinline__ void load_pointwise_tile_swizzled64(
    PointwiseOperandSwizzled64& tile, const __half* input,
    const __half* weight, int batch, int block_channel_base, int channel,
    int in_channels, int spatial_base, int spatial_count, int linear_thread)
{
    if (linear_thread >= 128)
        return;

    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    copy_async_16(
        tile.weight + swizzle_weight_offset(weight_row, weight_column),
        weight + (block_channel_base + weight_row) * in_channels + channel +
            weight_column,
        8);

    const int input_row = linear_thread / 8;
    const int input_column = (linear_thread % 8) * 8;
    const int remaining = spatial_count - spatial_base - input_column;
    const int valid_halves =
        remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
    copy_async_16(
        tile.input + swizzle_input_offset(input_row, input_column),
        input + (batch * in_channels + channel + input_row) * spatial_count +
            spatial_base + input_column,
        valid_halves);
}

__device__ __forceinline__ void load_pointwise_reglu_tile_swizzled(
    PointwiseOperandSwizzled& tile, const __half* input, const __half* weight,
    int batch, int block_channel_base, int gate_channels, int channel,
    int in_channels, int spatial_base, int spatial_count, int linear_thread)
{
    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    const int output_channel = weight_row < 64
        ? block_channel_base + weight_row
        : gate_channels + block_channel_base + weight_row - 64;
    copy_async_16(
        tile.weight + swizzle_weight_offset(weight_row, weight_column),
        weight + output_channel * in_channels + channel + weight_column, 8);

    if (linear_thread < 128) {
        const int input_row = linear_thread / 8;
        const int input_column = (linear_thread % 8) * 8;
        const int remaining = spatial_count - spatial_base - input_column;
        const int valid_halves =
            remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
        copy_async_16(
            tile.input + swizzle_input_offset(input_row, input_column),
            input + (batch * in_channels + channel + input_row) * spatial_count +
                spatial_base + input_column,
            valid_halves);
    }
}

__device__ __forceinline__ void load_pointwise_reglu_tile_swizzled_small(
    PointwiseOperandSwizzled64& tile, const __half* input,
    const __half* weight, int batch, int block_channel_base, int gate_channels,
    int channel, int in_channels, int spatial_base, int spatial_count,
    int linear_thread)
{
    if (linear_thread >= 128)
        return;

    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    const int output_channel = weight_row < 32
        ? block_channel_base + weight_row
        : gate_channels + block_channel_base + weight_row - 32;
    copy_async_16(
        tile.weight + swizzle_weight_offset(weight_row, weight_column),
        weight + output_channel * in_channels + channel + weight_column, 8);

    const int input_row = linear_thread / 8;
    const int input_column = (linear_thread % 8) * 8;
    const int remaining = spatial_count - spatial_base - input_column;
    const int valid_halves =
        remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
    copy_async_16(
        tile.input + swizzle_input_offset(input_row, input_column),
        input + (batch * in_channels + channel + input_row) * spatial_count +
            spatial_base + input_column,
        valid_halves);
}

struct MmaAccumulator {
    float values[4];
};

__device__ __forceinline__ void load_matrix_x4(
    const __half* source, unsigned int (&values)[4])
{
#if __CUDA_ARCH__ >= 800
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(source));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
        : "r"(address) : "memory");
#endif
}

__device__ __forceinline__ void load_matrix_x2_transpose(
    const __half* source, unsigned int (&values)[2])
{
#if __CUDA_ARCH__ >= 800
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(source));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(values[0]), "=r"(values[1])
        : "r"(address) : "memory");
#endif
}

__device__ __forceinline__ void load_matrix_x4_transpose(
    const __half* source, unsigned int (&values)[4])
{
#if __CUDA_ARCH__ >= 800
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(source));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
        : "r"(address) : "memory");
#endif
}

__device__ __forceinline__ void mma_m16n8k16(
    MmaAccumulator& accumulator, const unsigned int (&a)[4],
    const unsigned int* b)
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(accumulator.values[0]), "+f"(accumulator.values[1]),
          "+f"(accumulator.values[2]), "+f"(accumulator.values[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ __half pointwise_epilogue_value(
    float accumulator, const __half* residual, int epilogue, float alpha)
{
    const __half rounded = __float2half_rn(accumulator);
    if (epilogue == 1) {
        const float value = __half2float(rounded);
        return __float2half_rn(value >= 0.0F ? value : alpha * value);
    }
    if (epilogue == 2)
        return binary_value(rounded, *residual, 0);
    return rounded;
}

