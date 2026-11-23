// Fused y0 quantization and prior-input preparation.  The graph keeps the
// normalized tensor and clipped prior alive for the second quantization pass,
// so this kernel materializes those values as side outputs while writing the
// y0 output and the two following concat inputs directly.
extern "C" __global__ void mlvc_y0_tail_fp16(
    const __half* latent, const __half* prior, const __half* quant_scale,
    const __half* update_scale0, const __half* update_scale1,
    __half* normalized, __half* clipped, __half* y_raw, __half* concat2,
    __half* concat3, float clip_min, int batch_count, int quant_channels,
    int spatial_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int prior_channels = quant_channels * 4;
    const int normalized_channels = quant_channels * 2;
    const int concat3_channels = quant_channels * 6;
    const int quant_count = batch_count * quant_channels * spatial_count;
    const int prior_count = batch_count * prior_channels * spatial_count;

    if (index < quant_count) {
        const int plane = index / spatial_count;
        const int spatial = index - plane * spatial_count;
        const int batch = plane / quant_channels;
        const int channel = plane - batch * quant_channels;
        const std::size_t latent_base =
            (static_cast<std::size_t>(batch) * normalized_channels + channel) *
            spatial_count + spatial;
        const std::size_t latent_pair_base =
            latent_base + static_cast<std::size_t>(quant_channels) * spatial_count;
        const std::size_t prior_base =
            (static_cast<std::size_t>(batch) * prior_channels + channel) *
            spatial_count + spatial;
        const std::size_t prior_pair_base =
            prior_base + static_cast<std::size_t>(quant_channels) * spatial_count;
        const std::size_t prior_tail_base =
            prior_base + static_cast<std::size_t>(normalized_channels) * spatial_count;
        const std::size_t prior_tail_pair_base =
            prior_tail_base + static_cast<std::size_t>(quant_channels) * spatial_count;
        const std::size_t y_raw_base =
            (static_cast<std::size_t>(batch) * quant_channels + channel) *
            spatial_count + spatial;
        const std::size_t concat2_base =
            (static_cast<std::size_t>(batch) * normalized_channels + channel) *
            spatial_count + spatial;
        const std::size_t concat3_base =
            (static_cast<std::size_t>(batch) * concat3_channels + channel) *
            spatial_count + spatial;
        const std::size_t quant_pair =
            (static_cast<std::size_t>(batch) * normalized_channels + channel) *
            spatial_count + spatial;

        const __half prior0 = prior[prior_base];
        const __half prior1 = prior[prior_pair_base];
        const __half clip0 = __float2half_rn(
            fmaxf(__half2float(prior0), clip_min));
        const __half clip1 = __float2half_rn(
            fmaxf(__half2float(prior1), clip_min));
        const __half inverse0 = __float2half_rn(
            1.0F / __half2float(clip0));
        const __half inverse1 = __float2half_rn(
            1.0F / __half2float(clip1));
        const __half normalized0 = __float2half_rn(
            __half2float(latent[latent_base]) * __half2float(inverse0));
        const __half normalized1 = __float2half_rn(
            __half2float(latent[latent_pair_base]) * __half2float(inverse1));
        normalized[quant_pair] = normalized0;
        normalized[quant_pair + static_cast<std::size_t>(quant_channels) *
            spatial_count] = normalized1;
        clipped[quant_pair] = clip0;
        clipped[quant_pair + static_cast<std::size_t>(quant_channels) *
            spatial_count] = clip1;

        const __half scaled0 = binary_value(
            binary_value(normalized0, prior[prior_tail_base], 2),
            quant_scale[quant_pair], 1);
        const __half scaled1 = binary_value(
            binary_value(normalized1, prior[prior_tail_pair_base], 2),
            quant_scale[quant_pair +
                static_cast<std::size_t>(quant_channels) * spatial_count], 1);
        const __half rounded = __float2half_rn(
            nearbyintf(__half2float(binary_value(scaled0, scaled1, 0))));
        y_raw[y_raw_base] = rounded;

        const __half updated0 = binary_value(
            binary_value(rounded, prior[prior_tail_base], 0),
            update_scale0[y_raw_base], 1);
        const __half updated1 = binary_value(
            binary_value(rounded, prior[prior_tail_pair_base], 0),
            update_scale1[y_raw_base], 1);
        concat2[concat2_base] = updated0;
        concat2[concat2_base +
            static_cast<std::size_t>(quant_channels) * spatial_count] = updated1;
        concat3[concat3_base] = updated0;
        concat3[concat3_base +
            static_cast<std::size_t>(quant_channels) * spatial_count] = updated1;
    }

    if (index < prior_count) {
        const int plane = index / spatial_count;
        const int spatial = index - plane * spatial_count;
        const int batch = plane / prior_channels;
        const int channel = plane - batch * prior_channels;
        const std::size_t source =
            (static_cast<std::size_t>(batch) * prior_channels + channel) *
            spatial_count + spatial;
        const std::size_t destination =
            (static_cast<std::size_t>(batch) * concat3_channels +
             normalized_channels + channel) * spatial_count + spatial;
        concat3[destination] = prior[source];
    }
}

// Fused y1 quantization, prior update, residual add, and scale multiply.  The
// y1 output remains materialized for entropy coding while the final decoder
// input is written directly, including the graph's FP16 rounding boundaries.
extern "C" __global__ void mlvc_y1_tail_fp16(
    const __half* normalized, const __half* prior,
    const __half* quant_scale, const __half* update_scale0,
    const __half* update_scale1, const __half* concat2,
    const __half* clipped, __half* y_raw, __half* output, int batch_count,
    int quant_channels, int spatial_count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int normalized_channels = quant_channels * 2;
    const int quant_count = batch_count * quant_channels * spatial_count;
    if (index >= quant_count)
        return;

    const int plane = index / spatial_count;
    const int spatial = index - plane * spatial_count;
    const int batch = plane / quant_channels;
    const int channel = plane - batch * quant_channels;
    const std::size_t base =
        (static_cast<std::size_t>(batch) * normalized_channels + channel) *
        spatial_count + spatial;
    const std::size_t pair =
        base + static_cast<std::size_t>(quant_channels) * spatial_count;
    const std::size_t raw =
        (static_cast<std::size_t>(batch) * quant_channels + channel) *
        spatial_count + spatial;

    const __half scaled0 = binary_value(
        binary_value(normalized[base], prior[base], 2),
        quant_scale[base], 1);
    const __half scaled1 = binary_value(
        binary_value(normalized[pair], prior[pair], 2),
        quant_scale[pair], 1);
    const __half rounded = __float2half_rn(
        nearbyintf(__half2float(binary_value(scaled0, scaled1, 0))));
    y_raw[raw] = rounded;

    const __half updated0 = binary_value(
        binary_value(rounded, prior[base], 0), update_scale0[raw], 1);
    const __half updated1 = binary_value(
        binary_value(rounded, prior[pair], 0), update_scale1[raw], 1);
    const __half added0 = binary_value(concat2[base], updated0, 0);
    const __half added1 = binary_value(concat2[pair], updated1, 0);
    const __half output0 = binary_value(added0, clipped[base], 1);
    const __half output1 = binary_value(added1, clipped[pair], 1);
    output[base] = output0;
    output[pair] = output1;
}

