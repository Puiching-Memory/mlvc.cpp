// Fused y0 quantization and prior-input preparation.  The graph keeps the
// normalized tensor and clipped prior alive for the second quantization pass,
// so this kernel materializes those values as side outputs while writing the
// y0 output and the two following concat inputs directly.  3D grid eliminates
// integer divisions and reuses the 4 loaded prior channels to populate concat3.
extern "C" __global__ void mlvc_y0_tail_fp16(
    const __half* __restrict__ latent, const __half* __restrict__ prior,
    const __half* __restrict__ quant_scale,
    const __half* __restrict__ update_scale0,
    const __half* __restrict__ update_scale1,
    __half* __restrict__ normalized, __half* __restrict__ clipped,
    __half* __restrict__ y_raw, __half* __restrict__ concat2,
    __half* __restrict__ concat3, float clip_min, float clip_max,
    int batch_count, int quant_channels, int spatial_count)
{
    const int spatial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    if (spatial >= spatial_count || channel >= quant_channels ||
        batch >= batch_count) {
        return;
    }

    const std::size_t q_stride =
        static_cast<std::size_t>(quant_channels) * spatial_count;
    const std::size_t y_raw_base =
        (static_cast<std::size_t>(batch) * quant_channels + channel) *
        spatial_count + spatial;
    const std::size_t norm_base =
        (static_cast<std::size_t>(batch) * (quant_channels * 2) + channel) *
        spatial_count + spatial;
    const std::size_t prior_base =
        (static_cast<std::size_t>(batch) * (quant_channels * 4) + channel) *
        spatial_count + spatial;
    const std::size_t concat3_base =
        (static_cast<std::size_t>(batch) * (quant_channels * 6) + channel) *
        spatial_count + spatial;

    const __half prior0 = __ldg(prior + prior_base);
    const __half prior1 = __ldg(prior + prior_base + q_stride);
    const __half prior_tail0 = __ldg(prior + prior_base + q_stride * 2);
    const __half prior_tail1 = __ldg(prior + prior_base + q_stride * 3);
    const __half lat0 = __ldg(latent + norm_base);
    const __half lat1 = __ldg(latent + norm_base + q_stride);

    const __half clip0 = __float2half_rn(
        fminf(fmaxf(__half2float(prior0), clip_min), clip_max));
    const __half clip1 = __float2half_rn(
        fminf(fmaxf(__half2float(prior1), clip_min), clip_max));
    const __half inverse0 = __float2half_rn(1.0F / __half2float(clip0));
    const __half inverse1 = __float2half_rn(1.0F / __half2float(clip1));
    const __half normalized0 = __float2half_rn(
        __half2float(lat0) * __half2float(inverse0));
    const __half normalized1 = __float2half_rn(
        __half2float(lat1) * __half2float(inverse1));
    normalized[norm_base] = normalized0;
    normalized[norm_base + q_stride] = normalized1;
    clipped[norm_base] = clip0;
    clipped[norm_base + q_stride] = clip1;

    const __half scaled0 = binary_value(
        binary_value(normalized0, prior_tail0, 2),
        __ldg(quant_scale + norm_base), 1);
    const __half scaled1 = binary_value(
        binary_value(normalized1, prior_tail1, 2),
        __ldg(quant_scale + norm_base + q_stride), 1);
    const __half rounded = __float2half_rn(
        nearbyintf(__half2float(binary_value(scaled0, scaled1, 0))));
    y_raw[y_raw_base] = rounded;

    const __half updated0 = binary_value(
        binary_value(rounded, prior_tail0, 0),
        __ldg(update_scale0 + y_raw_base), 1);
    const __half updated1 = binary_value(
        binary_value(rounded, prior_tail1, 0),
        __ldg(update_scale1 + y_raw_base), 1);
    concat2[norm_base] = updated0;
    concat2[norm_base + q_stride] = updated1;
    concat3[concat3_base] = updated0;
    concat3[concat3_base + q_stride] = updated1;
    concat3[concat3_base + q_stride * 2] = prior0;
    concat3[concat3_base + q_stride * 3] = prior1;
    concat3[concat3_base + q_stride * 4] = prior_tail0;
    concat3[concat3_base + q_stride * 5] = prior_tail1;
}

// Fused y1 quantization, prior update, residual add, and scale multiply.  The
// y1 output remains materialized for entropy coding while the final decoder
// input is written directly, including the graph's FP16 rounding boundaries.
// Safe for in-place writes when output aliases prior, normalized, concat2, or
// clipped at the exact same base address.
extern "C" __global__ void mlvc_y1_tail_fp16(
    const __half* normalized, const __half* prior,
    const __half* __restrict__ quant_scale,
    const __half* __restrict__ update_scale0,
    const __half* __restrict__ update_scale1,
    const __half* concat2, const __half* clipped,
    __half* __restrict__ y_raw, __half* output, int batch_count,
    int quant_channels, int spatial_count)
{
    const int spatial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    if (spatial >= spatial_count || channel >= quant_channels ||
        batch >= batch_count) {
        return;
    }

    const std::size_t q_stride =
        static_cast<std::size_t>(quant_channels) * spatial_count;
    const std::size_t base =
        (static_cast<std::size_t>(batch) * (quant_channels * 2) + channel) *
        spatial_count + spatial;
    const std::size_t pair = base + q_stride;
    const std::size_t raw =
        (static_cast<std::size_t>(batch) * quant_channels + channel) *
        spatial_count + spatial;

    const __half norm0 = normalized[base];
    const __half norm1 = normalized[pair];
    const __half p0 = prior[base];
    const __half p1 = prior[pair];
    const __half c2_0 = concat2[base];
    const __half c2_1 = concat2[pair];
    const __half cl0 = clipped[base];
    const __half cl1 = clipped[pair];

    const __half scaled0 = binary_value(
        binary_value(norm0, p0, 2), __ldg(quant_scale + base), 1);
    const __half scaled1 = binary_value(
        binary_value(norm1, p1, 2), __ldg(quant_scale + pair), 1);
    const __half rounded = __float2half_rn(
        nearbyintf(__half2float(binary_value(scaled0, scaled1, 0))));
    y_raw[raw] = rounded;

    const __half updated0 = binary_value(
        binary_value(rounded, p0, 0), __ldg(update_scale0 + raw), 1);
    const __half updated1 = binary_value(
        binary_value(rounded, p1, 0), __ldg(update_scale1 + raw), 1);
    const __half added0 = binary_value(c2_0, updated0, 0);
    const __half added1 = binary_value(c2_1, updated1, 0);
    output[base] = binary_value(added0, cl0, 1);
    output[pair] = binary_value(added1, cl1, 1);
}

// Fused decoder y0 dual-prior inverse quantization and concat tail (11 nodes).
extern "C" __global__ void mlvc_dec_y0_tail_fp16(
    const __half* __restrict__ prior, const __half* __restrict__ y_raw,
    const __half* __restrict__ update_scale0,
    const __half* __restrict__ update_scale1,
    __half* __restrict__ clipped, __half* __restrict__ concat2,
    __half* __restrict__ concat3, float clip_min, float clip_max,
    int batch_count, int quant_channels, int spatial_count)
{
    const int spatial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    if (spatial >= spatial_count || channel >= quant_channels ||
        batch >= batch_count) {
        return;
    }

    const std::size_t q_stride =
        static_cast<std::size_t>(quant_channels) * spatial_count;
    const std::size_t y_raw_base =
        (static_cast<std::size_t>(batch) * quant_channels + channel) *
        spatial_count + spatial;
    const std::size_t norm_base =
        (static_cast<std::size_t>(batch) * (quant_channels * 2) + channel) *
        spatial_count + spatial;
    const std::size_t prior_base =
        (static_cast<std::size_t>(batch) * (quant_channels * 4) + channel) *
        spatial_count + spatial;
    const std::size_t concat3_base =
        (static_cast<std::size_t>(batch) * (quant_channels * 6) + channel) *
        spatial_count + spatial;

    const __half prior0 = __ldg(prior + prior_base);
    const __half prior1 = __ldg(prior + prior_base + q_stride);
    const __half prior_tail0 = __ldg(prior + prior_base + q_stride * 2);
    const __half prior_tail1 = __ldg(prior + prior_base + q_stride * 3);
    const __half y = __ldg(y_raw + y_raw_base);

    const __half clip0 = __float2half_rn(
        fminf(fmaxf(__half2float(prior0), clip_min), clip_max));
    const __half clip1 = __float2half_rn(
        fminf(fmaxf(__half2float(prior1), clip_min), clip_max));
    clipped[norm_base] = clip0;
    clipped[norm_base + q_stride] = clip1;

    const __half updated0 = binary_value(
        binary_value(y, prior_tail0, 0), __ldg(update_scale0 + y_raw_base), 1);
    const __half updated1 = binary_value(
        binary_value(y, prior_tail1, 0), __ldg(update_scale1 + y_raw_base), 1);
    concat2[norm_base] = updated0;
    concat2[norm_base + q_stride] = updated1;
    concat3[concat3_base] = updated0;
    concat3[concat3_base + q_stride] = updated1;
    concat3[concat3_base + q_stride * 2] = prior0;
    concat3[concat3_base + q_stride * 3] = prior1;
    concat3[concat3_base + q_stride * 4] = prior_tail0;
    concat3[concat3_base + q_stride * 5] = prior_tail1;
}

// Fused decoder y1 dual-prior inverse quantization, residual add, and scale
// multiply (9 nodes).  Safe for in-place writes when output aliases prior,
// concat2, or clipped at the exact same base address.
extern "C" __global__ void mlvc_dec_y1_tail_fp16(
    const __half* prior, const __half* __restrict__ y_raw,
    const __half* __restrict__ update_scale0,
    const __half* __restrict__ update_scale1,
    const __half* concat2, const __half* clipped, __half* output,
    int batch_count, int quant_channels, int spatial_count)
{
    const int spatial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    if (spatial >= spatial_count || channel >= quant_channels ||
        batch >= batch_count) {
        return;
    }

    const std::size_t q_stride =
        static_cast<std::size_t>(quant_channels) * spatial_count;
    const std::size_t base =
        (static_cast<std::size_t>(batch) * (quant_channels * 2) + channel) *
        spatial_count + spatial;
    const std::size_t pair = base + q_stride;
    const std::size_t raw =
        (static_cast<std::size_t>(batch) * quant_channels + channel) *
        spatial_count + spatial;

    const __half p0 = prior[base];
    const __half p1 = prior[pair];
    const __half c2_0 = concat2[base];
    const __half c2_1 = concat2[pair];
    const __half cl0 = clipped[base];
    const __half cl1 = clipped[pair];
    const __half y = __ldg(y_raw + raw);

    const __half updated0 = binary_value(
        binary_value(y, p0, 0), __ldg(update_scale0 + raw), 1);
    const __half updated1 = binary_value(
        binary_value(y, p1, 0), __ldg(update_scale1 + raw), 1);
    const __half added0 = binary_value(c2_0, updated0, 0);
    const __half added1 = binary_value(c2_1, updated1, 0);
    output[base] = binary_value(added0, cl0, 1);
    output[pair] = binary_value(added1, cl1, 1);
}
