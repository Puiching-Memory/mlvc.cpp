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

// NHWC variants of the fused y0/y1 encoder and decoder quantization tails.
// All activation tensors (latent, prior, normalized, clipped, concat2, concat3,
// output) are NHWC while y_raw and the constant scale masks remain NCHW,
// using 16x16 shared-memory tiles so both NCHW and NHWC accesses are coalesced.
// Launch: block = (16, 16, 1),
//         grid  = (ceil(quant_channels / 16), ceil(spatial_count / 16), batch).
extern "C" __global__ void mlvc_y0_tail_nhwc_fp16(
    const __half* __restrict__ latent, const __half* __restrict__ prior,
    const __half* __restrict__ quant_scale,
    const __half* __restrict__ update_scale0,
    const __half* __restrict__ update_scale1,
    __half* __restrict__ normalized, __half* __restrict__ clipped,
    __half* __restrict__ y_raw, __half* __restrict__ concat2,
    __half* __restrict__ concat3, float clip_min, float clip_max,
    int batch_count, int quant_channels, int spatial_count,
    int latent_stride, int prior_stride, int concat3_stride)
{
    __shared__ __half smem_q0[16][17];
    __shared__ __half smem_q1[16][17];
    __shared__ __half smem_u0[16][17];
    __shared__ __half smem_u1[16][17];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int c_base = static_cast<int>(blockIdx.x) * 16;
    const int s_base = static_cast<int>(blockIdx.y) * 16;
    const int batch = static_cast<int>(blockIdx.z);
    if (batch >= batch_count)
        return;

    // Coalesced NCHW loads of quant_scale[0..2Q), update_scale0[0..Q), update_scale1[0..Q).
    const int c_in = c_base + ty;
    const int s_in = s_base + tx;
    const std::size_t q_stride =
        static_cast<std::size_t>(quant_channels) * spatial_count;
    if (c_in < quant_channels && s_in < spatial_count) {
        const std::size_t raw_off =
            (static_cast<std::size_t>(batch) * quant_channels + c_in) *
                spatial_count + s_in;
        const std::size_t qs_off =
            (static_cast<std::size_t>(batch) * (quant_channels * 2) + c_in) *
                spatial_count + s_in;
        smem_q0[ty][tx] = __ldg(quant_scale + qs_off);
        smem_q1[ty][tx] = __ldg(quant_scale + qs_off + q_stride);
        smem_u0[ty][tx] = __ldg(update_scale0 + raw_off);
        smem_u1[ty][tx] = __ldg(update_scale1 + raw_off);
    }
    __syncthreads();

    const int c = c_base + tx;
    const int s = s_base + ty;
    __half rounded = __float2half(0.0F);
    if (c < quant_channels && s < spatial_count) {
        const std::size_t pix =
            static_cast<std::size_t>(batch) * spatial_count + s;
        const std::size_t lat_off = pix * latent_stride + c;
        const std::size_t prior_off = pix * prior_stride + c;
        const std::size_t norm_off = pix * (quant_channels * 2) + c;
        const std::size_t c3_off = pix * concat3_stride + c;
        const int Q = quant_channels;

        const __half prior0 = __ldg(prior + prior_off);
        const __half prior1 = __ldg(prior + prior_off + Q);
        const __half prior_tail0 = __ldg(prior + prior_off + Q * 2);
        const __half prior_tail1 = __ldg(prior + prior_off + Q * 3);
        const __half lat0 = __ldg(latent + lat_off);
        const __half lat1 = __ldg(latent + lat_off + Q);

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
        normalized[norm_off] = normalized0;
        normalized[norm_off + Q] = normalized1;
        clipped[norm_off] = clip0;
        clipped[norm_off + Q] = clip1;

        const __half scaled0 = binary_value(
            binary_value(normalized0, prior_tail0, 2), smem_q0[tx][ty], 1);
        const __half scaled1 = binary_value(
            binary_value(normalized1, prior_tail1, 2), smem_q1[tx][ty], 1);
        rounded = __float2half_rn(
            nearbyintf(__half2float(binary_value(scaled0, scaled1, 0))));

        const __half updated0 = binary_value(
            binary_value(rounded, prior_tail0, 0), smem_u0[tx][ty], 1);
        const __half updated1 = binary_value(
            binary_value(rounded, prior_tail1, 0), smem_u1[tx][ty], 1);
        concat2[norm_off] = updated0;
        concat2[norm_off + Q] = updated1;
        concat3[c3_off] = updated0;
        concat3[c3_off + Q] = updated1;
        concat3[c3_off + Q * 2] = prior0;
        concat3[c3_off + Q * 3] = prior1;
        concat3[c3_off + Q * 4] = prior_tail0;
        concat3[c3_off + Q * 5] = prior_tail1;
    }
    __syncthreads();
    smem_q0[tx][ty] = rounded;
    __syncthreads();
    if (c_in < quant_channels && s_in < spatial_count) {
        const std::size_t raw_off =
            (static_cast<std::size_t>(batch) * quant_channels + c_in) *
                spatial_count + s_in;
        y_raw[raw_off] = smem_q0[ty][tx];
    }
}

extern "C" __global__ void mlvc_y1_tail_nhwc_fp16(
    const __half* normalized, const __half* prior,
    const __half* __restrict__ quant_scale,
    const __half* __restrict__ update_scale0,
    const __half* __restrict__ update_scale1,
    const __half* concat2, const __half* clipped,
    __half* __restrict__ y_raw, __half* output, int batch_count,
    int quant_channels, int spatial_count,
    int prior_stride, int output_stride)
{
    __shared__ __half smem_q0[16][17];
    __shared__ __half smem_q1[16][17];
    __shared__ __half smem_u0[16][17];
    __shared__ __half smem_u1[16][17];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int c_base = static_cast<int>(blockIdx.x) * 16;
    const int s_base = static_cast<int>(blockIdx.y) * 16;
    const int batch = static_cast<int>(blockIdx.z);
    if (batch >= batch_count)
        return;

    const int c_in = c_base + ty;
    const int s_in = s_base + tx;
    const std::size_t q_stride =
        static_cast<std::size_t>(quant_channels) * spatial_count;
    if (c_in < quant_channels && s_in < spatial_count) {
        const std::size_t raw_off =
            (static_cast<std::size_t>(batch) * quant_channels + c_in) *
                spatial_count + s_in;
        const std::size_t qs_off =
            (static_cast<std::size_t>(batch) * (quant_channels * 2) + c_in) *
                spatial_count + s_in;
        smem_q0[ty][tx] = __ldg(quant_scale + qs_off);
        smem_q1[ty][tx] = __ldg(quant_scale + qs_off + q_stride);
        smem_u0[ty][tx] = __ldg(update_scale0 + raw_off);
        smem_u1[ty][tx] = __ldg(update_scale1 + raw_off);
    }
    __syncthreads();

    const int c = c_base + tx;
    const int s = s_base + ty;
    __half rounded = __float2half(0.0F);
    if (c < quant_channels && s < spatial_count) {
        const std::size_t pix =
            static_cast<std::size_t>(batch) * spatial_count + s;
        const int Q = quant_channels;
        const std::size_t base = pix * (Q * 2) + c;
        const std::size_t prior_off = pix * prior_stride + c;
        const std::size_t out_off = pix * output_stride + c;

        const __half norm0 = normalized[base];
        const __half norm1 = normalized[base + Q];
        const __half p0 = prior[prior_off];
        const __half p1 = prior[prior_off + Q];
        const __half c2_0 = concat2[base];
        const __half c2_1 = concat2[base + Q];
        const __half cl0 = clipped[base];
        const __half cl1 = clipped[base + Q];

        const __half scaled0 = binary_value(
            binary_value(norm0, p0, 2), smem_q0[tx][ty], 1);
        const __half scaled1 = binary_value(
            binary_value(norm1, p1, 2), smem_q1[tx][ty], 1);
        rounded = __float2half_rn(
            nearbyintf(__half2float(binary_value(scaled0, scaled1, 0))));

        const __half updated0 = binary_value(
            binary_value(rounded, p0, 0), smem_u0[tx][ty], 1);
        const __half updated1 = binary_value(
            binary_value(rounded, p1, 0), smem_u1[tx][ty], 1);
        const __half added0 = binary_value(c2_0, updated0, 0);
        const __half added1 = binary_value(c2_1, updated1, 0);
        output[out_off] = binary_value(added0, cl0, 1);
        output[out_off + Q] = binary_value(added1, cl1, 1);
    }
    __syncthreads();
    smem_q0[tx][ty] = rounded;
    __syncthreads();
    if (c_in < quant_channels && s_in < spatial_count) {
        const std::size_t raw_off =
            (static_cast<std::size_t>(batch) * quant_channels + c_in) *
                spatial_count + s_in;
        y_raw[raw_off] = smem_q0[ty][tx];
    }
}

extern "C" __global__ void mlvc_dec_y0_tail_nhwc_fp16(
    const __half* __restrict__ prior, const __half* __restrict__ y_raw,
    const __half* __restrict__ update_scale0,
    const __half* __restrict__ update_scale1,
    __half* __restrict__ clipped, __half* __restrict__ concat2,
    __half* __restrict__ concat3, float clip_min, float clip_max,
    int batch_count, int quant_channels, int spatial_count,
    int prior_stride, int concat3_stride)
{
    __shared__ __half smem_y[16][17];
    __shared__ __half smem_u0[16][17];
    __shared__ __half smem_u1[16][17];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int c_base = static_cast<int>(blockIdx.x) * 16;
    const int s_base = static_cast<int>(blockIdx.y) * 16;
    const int batch = static_cast<int>(blockIdx.z);
    if (batch >= batch_count)
        return;

    const int c_in = c_base + ty;
    const int s_in = s_base + tx;
    if (c_in < quant_channels && s_in < spatial_count) {
        const std::size_t raw_off =
            (static_cast<std::size_t>(batch) * quant_channels + c_in) *
                spatial_count + s_in;
        smem_y[ty][tx] = __ldg(y_raw + raw_off);
        smem_u0[ty][tx] = __ldg(update_scale0 + raw_off);
        smem_u1[ty][tx] = __ldg(update_scale1 + raw_off);
    }
    __syncthreads();

    const int c = c_base + tx;
    const int s = s_base + ty;
    if (c >= quant_channels || s >= spatial_count)
        return;

    const std::size_t pix =
        static_cast<std::size_t>(batch) * spatial_count + s;
    const int Q = quant_channels;
    const std::size_t prior_off = pix * prior_stride + c;
    const std::size_t norm_off = pix * (Q * 2) + c;
    const std::size_t c3_off = pix * concat3_stride + c;

    const __half prior0 = __ldg(prior + prior_off);
    const __half prior1 = __ldg(prior + prior_off + Q);
    const __half prior_tail0 = __ldg(prior + prior_off + Q * 2);
    const __half prior_tail1 = __ldg(prior + prior_off + Q * 3);
    const __half y = smem_y[tx][ty];

    const __half clip0 = __float2half_rn(
        fminf(fmaxf(__half2float(prior0), clip_min), clip_max));
    const __half clip1 = __float2half_rn(
        fminf(fmaxf(__half2float(prior1), clip_min), clip_max));
    clipped[norm_off] = clip0;
    clipped[norm_off + Q] = clip1;

    const __half updated0 = binary_value(
        binary_value(y, prior_tail0, 0), smem_u0[tx][ty], 1);
    const __half updated1 = binary_value(
        binary_value(y, prior_tail1, 0), smem_u1[tx][ty], 1);
    concat2[norm_off] = updated0;
    concat2[norm_off + Q] = updated1;
    concat3[c3_off] = updated0;
    concat3[c3_off + Q] = updated1;
    concat3[c3_off + Q * 2] = prior0;
    concat3[c3_off + Q * 3] = prior1;
    concat3[c3_off + Q * 4] = prior_tail0;
    concat3[c3_off + Q * 5] = prior_tail1;
}

extern "C" __global__ void mlvc_dec_y1_tail_nhwc_fp16(
    const __half* prior, const __half* __restrict__ y_raw,
    const __half* __restrict__ update_scale0,
    const __half* __restrict__ update_scale1,
    const __half* concat2, const __half* clipped, __half* output,
    int batch_count, int quant_channels, int spatial_count,
    int prior_stride, int output_stride)
{
    __shared__ __half smem_y[16][17];
    __shared__ __half smem_u0[16][17];
    __shared__ __half smem_u1[16][17];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int c_base = static_cast<int>(blockIdx.x) * 16;
    const int s_base = static_cast<int>(blockIdx.y) * 16;
    const int batch = static_cast<int>(blockIdx.z);
    if (batch >= batch_count)
        return;

    const int c_in = c_base + ty;
    const int s_in = s_base + tx;
    if (c_in < quant_channels && s_in < spatial_count) {
        const std::size_t raw_off =
            (static_cast<std::size_t>(batch) * quant_channels + c_in) *
                spatial_count + s_in;
        smem_y[ty][tx] = __ldg(y_raw + raw_off);
        smem_u0[ty][tx] = __ldg(update_scale0 + raw_off);
        smem_u1[ty][tx] = __ldg(update_scale1 + raw_off);
    }
    __syncthreads();

    const int c = c_base + tx;
    const int s = s_base + ty;
    if (c >= quant_channels || s >= spatial_count)
        return;

    const std::size_t pix =
        static_cast<std::size_t>(batch) * spatial_count + s;
    const int Q = quant_channels;
    const std::size_t base = pix * (Q * 2) + c;
    const std::size_t prior_off = pix * prior_stride + c;
    const std::size_t out_off = pix * output_stride + c;

    const __half p0 = prior[prior_off];
    const __half p1 = prior[prior_off + Q];
    const __half c2_0 = concat2[base];
    const __half c2_1 = concat2[base + Q];
    const __half cl0 = clipped[base];
    const __half cl1 = clipped[base + Q];
    const __half y = smem_y[tx][ty];

    const __half updated0 = binary_value(
        binary_value(y, p0, 0), smem_u0[tx][ty], 1);
    const __half updated1 = binary_value(
        binary_value(y, p1, 0), smem_u1[tx][ty], 1);
    const __half added0 = binary_value(c2_0, updated0, 0);
    const __half added1 = binary_value(c2_1, updated1, 0);
    output[out_off] = binary_value(added0, cl0, 1);
    output[out_off + Q] = binary_value(added1, cl1, 1);
}
