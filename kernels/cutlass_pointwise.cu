#include "mlvc/driver/cutlass.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>

namespace {

using Element = cutlass::half_t;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
inline constexpr int kAlignment = 8;
inline constexpr int kEpilogueStages = 1;
inline constexpr float kLeakyReluAlpha = 0.01F;

using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, Element, kAlignment, kEpilogueStages>;
using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
using Bias = cutlass::epilogue::threadblock::VisitorColBroadcast<
    ThreadMap, Element>;
using Add = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest>;
using AddBias = cutlass::epilogue::threadblock::Sm80EVT<
    Add, Accumulator, Bias>;
using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
    ThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, int64_t>>;
using BiasCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<Store, AddBias>;

template <typename Array>
struct LeakyRelu {
    CUTLASS_HOST_DEVICE
    Array operator()(Array values) const
    {
        CUTLASS_PRAGMA_UNROLL
        for (int index = 0; index < Array::kElements; ++index) {
            values[index] = values[index] >= 0.0F
                ? values[index] : kLeakyReluAlpha * values[index];
        }
        return values;
    }
};

using ApplyLeakyRelu = cutlass::epilogue::threadblock::VisitorCompute<
    LeakyRelu, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using BiasLeakyRelu = cutlass::epilogue::threadblock::Sm80EVT<
    ApplyLeakyRelu, AddBias>;
using LeakyReluCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<Store, BiasLeakyRelu>;

using Residual = cutlass::epilogue::threadblock::VisitorAuxLoad<
    ThreadMap, Element, cute::Stride<int64_t, cute::_1, int64_t>>;
using AddResidual = cutlass::epilogue::threadblock::Sm80EVT<
    Add, AddBias, Residual>;
using ResidualCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<Store, AddResidual>;

using MediumThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using MediumWarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using MediumThreadMap =
    cutlass::epilogue::threadblock::OutputTileThreadLayout<
        MediumThreadblockShape, MediumWarpShape, Element, kAlignment,
        kEpilogueStages>;
using MediumBias = cutlass::epilogue::threadblock::VisitorColBroadcast<
    MediumThreadMap, Element>;
using MediumAddBias = cutlass::epilogue::threadblock::Sm80EVT<
    Add, Accumulator, MediumBias>;
using MediumStore = cutlass::epilogue::threadblock::VisitorAuxStore<
    MediumThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, int64_t>>;
using MediumBiasCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<MediumStore, MediumAddBias>;
using MediumBiasLeakyRelu = cutlass::epilogue::threadblock::Sm80EVT<
    ApplyLeakyRelu, MediumAddBias>;
using MediumLeakyReluCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        MediumStore, MediumBiasLeakyRelu>;
using MediumResidual = cutlass::epilogue::threadblock::VisitorAuxLoad<
    MediumThreadMap, Element, cute::Stride<int64_t, cute::_1, int64_t>>;
using MediumAddResidual = cutlass::epilogue::threadblock::Sm80EVT<
    Add, MediumAddBias, MediumResidual>;
using MediumResidualCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        MediumStore, MediumAddResidual>;

using SpatialWideThreadblockShape = cutlass::gemm::GemmShape<64, 256, 32>;
using SpatialWideWarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using SpatialWideThreadMap =
    cutlass::epilogue::threadblock::OutputTileThreadLayout<
        SpatialWideThreadblockShape, SpatialWideWarpShape, Element, kAlignment,
        kEpilogueStages>;
using SpatialWideBias = cutlass::epilogue::threadblock::VisitorColBroadcast<
    SpatialWideThreadMap, Element>;
using SpatialWideAddBias = cutlass::epilogue::threadblock::Sm80EVT<
    Add, Accumulator, SpatialWideBias>;
using SpatialWideStore = cutlass::epilogue::threadblock::VisitorAuxStore<
    SpatialWideThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, int64_t>>;
using SpatialWideResidual = cutlass::epilogue::threadblock::VisitorAuxLoad<
    SpatialWideThreadMap, Element, cute::Stride<int64_t, cute::_1, int64_t>>;
using SpatialWideAddResidual = cutlass::epilogue::threadblock::Sm80EVT<
    Add, SpatialWideAddBias, SpatialWideResidual>;
using SpatialWideResidualCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        SpatialWideStore, SpatialWideAddResidual>;

template <typename Callbacks, typename TileShape = ThreadblockShape,
          typename TileWarpShape = WarpShape, int Stages = 3>
using PointwiseKernel =
    typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
        kAlignment,
        Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
        kAlignment,
        Element, cutlass::layout::RowMajor, kAlignment,
        float, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        TileShape, TileWarpShape, InstructionShape, Callbacks,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
        cutlass::arch::OpMultiplyAdd, kEpilogueStages>::GemmKernel;

using BiasKernel = PointwiseKernel<BiasCallbacks>;
using LeakyReluKernel = PointwiseKernel<LeakyReluCallbacks>;
using ResidualKernel = PointwiseKernel<ResidualCallbacks>;
using MediumBiasKernel = PointwiseKernel<
    MediumBiasCallbacks, MediumThreadblockShape, MediumWarpShape>;
using MediumStage4BiasKernel = PointwiseKernel<
    MediumBiasCallbacks, MediumThreadblockShape, MediumWarpShape, 4>;
using MediumLeakyReluKernel = PointwiseKernel<
    MediumLeakyReluCallbacks, MediumThreadblockShape, MediumWarpShape>;
using MediumResidualKernel = PointwiseKernel<
    MediumResidualCallbacks, MediumThreadblockShape, MediumWarpShape>;
using SpatialWideResidualKernel = PointwiseKernel<
    SpatialWideResidualCallbacks, SpatialWideThreadblockShape,
    SpatialWideWarpShape>;

using SpatialConvOutputOp =
    cutlass::epilogue::thread::LinearCombinationBiasElementwise<
        Element, float, float, Element, Element, kAlignment,
        cutlass::epilogue::thread::Identity<float>, cutlass::plus<float>, false,
        Element>;
using SpatialConvKernel =
    typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 256, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>, InstructionShape,
        SpatialConvOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

template <typename Kernel, typename OutputOp>
__device__ __forceinline__ void initialize_pointwise(
    void* params_storage, const Element* input, const Element* weight,
    Element* output, int out_channels, int spatial_count, int in_channels,
    OutputOp output_op, int tile_rows = 128, int tile_columns = 64)
{
    auto* params = static_cast<typename Kernel::Params*>(params_storage);
    params->problem_size = {out_channels, spatial_count, in_channels};
    params->grid_tiled_shape = {
        (out_channels + tile_rows - 1) / tile_rows,
        (spatial_count + tile_columns - 1) / tile_columns, 1};
    params->swizzle_log_tile = 0;
    params->mode = cutlass::gemm::GemmUniversalMode::kGemm;
    params->batch_count = 1;
    params->gemm_k_size = in_channels;
    params->batch_stride_D = 0;
    params->semaphore = nullptr;
    params->problem_shape = cute::make_shape(
        int32_t(out_channels), int32_t(spatial_count), int32_t(1));
    params->params_A = decltype(params->params_A){
        cutlass::layout::RowMajor{in_channels}};
    params->params_B = decltype(params->params_B){
        cutlass::layout::RowMajor{spatial_count}};
    params->output_op = output_op;
    params->ptr_A = const_cast<Element*>(weight);
    params->ptr_B = const_cast<Element*>(input);
    params->batch_stride_A = int64_t(out_channels) * in_channels;
    params->batch_stride_B = int64_t(in_channels) * spatial_count;
    params->ptr_gather_A_indices = nullptr;
    params->ptr_gather_B_indices = nullptr;
}

static_assert(sizeof(typename BiasKernel::Params) <=
              mlvc::driver::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename LeakyReluKernel::Params) <=
              mlvc::driver::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename ResidualKernel::Params) <=
              mlvc::driver::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumBiasKernel::Params) <=
              mlvc::driver::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumStage4BiasKernel::Params) <=
              mlvc::driver::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumLeakyReluKernel::Params) <=
              mlvc::driver::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumResidualKernel::Params) <=
              mlvc::driver::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SpatialWideResidualKernel::Params) <=
              mlvc::driver::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SpatialConvKernel::Params) <=
              mlvc::driver::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename BiasKernel::SharedStorage) == 36864);
static_assert(sizeof(typename LeakyReluKernel::SharedStorage) == 36864);
static_assert(sizeof(typename ResidualKernel::SharedStorage) == 36864);
static_assert(sizeof(typename MediumBiasKernel::SharedStorage) == 49152);
static_assert(sizeof(typename MediumStage4BiasKernel::SharedStorage) == 65536);
static_assert(sizeof(typename MediumLeakyReluKernel::SharedStorage) == 49152);
static_assert(sizeof(typename MediumResidualKernel::SharedStorage) == 49152);
static_assert(sizeof(typename SpatialWideResidualKernel::SharedStorage) == 61440);
static_assert(sizeof(typename SpatialConvKernel::SharedStorage) == 73728);

}  // namespace

extern "C" __global__ void mlvc_cutlass_spatial_conv_init_fp16(
    void* params_storage, const Element* input, const Element* weight,
    const Element* bias, Element* output, int batch_count, int in_channels,
    int input_height, int input_width, int out_channels, int output_height,
    int output_width, int kernel_height, int kernel_width, int stride_height,
    int stride_width, int pad_height, int pad_width)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const cutlass::Tensor4DCoord input_extent{
        batch_count, input_height, input_width, in_channels};
    const cutlass::Tensor4DCoord weight_extent{
        out_channels, kernel_height, kernel_width, in_channels};
    const cutlass::Tensor4DCoord output_extent{
        batch_count, output_height, output_width, out_channels};
    const cutlass::conv::Conv2dProblemSize problem{
        input_extent,
        weight_extent,
        {pad_height, pad_height, pad_width, pad_width},
        {stride_height, stride_width},
        {1, 1},
        output_extent,
        cutlass::conv::Mode::kCrossCorrelation,
        1};
    typename SpatialConvKernel::Arguments arguments{
        problem,
        {const_cast<Element*>(input),
         cutlass::layout::TensorNHWC::packed(input_extent)},
        {const_cast<Element*>(weight),
         cutlass::layout::TensorNHWC::packed(weight_extent)},
        {output, cutlass::layout::TensorNHWC::packed(output_extent)},
        {output, cutlass::layout::TensorNHWC::packed(output_extent)},
        typename SpatialConvOutputOp::Params{1.0F, 0.0F}};
    arguments.ptr_Vector = const_cast<Element*>(bias);
    arguments.ldr = 0;
    *static_cast<typename SpatialConvKernel::Params*>(params_storage) =
        typename SpatialConvKernel::Params{arguments};
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)batch_count;
    (void)in_channels;
    (void)input_height;
    (void)input_width;
    (void)out_channels;
    (void)output_height;
    (void)output_width;
    (void)kernel_height;
    (void)kernel_width;
    (void)stride_height;
    (void)stride_width;
    (void)pad_height;
    (void)pad_width;
#endif
}

extern "C" __global__ void mlvc_cutlass_spatial_conv_fp16(
    typename SpatialConvKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage = reinterpret_cast<
        typename SpatialConvKernel::SharedStorage*>(shared_storage);
    SpatialConvKernel{}(params, *storage);
#else
    (void)params;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_init_fp16(
    void* params_storage, const Element* input, const Element* weight,
    const Element* bias, const Element* residual, Element* output,
    int out_channels, int spatial_count, int in_channels)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    (void)residual;
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{
        int64_t(spatial_count), cute::_1{},
        int64_t(out_channels) * spatial_count};
    initialize_pointwise<BiasKernel>(
        params_storage, input, weight, output, out_channels, spatial_count,
        in_channels, typename BiasCallbacks::Params{
            {{}, {bias, Element(0), {}}, {}}, {output, stride}});
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)bias;
    (void)residual;
    (void)output;
    (void)out_channels;
    (void)spatial_count;
    (void)in_channels;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_leaky_relu_init_fp16(
    void* params_storage, const Element* input, const Element* weight,
    const Element* bias, const Element* residual, Element* output,
    int out_channels, int spatial_count, int in_channels)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    (void)residual;
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{
        int64_t(spatial_count), cute::_1{},
        int64_t(out_channels) * spatial_count};
    initialize_pointwise<LeakyReluKernel>(
        params_storage, input, weight, output, out_channels, spatial_count,
        in_channels, typename LeakyReluCallbacks::Params{
            {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride}});
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)bias;
    (void)residual;
    (void)output;
    (void)out_channels;
    (void)spatial_count;
    (void)in_channels;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_residual_init_fp16(
    void* params_storage, const Element* input, const Element* weight,
    const Element* bias, const Element* residual, Element* output,
    int out_channels, int spatial_count, int in_channels)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{
        int64_t(spatial_count), cute::_1{},
        int64_t(out_channels) * spatial_count};
    initialize_pointwise<ResidualKernel>(
        params_storage, input, weight, output, out_channels, spatial_count,
        in_channels, typename ResidualCallbacks::Params{
            {{{}, {bias, Element(0), {}}, {}},
             {const_cast<Element*>(residual), Element(0), stride},
             {}},
            {output, stride}});
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)bias;
    (void)residual;
    (void)output;
    (void)out_channels;
    (void)spatial_count;
    (void)in_channels;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_medium_init_fp16(
    void* params_storage, const Element* input, const Element* weight,
    const Element* bias, const Element* residual, Element* output,
    int out_channels, int spatial_count, int in_channels)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    (void)residual;
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{
        int64_t(spatial_count), cute::_1{},
        int64_t(out_channels) * spatial_count};
    initialize_pointwise<MediumBiasKernel>(
        params_storage, input, weight, output, out_channels, spatial_count,
        in_channels, typename MediumBiasCallbacks::Params{
            {{}, {bias, Element(0), {}}, {}}, {output, stride}});
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)bias;
    (void)residual;
    (void)output;
    (void)out_channels;
    (void)spatial_count;
    (void)in_channels;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_medium_stage4_init_fp16(
    void* params_storage, const Element* input, const Element* weight,
    const Element* bias, const Element* residual, Element* output,
    int out_channels, int spatial_count, int in_channels)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    (void)residual;
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{
        int64_t(spatial_count), cute::_1{},
        int64_t(out_channels) * spatial_count};
    initialize_pointwise<MediumStage4BiasKernel>(
        params_storage, input, weight, output, out_channels, spatial_count,
        in_channels, typename MediumBiasCallbacks::Params{
            {{}, {bias, Element(0), {}}, {}}, {output, stride}},
        128, 128);
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)bias;
    (void)residual;
    (void)output;
    (void)out_channels;
    (void)spatial_count;
    (void)in_channels;
#endif
}

extern "C" __global__ void
mlvc_cutlass_pointwise_medium_leaky_relu_init_fp16(
    void* params_storage, const Element* input, const Element* weight,
    const Element* bias, const Element* residual, Element* output,
    int out_channels, int spatial_count, int in_channels)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    (void)residual;
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{
        int64_t(spatial_count), cute::_1{},
        int64_t(out_channels) * spatial_count};
    initialize_pointwise<MediumLeakyReluKernel>(
        params_storage, input, weight, output, out_channels, spatial_count,
        in_channels, typename MediumLeakyReluCallbacks::Params{
            {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride}});
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)bias;
    (void)residual;
    (void)output;
    (void)out_channels;
    (void)spatial_count;
    (void)in_channels;
#endif
}

extern "C" __global__ void
mlvc_cutlass_pointwise_medium_residual_init_fp16(
    void* params_storage, const Element* input, const Element* weight,
    const Element* bias, const Element* residual, Element* output,
    int out_channels, int spatial_count, int in_channels)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{
        int64_t(spatial_count), cute::_1{},
        int64_t(out_channels) * spatial_count};
    initialize_pointwise<MediumResidualKernel>(
        params_storage, input, weight, output, out_channels, spatial_count,
        in_channels, typename MediumResidualCallbacks::Params{
            {{{}, {bias, Element(0), {}}, {}},
             {const_cast<Element*>(residual), Element(0), stride},
             {}},
            {output, stride}});
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)bias;
    (void)residual;
    (void)output;
    (void)out_channels;
    (void)spatial_count;
    (void)in_channels;
#endif
}

extern "C" __global__ void
mlvc_cutlass_pointwise_spatial_wide_residual_init_fp16(
    void* params_storage, const Element* input, const Element* weight,
    const Element* bias, const Element* residual, Element* output,
    int out_channels, int spatial_count, int in_channels)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{
        int64_t(spatial_count), cute::_1{},
        int64_t(out_channels) * spatial_count};
    initialize_pointwise<SpatialWideResidualKernel>(
        params_storage, input, weight, output, out_channels, spatial_count,
        in_channels, typename SpatialWideResidualCallbacks::Params{
            {{{}, {bias, Element(0), {}}, {}},
             {const_cast<Element*>(residual), Element(0), stride},
             {}},
            {output, stride}},
        64, 256);
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)bias;
    (void)residual;
    (void)output;
    (void)out_channels;
    (void)spatial_count;
    (void)in_channels;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_fp16(
    typename BiasKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage =
        reinterpret_cast<typename BiasKernel::SharedStorage*>(shared_storage);
    BiasKernel{}(params, *storage);
#else
    (void)params;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_medium_fp16(
    typename MediumBiasKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage = reinterpret_cast<typename MediumBiasKernel::SharedStorage*>(
        shared_storage);
    MediumBiasKernel{}(params, *storage);
#else
    (void)params;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_medium_stage4_fp16(
    typename MediumStage4BiasKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage = reinterpret_cast<
        typename MediumStage4BiasKernel::SharedStorage*>(shared_storage);
    MediumStage4BiasKernel{}(params, *storage);
#else
    (void)params;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_medium_leaky_relu_fp16(
    typename MediumLeakyReluKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage = reinterpret_cast<
        typename MediumLeakyReluKernel::SharedStorage*>(shared_storage);
    MediumLeakyReluKernel{}(params, *storage);
#else
    (void)params;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_medium_residual_fp16(
    typename MediumResidualKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage = reinterpret_cast<
        typename MediumResidualKernel::SharedStorage*>(shared_storage);
    MediumResidualKernel{}(params, *storage);
#else
    (void)params;
#endif
}

extern "C" __global__ void
mlvc_cutlass_pointwise_spatial_wide_residual_fp16(
    typename SpatialWideResidualKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage = reinterpret_cast<
        typename SpatialWideResidualKernel::SharedStorage*>(shared_storage);
    SpatialWideResidualKernel{}(params, *storage);
#else
    (void)params;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_leaky_relu_fp16(
    typename LeakyReluKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage = reinterpret_cast<typename LeakyReluKernel::SharedStorage*>(
        shared_storage);
    LeakyReluKernel{}(params, *storage);
#else
    (void)params;
#endif
}

extern "C" __global__ void mlvc_cutlass_pointwise_residual_fp16(
    typename ResidualKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage = reinterpret_cast<typename ResidualKernel::SharedStorage*>(
        shared_storage);
    ResidualKernel{}(params, *storage);
#else
    (void)params;
#endif
}
