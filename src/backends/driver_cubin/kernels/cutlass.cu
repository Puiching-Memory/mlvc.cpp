#include "backends/driver_cubin/cutlass.hpp"

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
using SpatialWideBiasCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        SpatialWideStore, SpatialWideAddBias>;
using SpatialWideBiasLeakyRelu = cutlass::epilogue::threadblock::Sm80EVT<
    ApplyLeakyRelu, SpatialWideAddBias>;
using SpatialWideLeakyReluCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        SpatialWideStore, SpatialWideBiasLeakyRelu>;

using SmallM128ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
using SmallM128WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
using SmallM128ThreadMap =
    cutlass::epilogue::threadblock::OutputTileThreadLayout<
        SmallM128ThreadblockShape, SmallM128WarpShape, Element, kAlignment,
        kEpilogueStages>;
using SmallM128Bias = cutlass::epilogue::threadblock::VisitorColBroadcast<
    SmallM128ThreadMap, Element>;
using SmallM128AddBias = cutlass::epilogue::threadblock::Sm80EVT<
    Add, Accumulator, SmallM128Bias>;
using SmallM128Store = cutlass::epilogue::threadblock::VisitorAuxStore<
    SmallM128ThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, int64_t>>;
using SmallM128BiasCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<SmallM128Store, SmallM128AddBias>;
using SmallM128BiasLeakyRelu = cutlass::epilogue::threadblock::Sm80EVT<
    ApplyLeakyRelu, SmallM128AddBias>;
using SmallM128LeakyReluCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        SmallM128Store, SmallM128BiasLeakyRelu>;
using SmallM128Residual = cutlass::epilogue::threadblock::VisitorAuxLoad<
    SmallM128ThreadMap, Element, cute::Stride<int64_t, cute::_1, int64_t>>;
using SmallM128AddResidual = cutlass::epilogue::threadblock::Sm80EVT<
    Add, SmallM128AddBias, SmallM128Residual>;
using SmallM128ResidualCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        SmallM128Store, SmallM128AddResidual>;

using SmallM64N64ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
using SmallM64N64WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
using SmallM64N64ThreadMap =
    cutlass::epilogue::threadblock::OutputTileThreadLayout<
        SmallM64N64ThreadblockShape, SmallM64N64WarpShape, Element, kAlignment,
        kEpilogueStages>;
using SmallM64N64Bias = cutlass::epilogue::threadblock::VisitorColBroadcast<
    SmallM64N64ThreadMap, Element>;
using SmallM64N64AddBias = cutlass::epilogue::threadblock::Sm80EVT<
    Add, Accumulator, SmallM64N64Bias>;
using SmallM64N64Store = cutlass::epilogue::threadblock::VisitorAuxStore<
    SmallM64N64ThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, int64_t>>;
using SmallM64N64BiasCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<SmallM64N64Store, SmallM64N64AddBias>;
using SmallM64N64BiasLeakyRelu = cutlass::epilogue::threadblock::Sm80EVT<
    ApplyLeakyRelu, SmallM64N64AddBias>;
using SmallM64N64LeakyReluCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        SmallM64N64Store, SmallM64N64BiasLeakyRelu>;
using SmallM64N64Residual = cutlass::epilogue::threadblock::VisitorAuxLoad<
    SmallM64N64ThreadMap, Element, cute::Stride<int64_t, cute::_1, int64_t>>;
using SmallM64N64AddResidual = cutlass::epilogue::threadblock::Sm80EVT<
    Add, SmallM64N64AddBias, SmallM64N64Residual>;
using SmallM64N64ResidualCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        SmallM64N64Store, SmallM64N64AddResidual>;

// Custom EVT store that fuses ReGLU in registers on top of weight-interleaved
// 16-row blocks (rows 0..7 = gate, rows 8..15 = linear). Because
// DefaultThreadMapTensorOp steps 8 rows per epilogue iteration, even steps
// capture FP16-rounded gate values in registers and odd steps multiply
// clamp(gate, min, max) * linear and write only the G output rows to HBM.
template <class ThreadMapT, class ElementType,
          cutlass::FloatRoundStyle RoundStyle, class StrideMNL>
struct VisitorReGLUStore {
    struct Arguments {
        ElementType* ptr_aux = nullptr;
        StrideMNL dAux = {};
        float minimum = 0.0F;
        float maximum = 65504.0F;
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
        ProblemShape const&, Arguments const& args, void*)
    {
        return args;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&)
    {
        return 0;
    }

    struct SharedStorage {};

    static int constexpr vec_bits =
        ThreadMapT::kElementsPerAccess * cutlass::sizeof_bits<ElementType>::value;
    using VecType = cutlass::uint_bit_t<cute::min(128, vec_bits)>;
    static int constexpr VecLength = sizeof(VecType) / sizeof(ElementType);

    CUTLASS_HOST_DEVICE
    VisitorReGLUStore() {}

    CUTLASS_HOST_DEVICE
    VisitorReGLUStore(Params const& params, SharedStorage const&)
        : params_ptr(&params)
    {
    }

    Params const* params_ptr;

    template <class RTensor, class CTensor, class ProblemShape>
    struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
        CUTLASS_DEVICE
        Callbacks(RTensor&& tC_rGate, RTensor&& tC_rAux, CTensor&& tC_cAux,
                  ProblemShape problem_shape, Params const* params_ptr)
            : tC_rGate(cute::forward<RTensor>(tC_rGate)),
              tC_rAux(cute::forward<RTensor>(tC_rAux)),
              tC_cAux(cute::forward<CTensor>(tC_cAux)),
              problem_shape(problem_shape),
              params_ptr(params_ptr)
        {
        }

        RTensor tC_rGate;
        RTensor tC_rAux;
        CTensor tC_cAux;
        ProblemShape problem_shape;
        Params const* params_ptr;

        CUTLASS_DEVICE void begin_step(int step_idx)
        {
            if ((step_idx & 1) == 1)
                cute::clear(tC_rAux);
        }

        template <class ElementAccumulator, class ElementInput, int FragmentSize>
        CUTLASS_DEVICE auto visit(
            int iter_idx, int row_idx, int column_idx, int frg_idx,
            cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc,
            cutlass::Array<ElementInput, FragmentSize> const& frg_input)
        {
            (void)row_idx;
            (void)column_idx;
            (void)frg_acc;
            using ConvertInput = cutlass::NumericArrayConverter<
                ElementType, ElementInput, FragmentSize, RoundStyle>;
            using ConvertScalar = cutlass::NumericConverter<
                ElementType, float, RoundStyle>;
            ConvertInput convert_input{};
            ConvertScalar convert_scalar{};

            auto tC_rGate_frg = cute::recast<
                cutlass::Array<ElementType, FragmentSize>>(
                cute::coalesce(tC_rGate));
            auto tC_rAux_frg = cute::recast<
                cutlass::Array<ElementType, FragmentSize>>(
                cute::coalesce(tC_rAux));

            const cutlass::Array<ElementType, FragmentSize> rounded =
                convert_input(frg_input);
            if ((iter_idx & 1) == 0) {
                tC_rGate_frg(frg_idx) = rounded;
            } else {
                const auto gate_h = tC_rGate_frg(frg_idx);
                cutlass::Array<ElementType, FragmentSize> out;
                const float lo = params_ptr->minimum;
                const float hi = params_ptr->maximum;
                CUTLASS_PRAGMA_UNROLL
                for (int e = 0; e < FragmentSize; ++e) {
                    const float g =
                        fminf(fmaxf(static_cast<float>(gate_h[e]), lo), hi);
                    const float l = static_cast<float>(rounded[e]);
                    out[e] = convert_scalar(g * l);
                }
                tC_rAux_frg(frg_idx) = out;
            }
            return frg_input;
        }

        CUTLASS_DEVICE void end_step(int step_idx)
        {
            if ((step_idx & 1) == 0)
                return;
            auto src_v = cute::filter(tC_rAux);
            auto coord_v = cute::filter(
                tC_cAux(cute::_, cute::_, cute::_, step_idx));
            const int gate_channels = cute::get<0>(problem_shape) >> 1;
            const int spatial_count = cute::get<1>(problem_shape);
            const int64_t row_stride = cute::get<0>(params_ptr->dAux);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(src_v); ++i) {
                const auto coord = coord_v(i);
                const int m_inter = cute::get<0>(coord);
                const int n = cute::get<1>(coord);
                const int out_m = ((m_inter >> 4) << 3) | (m_inter & 7);
                const bool guard =
                    (out_m < gate_channels) && (n < spatial_count);
                ElementType* dst = params_ptr->ptr_aux +
                    static_cast<int64_t>(out_m) * row_stride + n;
                cutlass::arch::global_store<VecType, sizeof(VecType)>(
                    src_v(i), static_cast<void*>(dst), guard);
            }
        }
    };

    template <class ProblemShape>
    CUTLASS_DEVICE auto get_callbacks(
        cutlass::gemm::GemmCoord threadblock_tile_offset, int thread_idx,
        ProblemShape problem_shape)
    {
        using namespace cute;
        Tensor mDummy = make_tensor(
            make_gmem_ptr(params_ptr->ptr_aux), problem_shape,
            params_ptr->dAux);
        Tensor tC_gDummy = recast<VecType>(group_modes<3, 6>(
            ThreadMapT::partition(mDummy, thread_idx, threadblock_tile_offset)));
        Tensor tC_rGate = make_tensor_like(take<0, 3>(tC_gDummy));
        Tensor tC_rAux = make_tensor_like(take<0, 3>(tC_gDummy));

        Tensor cAux = make_identity_tensor(mDummy.shape());
        Tensor tC_cAux = outer_partition(
            group_modes<3, 6>(ThreadMapT::partition(
                cAux, thread_idx, threadblock_tile_offset)),
            Shape<Int<VecLength>>{}, (_0{}));

        return Callbacks<decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
            cute::move(tC_rGate), cute::move(tC_rAux), cute::move(tC_cAux),
            problem_shape, params_ptr);
    }
};

using ReGLUStore = VisitorReGLUStore<
    ThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, int64_t>>;
using ReGLUCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<ReGLUStore, AddBias>;

using MediumReGLUStore = VisitorReGLUStore<
    MediumThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, int64_t>>;
using MediumReGLUCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<MediumReGLUStore, MediumAddBias>;

using SmallM64N64ReGLUStore = VisitorReGLUStore<
    SmallM64N64ThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, int64_t>>;
using SmallM64N64ReGLUCallbacks =
    cutlass::epilogue::threadblock::Sm80EVT<
        SmallM64N64ReGLUStore, SmallM64N64AddBias>;

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
using SpatialWideBiasKernel = PointwiseKernel<
    SpatialWideBiasCallbacks, SpatialWideThreadblockShape,
    SpatialWideWarpShape>;
using SpatialWideLeakyReluKernel = PointwiseKernel<
    SpatialWideLeakyReluCallbacks, SpatialWideThreadblockShape,
    SpatialWideWarpShape>;
using SmallM128BiasKernel = PointwiseKernel<
    SmallM128BiasCallbacks, SmallM128ThreadblockShape, SmallM128WarpShape>;
using SmallM128LeakyReluKernel = PointwiseKernel<
    SmallM128LeakyReluCallbacks, SmallM128ThreadblockShape, SmallM128WarpShape>;
using SmallM128ResidualKernel = PointwiseKernel<
    SmallM128ResidualCallbacks, SmallM128ThreadblockShape, SmallM128WarpShape>;
using SmallM64N64BiasKernel = PointwiseKernel<
    SmallM64N64BiasCallbacks, SmallM64N64ThreadblockShape, SmallM64N64WarpShape>;
using SmallM64N64LeakyReluKernel = PointwiseKernel<
    SmallM64N64LeakyReluCallbacks, SmallM64N64ThreadblockShape,
    SmallM64N64WarpShape>;
using SmallM64N64ResidualKernel = PointwiseKernel<
    SmallM64N64ResidualCallbacks, SmallM64N64ThreadblockShape,
    SmallM64N64WarpShape>;
using ReGLUKernel = PointwiseKernel<ReGLUCallbacks>;
using MediumReGLUKernel = PointwiseKernel<
    MediumReGLUCallbacks, MediumThreadblockShape, MediumWarpShape>;
using MediumStage4ReGLUKernel = PointwiseKernel<
    MediumReGLUCallbacks, MediumThreadblockShape, MediumWarpShape, 4>;
using SmallM64N64ReGLUKernel = PointwiseKernel<
    SmallM64N64ReGLUCallbacks, SmallM64N64ThreadblockShape,
    SmallM64N64WarpShape>;

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

using SpatialConvSmallKernel =
    typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>, InstructionShape,
        SpatialConvOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

template <typename Kernel, typename OutputOp>
__device__ __forceinline__ void initialize_pointwise(
    void* params_storage, const Element* input, const Element* weight,
    Element* output, int out_channels, int spatial_count, int in_channels,
    OutputOp output_op, int tile_rows = 128, int tile_columns = 64,
    int log_tile = 0)
{
    auto* params = static_cast<typename Kernel::Params*>(params_storage);
    params->problem_size = {out_channels, spatial_count, in_channels};
    params->grid_tiled_shape = {
        (out_channels + tile_rows - 1) / tile_rows,
        (spatial_count + tile_columns - 1) / tile_columns, 1};
    params->swizzle_log_tile = log_tile;
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
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename LeakyReluKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename ResidualKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumBiasKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumStage4BiasKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumLeakyReluKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumResidualKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SpatialWideResidualKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SpatialWideBiasKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SpatialWideLeakyReluKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SmallM128BiasKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SmallM128LeakyReluKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SmallM128ResidualKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SmallM64N64BiasKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SmallM64N64LeakyReluKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SmallM64N64ResidualKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename ReGLUKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumReGLUKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename MediumStage4ReGLUKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SmallM64N64ReGLUKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SpatialConvKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename SpatialConvSmallKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename BiasKernel::SharedStorage) == 36864);
static_assert(sizeof(typename LeakyReluKernel::SharedStorage) == 36864);
static_assert(sizeof(typename ResidualKernel::SharedStorage) == 36864);
static_assert(sizeof(typename MediumBiasKernel::SharedStorage) == 49152);
static_assert(sizeof(typename MediumStage4BiasKernel::SharedStorage) == 65536);
static_assert(sizeof(typename MediumLeakyReluKernel::SharedStorage) == 49152);
static_assert(sizeof(typename MediumResidualKernel::SharedStorage) == 49152);
static_assert(sizeof(typename SpatialWideResidualKernel::SharedStorage) == 61440);
static_assert(sizeof(typename SpatialWideBiasKernel::SharedStorage) == 61440);
static_assert(sizeof(typename SpatialWideLeakyReluKernel::SharedStorage) == 61440);
static_assert(sizeof(typename SmallM128BiasKernel::SharedStorage) <= 36864);
static_assert(sizeof(typename SmallM128LeakyReluKernel::SharedStorage) <= 36864);
static_assert(sizeof(typename SmallM128ResidualKernel::SharedStorage) <= 36864);
static_assert(sizeof(typename SmallM64N64BiasKernel::SharedStorage) <= 24576);
static_assert(sizeof(typename SmallM64N64LeakyReluKernel::SharedStorage) <= 24576);
static_assert(sizeof(typename SmallM64N64ResidualKernel::SharedStorage) <= 24576);
static_assert(sizeof(typename ReGLUKernel::SharedStorage) == 36864);
static_assert(sizeof(typename MediumReGLUKernel::SharedStorage) == 49152);
static_assert(sizeof(typename MediumStage4ReGLUKernel::SharedStorage) == 65536);
static_assert(sizeof(typename SmallM64N64ReGLUKernel::SharedStorage) <= 24576);
static_assert(sizeof(typename SpatialConvKernel::SharedStorage) == 73728);
static_assert(SpatialConvSmallKernel::kThreadCount == 128);
static_assert(sizeof(typename SpatialConvSmallKernel::SharedStorage) <= 32768);

template <typename Kernel, typename CallbacksParams>
__device__ __forceinline__ void pointwise_init_body(
    void* params_storage, const Element* input, const Element* weight,
    Element* output, int out_channels, int spatial_count, int in_channels,
    const CallbacksParams& callbacks_params, int tile_rows, int tile_columns,
    int log_tile)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    initialize_pointwise<Kernel>(
        params_storage, input, weight, output, out_channels, spatial_count,
        in_channels, callbacks_params, tile_rows, tile_columns, log_tile);
#else
    (void)params_storage;
    (void)input;
    (void)weight;
    (void)output;
    (void)out_channels;
    (void)spatial_count;
    (void)in_channels;
    (void)callbacks_params;
    (void)tile_rows;
    (void)tile_columns;
    (void)log_tile;
#endif
}

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

extern "C" __global__ void mlvc_cutlass_spatial_conv_small_init_fp16(
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
    typename SpatialConvSmallKernel::Arguments arguments{
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
    *static_cast<typename SpatialConvSmallKernel::Params*>(params_storage) =
        typename SpatialConvSmallKernel::Params{arguments};
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

extern "C" __global__ void mlvc_cutlass_spatial_conv_small_fp16(
    typename SpatialConvSmallKernel::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* storage = reinterpret_cast<
        typename SpatialConvSmallKernel::SharedStorage*>(shared_storage);
    SpatialConvSmallKernel{}(params, *storage);
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
            {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride}},
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
            {output, stride}},
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

#define MLVC_POINTWISE_INIT(ExportName, KernelType, CallbacksType,           \
                            TileRows, TileColumns, ...)                        \
extern "C" __global__ void ExportName(                                         \
    void* params_storage, const Element* input, const Element* weight,         \
    const Element* bias, const Element* residual, Element* output,             \
    int out_channels, int spatial_count, int in_channels, int log_tile)        \
{                                                                              \
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{              \
        int64_t(spatial_count), cute::_1{},                                    \
        int64_t(out_channels) * spatial_count};                                \
    pointwise_init_body<KernelType>(                                           \
        params_storage, input, weight, output, out_channels, spatial_count,    \
        in_channels, typename CallbacksType::Params{__VA_ARGS__}, TileRows,    \
        TileColumns, log_tile);                                                \
}

// L2-swizzle variants reuse the base kernels; only the parameter
// initialization differs (nonzero swizzle_log_tile).
MLVC_POINTWISE_INIT(mlvc_cutlass_pointwise_swizzle_init_fp16,
                    BiasKernel, BiasCallbacks, 128, 64,
                    {{}, {bias, Element(0), {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(mlvc_cutlass_pointwise_leaky_relu_swizzle_init_fp16,
                    LeakyReluKernel, LeakyReluCallbacks, 128, 64,
                    {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(mlvc_cutlass_pointwise_residual_swizzle_init_fp16,
                    ResidualKernel, ResidualCallbacks, 128, 64,
                    {{{}, {bias, Element(0), {}}, {}},
                     {const_cast<Element*>(residual), Element(0), stride}, {}},
                    {output, stride})
MLVC_POINTWISE_INIT(mlvc_cutlass_pointwise_medium_swizzle_init_fp16,
                    MediumBiasKernel, MediumBiasCallbacks, 128, 128,
                    {{}, {bias, Element(0), {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(mlvc_cutlass_pointwise_medium_leaky_relu_swizzle_init_fp16,
                    MediumLeakyReluKernel, MediumLeakyReluCallbacks, 128, 128,
                    {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(mlvc_cutlass_pointwise_medium_residual_swizzle_init_fp16,
                    MediumResidualKernel, MediumResidualCallbacks, 128, 128,
                    {{{}, {bias, Element(0), {}}, {}},
                     {const_cast<Element*>(residual), Element(0), stride}, {}},
                    {output, stride})
MLVC_POINTWISE_INIT(
    mlvc_cutlass_pointwise_spatial_wide_residual_swizzle_init_fp16,
    SpatialWideResidualKernel, SpatialWideResidualCallbacks, 64, 256,
    {{{}, {bias, Element(0), {}}, {}},
     {const_cast<Element*>(residual), Element(0), stride}, {}},
    {output, stride})

#define MLVC_POINTWISE_BASE_INIT(ExportName, KernelType, CallbacksType,        \
                                 TileRows, TileColumns, ...)                   \
extern "C" __global__ void ExportName(                                         \
    void* params_storage, const Element* input, const Element* weight,         \
    const Element* bias, const Element* residual, Element* output,             \
    int out_channels, int spatial_count, int in_channels)                      \
{                                                                              \
    (void)residual;                                                            \
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{              \
        int64_t(spatial_count), cute::_1{},                                    \
        int64_t(out_channels) * spatial_count};                                \
    pointwise_init_body<KernelType>(                                           \
        params_storage, input, weight, output, out_channels, spatial_count,    \
        in_channels, typename CallbacksType::Params{__VA_ARGS__}, TileRows,    \
        TileColumns, 0);                                                       \
}

#define MLVC_POINTWISE_MAIN(ExportName, KernelType)                            \
extern "C" __global__ void ExportName(typename KernelType::Params params)      \
{                                                                              \
    __if_sm80_run_kernel<KernelType>(params);                                  \
}

template <typename KernelType>
__device__ __forceinline__ void __if_sm80_run_kernel(
    typename KernelType::Params params)
{
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char mlvc_cutlass_dynamic_smem[];
    auto* storage = reinterpret_cast<
        typename KernelType::SharedStorage*>(mlvc_cutlass_dynamic_smem);
    KernelType{}(params, *storage);
#else
    (void)params;
#endif
}

MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_spatial_wide_fp16,
                    SpatialWideBiasKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_spatial_wide_leaky_relu_fp16,
                    SpatialWideLeakyReluKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_m64n128_fp16, SmallM128BiasKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_m64n128_leaky_relu_fp16,
                    SmallM128LeakyReluKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_m64n128_residual_fp16,
                    SmallM128ResidualKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_m64n64_fp16, SmallM64N64BiasKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_m64n64_leaky_relu_fp16,
                    SmallM64N64LeakyReluKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_m64n64_residual_fp16,
                    SmallM64N64ResidualKernel)

MLVC_POINTWISE_BASE_INIT(mlvc_cutlass_pointwise_spatial_wide_init_fp16,
                         SpatialWideBiasKernel, SpatialWideBiasCallbacks,
                         64, 256,
                         {{}, {bias, Element(0), {}}, {}}, {output, stride})
MLVC_POINTWISE_BASE_INIT(
    mlvc_cutlass_pointwise_spatial_wide_leaky_relu_init_fp16,
    SpatialWideLeakyReluKernel, SpatialWideLeakyReluCallbacks, 64, 256,
    {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride})
MLVC_POINTWISE_BASE_INIT(mlvc_cutlass_pointwise_m64n128_init_fp16,
                         SmallM128BiasKernel, SmallM128BiasCallbacks, 64, 128,
                         {{}, {bias, Element(0), {}}, {}}, {output, stride})
MLVC_POINTWISE_BASE_INIT(mlvc_cutlass_pointwise_m64n128_leaky_relu_init_fp16,
                         SmallM128LeakyReluKernel, SmallM128LeakyReluCallbacks,
                         64, 128,
                         {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride})
MLVC_POINTWISE_BASE_INIT(
    mlvc_cutlass_pointwise_m64n128_residual_init_fp16,
    SmallM128ResidualKernel, SmallM128ResidualCallbacks, 64, 128,
    {{{}, {bias, Element(0), {}}, {}},
     {const_cast<Element*>(residual), Element(0), stride}, {}},
    {output, stride})
MLVC_POINTWISE_BASE_INIT(mlvc_cutlass_pointwise_m64n64_init_fp16,
                         SmallM64N64BiasKernel, SmallM64N64BiasCallbacks,
                         64, 64,
                         {{}, {bias, Element(0), {}}, {}}, {output, stride})
MLVC_POINTWISE_BASE_INIT(mlvc_cutlass_pointwise_m64n64_leaky_relu_init_fp16,
                         SmallM64N64LeakyReluKernel,
                         SmallM64N64LeakyReluCallbacks, 64, 64,
                         {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride})
MLVC_POINTWISE_BASE_INIT(
    mlvc_cutlass_pointwise_m64n64_residual_init_fp16,
    SmallM64N64ResidualKernel, SmallM64N64ResidualCallbacks, 64, 64,
    {{{}, {bias, Element(0), {}}, {}},
     {const_cast<Element*>(residual), Element(0), stride}, {}},
    {output, stride})

MLVC_POINTWISE_INIT(mlvc_cutlass_pointwise_spatial_wide_swizzle_init_fp16,
                    SpatialWideBiasKernel, SpatialWideBiasCallbacks, 64, 256,
                    {{}, {bias, Element(0), {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(
    mlvc_cutlass_pointwise_spatial_wide_leaky_relu_swizzle_init_fp16,
    SpatialWideLeakyReluKernel, SpatialWideLeakyReluCallbacks, 64, 256,
    {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(mlvc_cutlass_pointwise_m64n128_swizzle_init_fp16,
                    SmallM128BiasKernel, SmallM128BiasCallbacks, 64, 128,
                    {{}, {bias, Element(0), {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(
    mlvc_cutlass_pointwise_m64n128_leaky_relu_swizzle_init_fp16,
    SmallM128LeakyReluKernel, SmallM128LeakyReluCallbacks, 64, 128,
    {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(
    mlvc_cutlass_pointwise_m64n128_residual_swizzle_init_fp16,
    SmallM128ResidualKernel, SmallM128ResidualCallbacks, 64, 128,
    {{{}, {bias, Element(0), {}}, {}},
     {const_cast<Element*>(residual), Element(0), stride}, {}},
    {output, stride})
MLVC_POINTWISE_INIT(mlvc_cutlass_pointwise_m64n64_swizzle_init_fp16,
                    SmallM64N64BiasKernel, SmallM64N64BiasCallbacks, 64, 64,
                    {{}, {bias, Element(0), {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(
    mlvc_cutlass_pointwise_m64n64_leaky_relu_swizzle_init_fp16,
    SmallM64N64LeakyReluKernel, SmallM64N64LeakyReluCallbacks, 64, 64,
    {{{}, {bias, Element(0), {}}, {}}, {}}, {output, stride})
MLVC_POINTWISE_INIT(
    mlvc_cutlass_pointwise_m64n64_residual_swizzle_init_fp16,
    SmallM64N64ResidualKernel, SmallM64N64ResidualCallbacks, 64, 64,
    {{{}, {bias, Element(0), {}}, {}},
     {const_cast<Element*>(residual), Element(0), stride}, {}},
    {output, stride})

#define MLVC_POINTWISE_REGLU_INIT(ExportName, KernelType, CallbacksType,       \
                                  TileRows, TileColumns)                       \
extern "C" __global__ void ExportName(                                         \
    void* params_storage, const Element* input, const Element* weight,         \
    const Element* bias, Element* output, int out_channels, int spatial_count, \
    int in_channels, float minimum, float maximum, int log_tile)               \
{                                                                              \
    const auto stride = cute::Stride<int64_t, cute::_1, int64_t>{              \
        int64_t(spatial_count), cute::_1{},                                    \
        int64_t(out_channels >> 1) * spatial_count};                           \
    pointwise_init_body<KernelType>(                                           \
        params_storage, input, weight, output, out_channels, spatial_count,    \
        in_channels,                                                           \
        typename CallbacksType::Params{                                        \
            {{}, {bias, Element(0), {}}, {}},                                  \
            {output, stride, minimum, maximum}},                               \
        TileRows, TileColumns, log_tile);                                      \
}

MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_reglu_fp16, ReGLUKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_medium_reglu_fp16, MediumReGLUKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_medium_stage4_reglu_fp16,
                    MediumStage4ReGLUKernel)
MLVC_POINTWISE_MAIN(mlvc_cutlass_pointwise_m64n64_reglu_fp16,
                    SmallM64N64ReGLUKernel)

MLVC_POINTWISE_REGLU_INIT(mlvc_cutlass_pointwise_reglu_init_fp16,
                          ReGLUKernel, ReGLUCallbacks, 128, 64)
MLVC_POINTWISE_REGLU_INIT(mlvc_cutlass_pointwise_medium_reglu_init_fp16,
                          MediumReGLUKernel, MediumReGLUCallbacks, 128, 128)
MLVC_POINTWISE_REGLU_INIT(mlvc_cutlass_pointwise_medium_stage4_reglu_init_fp16,
                          MediumStage4ReGLUKernel, MediumReGLUCallbacks,
                          128, 128)
MLVC_POINTWISE_REGLU_INIT(mlvc_cutlass_pointwise_m64n64_reglu_init_fp16,
                          SmallM64N64ReGLUKernel, SmallM64N64ReGLUCallbacks,
                          64, 64)

#undef MLVC_POINTWISE_REGLU_INIT
#undef MLVC_POINTWISE_MAIN
#undef MLVC_POINTWISE_BASE_INIT
#undef MLVC_POINTWISE_INIT

// -----------------------------------------------------------------------------
// Global NHWC Pointwise GEMMs: D[SP, OC] = X[SP, IC] * W^T[IC, OC]
//   A = X (RowMajor, lda = input_stride)
//   B = W^T (ColumnMajor, ldb = in_channels; identical in memory to W[OC, IC])
//   Bias = VisitorRowBroadcast (128-bit coalesced vector load along OC)
//   Residual = VisitorAuxLoad (RowMajor, ldc = residual_stride)
//   Output = VisitorAuxStore / VisitorReGLUStoreNHWC (RowMajor, ldd = out_stride)
// -----------------------------------------------------------------------------
namespace {

template <class ThreadMapT, class ElementType,
          cutlass::FloatRoundStyle RoundStyle, class StrideMNL>
struct VisitorReGLUStoreNHWC {
    struct Arguments {
        ElementType* ptr_aux = nullptr;
        StrideMNL dAux = {};
        float minimum = 0.0F;
        float maximum = 65504.0F;
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
        ProblemShape const&, Arguments const& args, void*)
    {
        return args;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&)
    {
        return 0;
    }

    struct SharedStorage {};

    static int constexpr vec_bits =
        ThreadMapT::kElementsPerAccess * cutlass::sizeof_bits<ElementType>::value;
    using VecType = cutlass::uint_bit_t<cute::min(128, vec_bits)>;
    using OutVecType = cutlass::uint_bit_t<cute::min(128, vec_bits) / 2>;
    static int constexpr VecLength = sizeof(VecType) / sizeof(ElementType);

    CUTLASS_HOST_DEVICE
    VisitorReGLUStoreNHWC() {}

    CUTLASS_HOST_DEVICE
    VisitorReGLUStoreNHWC(Params const& params, SharedStorage const&)
        : params_ptr(&params)
    {
    }

    Params const* params_ptr;

    template <class RTensor, class CTensor, class ProblemShape>
    struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
        CUTLASS_DEVICE
        Callbacks(RTensor&& tC_rAux, CTensor&& tC_cAux,
                  ProblemShape problem_shape, Params const* params_ptr)
            : tC_rAux(cute::forward<RTensor>(tC_rAux)),
              tC_cAux(cute::forward<CTensor>(tC_cAux)),
              problem_shape(problem_shape),
              params_ptr(params_ptr)
        {
        }

        RTensor tC_rAux;
        CTensor tC_cAux;
        ProblemShape problem_shape;
        Params const* params_ptr;

        CUTLASS_DEVICE void begin_step(int) { cute::clear(tC_rAux); }

        template <class ElementAccumulator, class ElementInput, int FragmentSize>
        CUTLASS_DEVICE auto visit(
            int, int, int, int frg_idx,
            cutlass::Array<ElementAccumulator, FragmentSize> const&,
            cutlass::Array<ElementInput, FragmentSize> const& frg_input)
        {
            using ConvertInput = cutlass::NumericArrayConverter<
                ElementType, ElementInput, FragmentSize, RoundStyle>;
            using ConvertScalar = cutlass::NumericConverter<
                ElementType, float, RoundStyle>;
            ConvertInput convert_input{};
            ConvertScalar convert_scalar{};

            const cutlass::Array<ElementType, FragmentSize> rounded =
                convert_input(frg_input);
            cutlass::Array<ElementType, FragmentSize / 2> out;
            const float lo = params_ptr->minimum;
            const float hi = params_ptr->maximum;
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < FragmentSize / 2; ++k) {
                const float g =
                    fminf(fmaxf(static_cast<float>(rounded[2 * k]), lo), hi);
                const float l = static_cast<float>(rounded[2 * k + 1]);
                out[k] = convert_scalar(g * l);
            }
            auto tC_rAux_frg = cute::recast<
                cutlass::Array<ElementType, FragmentSize / 2>>(
                cute::coalesce(tC_rAux));
            tC_rAux_frg(frg_idx) = out;
            return frg_input;
        }

        CUTLASS_DEVICE void end_step(int step_idx)
        {
            auto src_v = cute::filter(tC_rAux);
            auto coord_v = cute::filter(
                tC_cAux(cute::_, cute::_, cute::_, step_idx));
            const int spatial_count = cute::get<0>(problem_shape);
            const int gate_channels = cute::get<1>(problem_shape) >> 1;
            const int64_t row_stride = cute::get<0>(params_ptr->dAux);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(src_v); ++i) {
                const auto coord = coord_v(i);
                const int m = cute::get<0>(coord);
                const int n_out = cute::get<1>(coord) >> 1;
                const bool guard = (m < spatial_count) && (n_out < gate_channels);
                ElementType* dst = params_ptr->ptr_aux +
                    static_cast<int64_t>(m) * row_stride + n_out;
                cutlass::arch::global_store<OutVecType, sizeof(OutVecType)>(
                    src_v(i), static_cast<void*>(dst), guard);
            }
        }
    };

    template <class ProblemShape>
    CUTLASS_DEVICE auto get_callbacks(
        cutlass::gemm::GemmCoord threadblock_tile_offset, int thread_idx,
        ProblemShape problem_shape)
    {
        using namespace cute;
        Tensor mDummy = make_tensor(
            make_gmem_ptr(params_ptr->ptr_aux), problem_shape,
            params_ptr->dAux);
        Tensor tC_gDummy = recast<VecType>(group_modes<3, 6>(
            ThreadMapT::partition(mDummy, thread_idx, threadblock_tile_offset)));
        Tensor tC_rAux = make_tensor<OutVecType>(shape(take<0, 3>(tC_gDummy)));

        Tensor cAux = make_identity_tensor(mDummy.shape());
        Tensor tC_cAux = outer_partition(
            group_modes<3, 6>(ThreadMapT::partition(
                cAux, thread_idx, threadblock_tile_offset)),
            Shape<Int<VecLength>>{}, (_0{}));

        return Callbacks<decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
            cute::move(tC_rAux), cute::move(tC_cAux), problem_shape, params_ptr);
    }
};

template <typename ThreadMapT>
struct NhwcEpilogueTypes {
    using RowBias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMapT, Element, cute::Stride<cute::_0, cute::_1, int64_t>>;
    using AddBias = cutlass::epilogue::threadblock::Sm80EVT<
        Add, Accumulator, RowBias>;
    using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMapT, Element, cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, cute::_1, int64_t>>;
    using BiasCallbacks =
        cutlass::epilogue::threadblock::Sm80EVT<Store, AddBias>;
    using BiasLeakyRelu =
        cutlass::epilogue::threadblock::Sm80EVT<ApplyLeakyRelu, AddBias>;
    using LeakyReluCallbacks =
        cutlass::epilogue::threadblock::Sm80EVT<Store, BiasLeakyRelu>;
    using Residual = cutlass::epilogue::threadblock::VisitorAuxLoad<
        ThreadMapT, Element, cute::Stride<int64_t, cute::_1, int64_t>>;
    using AddResidual =
        cutlass::epilogue::threadblock::Sm80EVT<Add, AddBias, Residual>;
    using ResidualCallbacks =
        cutlass::epilogue::threadblock::Sm80EVT<Store, AddResidual>;
    using ReGLUStore = VisitorReGLUStoreNHWC<
        ThreadMapT, Element, cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, cute::_1, int64_t>>;
    using ReGLUCallbacks =
        cutlass::epilogue::threadblock::Sm80EVT<ReGLUStore, AddBias>;
};

template <typename Callbacks, typename TileShape, typename TileWarpShape,
          int Stages = 3>
using PointwiseNhwcKernel =
    typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
        kAlignment,
        Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone,
        kAlignment,
        Element, cutlass::layout::RowMajor, kAlignment,
        float, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        TileShape, TileWarpShape, InstructionShape, Callbacks,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
        cutlass::arch::OpMultiplyAdd, kEpilogueStages>::GemmKernel;

using NhwcM128N64Ep = NhwcEpilogueTypes<ThreadMap>;
using NhwcM128N128Ep = NhwcEpilogueTypes<MediumThreadMap>;
using NhwcM64N128Ep = NhwcEpilogueTypes<SmallM128ThreadMap>;
using NhwcM64N64Ep = NhwcEpilogueTypes<SmallM64N64ThreadMap>;

using NhwcM128N64BiasKernel = PointwiseNhwcKernel<
    NhwcM128N64Ep::BiasCallbacks, ThreadblockShape, WarpShape, 3>;
using NhwcM128N64LeakyReluKernel = PointwiseNhwcKernel<
    NhwcM128N64Ep::LeakyReluCallbacks, ThreadblockShape, WarpShape, 3>;
using NhwcM128N64ResidualKernel = PointwiseNhwcKernel<
    NhwcM128N64Ep::ResidualCallbacks, ThreadblockShape, WarpShape, 3>;

using NhwcM128N128BiasKernel = PointwiseNhwcKernel<
    NhwcM128N128Ep::BiasCallbacks, MediumThreadblockShape, MediumWarpShape, 3>;
using NhwcM128N128Stage4BiasKernel = PointwiseNhwcKernel<
    NhwcM128N128Ep::BiasCallbacks, MediumThreadblockShape, MediumWarpShape, 4>;
using NhwcM128N128LeakyReluKernel = PointwiseNhwcKernel<
    NhwcM128N128Ep::LeakyReluCallbacks, MediumThreadblockShape, MediumWarpShape, 3>;
using NhwcM128N128ResidualKernel = PointwiseNhwcKernel<
    NhwcM128N128Ep::ResidualCallbacks, MediumThreadblockShape, MediumWarpShape, 3>;
using NhwcM128N128ReGLUKernel = PointwiseNhwcKernel<
    NhwcM128N128Ep::ReGLUCallbacks, MediumThreadblockShape, MediumWarpShape, 3>;
using NhwcM128N128Stage4ReGLUKernel = PointwiseNhwcKernel<
    NhwcM128N128Ep::ReGLUCallbacks, MediumThreadblockShape, MediumWarpShape, 4>;

using NhwcM64N128BiasKernel = PointwiseNhwcKernel<
    NhwcM64N128Ep::BiasCallbacks, SmallM128ThreadblockShape, SmallM128WarpShape, 3>;
using NhwcM64N128LeakyReluKernel = PointwiseNhwcKernel<
    NhwcM64N128Ep::LeakyReluCallbacks, SmallM128ThreadblockShape, SmallM128WarpShape, 3>;
using NhwcM64N128ResidualKernel = PointwiseNhwcKernel<
    NhwcM64N128Ep::ResidualCallbacks, SmallM128ThreadblockShape, SmallM128WarpShape, 3>;
using NhwcM64N128ReGLUKernel = PointwiseNhwcKernel<
    NhwcM64N128Ep::ReGLUCallbacks, SmallM128ThreadblockShape, SmallM128WarpShape, 3>;

using NhwcM64N64BiasKernel = PointwiseNhwcKernel<
    NhwcM64N64Ep::BiasCallbacks, SmallM64N64ThreadblockShape, SmallM64N64WarpShape, 3>;
using NhwcM64N64LeakyReluKernel = PointwiseNhwcKernel<
    NhwcM64N64Ep::LeakyReluCallbacks, SmallM64N64ThreadblockShape, SmallM64N64WarpShape, 3>;
using NhwcM64N64ResidualKernel = PointwiseNhwcKernel<
    NhwcM64N64Ep::ResidualCallbacks, SmallM64N64ThreadblockShape, SmallM64N64WarpShape, 3>;
using NhwcM64N64ReGLUKernel = PointwiseNhwcKernel<
    NhwcM64N64Ep::ReGLUCallbacks, SmallM64N64ThreadblockShape, SmallM64N64WarpShape, 3>;

static_assert(sizeof(typename NhwcM128N64BiasKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename NhwcM128N64ResidualKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename NhwcM128N128Stage4ReGLUKernel::Params) <=
              mlvc::driver_cubin::kCutlassPointwiseParamsStorageBytes);
static_assert(sizeof(typename NhwcM128N64BiasKernel::SharedStorage) <= 36864);
static_assert(sizeof(typename NhwcM128N128BiasKernel::SharedStorage) <= 49152);
static_assert(sizeof(typename NhwcM128N128Stage4BiasKernel::SharedStorage) <= 65536);
static_assert(sizeof(typename NhwcM64N128BiasKernel::SharedStorage) <= 36864);
static_assert(sizeof(typename NhwcM64N64BiasKernel::SharedStorage) <= 24576);

template <typename Kernel, typename OutputOp>
__device__ __forceinline__ void initialize_pointwise_nhwc(
    void* params_storage, const Element* input, const Element* weight,
    Element* output, int spatial_count, int out_channels, int in_channels,
    int input_stride, OutputOp output_op, int tile_m, int tile_n, int log_tile)
{
    (void)output;
    auto* params = static_cast<typename Kernel::Params*>(params_storage);
    params->problem_size = {spatial_count, out_channels, in_channels};
    params->grid_tiled_shape = {
        (spatial_count + tile_m - 1) / tile_m,
        (out_channels + tile_n - 1) / tile_n, 1};
    params->swizzle_log_tile = log_tile;
    params->mode = cutlass::gemm::GemmUniversalMode::kGemm;
    params->batch_count = 1;
    params->gemm_k_size = in_channels;
    params->batch_stride_D = 0;
    params->semaphore = nullptr;
    params->problem_shape = cute::make_shape(
        int32_t(spatial_count), int32_t(out_channels), int32_t(1));
    params->params_A = decltype(params->params_A){
        cutlass::layout::RowMajor{input_stride}};
    params->params_B = decltype(params->params_B){
        cutlass::layout::ColumnMajor{in_channels}};
    params->output_op = output_op;
    params->ptr_A = const_cast<Element*>(input);
    params->ptr_B = const_cast<Element*>(weight);
    params->batch_stride_A = int64_t(spatial_count) * input_stride;
    params->batch_stride_B = int64_t(out_channels) * in_channels;
    params->ptr_gather_A_indices = nullptr;
    params->ptr_gather_B_indices = nullptr;
}

template <typename Kernel, typename CallbacksParams>
__device__ __forceinline__ void pointwise_nhwc_init_body(
    void* params_storage, const Element* input, const Element* weight,
    Element* output, int spatial_count, int out_channels, int in_channels,
    int input_stride, const CallbacksParams& callbacks_params,
    int tile_m, int tile_n, int log_tile)
{
#if __CUDA_ARCH__ >= 800
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    initialize_pointwise_nhwc<Kernel>(
        params_storage, input, weight, output, spatial_count, out_channels,
        in_channels, input_stride, callbacks_params, tile_m, tile_n, log_tile);
#else
    (void)params_storage; (void)input; (void)weight; (void)output;
    (void)spatial_count; (void)out_channels; (void)in_channels;
    (void)input_stride; (void)callbacks_params;
    (void)tile_m; (void)tile_n; (void)log_tile;
#endif
}

}  // namespace

#define MLVC_NHWC_MAIN(ExportName, KernelType)                                 \
extern "C" __global__ void ExportName(typename KernelType::Params params)      \
{                                                                              \
    __if_sm80_run_kernel<KernelType>(params);                                  \
}

#define MLVC_NHWC_INIT(ExportName, KernelType, CallbacksType, TileM, TileN,    \
                       ...)                                                    \
extern "C" __global__ void ExportName(                                         \
    void* params_storage, const Element* input, const Element* weight,         \
    const Element* bias, const Element* residual, Element* output,             \
    int spatial_count, int out_channels, int in_channels,                      \
    int input_stride, int residual_stride, int output_stride, int log_tile)    \
{                                                                              \
    (void)residual; (void)residual_stride;                                     \
    const auto row_stride = cute::Stride<cute::_0, cute::_1, int64_t>{         \
        cute::_0{}, cute::_1{}, int64_t(out_channels)};                        \
    const auto res_stride = cute::Stride<int64_t, cute::_1, int64_t>{          \
        int64_t(residual_stride), cute::_1{},                                  \
        int64_t(spatial_count) * residual_stride};                             \
    const auto out_stride = cute::Stride<int64_t, cute::_1, int64_t>{          \
        int64_t(output_stride), cute::_1{},                                    \
        int64_t(spatial_count) * output_stride};                               \
    (void)res_stride;                                                          \
    pointwise_nhwc_init_body<KernelType>(                                      \
        params_storage, input, weight, output, spatial_count, out_channels,    \
        in_channels, input_stride,                                             \
        typename CallbacksType::Params{__VA_ARGS__}, TileM, TileN, log_tile);  \
}

#define MLVC_NHWC_REGLU_INIT(ExportName, KernelType, CallbacksType, TileM,     \
                             TileN)                                            \
extern "C" __global__ void ExportName(                                         \
    void* params_storage, const Element* input, const Element* weight,         \
    const Element* bias, Element* output, int spatial_count, int out_channels, \
    int in_channels, int input_stride, int output_stride,                      \
    float minimum, float maximum, int log_tile)                                \
{                                                                              \
    const auto row_stride = cute::Stride<cute::_0, cute::_1, int64_t>{         \
        cute::_0{}, cute::_1{}, int64_t(out_channels)};                        \
    const auto out_stride = cute::Stride<int64_t, cute::_1, int64_t>{          \
        int64_t(output_stride), cute::_1{},                                    \
        int64_t(spatial_count) * output_stride};                               \
    pointwise_nhwc_init_body<KernelType>(                                      \
        params_storage, input, weight, output, spatial_count, out_channels,    \
        in_channels, input_stride,                                             \
        typename CallbacksType::Params{                                        \
            {{}, {bias, Element(0), row_stride}, {}},                          \
            {output, out_stride, minimum, maximum}},                           \
        TileM, TileN, log_tile);                                               \
}

MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m128n64_fp16, NhwcM128N64BiasKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m128n64_leaky_relu_fp16, NhwcM128N64LeakyReluKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m128n64_residual_fp16, NhwcM128N64ResidualKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m128n128_fp16, NhwcM128N128BiasKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m128n128_stage4_fp16, NhwcM128N128Stage4BiasKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m128n128_leaky_relu_fp16, NhwcM128N128LeakyReluKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m128n128_residual_fp16, NhwcM128N128ResidualKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m128n128_reglu_fp16, NhwcM128N128ReGLUKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m128n128_stage4_reglu_fp16, NhwcM128N128Stage4ReGLUKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m64n128_fp16, NhwcM64N128BiasKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m64n128_leaky_relu_fp16, NhwcM64N128LeakyReluKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m64n128_residual_fp16, NhwcM64N128ResidualKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m64n128_reglu_fp16, NhwcM64N128ReGLUKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m64n64_fp16, NhwcM64N64BiasKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m64n64_leaky_relu_fp16, NhwcM64N64LeakyReluKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m64n64_residual_fp16, NhwcM64N64ResidualKernel)
MLVC_NHWC_MAIN(mlvc_cutlass_nhwc_m64n64_reglu_fp16, NhwcM64N64ReGLUKernel)

MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m128n64_init_fp16, NhwcM128N64BiasKernel,
               NhwcM128N64Ep::BiasCallbacks, 128, 64,
               {{}, {bias, Element(0), row_stride}, {}}, {output, out_stride})
MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m128n64_leaky_relu_init_fp16,
               NhwcM128N64LeakyReluKernel, NhwcM128N64Ep::LeakyReluCallbacks,
               128, 64,
               {{{}, {bias, Element(0), row_stride}, {}}, {}},
               {output, out_stride})
MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m128n64_residual_init_fp16,
               NhwcM128N64ResidualKernel, NhwcM128N64Ep::ResidualCallbacks,
               128, 64,
               {{{}, {bias, Element(0), row_stride}, {}},
                {const_cast<Element*>(residual), Element(0), res_stride}, {}},
               {output, out_stride})

MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m128n128_init_fp16, NhwcM128N128BiasKernel,
               NhwcM128N128Ep::BiasCallbacks, 128, 128,
               {{}, {bias, Element(0), row_stride}, {}}, {output, out_stride})
MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m128n128_stage4_init_fp16,
               NhwcM128N128Stage4BiasKernel, NhwcM128N128Ep::BiasCallbacks,
               128, 128,
               {{}, {bias, Element(0), row_stride}, {}}, {output, out_stride})
MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m128n128_leaky_relu_init_fp16,
               NhwcM128N128LeakyReluKernel, NhwcM128N128Ep::LeakyReluCallbacks,
               128, 128,
               {{{}, {bias, Element(0), row_stride}, {}}, {}},
               {output, out_stride})
MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m128n128_residual_init_fp16,
               NhwcM128N128ResidualKernel, NhwcM128N128Ep::ResidualCallbacks,
               128, 128,
               {{{}, {bias, Element(0), row_stride}, {}},
                {const_cast<Element*>(residual), Element(0), res_stride}, {}},
               {output, out_stride})
MLVC_NHWC_REGLU_INIT(mlvc_cutlass_nhwc_m128n128_reglu_init_fp16,
                     NhwcM128N128ReGLUKernel, NhwcM128N128Ep::ReGLUCallbacks,
                     128, 128)
MLVC_NHWC_REGLU_INIT(mlvc_cutlass_nhwc_m128n128_stage4_reglu_init_fp16,
                     NhwcM128N128Stage4ReGLUKernel,
                     NhwcM128N128Ep::ReGLUCallbacks, 128, 128)

MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m64n128_init_fp16, NhwcM64N128BiasKernel,
               NhwcM64N128Ep::BiasCallbacks, 64, 128,
               {{}, {bias, Element(0), row_stride}, {}}, {output, out_stride})
MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m64n128_leaky_relu_init_fp16,
               NhwcM64N128LeakyReluKernel, NhwcM64N128Ep::LeakyReluCallbacks,
               64, 128,
               {{{}, {bias, Element(0), row_stride}, {}}, {}},
               {output, out_stride})
MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m64n128_residual_init_fp16,
               NhwcM64N128ResidualKernel, NhwcM64N128Ep::ResidualCallbacks,
               64, 128,
               {{{}, {bias, Element(0), row_stride}, {}},
                {const_cast<Element*>(residual), Element(0), res_stride}, {}},
               {output, out_stride})
MLVC_NHWC_REGLU_INIT(mlvc_cutlass_nhwc_m64n128_reglu_init_fp16,
                     NhwcM64N128ReGLUKernel, NhwcM64N128Ep::ReGLUCallbacks,
                     64, 128)

MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m64n64_init_fp16, NhwcM64N64BiasKernel,
               NhwcM64N64Ep::BiasCallbacks, 64, 64,
               {{}, {bias, Element(0), row_stride}, {}}, {output, out_stride})
MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m64n64_leaky_relu_init_fp16,
               NhwcM64N64LeakyReluKernel, NhwcM64N64Ep::LeakyReluCallbacks,
               64, 64,
               {{{}, {bias, Element(0), row_stride}, {}}, {}},
               {output, out_stride})
MLVC_NHWC_INIT(mlvc_cutlass_nhwc_m64n64_residual_init_fp16,
               NhwcM64N64ResidualKernel, NhwcM64N64Ep::ResidualCallbacks,
               64, 64,
               {{{}, {bias, Element(0), row_stride}, {}},
                {const_cast<Element*>(residual), Element(0), res_stride}, {}},
               {output, out_stride})
MLVC_NHWC_REGLU_INIT(mlvc_cutlass_nhwc_m64n64_reglu_init_fp16,
                     NhwcM64N64ReGLUKernel, NhwcM64N64Ep::ReGLUCallbacks,
                     64, 64)

#undef MLVC_NHWC_REGLU_INIT
#undef MLVC_NHWC_INIT
#undef MLVC_NHWC_MAIN
