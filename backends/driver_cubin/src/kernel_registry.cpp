#include "aot_graph.hpp"

namespace mlvc::driver_cubin_backend {

void AotGraph::register_kernels()
{
    binary_ = module_.function("mlvc_binary_fp16");
    binary_contiguous_ = module_.function("mlvc_binary_contiguous_fp16");
    unary_ = module_.function("mlvc_unary_fp16");
    feature_update_ = module_.function("mlvc_feature_update_fp16");
    feature_update_outputs_ =
        module_.function("mlvc_feature_update_outputs_fp16");
    y0_tail_ = module_.function("mlvc_y0_tail_fp16");
    y1_tail_ = module_.function("mlvc_y1_tail_fp16");
    reglu_ = module_.function("mlvc_reglu_fp16");
    reglu_vec8_ = module_.function("mlvc_reglu_vec8_fp16");
    convolution_ = module_.function("mlvc_conv_fp16");
    pointwise_convolution_ = module_.function("mlvc_pointwise_conv_fp16");
    pointwise_convolution_wide_ =
        module_.function("mlvc_pointwise_conv_wide_fp16");
    pointwise_convolution_balanced_ =
        module_.function("mlvc_pointwise_conv_balanced_fp16");
    pointwise_convolution_mma_ =
        module_.function("mlvc_pointwise_conv_mma_fp16");
    pointwise_convolution_mma_small_ =
        module_.function("mlvc_pointwise_conv_mma_small_fp16");
    pointwise_reglu_mma_ = module_.function("mlvc_pointwise_reglu_mma_fp16");
    pointwise_reglu_mma_small_ =
        module_.function("mlvc_pointwise_reglu_mma_small_fp16");
    if (driver_.device_info().compute_major >= 8) {
        cutlass_pointwise_init_ =
            module_.function("mlvc_cutlass_pointwise_init_fp16");
        cutlass_pointwise_ =
            module_.function("mlvc_cutlass_pointwise_fp16");
        cutlass_pointwise_leaky_relu_init_ = module_.function(
            "mlvc_cutlass_pointwise_leaky_relu_init_fp16");
        cutlass_pointwise_leaky_relu_ = module_.function(
            "mlvc_cutlass_pointwise_leaky_relu_fp16");
        cutlass_pointwise_residual_init_ = module_.function(
            "mlvc_cutlass_pointwise_residual_init_fp16");
        cutlass_pointwise_residual_ = module_.function(
            "mlvc_cutlass_pointwise_residual_fp16");
        cutlass_pointwise_medium_init_ = module_.function(
            "mlvc_cutlass_pointwise_medium_init_fp16");
        cutlass_pointwise_medium_ = module_.function(
            "mlvc_cutlass_pointwise_medium_fp16");
        cutlass_pointwise_medium_stage4_init_ = module_.function(
            "mlvc_cutlass_pointwise_medium_stage4_init_fp16");
        cutlass_pointwise_medium_stage4_ = module_.function(
            "mlvc_cutlass_pointwise_medium_stage4_fp16");
        driver_.set_max_dynamic_shared_memory(
            cutlass_pointwise_medium_stage4_, 65536U);
        cutlass_pointwise_medium_leaky_relu_init_ = module_.function(
            "mlvc_cutlass_pointwise_medium_leaky_relu_init_fp16");
        cutlass_pointwise_medium_leaky_relu_ = module_.function(
            "mlvc_cutlass_pointwise_medium_leaky_relu_fp16");
        cutlass_pointwise_medium_residual_init_ = module_.function(
            "mlvc_cutlass_pointwise_medium_residual_init_fp16");
        cutlass_pointwise_medium_residual_ = module_.function(
            "mlvc_cutlass_pointwise_medium_residual_fp16");
        cutlass_pointwise_spatial_wide_residual_init_ = module_.function(
            "mlvc_cutlass_pointwise_spatial_wide_residual_init_fp16");
        cutlass_pointwise_spatial_wide_residual_ = module_.function(
            "mlvc_cutlass_pointwise_spatial_wide_residual_fp16");
        driver_.set_max_dynamic_shared_memory(
            cutlass_pointwise_spatial_wide_residual_, 61440U);
        cutlass_spatial_convolution_init_ =
            module_.function("mlvc_cutlass_spatial_conv_init_fp16");
        cutlass_spatial_convolution_ =
            module_.function("mlvc_cutlass_spatial_conv_fp16");
        driver_.set_max_dynamic_shared_memory(
            cutlass_spatial_convolution_, 73728U);
        transpose_ = module_.function("mlvc_transpose_fp16");
        oihw_to_krsc_ = module_.function("mlvc_oihw_to_krsc_fp16");
    }
    spatial_convolution_mma_ =
        module_.function("mlvc_spatial_conv_mma_fp16");
    spatial_convolution_mma_wide_ =
        module_.function("mlvc_spatial_conv_mma_wide_fp16");
    spatial_convolution_ = module_.function("mlvc_spatial_conv_fp16");
    depthwise_convolution_ = module_.function("mlvc_depthwise_conv_fp16");
    depthwise_convolution_pair_ =
        module_.function("mlvc_depthwise_conv_pair_fp16");
    depthwise_convolution_quad_ =
        module_.function("mlvc_depthwise_conv_quad_fp16");
    gather_ = module_.function("mlvc_gather_axis0_fp16");
    slice_ = module_.function("mlvc_slice_fp16");
    concat_ = module_.function("mlvc_concat_copy_fp16");
    depth_to_space_ = module_.function("mlvc_depth_to_space_fp16");
    space_to_depth_ = module_.function("mlvc_space_to_depth_fp16");
    yuv420_to_nchw_ = module_.function("mlvc_yuv420_to_nchw_fp16");
    nchw_to_yuv420_ = module_.function("mlvc_nchw_to_yuv420");
}

}  // namespace mlvc::driver_cubin_backend
