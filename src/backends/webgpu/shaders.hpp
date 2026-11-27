#pragma once

#include <cstdint>
#include <string>

namespace mlvc::webgpu_backend {

// Generates WGSL source strings parameterized by whether native `enable f16;`
// is supported on the active WebGPU device. When `use_f16` is false, storage
// buffers store packed 2xFP16 in `u32` via `unpack2x16float` / `pack2x16float`.

std::string wgsl_unary(bool use_f16);
std::string wgsl_unary_vec4();
std::string wgsl_binary(bool use_f16);
std::string wgsl_binary_vec4();
std::string wgsl_gather(bool use_f16);
std::string wgsl_slice_nhwc(bool use_f16);
std::string wgsl_concat_nhwc(bool use_f16);
std::string wgsl_depth_to_space_nhwc(bool use_f16);
std::string wgsl_space_to_depth_nhwc(bool use_f16);
std::string wgsl_nchw_to_nhwc(bool use_f16);
std::string wgsl_nhwc_to_nchw(bool use_f16);

// Convolutions & Tiled GEMM + Epilogue / ReGLU fusion.
// tile_variant: 0 = small-M (16x64), 1 = medium-N (64x64), 2 = wide-N (64x128).
std::string wgsl_conv_generic_nhwc(bool use_f16);
std::string wgsl_pointwise_gemm(bool use_f16, int tile_variant, bool is_reglu);
std::string wgsl_spatial_conv_gemm(bool use_f16);
std::string wgsl_depthwise_3x3_s1p1(bool use_f16);

// Multi-node fused tails & activations.
std::string wgsl_reglu_nhwc(bool use_f16);
std::string wgsl_feature_update_nhwc(bool use_f16, bool emit_side_outputs);
std::string wgsl_y0_tail_nhwc(bool use_f16);
std::string wgsl_y1_tail_nhwc(bool use_f16);
std::string wgsl_dec_y0_tail_nhwc(bool use_f16);
std::string wgsl_dec_y1_tail_nhwc(bool use_f16);

// GPU codec I/O (YUV420 <-> NHWC FP16 + zero-readback Canvas Render Shader).
std::string wgsl_yuv420_to_nhwc(bool use_f16);
std::string wgsl_nhwc_to_yuv420(bool use_f16, bool fused_d2s8);
std::string wgsl_render_yuv444_to_rgba(bool use_f16, bool fused_d2s8);

}  // namespace mlvc::webgpu_backend
