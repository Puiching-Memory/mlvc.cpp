#pragma once

// NVIDIA's official header defines the Driver API ABI. The driver-cubin
// executable links only the driver library (libcuda.so.1/nvcuda.dll), never
// CUDA Runtime, cuDNN, cuBLAS, or an inference framework.

#include <cuda.h>

namespace mlvc::driver::abi {

using Result = CUresult;
using Device = CUdevice;
using DeviceAddress = CUdeviceptr;
using Context = CUcontext;
using Stream = CUstream;
using Module = CUmodule;
using Function = CUfunction;

inline constexpr Result kSuccess = CUDA_SUCCESS;
inline constexpr unsigned int kStreamNonBlocking = CU_STREAM_NON_BLOCKING;
inline constexpr CUdevice_attribute kComputeCapabilityMajor =
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
inline constexpr CUdevice_attribute kComputeCapabilityMinor =
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;

}  // namespace mlvc::driver::abi
