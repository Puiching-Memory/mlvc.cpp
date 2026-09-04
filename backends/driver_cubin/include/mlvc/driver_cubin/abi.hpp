#pragma once

// NVIDIA's official header defines the Driver API ABI. The driver-cubin
// executable links only the driver library (libcuda.so.1/nvcuda.dll), never
// CUDA Runtime, cuDNN, cuBLAS, or an inference framework.

#include <cuda.h>

namespace mlvc::driver_cubin::abi {

using Result = CUresult;
using Device = CUdevice;
using DeviceAddress = CUdeviceptr;
using Context = CUcontext;
using Stream = CUstream;
using Module = CUmodule;
using Function = CUfunction;
using Graph = CUgraph;
using GraphExec = CUgraphExec;

inline constexpr Result kSuccess = CUDA_SUCCESS;
inline constexpr unsigned int kStreamNonBlocking = CU_STREAM_NON_BLOCKING;
inline constexpr CUstreamCaptureMode kStreamCaptureThreadLocal =
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
inline constexpr CUdevice_attribute kComputeCapabilityMajor =
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
inline constexpr CUdevice_attribute kComputeCapabilityMinor =
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
inline constexpr CUdevice_attribute kMultiprocessorCount =
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;

}  // namespace mlvc::driver_cubin::abi
