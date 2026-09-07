#pragma once

#include <cstddef>

namespace mlvc::driver_cubin {

// CUTLASS owns the parameter layout. Keep the runtime side opaque and let a
// one-thread device initializer construct the object in this storage.
// CUTLASS 4.8 spatial convolution parameters need 520 bytes.
inline constexpr std::size_t kCutlassPointwiseParamsStorageBytes = 576;

}  // namespace mlvc::driver_cubin
