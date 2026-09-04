#pragma once

#include <cstddef>

namespace mlvc::driver_cubin {

// CUTLASS owns the parameter layout. Keep the runtime side opaque and let a
// one-thread device initializer construct the object in this storage.
inline constexpr std::size_t kCutlassPointwiseParamsStorageBytes = 512;

}  // namespace mlvc::driver_cubin
