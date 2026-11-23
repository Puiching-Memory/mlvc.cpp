#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

namespace {

__device__ __forceinline__ int offset4(int linear, int d1, int d2, int d3,
                                       int s0, int s1, int s2, int s3)
{
    const int c3 = linear % d3;
    linear /= d3;
    const int c2 = linear % d2;
    linear /= d2;
    const int c1 = linear % d1;
    const int c0 = linear / d1;
    return c0 * s0 + c1 * s1 + c2 * s2 + c3 * s3;
}

}  // namespace

