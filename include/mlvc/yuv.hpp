#pragma once

#include "mlvc/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

namespace mlvc {

struct Yuv420Frame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> y;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;
};

struct FramePadding {
    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;
    bool rotated = false;
};

bool read_yuv420_frame(std::istream& input, int width, int height,
                       Yuv420Frame& frame);
void write_yuv420_frame(std::ostream& output, const Yuv420Frame& frame);

FramePadding frame_padding(int width, int height,
                           int model_width, int model_height);
Tensor prepare_model_input(const Yuv420Frame& frame,
                           int model_width, int model_height,
                           float pixel_range, FramePadding& padding);
Yuv420Frame prepare_model_output(const Tensor& tensor, int width, int height,
                                 float pixel_range,
                                 const FramePadding& padding);

std::size_t yuv420_frame_bytes(int width, int height);

}  // namespace mlvc
