#pragma once

#include "mlvc/core/tensor.hpp"

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

struct Yuv420FrameView {
    int width = 0;
    int height = 0;
    const std::uint8_t* y = nullptr;
    const std::uint8_t* u = nullptr;
    const std::uint8_t* v = nullptr;
};

struct MutableYuv420FrameView {
    int width = 0;
    int height = 0;
    std::uint8_t* y = nullptr;
    std::uint8_t* u = nullptr;
    std::uint8_t* v = nullptr;

    operator Yuv420FrameView() const noexcept
    {
        return Yuv420FrameView{width, height, y, u, v};
    }
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
bool read_yuv420_frame(std::istream& input, MutableYuv420FrameView frame);
void write_yuv420_frame(std::ostream& output, const Yuv420Frame& frame);
void write_yuv420_frame(std::ostream& output, Yuv420FrameView frame);

Yuv420FrameView yuv420_view(const Yuv420Frame& frame) noexcept;
MutableYuv420FrameView mutable_yuv420_view(Yuv420Frame& frame) noexcept;

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
