#pragma once

#include "mlvc/core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>
#include <vector>

namespace mlvc {

struct Yuv420Frame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> y;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;
};

class Yuv420FrameView {
public:
    Yuv420FrameView(int width, int height, std::span<const std::uint8_t> y,
                    std::span<const std::uint8_t> u,
                    std::span<const std::uint8_t> v);

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    std::span<const std::uint8_t> y() const noexcept { return y_; }
    std::span<const std::uint8_t> u() const noexcept { return u_; }
    std::span<const std::uint8_t> v() const noexcept { return v_; }

private:
    int width_;
    int height_;
    std::span<const std::uint8_t> y_;
    std::span<const std::uint8_t> u_;
    std::span<const std::uint8_t> v_;
};

class MutableYuv420FrameView {
public:
    MutableYuv420FrameView(int width, int height, std::span<std::uint8_t> y,
                           std::span<std::uint8_t> u,
                           std::span<std::uint8_t> v);

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    std::span<std::uint8_t> y() const noexcept { return y_; }
    std::span<std::uint8_t> u() const noexcept { return u_; }
    std::span<std::uint8_t> v() const noexcept { return v_; }

    operator Yuv420FrameView() const;

private:
    int width_;
    int height_;
    std::span<std::uint8_t> y_;
    std::span<std::uint8_t> u_;
    std::span<std::uint8_t> v_;
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

Yuv420FrameView yuv420_view(const Yuv420Frame& frame);
MutableYuv420FrameView mutable_yuv420_view(Yuv420Frame& frame);

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
