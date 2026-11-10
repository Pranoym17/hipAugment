#include "common/validate.hpp"

#include <cmath>

namespace hipaug {

TensorDesc make_desc(int n, int h_max, int w_max, Layout layout, DType dtype) {
  TensorDesc d{};
  d.n = n;
  d.c = 3;
  d.h_max = h_max;
  d.w_max = w_max;
  d.layout = layout;
  d.dtype = dtype;
  const size_t c = 3, h = static_cast<size_t>(h_max), w = static_cast<size_t>(w_max);
  if (layout == Layout::NHWC) { // channels interleaved: [n][y][x][c]
    d.stride_c = 1;
    d.stride_w = c;
    d.stride_h = w * c;
  } else { // planar: [n][c][y][x]
    d.stride_w = 1;
    d.stride_h = w;
    d.stride_c = h * w;
  }
  d.stride_n = c * h * w;
  return d;
}

size_t num_elements(const TensorDesc& d) { return static_cast<size_t>(d.n) * d.stride_n; }

const char* to_string(Status s) {
  switch (s) {
  case Status::Ok:
    return "Ok";
  case Status::InvalidArg:
    return "InvalidArg";
  case Status::HipError:
    return "HipError";
  }
  return "Unknown";
}

namespace detail {

namespace {

bool desc_ok(const TensorDesc& d, Layout layout, DType dtype) {
  if (d.n <= 0 || d.c != 3 || d.h_max <= 0 || d.w_max <= 0)
    return false;
  if (d.layout != layout || d.dtype != dtype)
    return false;
  if (d.stride_n == 0 || d.stride_h == 0 || d.stride_w == 0 || d.stride_c == 0)
    return false;
  // The strides must leave room for a full image: the last element of an image
  // has to sit before the start of the next one.
  const size_t last = static_cast<size_t>(d.h_max - 1) * d.stride_h +
                      static_cast<size_t>(d.w_max - 1) * d.stride_w +
                      static_cast<size_t>(d.c - 1) * d.stride_c;
  return last < d.stride_n;
}

} // namespace

Status validate_resize(const TensorDesc& in, const void* src, const void* roi,
                       const TensorDesc& out, const void* dst) {
  if (!src || !roi || !dst || src == dst)
    return Status::InvalidArg;
  if (!desc_ok(in, Layout::NHWC, DType::U8) || !desc_ok(out, Layout::NHWC, DType::U8))
    return Status::InvalidArg;
  if (in.n != out.n)
    return Status::InvalidArg;
  return Status::Ok;
}

Status validate_blur(const TensorDesc& in, const void* src, const void* roi, const void* dst,
                     int ksize, float sigma) {
  if (!src || !roi || !dst || src == dst)
    return Status::InvalidArg;
  if (!desc_ok(in, Layout::NHWC, DType::U8))
    return Status::InvalidArg;
  if (ksize != 3 && ksize != 5)
    return Status::InvalidArg;
  if (!std::isfinite(sigma) || sigma <= 0.0f)
    return Status::InvalidArg;
  return Status::Ok;
}

Status validate_normalize(const TensorDesc& in, const void* src, const void* roi,
                          const void* mirror, const float* mean, const float* std,
                          const TensorDesc& out, const void* dst) {
  if (!src || !roi || !mirror || !mean || !std || !dst)
    return Status::InvalidArg;
  if (!desc_ok(in, Layout::NHWC, DType::U8) || !desc_ok(out, Layout::NCHW, DType::F32))
    return Status::InvalidArg;
  if (in.n != out.n)
    return Status::InvalidArg;
  for (int c = 0; c < 3; ++c) {
    if (!std::isfinite(mean[c]) || !std::isfinite(std[c]) || std[c] == 0.0f)
      return Status::InvalidArg;
  }
  return Status::Ok;
}

void gaussian_kernel_2d(int ksize, float sigma, float k2[25]) {
  float k1[5] = {0, 0, 0, 0, 0};
  const int r = ksize / 2;
  float sum = 0.0f;
  for (int i = 0; i < ksize; ++i) {
    const float d = static_cast<float>(i - r);
    k1[i] = std::exp(-(d * d) / (2.0f * sigma * sigma));
    sum += k1[i];
  }
  for (int i = 0; i < ksize; ++i)
    k1[i] /= sum;
  for (int i = 0; i < 25; ++i)
    k2[i] = 0.0f;
  for (int j = 0; j < ksize; ++j)
    for (int i = 0; i < ksize; ++i)
      k2[j * ksize + i] = k1[j] * k1[i];
}

} // namespace detail
} // namespace hipaug
