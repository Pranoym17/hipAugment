// Small helpers shared by the CPU reference implementations.
#pragma once

#include <algorithm>
#include <cstdint>

#include "hipaug/hipaug.hpp"

namespace hipaug::cpu::detail {

// A ROI clamped to the padded image: columns [x0, x1) and rows [y0, y1).
struct Box {
  int x0, y0, x1, y1;
  bool empty() const { return x1 <= x0 || y1 <= y0; }
};

inline Box clamp_roi(const Roi& r, const TensorDesc& d) {
  Box b;
  b.x0 = std::max(r.x, 0);
  b.y0 = std::max(r.y, 0);
  b.x1 = std::min(r.x + r.w, d.w_max);
  b.y1 = std::min(r.y + r.h, d.h_max);
  return b;
}

// Offset of element (n, y, x, c) in a buffer described by d.
inline size_t offset(const TensorDesc& d, int n, int y, int x, int c) {
  return static_cast<size_t>(n) * d.stride_n + static_cast<size_t>(y) * d.stride_h +
         static_cast<size_t>(x) * d.stride_w + static_cast<size_t>(c) * d.stride_c;
}

// One axis of a bilinear sample: the two source taps and the weight of the second.
struct Tap {
  int i0, i1;
  float w;
};

// Pixel-centre convention: src = (dst + 0.5) * scale - 0.5, in ROI-local coordinates,
// clamped to the ROI edge. `size` is the ROI length along this axis.
inline Tap bilinear_tap(int dst, float scale, int size) {
  float f = (static_cast<float>(dst) + 0.5f) * scale - 0.5f;
  f = std::min(std::max(f, 0.0f), static_cast<float>(size - 1));
  Tap t;
  t.i0 = static_cast<int>(f); // f >= 0, so truncation is floor
  t.i1 = std::min(t.i0 + 1, size - 1);
  t.w = f - static_cast<float>(t.i0);
  return t;
}

// Bilinear blend of four taps. Same expression order as the GPU kernels.
inline float bilerp(float p00, float p01, float p10, float p11, float wx, float wy) {
  const float top = p00 + wx * (p01 - p00);
  const float bot = p10 + wx * (p11 - p10);
  return top + wy * (bot - top);
}

} // namespace hipaug::cpu::detail
