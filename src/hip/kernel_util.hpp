// Device-side helpers shared by the HIP kernels.
#pragma once

#include "hipaug/runtime.hpp"

#include "hipaug/hipaug.hpp"

namespace hipaug::kernel {

// 16x16 = 256 threads: four 64-thread AMD wavefronts or eight 32-thread NVIDIA warps.
constexpr int kBlockW = 16;
constexpr int kBlockH = 16;

// Grid that covers a w x h image plane for n images: x and y tile the plane in
// 16x16 blocks, z is the image index.
inline dim3 grid_for(int w, int h, int n) {
  return dim3((w + kBlockW - 1) / kBlockW, (h + kBlockH - 1) / kBlockH, n);
}

__device__ inline int imin(int a, int b) {
  return a < b ? a : b;
}
__device__ inline int imax(int a, int b) {
  return a > b ? a : b;
}
__device__ inline int iclamp(int v, int lo, int hi) {
  return imin(imax(v, lo), hi);
}

// A ROI clamped to the padded image: columns [x0, x1) and rows [y0, y1).
// The ROI array comes from the caller's device memory and cannot be validated on
// the host, so every kernel clamps it before use.
struct Box {
  int x0, y0, x1, y1;
};

__device__ inline Box clamp_roi(const Roi& r, int w_max, int h_max) {
  Box b;
  b.x0 = imax(r.x, 0);
  b.y0 = imax(r.y, 0);
  b.x1 = imin(r.x + r.w, w_max);
  b.y1 = imin(r.y + r.h, h_max);
  return b;
}

__device__ inline bool is_empty(const Box& b) {
  return b.x1 <= b.x0 || b.y1 <= b.y0;
}

// Offset of element (n, y, x, c): n * stride_n + y * stride_h + x * stride_w + c * stride_c.
__device__ inline size_t offset(const TensorDesc& d, int n, int y, int x, int c) {
  return static_cast<size_t>(n) * d.stride_n + static_cast<size_t>(y) * d.stride_h +
         static_cast<size_t>(x) * d.stride_w + static_cast<size_t>(c) * d.stride_c;
}

// One axis of a bilinear sample: the two source taps and the weight of the second.
struct Tap {
  int i0, i1;
  float w;
};

// Pixel-centre convention: src = (dst + 0.5) * scale - 0.5 in ROI-local coordinates,
// clamped to the ROI edge. `size` is the ROI length along this axis.
__device__ inline Tap bilinear_tap(int dst, float scale, int size) {
  float f = (static_cast<float>(dst) + 0.5f) * scale - 0.5f;
  f = fminf(fmaxf(f, 0.0f), static_cast<float>(size - 1));
  Tap t;
  t.i0 = static_cast<int>(f); // f >= 0, so truncation is floor
  t.i1 = imin(t.i0 + 1, size - 1);
  t.w = f - static_cast<float>(t.i0);
  return t;
}

// Bilinear blend of four taps. Same expression order as the CPU reference.
__device__ inline float bilerp(float p00, float p01, float p10, float p11, float wx, float wy) {
  const float top = p00 + wx * (p01 - p00);
  const float bot = p10 + wx * (p11 - p10);
  return top + wy * (bot - top);
}

// Per-channel normalization constants, passed to kernels by value.
struct Norm {
  float mean[3];
  float inv_std[3];
};

inline Norm make_norm(const float mean[3], const float std[3]) {
  Norm nm;
  for (int c = 0; c < 3; ++c) {
    nm.mean[c] = mean[c];
    nm.inv_std[c] = 1.0f / std[c];
  }
  return nm;
}

} // namespace hipaug::kernel
