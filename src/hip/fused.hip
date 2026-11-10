// HIP: ROI crop, bilinear resize, mirror, normalize and layout change in one kernel.
// u8 NHWC -> f32 NCHW.
//
// Equal to resize followed by crop_mirror_normalize, without the u8 intermediate image:
// each thread reads four source pixels and writes three floats, and nothing else
// touches global memory. The bilinear result is not rounded to u8 before it is
// normalized, so the two paths differ by at most 0.5 / std.
#include "common/validate.hpp"
#include "hip/kernel_util.hpp"
#include "hipaug/hip_check.hpp"

namespace hipaug {
namespace {

// Thread mapping: one thread per output pixel, all three channels.
//   dx = blockIdx.x * 16 + threadIdx.x   output column
//   dy = blockIdx.y * 16 + threadIdx.y   output row
//   n  = blockIdx.z                      image index
__global__ void fused_rcmn_kernel(TensorDesc in, const uint8_t* __restrict__ src,
                                  const Roi* __restrict__ roi, const uint8_t* __restrict__ mirror,
                                  kernel::Norm norm, TensorDesc out, float* __restrict__ dst) {
  const int dx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dy = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;
  if (dx >= out.w_max || dy >= out.h_max)
    return;

  // Crop: the ROI of image n, clamped to the padded image.
  const kernel::Box b = kernel::clamp_roi(roi[n], in.w_max, in.h_max);
  if (kernel::is_empty(b))
    return;
  const int rw = b.x1 - b.x0, rh = b.y1 - b.y0;

  // Mirror: flipping the resized image means output column dx holds resized column xs.
  const int xs = mirror[n] ? out.w_max - 1 - dx : dx;

  // Resize: map the pixel centre of (xs, dy) into the ROI. Taps are ROI-local.
  const float scale_x = static_cast<float>(rw) / static_cast<float>(out.w_max);
  const float scale_y = static_cast<float>(rh) / static_cast<float>(out.h_max);
  const kernel::Tap tx = kernel::bilinear_tap(xs, scale_x, rw);
  const kernel::Tap ty = kernel::bilinear_tap(dy, scale_y, rh);

  // ROI-local taps + ROI origin (b.x0, b.y0) = padded-image coordinates, then input
  // strides. Each pointer is channel 0 of one interleaved source pixel.
  const uint8_t* p00 = src + kernel::offset(in, n, b.y0 + ty.i0, b.x0 + tx.i0, 0);
  const uint8_t* p01 = src + kernel::offset(in, n, b.y0 + ty.i0, b.x0 + tx.i1, 0);
  const uint8_t* p10 = src + kernel::offset(in, n, b.y0 + ty.i1, b.x0 + tx.i0, 0);
  const uint8_t* p11 = src + kernel::offset(in, n, b.y0 + ty.i1, b.x0 + tx.i1, 0);

  // Normalize and layout change: the output is planar, so the three channels of
  // pixel (dx, dy) are written stride_c (h * w) elements apart.
  float* d = dst + kernel::offset(out, n, dy, dx, 0);
  for (int c = 0; c < 3; ++c) {
    const size_t ci = c * in.stride_c;
    const float v = kernel::bilerp(p00[ci], p01[ci], p10[ci], p11[ci], tx.w, ty.w);
    d[c * out.stride_c] = (v - norm.mean[c]) * norm.inv_std[c];
  }
}

} // namespace

Status fused_rcmn(const TensorDesc& in, const uint8_t* src, const Roi* roi, const uint8_t* mirror,
                  const float mean[3], const float std[3], const TensorDesc& out, float* dst,
                  hipStream_t stream) {
  const Status st = detail::validate_normalize(in, src, roi, mirror, mean, std, out, dst);
  if (st != Status::Ok)
    return st;
  const dim3 block(kernel::kBlockW, kernel::kBlockH, 1);
  const dim3 grid = kernel::grid_for(out.w_max, out.h_max, out.n);
  hipLaunchKernelGGL(fused_rcmn_kernel, grid, block, 0, stream, in, src, roi, mirror,
                     kernel::make_norm(mean, std), out, dst);
  HIP_CHECK(hipGetLastError());
  return Status::Ok;
}

} // namespace hipaug
