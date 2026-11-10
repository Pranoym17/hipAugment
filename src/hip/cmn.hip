// HIP: crop the ROI, mirror in x, normalize, change layout. u8 NHWC -> f32 NCHW.
#include "common/validate.hpp"
#include "hip/kernel_util.hpp"
#include "hipaug/hip_check.hpp"

namespace hipaug {
namespace {

// Thread mapping: one thread per output pixel, all three channels.
//   x = blockIdx.x * 16 + threadIdx.x   output column
//   y = blockIdx.y * 16 + threadIdx.y   output row
//   n = blockIdx.z                      image index
__global__ void cmn_kernel(TensorDesc in, const uint8_t* __restrict__ src,
                           const Roi* __restrict__ roi, const uint8_t* __restrict__ mirror,
                           kernel::Norm norm, TensorDesc out, float* __restrict__ dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;
  if (x >= out.w_max || y >= out.h_max)
    return;

  const Roi r = roi[n];
  const kernel::Box b = kernel::clamp_roi(r, in.w_max, in.h_max);
  if (kernel::is_empty(b))
    return;

  // Output column x reads crop column xs: the same column, or its mirror image.
  const int xs = mirror[n] ? out.w_max - 1 - x : x;
  // The crop starts at the ROI origin (r.x, r.y). The ROI should have the output
  // size; if it does not, the clamp keeps the read inside the ROI.
  const int sx = kernel::iclamp(r.x + xs, b.x0, b.x1 - 1);
  const int sy = kernel::iclamp(r.y + y, b.y0, b.y1 - 1);

  // Input is interleaved: the three channels of one pixel sit stride_c (1) apart.
  // Output is planar: the three channels sit stride_c (h * w) apart.
  const uint8_t* p = src + kernel::offset(in, n, sy, sx, 0);
  float* d = dst + kernel::offset(out, n, y, x, 0);
  for (int c = 0; c < 3; ++c) {
    const float v = static_cast<float>(p[c * in.stride_c]);
    d[c * out.stride_c] = (v - norm.mean[c]) * norm.inv_std[c];
  }
}

} // namespace

Status crop_mirror_normalize(const TensorDesc& in, const uint8_t* src, const Roi* roi,
                             const uint8_t* mirror, const float mean[3], const float std[3],
                             const TensorDesc& out, float* dst, hipStream_t stream) {
  const Status st = detail::validate_normalize(in, src, roi, mirror, mean, std, out, dst);
  if (st != Status::Ok)
    return st;
  const dim3 block(kernel::kBlockW, kernel::kBlockH, 1);
  const dim3 grid = kernel::grid_for(out.w_max, out.h_max, out.n);
  hipLaunchKernelGGL(cmn_kernel, grid, block, 0, stream, in, src, roi, mirror,
                     kernel::make_norm(mean, std), out, dst);
  HIP_CHECK(hipGetLastError());
  return Status::Ok;
}

} // namespace hipaug
