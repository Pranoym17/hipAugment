// HIP: bilinear resize of each image's ROI into the full output size. u8 NHWC -> u8 NHWC.
#include "common/validate.hpp"
#include "hip/kernel_util.hpp"
#include "hipaug/hip_check.hpp"

namespace hipaug {
namespace {

// Thread mapping: one thread per output pixel.
//   dx = blockIdx.x * 16 + threadIdx.x   output column
//   dy = blockIdx.y * 16 + threadIdx.y   output row
//   n  = blockIdx.z                      image index
__global__ void resize_kernel(TensorDesc in, const uint8_t* __restrict__ src,
                              const Roi* __restrict__ roi, TensorDesc out,
                              uint8_t* __restrict__ dst) {
  const int dx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dy = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;
  // The grid is rounded up to whole blocks, so some threads fall past the output edge.
  if (dx >= out.w_max || dy >= out.h_max)
    return;

  // ROI of image n, clamped to the padded image. An empty ROI writes nothing.
  const kernel::Box b = kernel::clamp_roi(roi[n], in.w_max, in.h_max);
  if (kernel::is_empty(b))
    return;
  const int rw = b.x1 - b.x0, rh = b.y1 - b.y0;

  // Map the output pixel centre into the ROI. Taps are ROI-local: 0 .. rw-1, 0 .. rh-1.
  const float scale_x = static_cast<float>(rw) / static_cast<float>(out.w_max);
  const float scale_y = static_cast<float>(rh) / static_cast<float>(out.h_max);
  const kernel::Tap tx = kernel::bilinear_tap(dx, scale_x, rw);
  const kernel::Tap ty = kernel::bilinear_tap(dy, scale_y, rh);

  // Add the ROI origin (b.x0, b.y0) to turn ROI-local taps into padded-image
  // coordinates, then apply the input strides. Each pointer is channel 0 of a tap.
  const uint8_t* p00 = src + kernel::offset(in, n, b.y0 + ty.i0, b.x0 + tx.i0, 0);
  const uint8_t* p01 = src + kernel::offset(in, n, b.y0 + ty.i0, b.x0 + tx.i1, 0);
  const uint8_t* p10 = src + kernel::offset(in, n, b.y0 + ty.i1, b.x0 + tx.i0, 0);
  const uint8_t* p11 = src + kernel::offset(in, n, b.y0 + ty.i1, b.x0 + tx.i1, 0);

  // Output pixel (dx, dy) of image n. Channels are stride_c apart in both tensors.
  uint8_t* d = dst + kernel::offset(out, n, dy, dx, 0);
  for (int c = 0; c < 3; ++c) {
    const size_t ci = c * in.stride_c;
    const float v = kernel::bilerp(p00[ci], p01[ci], p10[ci], p11[ci], tx.w, ty.w);
    d[c * out.stride_c] = static_cast<uint8_t>(v + 0.5f); // round to nearest
  }
}

} // namespace

Status resize(const TensorDesc& in, const uint8_t* src, const Roi* roi, const TensorDesc& out,
              uint8_t* dst, hipStream_t stream) {
  const Status st = detail::validate_resize(in, src, roi, out, dst);
  if (st != Status::Ok)
    return st;
  const dim3 block(kernel::kBlockW, kernel::kBlockH, 1);
  const dim3 grid = kernel::grid_for(out.w_max, out.h_max, out.n);
  hipLaunchKernelGGL(resize_kernel, grid, block, 0, stream, in, src, roi, out, dst);
  HIP_CHECK(hipGetLastError());
  return Status::Ok;
}

} // namespace hipaug
