// HIP: 3x3 or 5x5 Gaussian blur inside each image's ROI. u8 NHWC -> u8 NHWC, same size.
//
// Two kernels with identical results:
//   blur_direct_kernel  v1, every tap is read from global memory
//   blur_lds_kernel     v2, each block first copies its tile plus halo into LDS
#include "common/validate.hpp"
#include "hip/kernel_util.hpp"
#include "hipaug/hip_check.hpp"

namespace hipaug {
namespace {

// The 2D kernel weights, row-major, passed by value. sigma is a run-time argument,
// so the weights are computed on the host for each call.
struct Weights {
  float k[25];
};

// v1. Thread mapping: one thread per pixel of the padded image.
//   x = blockIdx.x * 16 + threadIdx.x   column in the padded image
//   y = blockIdx.y * 16 + threadIdx.y   row in the padded image
//   n = blockIdx.z                      image index
// Threads outside the ROI exit without writing.
__global__ void blur_direct_kernel(TensorDesc in, const uint8_t* __restrict__ src,
                                   const Roi* __restrict__ roi, uint8_t* __restrict__ dst,
                                   int ksize, Weights w) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const kernel::Box b = kernel::clamp_roi(roi[n], in.w_max, in.h_max);
  // Covers both threads past the image edge and threads outside the ROI.
  if (x < b.x0 || x >= b.x1 || y < b.y0 || y >= b.y1)
    return;

  const int r = ksize / 2;
  float acc[3] = {0.0f, 0.0f, 0.0f};
  for (int j = 0; j < ksize; ++j) {
    const int yy = kernel::iclamp(y + j - r, b.y0, b.y1 - 1); // clamp taps to the ROI edge
    for (int i = 0; i < ksize; ++i) {
      const int xx = kernel::iclamp(x + i - r, b.x0, b.x1 - 1);
      const uint8_t* p = src + kernel::offset(in, n, yy, xx, 0);
      const float wt = w.k[j * ksize + i];
      for (int c = 0; c < 3; ++c)
        acc[c] += wt * static_cast<float>(p[c * in.stride_c]);
    }
  }
  uint8_t* d = dst + kernel::offset(in, n, y, x, 0);
  for (int c = 0; c < 3; ++c)
    d[c * in.stride_c] = static_cast<uint8_t>(fminf(acc[c], 255.0f) + 0.5f);
}

// v2. Same thread mapping as v1. Each 16x16 block first loads its pixels plus a halo
// of r pixels on every side into LDS, then every thread reads its taps from LDS.
//
// Tile layout: (16 + 2r) x (16 + 2r) slots, row-major. Slot (tx, ty) holds the pixel
// at padded-image position (bx - r + tx, by - r + ty), clamped to the ROI, where
// (bx, by) is the block's top-left pixel. Each slot packs R, G, B into one 32-bit word.
constexpr int kMaxRadius = 2;
constexpr int kTileMax = (kernel::kBlockW + 2 * kMaxRadius) * (kernel::kBlockH + 2 * kMaxRadius);

__global__ void blur_lds_kernel(TensorDesc in, const uint8_t* __restrict__ src,
                                const Roi* __restrict__ roi, uint8_t* __restrict__ dst, int ksize,
                                Weights w) {
  __shared__ uint32_t tile[kTileMax];

  const int n = blockIdx.z;
  const int bx = blockIdx.x * kernel::kBlockW; // top-left pixel of this block
  const int by = blockIdx.y * kernel::kBlockH;

  const kernel::Box b = kernel::clamp_roi(roi[n], in.w_max, in.h_max);
  // Skip blocks that do not touch the ROI. Every thread in the block takes the same
  // branch, so returning before the barrier is safe.
  if (kernel::is_empty(b) || bx >= b.x1 || bx + kernel::kBlockW <= b.x0 || by >= b.y1 ||
      by + kernel::kBlockH <= b.y0)
    return;

  const int r = ksize / 2;
  const int tw = kernel::kBlockW + 2 * r; // tile width in slots
  const int th = kernel::kBlockH + 2 * r;

  // Load phase. The tile has more slots than the block has threads (324 or 400 vs 256),
  // so thread t fills slots t, t + 256, ...
  const int tid = threadIdx.y * kernel::kBlockW + threadIdx.x;
  for (int slot = tid; slot < tw * th; slot += kernel::kBlockW * kernel::kBlockH) {
    const int tx = slot % tw, ty = slot / tw;
    const int sx = kernel::iclamp(bx - r + tx, b.x0, b.x1 - 1);
    const int sy = kernel::iclamp(by - r + ty, b.y0, b.y1 - 1);
    const uint8_t* p = src + kernel::offset(in, n, sy, sx, 0);
    tile[slot] = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[in.stride_c]) << 8) |
                 (static_cast<uint32_t>(p[2 * in.stride_c]) << 16);
  }
  __syncthreads();

  // Compute phase. This thread's pixel is tile slot (threadIdx.x + r, threadIdx.y + r),
  // so tap (i, j) of its window is slot (threadIdx.x + i, threadIdx.y + j).
  const int x = bx + threadIdx.x;
  const int y = by + threadIdx.y;
  if (x < b.x0 || x >= b.x1 || y < b.y0 || y >= b.y1)
    return;

  float acc[3] = {0.0f, 0.0f, 0.0f};
  for (int j = 0; j < ksize; ++j) {
    for (int i = 0; i < ksize; ++i) {
      const uint32_t px = tile[(threadIdx.y + j) * tw + threadIdx.x + i];
      const float wt = w.k[j * ksize + i];
      acc[0] += wt * static_cast<float>(px & 0xFF);
      acc[1] += wt * static_cast<float>((px >> 8) & 0xFF);
      acc[2] += wt * static_cast<float>((px >> 16) & 0xFF);
    }
  }
  uint8_t* d = dst + kernel::offset(in, n, y, x, 0);
  for (int c = 0; c < 3; ++c)
    d[c * in.stride_c] = static_cast<uint8_t>(fminf(acc[c], 255.0f) + 0.5f);
}

enum class BlurImpl { Direct, Lds };

Status launch_blur(BlurImpl impl, const TensorDesc& in, const uint8_t* src, const Roi* roi,
                   uint8_t* dst, int ksize, float sigma, hipStream_t stream) {
  const Status st = detail::validate_blur(in, src, roi, dst, ksize, sigma);
  if (st != Status::Ok)
    return st;
  Weights w;
  detail::gaussian_kernel_2d(ksize, sigma, w.k);
  const dim3 block(kernel::kBlockW, kernel::kBlockH, 1);
  const dim3 grid = kernel::grid_for(in.w_max, in.h_max, in.n);
  if (impl == BlurImpl::Lds)
    hipLaunchKernelGGL(blur_lds_kernel, grid, block, 0, stream, in, src, roi, dst, ksize, w);
  else
    hipLaunchKernelGGL(blur_direct_kernel, grid, block, 0, stream, in, src, roi, dst, ksize, w);
  HIP_CHECK(hipGetLastError());
  return Status::Ok;
}

} // namespace

Status gaussian_blur(const TensorDesc& in, const uint8_t* src, const Roi* roi, uint8_t* dst,
                     int ksize, float sigma, hipStream_t stream) {
  return launch_blur(BlurImpl::Lds, in, src, roi, dst, ksize, sigma, stream);
}

namespace detail {

Status gaussian_blur_direct(const TensorDesc& in, const uint8_t* src, const Roi* roi, uint8_t* dst,
                            int ksize, float sigma, hipStream_t stream) {
  return launch_blur(BlurImpl::Direct, in, src, roi, dst, ksize, sigma, stream);
}

Status gaussian_blur_lds(const TensorDesc& in, const uint8_t* src, const Roi* roi, uint8_t* dst,
                         int ksize, float sigma, hipStream_t stream) {
  return launch_blur(BlurImpl::Lds, in, src, roi, dst, ksize, sigma, stream);
}

} // namespace detail
} // namespace hipaug
