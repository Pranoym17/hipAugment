// hipAugment public API.
//
// Every op works on a batch of N three-channel images. Images are stored padded to the
// largest size in the batch (h_max, w_max). Each image has its own region of interest (ROI).
// The caller allocates every buffer; no op allocates memory.
//
// Element offset for image n, row y, column x, channel c, in either layout:
//   n * stride_n + y * stride_h + x * stride_w + c * stride_c
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(HIPAUG_HAS_HIP)
#include <hip/hip_runtime.h>
#endif

namespace hipaug {

enum class Status { Ok, InvalidArg, HipError };
enum class Layout { NHWC, NCHW };
enum class DType { U8, F32 };

struct TensorDesc { // strides in elements
  int n, c, h_max, w_max;
  Layout layout;
  DType dtype;
  size_t stride_n, stride_h, stride_w, stride_c;
};

// ROI in pixels, relative to the top-left of the padded image.
struct Roi {
  int x, y, w, h;
};

// Build a densely packed descriptor (no extra row or image padding) for the given layout.
TensorDesc make_desc(int n, int h_max, int w_max, Layout layout, DType dtype);

// Number of elements a buffer described by `d` must hold.
size_t num_elements(const TensorDesc& d);

const char* to_string(Status s);

// CPU reference implementations. Same arguments as the GPU ops, minus the stream,
// and every pointer is a host pointer.
namespace cpu {

Status resize(const TensorDesc& in, const uint8_t* src, const Roi* roi, const TensorDesc& out,
              uint8_t* dst);

Status gaussian_blur(const TensorDesc& in, const uint8_t* src, const Roi* roi, uint8_t* dst,
                     int ksize, float sigma);

Status crop_mirror_normalize(const TensorDesc& in, const uint8_t* src, const Roi* roi,
                             const uint8_t* mirror, const float mean[3], const float std[3],
                             const TensorDesc& out, float* dst);

Status fused_rcmn(const TensorDesc& in, const uint8_t* src, const Roi* roi, const uint8_t* mirror,
                  const float mean[3], const float std[3], const TensorDesc& out, float* dst);

} // namespace cpu

#if defined(HIPAUG_HAS_HIP)

// GPU ops. src, dst, roi and mirror are device pointers. mean and std are host pointers.
// Ops are asynchronous: they return after the kernel is queued on `stream`.

// Bilinear resize of each image's ROI into the full output size. u8 NHWC -> u8 NHWC.
Status resize(const TensorDesc& in, const uint8_t* src, const Roi* roi, const TensorDesc& out,
              uint8_t* dst, hipStream_t stream);

// 3x3 or 5x5 Gaussian blur inside the ROI. Edges clamp to the ROI. Pixels outside the ROI
// are not written, so fill dst first if you need them. src and dst must not alias.
Status gaussian_blur(const TensorDesc& in, const uint8_t* src, const Roi* roi, uint8_t* dst,
                     int ksize, float sigma, hipStream_t stream);

// Crop the ROI (ROI size must equal the output size), flip x where mirror[n] != 0,
// then (x - mean) / std per channel. u8 NHWC -> f32 NCHW.
Status crop_mirror_normalize(const TensorDesc& in, const uint8_t* src, const Roi* roi,
                             const uint8_t* mirror, const float mean[3], const float std[3],
                             const TensorDesc& out, float* dst, hipStream_t stream);

// ROI crop, bilinear resize, mirror, normalize and layout change in one kernel.
Status fused_rcmn(const TensorDesc& in, const uint8_t* src, const Roi* roi, const uint8_t* mirror,
                  const float mean[3], const float std[3], const TensorDesc& out, float* dst,
                  hipStream_t stream);

// The two blur kernels behind gaussian_blur, exposed so tests and benchmarks can run both.
namespace detail {
Status gaussian_blur_direct(const TensorDesc& in, const uint8_t* src, const Roi* roi, uint8_t* dst,
                            int ksize, float sigma, hipStream_t stream);
Status gaussian_blur_lds(const TensorDesc& in, const uint8_t* src, const Roi* roi, uint8_t* dst,
                         int ksize, float sigma, hipStream_t stream);
} // namespace detail

#endif // HIPAUG_HAS_HIP

} // namespace hipaug
