// Host-side argument validation shared by the CPU reference and the HIP ops.
// ROI contents are not checked here: on the GPU path they live in device memory.
// Both backends clamp each ROI to the padded image instead (see clamp_roi).
#pragma once

#include "hipaug/hipaug.hpp"

namespace hipaug::detail {

Status validate_resize(const TensorDesc& in, const void* src, const void* roi,
                       const TensorDesc& out, const void* dst);

Status validate_blur(const TensorDesc& in, const void* src, const void* roi, const void* dst,
                     int ksize, float sigma);

// Used by crop_mirror_normalize and fused_rcmn.
Status validate_normalize(const TensorDesc& in, const void* src, const void* roi,
                          const void* mirror, const float* mean, const float* std,
                          const TensorDesc& out, const void* dst);

// Fill k2[0 .. ksize*ksize) with a normalized 2D Gaussian kernel, row-major.
// ksize must be 3 or 5. Unused entries of k2[25] are set to 0.
void gaussian_kernel_2d(int ksize, float sigma, float k2[25]);

} // namespace hipaug::detail
