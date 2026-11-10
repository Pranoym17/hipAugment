// CPU reference: bilinear resize of each image's ROI into the full output size.
#include "common/validate.hpp"
#include "cpu/cpu_util.hpp"

namespace hipaug::cpu {

Status resize(const TensorDesc& in, const uint8_t* src, const Roi* roi, const TensorDesc& out,
              uint8_t* dst) {
  const Status st = hipaug::detail::validate_resize(in, src, roi, out, dst);
  if (st != Status::Ok)
    return st;

#pragma omp parallel for
  for (int n = 0; n < in.n; ++n) {
    const detail::Box b = detail::clamp_roi(roi[n], in);
    if (b.empty())
      continue; // nothing to sample; this image's output is left untouched
    const int rw = b.x1 - b.x0, rh = b.y1 - b.y0;
    const float scale_x = static_cast<float>(rw) / static_cast<float>(out.w_max);
    const float scale_y = static_cast<float>(rh) / static_cast<float>(out.h_max);

    for (int dy = 0; dy < out.h_max; ++dy) {
      const detail::Tap ty = detail::bilinear_tap(dy, scale_y, rh);
      for (int dx = 0; dx < out.w_max; ++dx) {
        const detail::Tap tx = detail::bilinear_tap(dx, scale_x, rw);
        // Taps are ROI-local; add the ROI origin to get padded-image coordinates.
        const uint8_t* p00 = src + detail::offset(in, n, b.y0 + ty.i0, b.x0 + tx.i0, 0);
        const uint8_t* p01 = src + detail::offset(in, n, b.y0 + ty.i0, b.x0 + tx.i1, 0);
        const uint8_t* p10 = src + detail::offset(in, n, b.y0 + ty.i1, b.x0 + tx.i0, 0);
        const uint8_t* p11 = src + detail::offset(in, n, b.y0 + ty.i1, b.x0 + tx.i1, 0);
        uint8_t* d = dst + detail::offset(out, n, dy, dx, 0);
        for (int c = 0; c < 3; ++c) {
          const size_t ci = c * in.stride_c;
          const float v = detail::bilerp(p00[ci], p01[ci], p10[ci], p11[ci], tx.w, ty.w);
          d[c * out.stride_c] = static_cast<uint8_t>(v + 0.5f); // round to nearest
        }
      }
    }
  }
  return Status::Ok;
}

} // namespace hipaug::cpu
