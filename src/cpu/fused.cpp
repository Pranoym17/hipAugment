// CPU reference: ROI crop, bilinear resize, mirror, normalize and layout change in one pass.
//
// Equal to resize followed by crop_mirror_normalize, except that the bilinear result is
// not rounded to u8 in between. The two paths therefore differ by at most 0.5 / std.
#include "common/validate.hpp"
#include "cpu/cpu_util.hpp"

namespace hipaug::cpu {

Status fused_rcmn(const TensorDesc& in, const uint8_t* src, const Roi* roi, const uint8_t* mirror,
                  const float mean[3], const float std[3], const TensorDesc& out, float* dst) {
  const Status st = hipaug::detail::validate_normalize(in, src, roi, mirror, mean, std, out, dst);
  if (st != Status::Ok)
    return st;
  const float inv_std[3] = {1.0f / std[0], 1.0f / std[1], 1.0f / std[2]};

#pragma omp parallel for
  for (int n = 0; n < in.n; ++n) {
    const detail::Box b = detail::clamp_roi(roi[n], in);
    if (b.empty())
      continue;
    const int rw = b.x1 - b.x0, rh = b.y1 - b.y0;
    const float scale_x = static_cast<float>(rw) / static_cast<float>(out.w_max);
    const float scale_y = static_cast<float>(rh) / static_cast<float>(out.h_max);

    for (int dy = 0; dy < out.h_max; ++dy) {
      const detail::Tap ty = detail::bilinear_tap(dy, scale_y, rh);
      for (int dx = 0; dx < out.w_max; ++dx) {
        // Mirror flips the resized image, so output column dx samples as if it
        // were resized column xs.
        const int xs = mirror[n] ? out.w_max - 1 - dx : dx;
        const detail::Tap tx = detail::bilinear_tap(xs, scale_x, rw);
        const uint8_t* p00 = src + detail::offset(in, n, b.y0 + ty.i0, b.x0 + tx.i0, 0);
        const uint8_t* p01 = src + detail::offset(in, n, b.y0 + ty.i0, b.x0 + tx.i1, 0);
        const uint8_t* p10 = src + detail::offset(in, n, b.y0 + ty.i1, b.x0 + tx.i0, 0);
        const uint8_t* p11 = src + detail::offset(in, n, b.y0 + ty.i1, b.x0 + tx.i1, 0);
        for (int c = 0; c < 3; ++c) {
          const size_t ci = c * in.stride_c;
          const float v = detail::bilerp(p00[ci], p01[ci], p10[ci], p11[ci], tx.w, ty.w);
          dst[detail::offset(out, n, dy, dx, c)] = (v - mean[c]) * inv_std[c];
        }
      }
    }
  }
  return Status::Ok;
}

} // namespace hipaug::cpu
