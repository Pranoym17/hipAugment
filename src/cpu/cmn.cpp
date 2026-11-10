// CPU reference: crop the ROI, mirror in x, normalize, and change layout to planar float.
#include "common/validate.hpp"
#include "cpu/cpu_util.hpp"

namespace hipaug::cpu {

Status crop_mirror_normalize(const TensorDesc& in, const uint8_t* src, const Roi* roi,
                             const uint8_t* mirror, const float mean[3], const float std[3],
                             const TensorDesc& out, float* dst) {
  const Status st = hipaug::detail::validate_normalize(in, src, roi, mirror, mean, std, out, dst);
  if (st != Status::Ok)
    return st;
  const float inv_std[3] = {1.0f / std[0], 1.0f / std[1], 1.0f / std[2]};

#pragma omp parallel for
  for (int n = 0; n < in.n; ++n) {
    const detail::Box b = detail::clamp_roi(roi[n], in);
    if (b.empty())
      continue;
    for (int y = 0; y < out.h_max; ++y) {
      for (int x = 0; x < out.w_max; ++x) {
        // Output column x reads crop column xs. The crop starts at the ROI origin.
        const int xs = mirror[n] ? out.w_max - 1 - x : x;
        // The ROI should have the output size. If it does not, clamping keeps the
        // read inside the ROI instead of walking off the image.
        const int sx = std::min(std::max(roi[n].x + xs, b.x0), b.x1 - 1);
        const int sy = std::min(std::max(roi[n].y + y, b.y0), b.y1 - 1);
        const uint8_t* p = src + detail::offset(in, n, sy, sx, 0);
        for (int c = 0; c < 3; ++c) {
          const float v = static_cast<float>(p[c * in.stride_c]);
          dst[detail::offset(out, n, y, x, c)] = (v - mean[c]) * inv_std[c];
        }
      }
    }
  }
  return Status::Ok;
}

} // namespace hipaug::cpu
