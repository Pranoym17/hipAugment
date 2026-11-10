// CPU reference: 3x3 or 5x5 Gaussian blur inside each image's ROI.
#include "common/validate.hpp"
#include "cpu/cpu_util.hpp"

namespace hipaug::cpu {

Status gaussian_blur(const TensorDesc& in, const uint8_t* src, const Roi* roi, uint8_t* dst,
                     int ksize, float sigma) {
  const Status st = hipaug::detail::validate_blur(in, src, roi, dst, ksize, sigma);
  if (st != Status::Ok)
    return st;
  float k2[25];
  hipaug::detail::gaussian_kernel_2d(ksize, sigma, k2);
  const int r = ksize / 2;

#pragma omp parallel for
  for (int n = 0; n < in.n; ++n) {
    const detail::Box b = detail::clamp_roi(roi[n], in);
    if (b.empty())
      continue;
    // Only pixels inside the ROI are written. dst outside the ROI is left as it was.
    for (int y = b.y0; y < b.y1; ++y) {
      for (int x = b.x0; x < b.x1; ++x) {
        float acc[3] = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < ksize; ++j) {
          const int yy = std::min(std::max(y + j - r, b.y0), b.y1 - 1); // clamp to ROI edge
          for (int i = 0; i < ksize; ++i) {
            const int xx = std::min(std::max(x + i - r, b.x0), b.x1 - 1);
            const uint8_t* p = src + detail::offset(in, n, yy, xx, 0);
            const float w = k2[j * ksize + i];
            for (int c = 0; c < 3; ++c)
              acc[c] += w * static_cast<float>(p[c * in.stride_c]);
          }
        }
        uint8_t* d = dst + detail::offset(in, n, y, x, 0);
        for (int c = 0; c < 3; ++c)
          d[c * in.stride_c] = static_cast<uint8_t>(std::min(acc[c], 255.0f) + 0.5f);
      }
    }
  }
  return Status::Ok;
}

} // namespace hipaug::cpu
