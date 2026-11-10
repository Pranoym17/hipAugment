// CPU reference tests for gaussian_blur.
#include <gtest/gtest.h>

#include "common/validate.hpp"
#include "test_util.hpp"

using namespace hipaug;
using namespace testutil;

namespace {

// dst starts as a copy of src with every byte inverted, so any pixel the op
// does not write is easy to tell apart from a blurred one.
std::vector<uint8_t> run_blur(const Batch& b, int ksize, float sigma) {
  std::vector<uint8_t> out(b.data.size());
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = b.data[i] ^ 0xFF;
  EXPECT_EQ(cpu::gaussian_blur(b.desc, b.data.data(), b.rois.data(), out.data(), ksize, sigma),
            Status::Ok);
  return out;
}

size_t at(const TensorDesc& d, int n, int y, int x, int c) {
  return n * d.stride_n + y * d.stride_h + x * d.stride_w + c;
}

} // namespace

TEST(CpuBlur, ConstantImageStaysConstant) {
  for (int ksize : {3, 5}) {
    Batch b = make_batch({{12, 9}}, 30);
    std::fill(b.data.begin(), b.data.end(), 200);
    const auto out = run_blur(b, ksize, 1.5f);
    for (uint8_t v : out)
      EXPECT_EQ(v, 200);
  }
}

TEST(CpuBlur, ImpulseResponseIsTheKernel) {
  for (int ksize : {3, 5}) {
    Batch b = make_batch({{11, 11}}, 31);
    std::fill(b.data.begin(), b.data.end(), 0);
    for (int c = 0; c < 3; ++c)
      b.data[at(b.desc, 0, 5, 5, c)] = 255;
    const float sigma = 1.0f;
    const auto out = run_blur(b, ksize, sigma);
    float k[25];
    detail::gaussian_kernel_2d(ksize, sigma, k);
    const int r = ksize / 2;
    for (int y = 0; y < 11; ++y)
      for (int x = 0; x < 11; ++x) {
        const int j = y - 5 + r, i = x - 5 + r;
        const bool inside = j >= 0 && j < ksize && i >= 0 && i < ksize;
        const int expect = inside ? static_cast<int>(255.0f * k[j * ksize + i] + 0.5f) : 0;
        EXPECT_NEAR(out[at(b.desc, 0, y, x, 0)], expect, 1) << "x=" << x << " y=" << y;
      }
  }
}

TEST(CpuBlur, TinySigmaIsIdentity) {
  Batch b = make_batch({{10, 8}}, 32);
  EXPECT_EQ(run_blur(b, 3, 0.05f), b.data);
}

TEST(CpuBlur, PixelsOutsideTheRoiAreUntouched) {
  for (const Case& cs : standard_cases(false)) {
    const Batch& b = cs.batch;
    const auto out = run_blur(b, 5, 1.2f);
    for (int n = 0; n < b.desc.n; ++n) {
      const Roi r = b.rois[n];
      for (int y = 0; y < b.desc.h_max; ++y)
        for (int x = 0; x < b.desc.w_max; ++x) {
          if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h)
            continue;
          const size_t i = at(b.desc, n, y, x, 0);
          ASSERT_EQ(out[i], b.data[i] ^ 0xFF) << cs.name << " n=" << n << " x=" << x << " y=" << y;
        }
    }
  }
}

TEST(CpuBlur, RoiResultEqualsBlurOfStandaloneCrop) {
  // Edges clamp to the ROI, so blurring in place must equal blurring the crop alone.
  for (int ksize : {3, 5}) {
    for (const Case& cs : standard_cases(false)) {
      const Batch& b = cs.batch;
      const auto out = run_blur(b, ksize, 1.2f);
      for (int n = 0; n < b.desc.n; ++n) {
        const Batch crop = crop_image(b, n);
        const auto ref = run_blur(crop, ksize, 1.2f);
        const Roi r = b.rois[n];
        for (int y = 0; y < r.h; ++y)
          for (int x = 0; x < r.w; ++x)
            for (int c = 0; c < 3; ++c)
              ASSERT_EQ(out[at(b.desc, n, r.y + y, r.x + x, c)], ref[at(crop.desc, 0, y, x, c)])
                  << cs.name << " n=" << n;
      }
    }
  }
}

TEST(CpuBlur, ReadsOnlyInsideTheRoi) {
  for (const Case& cs : standard_cases(false)) {
    const Batch& b = cs.batch;
    Batch scrambled = b;
    scramble_outside_roi(scrambled);
    const auto before = run_blur(b, 5, 1.2f);
    const auto after = run_blur(scrambled, 5, 1.2f);
    for (int n = 0; n < b.desc.n; ++n) {
      const Roi r = b.rois[n];
      for (int y = r.y; y < r.y + r.h; ++y)
        for (int x = r.x; x < r.x + r.w; ++x)
          ASSERT_EQ(before[at(b.desc, n, y, x, 0)], after[at(b.desc, n, y, x, 0)]) << cs.name;
    }
  }
}
