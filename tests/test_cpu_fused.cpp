// CPU reference tests for fused_rcmn, including equality with the separate path.
#include <gtest/gtest.h>

#include "test_util.hpp"

using namespace hipaug;
using namespace testutil;

namespace {

std::vector<float> run_fused(const Batch& b, int out_w, int out_h, TensorDesc* out_desc) {
  *out_desc = make_desc(b.desc.n, out_h, out_w, Layout::NCHW, DType::F32);
  std::vector<float> out(num_elements(*out_desc), -777.0f);
  EXPECT_EQ(cpu::fused_rcmn(b.desc, b.data.data(), b.rois.data(), b.mirror.data(), kMean, kStd,
                            *out_desc, out.data()),
            Status::Ok);
  return out;
}

// resize into a u8 intermediate, then crop_mirror_normalize over the whole intermediate.
std::vector<float> run_separate(const Batch& b, int out_w, int out_h) {
  const TensorDesc mid = make_desc(b.desc.n, out_h, out_w, Layout::NHWC, DType::U8);
  std::vector<uint8_t> resized(num_elements(mid));
  EXPECT_EQ(cpu::resize(b.desc, b.data.data(), b.rois.data(), mid, resized.data()), Status::Ok);
  const std::vector<Roi> full(b.desc.n, Roi{0, 0, out_w, out_h});
  const TensorDesc od = make_desc(b.desc.n, out_h, out_w, Layout::NCHW, DType::F32);
  std::vector<float> out(num_elements(od), -777.0f);
  EXPECT_EQ(cpu::crop_mirror_normalize(mid, resized.data(), full.data(), b.mirror.data(), kMean,
                                       kStd, od, out.data()),
            Status::Ok);
  return out;
}

} // namespace

TEST(CpuFused, MatchesResizeThenCropMirrorNormalize) {
  // The separate path rounds the bilinear result to u8 before normalizing and the
  // fused path does not, so they may differ by half a u8 step: 0.5 / std.
  for (const Case& cs : standard_cases(false)) {
    TensorDesc od;
    const auto fused = run_fused(cs.batch, cs.out_w, cs.out_h, &od);
    const auto separate = run_separate(cs.batch, cs.out_w, cs.out_h);
    ASSERT_EQ(fused.size(), separate.size());
    for (int n = 0; n < od.n; ++n)
      for (int c = 0; c < 3; ++c) {
        const float tol = 0.5f / kStd[c] + kTolF32;
        for (size_t i = 0; i < od.stride_c; ++i) {
          const size_t idx = n * od.stride_n + c * od.stride_c + i;
          ASSERT_NEAR(fused[idx], separate[idx], tol) << cs.name << " n=" << n << " c=" << c;
        }
      }
  }
}

TEST(CpuFused, WithoutScalingEqualsCropMirrorNormalizeExactly) {
  // When the ROI already has the output size the bilinear weights are zero,
  // so there is no rounding difference at all.
  for (const Case& cs : standard_cases(true)) {
    TensorDesc od;
    const auto fused = run_fused(cs.batch, cs.out_w, cs.out_h, &od);
    std::vector<float> cmn(num_elements(od), -777.0f);
    ASSERT_EQ(cpu::crop_mirror_normalize(cs.batch.desc, cs.batch.data.data(), cs.batch.rois.data(),
                                         cs.batch.mirror.data(), kMean, kStd, od, cmn.data()),
              Status::Ok);
    EXPECT_EQ(fused, cmn) << cs.name;
  }
}

TEST(CpuFused, MirrorFlipsTheResizedImage) {
  Batch b = make_batch({{13, 9}}, 40);
  TensorDesc od;
  b.mirror = {0};
  const auto plain = run_fused(b, 7, 5, &od);
  b.mirror = {1};
  const auto flipped = run_fused(b, 7, 5, &od);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < 5; ++y)
      for (int x = 0; x < 7; ++x)
        EXPECT_EQ(flipped[c * od.stride_c + y * od.stride_h + x],
                  plain[c * od.stride_c + y * od.stride_h + (6 - x)]);
}

TEST(CpuFused, RoiResultEqualsStandaloneCrop) {
  for (const Case& cs : standard_cases(false)) {
    TensorDesc od;
    const auto out = run_fused(cs.batch, cs.out_w, cs.out_h, &od);
    for (int n = 0; n < cs.batch.desc.n; ++n) {
      const Batch crop = crop_image(cs.batch, n);
      TensorDesc cd;
      EXPECT_EQ(image_slice(out, od, n), run_fused(crop, cs.out_w, cs.out_h, &cd))
          << cs.name << " image " << n;
    }
  }
}

TEST(CpuFused, ReadsOnlyInsideTheRoi) {
  for (const Case& cs : standard_cases(false)) {
    TensorDesc od;
    const auto before = run_fused(cs.batch, cs.out_w, cs.out_h, &od);
    Batch scrambled = cs.batch;
    scramble_outside_roi(scrambled);
    EXPECT_EQ(before, run_fused(scrambled, cs.out_w, cs.out_h, &od)) << cs.name;
  }
}
