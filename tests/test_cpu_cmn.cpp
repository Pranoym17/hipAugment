// CPU reference tests for crop_mirror_normalize.
#include <gtest/gtest.h>

#include "test_util.hpp"

using namespace hipaug;
using namespace testutil;

namespace {

std::vector<float> run_cmn(const Batch& b, int out_w, int out_h, TensorDesc* out_desc,
                           const float* mean = kMean, const float* std = kStd) {
  *out_desc = make_desc(b.desc.n, out_h, out_w, Layout::NCHW, DType::F32);
  std::vector<float> out(num_elements(*out_desc), -777.0f);
  EXPECT_EQ(cpu::crop_mirror_normalize(b.desc, b.data.data(), b.rois.data(), b.mirror.data(), mean,
                                       std, *out_desc, out.data()),
            Status::Ok);
  return out;
}

} // namespace

TEST(CpuCmn, NormalizesAndWritesPlanarLayout) {
  // One 2x1 image: pixel 0 = (10, 20, 30), pixel 1 = (40, 50, 60).
  Batch b = make_batch({{2, 1}}, 20);
  b.data = {10, 20, 30, 40, 50, 60};
  b.mirror = {0};
  const float mean[3] = {10, 20, 30}, std[3] = {10, 5, 2};
  TensorDesc od;
  const auto out = run_cmn(b, 2, 1, &od, mean, std);
  // NCHW: plane c holds channel c of every pixel.
  const std::vector<float> expect = {0, 3, /* R */ 0, 6, /* G */ 0, 15 /* B */};
  ASSERT_EQ(out.size(), expect.size());
  for (size_t i = 0; i < expect.size(); ++i)
    EXPECT_NEAR(out[i], expect[i], 1e-5f) << "element " << i;
}

TEST(CpuCmn, MirrorFlipsColumnsWithinTheCrop) {
  Batch b = make_batch({{6, 3}}, 21);
  b.rois[0] = Roi{1, 1, 4, 2};
  TensorDesc od;
  b.mirror = {0};
  const auto plain = run_cmn(b, 4, 2, &od);
  b.mirror = {1};
  const auto flipped = run_cmn(b, 4, 2, &od);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 4; ++x)
        EXPECT_EQ(flipped[c * od.stride_c + y * od.stride_h + x],
                  plain[c * od.stride_c + y * od.stride_h + (3 - x)]);
}

TEST(CpuCmn, RoiResultEqualsStandaloneCrop) {
  for (const Case& cs : standard_cases(true)) {
    TensorDesc od;
    const auto out = run_cmn(cs.batch, cs.out_w, cs.out_h, &od);
    for (int n = 0; n < cs.batch.desc.n; ++n) {
      const Batch crop = crop_image(cs.batch, n);
      TensorDesc cd;
      const auto ref = run_cmn(crop, cs.out_w, cs.out_h, &cd);
      EXPECT_EQ(image_slice(out, od, n), ref) << cs.name << " image " << n;
    }
  }
}

TEST(CpuCmn, ReadsOnlyInsideTheRoi) {
  for (const Case& cs : standard_cases(true)) {
    TensorDesc od;
    const auto before = run_cmn(cs.batch, cs.out_w, cs.out_h, &od);
    Batch scrambled = cs.batch;
    scramble_outside_roi(scrambled);
    EXPECT_EQ(before, run_cmn(scrambled, cs.out_w, cs.out_h, &od)) << cs.name;
  }
}

TEST(CpuCmn, RoiSmallerThanOutputReplicatesTheEdge) {
  // The ROI should match the output size. When it is smaller, reads clamp to the
  // ROI edge instead of leaving it.
  Batch b = make_batch({{8, 8}}, 22);
  b.rois[0] = Roi{2, 2, 2, 2};
  b.mirror = {0};
  TensorDesc od;
  const auto before = run_cmn(b, 4, 4, &od);
  scramble_outside_roi(b);
  EXPECT_EQ(before, run_cmn(b, 4, 4, &od));
  // Columns 1, 2 and 3 all read ROI column 1.
  EXPECT_EQ(before[1], before[2]);
  EXPECT_EQ(before[2], before[3]);
}
