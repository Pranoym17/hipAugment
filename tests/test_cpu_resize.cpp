// CPU reference tests for resize: hand-computed values and invariants.
#include <gtest/gtest.h>

#include "test_util.hpp"

using namespace hipaug;
using namespace testutil;

namespace {

// Run the CPU resize. The output starts filled with 0xAB so untouched bytes are visible.
std::vector<uint8_t> run_resize(const Batch& b, int out_w, int out_h, TensorDesc* out_desc) {
  *out_desc = make_desc(b.desc.n, out_h, out_w, Layout::NHWC, DType::U8);
  std::vector<uint8_t> out(num_elements(*out_desc), 0xAB);
  EXPECT_EQ(cpu::resize(b.desc, b.data.data(), b.rois.data(), *out_desc, out.data()), Status::Ok);
  return out;
}

// Set all three channels of pixel (x, y) in a single-image batch.
void set_px(Batch& b, int x, int y, uint8_t v) {
  for (int c = 0; c < 3; ++c)
    b.data[y * b.desc.stride_h + x * b.desc.stride_w + c] = v;
}

} // namespace

TEST(CpuResize, SameSizeIsIdentity) {
  Batch b = make_batch({{9, 7}}, 10);
  TensorDesc od;
  const auto out = run_resize(b, 9, 7, &od);
  EXPECT_EQ(out, b.data);
}

TEST(CpuResize, Upsample2To4MatchesHandComputed) {
  // Source row [0, 100]. Pixel centres: src = (dst + 0.5) * 0.5 - 0.5
  //   dst 0 -> -0.25 (clamped to 0) -> 0      dst 1 -> 0.25 -> 25
  //   dst 2 ->  0.75 -> 75                    dst 3 -> 1.25 (clamped to 1) -> 100
  Batch b = make_batch({{2, 1}}, 11);
  set_px(b, 0, 0, 0);
  set_px(b, 1, 0, 100);
  TensorDesc od;
  const auto out = run_resize(b, 4, 1, &od);
  const uint8_t expect[4] = {0, 25, 75, 100};
  for (int x = 0; x < 4; ++x)
    for (int c = 0; c < 3; ++c)
      EXPECT_EQ(out[x * 3 + c], expect[x]) << "x=" << x;
}

TEST(CpuResize, Downsample2xAveragesEach2x2Block) {
  // 4x4 -> 2x2: src = (dst + 0.5) * 2 - 0.5 = 0.5 or 2.5, so each output is the
  // mean of one 2x2 block.
  Batch b = make_batch({{4, 4}}, 12);
  TensorDesc od;
  const auto out = run_resize(b, 2, 2, &od);
  for (int oy = 0; oy < 2; ++oy)
    for (int ox = 0; ox < 2; ++ox)
      for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int j = 0; j < 2; ++j)
          for (int i = 0; i < 2; ++i)
            sum += b.data[(2 * oy + j) * b.desc.stride_h + (2 * ox + i) * 3 + c];
        const uint8_t expect = static_cast<uint8_t>(sum / 4.0f + 0.5f);
        EXPECT_EQ(out[oy * od.stride_h + ox * 3 + c], expect);
      }
}

TEST(CpuResize, OneByOneRoiGivesConstantImage) {
  Batch b = make_batch({{16, 12}}, 13);
  b.rois[0] = Roi{5, 7, 1, 1};
  TensorDesc od;
  const auto out = run_resize(b, 8, 8, &od);
  for (int i = 0; i < 64; ++i)
    for (int c = 0; c < 3; ++c)
      EXPECT_EQ(out[i * 3 + c], b.data[7 * b.desc.stride_h + 5 * 3 + c]);
}

TEST(CpuResize, RoiResultEqualsResizeOfStandaloneCrop) {
  for (const Case& cs : standard_cases(false)) {
    TensorDesc od;
    const auto out = run_resize(cs.batch, cs.out_w, cs.out_h, &od);
    for (int n = 0; n < cs.batch.desc.n; ++n) {
      const Batch crop = crop_image(cs.batch, n);
      TensorDesc cd;
      const auto ref = run_resize(crop, cs.out_w, cs.out_h, &cd);
      EXPECT_EQ(image_slice(out, od, n), ref) << cs.name << " image " << n;
    }
  }
}

TEST(CpuResize, ReadsOnlyInsideTheRoi) {
  for (const Case& cs : standard_cases(false)) {
    TensorDesc od;
    const auto before = run_resize(cs.batch, cs.out_w, cs.out_h, &od);
    Batch scrambled = cs.batch;
    scramble_outside_roi(scrambled);
    const auto after = run_resize(scrambled, cs.out_w, cs.out_h, &od);
    EXPECT_EQ(before, after) << cs.name;
  }
}

TEST(CpuResize, RoiPartlyOutsideImageIsClamped) {
  // A ROI hanging off the bottom-right corner behaves like its clamped version.
  Batch a = make_batch({{20, 20}}, 14);
  Batch b = a;
  a.rois[0] = Roi{10, 12, 50, 50};
  b.rois[0] = Roi{10, 12, 10, 8};
  TensorDesc od;
  EXPECT_EQ(run_resize(a, 6, 6, &od), run_resize(b, 6, 6, &od));
}

TEST(CpuResize, EmptyRoiLeavesOutputUntouched) {
  Batch b = make_batch({{8, 8}, {8, 8}}, 15);
  b.rois[1] = Roi{3, 3, 0, 4};
  TensorDesc od;
  const auto out = run_resize(b, 4, 4, &od);
  for (uint8_t v : image_slice(out, od, 1))
    EXPECT_EQ(v, 0xAB);
}
