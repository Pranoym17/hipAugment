// Descriptor helpers and argument validation.
#include <gtest/gtest.h>

#include "common/validate.hpp"
#include "hipaug/hipaug.hpp"

using namespace hipaug;

TEST(Desc, NhwcStrides) {
  const TensorDesc d = make_desc(4, 10, 20, Layout::NHWC, DType::U8);
  EXPECT_EQ(d.stride_c, 1u);
  EXPECT_EQ(d.stride_w, 3u);
  EXPECT_EQ(d.stride_h, 60u);
  EXPECT_EQ(d.stride_n, 600u);
  EXPECT_EQ(num_elements(d), 2400u);
}

TEST(Desc, NchwStrides) {
  const TensorDesc d = make_desc(2, 10, 20, Layout::NCHW, DType::F32);
  EXPECT_EQ(d.stride_w, 1u);
  EXPECT_EQ(d.stride_h, 20u);
  EXPECT_EQ(d.stride_c, 200u);
  EXPECT_EQ(d.stride_n, 600u);
  EXPECT_EQ(num_elements(d), 1200u);
}

TEST(Status, Strings) {
  EXPECT_STREQ(to_string(Status::Ok), "Ok");
  EXPECT_STREQ(to_string(Status::InvalidArg), "InvalidArg");
  EXPECT_STREQ(to_string(Status::HipError), "HipError");
}

TEST(Validate, ResizeRejectsBadArguments) {
  const TensorDesc in = make_desc(2, 8, 8, Layout::NHWC, DType::U8);
  const TensorDesc out = make_desc(2, 4, 4, Layout::NHWC, DType::U8);
  int a = 0, b = 0, r = 0; // any non-null addresses
  EXPECT_EQ(detail::validate_resize(in, &a, &r, out, &b), Status::Ok);
  EXPECT_EQ(detail::validate_resize(in, nullptr, &r, out, &b), Status::InvalidArg);
  EXPECT_EQ(detail::validate_resize(in, &a, nullptr, out, &b), Status::InvalidArg);
  EXPECT_EQ(detail::validate_resize(in, &a, &r, out, nullptr), Status::InvalidArg);
  EXPECT_EQ(detail::validate_resize(in, &a, &r, out, &a), Status::InvalidArg); // aliasing

  TensorDesc bad = out;
  bad.n = 3; // batch size mismatch
  EXPECT_EQ(detail::validate_resize(in, &a, &r, bad, &b), Status::InvalidArg);
  bad = out;
  bad.layout = Layout::NCHW;
  EXPECT_EQ(detail::validate_resize(in, &a, &r, bad, &b), Status::InvalidArg);
  bad = in;
  bad.c = 1;
  EXPECT_EQ(detail::validate_resize(bad, &a, &r, out, &b), Status::InvalidArg);
  bad = in;
  bad.stride_n = 10; // too small to hold one image
  EXPECT_EQ(detail::validate_resize(bad, &a, &r, out, &b), Status::InvalidArg);
  bad = in;
  bad.h_max = 0;
  EXPECT_EQ(detail::validate_resize(bad, &a, &r, out, &b), Status::InvalidArg);
}

TEST(Validate, BlurRejectsBadArguments) {
  const TensorDesc in = make_desc(1, 8, 8, Layout::NHWC, DType::U8);
  int a = 0, b = 0, r = 0;
  EXPECT_EQ(detail::validate_blur(in, &a, &r, &b, 3, 1.0f), Status::Ok);
  EXPECT_EQ(detail::validate_blur(in, &a, &r, &b, 5, 1.0f), Status::Ok);
  EXPECT_EQ(detail::validate_blur(in, &a, &r, &b, 7, 1.0f), Status::InvalidArg);
  EXPECT_EQ(detail::validate_blur(in, &a, &r, &b, 4, 1.0f), Status::InvalidArg);
  EXPECT_EQ(detail::validate_blur(in, &a, &r, &b, 3, 0.0f), Status::InvalidArg);
  EXPECT_EQ(detail::validate_blur(in, &a, &r, &b, 3, -1.0f), Status::InvalidArg);
  EXPECT_EQ(detail::validate_blur(in, &a, &r, &a, 3, 1.0f), Status::InvalidArg); // in place
}

TEST(Validate, NormalizeRejectsBadArguments) {
  const TensorDesc in = make_desc(2, 8, 8, Layout::NHWC, DType::U8);
  const TensorDesc out = make_desc(2, 4, 4, Layout::NCHW, DType::F32);
  const float mean[3] = {0, 0, 0}, std[3] = {1, 1, 1};
  int a = 0, b = 0, r = 0, m = 0;
  EXPECT_EQ(detail::validate_normalize(in, &a, &r, &m, mean, std, out, &b), Status::Ok);
  EXPECT_EQ(detail::validate_normalize(in, &a, &r, nullptr, mean, std, out, &b),
            Status::InvalidArg);
  EXPECT_EQ(detail::validate_normalize(in, &a, &r, &m, nullptr, std, out, &b), Status::InvalidArg);
  const float zero_std[3] = {1, 0, 1};
  EXPECT_EQ(detail::validate_normalize(in, &a, &r, &m, mean, zero_std, out, &b),
            Status::InvalidArg);
  TensorDesc bad = out;
  bad.layout = Layout::NHWC; // output must be planar
  EXPECT_EQ(detail::validate_normalize(in, &a, &r, &m, mean, std, bad, &b), Status::InvalidArg);
  bad = out;
  bad.dtype = DType::U8;
  EXPECT_EQ(detail::validate_normalize(in, &a, &r, &m, mean, std, bad, &b), Status::InvalidArg);
}

TEST(Gaussian, KernelIsNormalizedAndSymmetric) {
  for (int ksize : {3, 5}) {
    float k[25];
    detail::gaussian_kernel_2d(ksize, 1.2f, k);
    float sum = 0.0f;
    for (int i = 0; i < ksize * ksize; ++i)
      sum += k[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    for (int j = 0; j < ksize; ++j)
      for (int i = 0; i < ksize; ++i)
        EXPECT_FLOAT_EQ(k[j * ksize + i], k[i * ksize + j]);
    const int centre = (ksize / 2) * ksize + ksize / 2;
    for (int i = 0; i < ksize * ksize; ++i)
      EXPECT_LE(k[i], k[centre]);
  }
}
