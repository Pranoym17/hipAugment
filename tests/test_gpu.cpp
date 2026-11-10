// GPU vs CPU tests. Every HIP op must match its CPU reference within tolerance on the
// five standard cases, and fused_rcmn must match the separate GPU path.
#include <gtest/gtest.h>

#include "common/device_buffer.hpp"
#include "test_util.hpp"

using namespace hipaug;
using namespace testutil;

namespace {

// A batch uploaded to the device.
struct DeviceBatch {
  DeviceBuffer<uint8_t> data;
  DeviceBuffer<Roi> rois;
  DeviceBuffer<uint8_t> mirror;

  Status upload(const Batch& b) {
    Status st = data.upload(b.data);
    if (st != Status::Ok)
      return st;
    st = rois.upload(b.rois);
    if (st != Status::Ok)
      return st;
    return mirror.upload(b.mirror);
  }
};

// Each helper runs one op on the GPU, starting from the same pre-filled output the
// CPU run uses, and copies the result back.

Status gpu_resize(const Batch& b, const TensorDesc& od, std::vector<uint8_t>& out,
                  hipStream_t stream = nullptr) {
  DeviceBatch dev;
  DeviceBuffer<uint8_t> dst;
  Status st = dev.upload(b);
  if (st != Status::Ok)
    return st;
  st = dst.upload(out);
  if (st != Status::Ok)
    return st;
  st = resize(b.desc, dev.data.get(), dev.rois.get(), od, dst.get(), stream);
  if (st != Status::Ok)
    return st;
  HIP_CHECK(hipStreamSynchronize(stream));
  return dst.download(out);
}

using BlurFn = Status (*)(const TensorDesc&, const uint8_t*, const Roi*, uint8_t*, int, float,
                          hipStream_t);

Status gpu_blur(BlurFn fn, const Batch& b, int ksize, float sigma, std::vector<uint8_t>& out) {
  DeviceBatch dev;
  DeviceBuffer<uint8_t> dst;
  Status st = dev.upload(b);
  if (st != Status::Ok)
    return st;
  st = dst.upload(out);
  if (st != Status::Ok)
    return st;
  st = fn(b.desc, dev.data.get(), dev.rois.get(), dst.get(), ksize, sigma, nullptr);
  if (st != Status::Ok)
    return st;
  return dst.download(out);
}

using NormFn = Status (*)(const TensorDesc&, const uint8_t*, const Roi*, const uint8_t*,
                          const float*, const float*, const TensorDesc&, float*, hipStream_t);

Status gpu_normalize(NormFn fn, const Batch& b, const TensorDesc& od, std::vector<float>& out) {
  DeviceBatch dev;
  DeviceBuffer<float> dst;
  Status st = dev.upload(b);
  if (st != Status::Ok)
    return st;
  st = dst.upload(out);
  if (st != Status::Ok)
    return st;
  st = fn(b.desc, dev.data.get(), dev.rois.get(), dev.mirror.get(), kMean, kStd, od, dst.get(),
          nullptr);
  if (st != Status::Ok)
    return st;
  return dst.download(out);
}

// resize then crop_mirror_normalize, both on the GPU, with a device-side u8 intermediate.
Status gpu_separate(const Batch& b, const TensorDesc& od, std::vector<float>& out) {
  const TensorDesc mid = make_desc(b.desc.n, od.h_max, od.w_max, Layout::NHWC, DType::U8);
  DeviceBatch dev;
  DeviceBuffer<uint8_t> resized;
  DeviceBuffer<Roi> full;
  DeviceBuffer<float> dst;
  Status st = dev.upload(b);
  if (st != Status::Ok)
    return st;
  st = resized.alloc(num_elements(mid));
  if (st != Status::Ok)
    return st;
  st = full.upload(std::vector<Roi>(b.desc.n, Roi{0, 0, od.w_max, od.h_max}));
  if (st != Status::Ok)
    return st;
  st = dst.upload(out);
  if (st != Status::Ok)
    return st;
  st = resize(b.desc, dev.data.get(), dev.rois.get(), mid, resized.get(), nullptr);
  if (st != Status::Ok)
    return st;
  st = crop_mirror_normalize(mid, resized.get(), full.get(), dev.mirror.get(), kMean, kStd, od,
                             dst.get(), nullptr);
  if (st != Status::Ok)
    return st;
  return dst.download(out);
}

// The standard cases plus one batch at realistic sizes.
std::vector<Case> gpu_cases(bool roi_equals_out) {
  std::vector<Case> cases = standard_cases(roi_equals_out);
  Case big{"large_224",
           make_batch({{500, 375}, {640, 480}, {333, 500}, {224, 224}, {800, 600}, {256, 341}}, 9),
           224, 224};
  if (roi_equals_out)
    for (auto& r : big.batch.rois)
      r = Roi{(r.w - 224) / 2, (r.h - 224) / 2, 224, 224}; // centre crop
  cases.push_back(std::move(big));
  return cases;
}

std::vector<uint8_t> inverted(const std::vector<uint8_t>& v) {
  std::vector<uint8_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = v[i] ^ 0xFF;
  return out;
}

} // namespace

TEST(GpuResize, MatchesCpuOnStandardCases) {
  for (const Case& cs : gpu_cases(false)) {
    const Batch& b = cs.batch;
    const TensorDesc od = make_desc(b.desc.n, cs.out_h, cs.out_w, Layout::NHWC, DType::U8);
    std::vector<uint8_t> ref(num_elements(od), 0xAB), out(num_elements(od), 0xAB);
    ASSERT_EQ(cpu::resize(b.desc, b.data.data(), b.rois.data(), od, ref.data()), Status::Ok);
    ASSERT_EQ(gpu_resize(b, od, out), Status::Ok);
    EXPECT_LE(max_abs_diff(out, ref), kTolU8) << cs.name;
  }
}

TEST(GpuResize, WorksOnACallerStream) {
  hipStream_t stream = nullptr;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
  const Case cs = gpu_cases(false)[1];
  const Batch& b = cs.batch;
  const TensorDesc od = make_desc(b.desc.n, cs.out_h, cs.out_w, Layout::NHWC, DType::U8);
  std::vector<uint8_t> ref(num_elements(od), 0xAB), out(num_elements(od), 0xAB);
  ASSERT_EQ(cpu::resize(b.desc, b.data.data(), b.rois.data(), od, ref.data()), Status::Ok);
  ASSERT_EQ(gpu_resize(b, od, out, stream), Status::Ok);
  EXPECT_LE(max_abs_diff(out, ref), kTolU8);
  EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST(GpuResize, EmptyRoiLeavesOutputUntouched) {
  Batch b = make_batch({{8, 8}, {8, 8}}, 15);
  b.rois[1] = Roi{3, 3, 0, 4};
  const TensorDesc od = make_desc(2, 4, 4, Layout::NHWC, DType::U8);
  std::vector<uint8_t> out(num_elements(od), 0xAB);
  ASSERT_EQ(gpu_resize(b, od, out), Status::Ok);
  for (uint8_t v : image_slice(out, od, 1))
    EXPECT_EQ(v, 0xAB);
}

TEST(GpuCmn, MatchesCpuOnStandardCases) {
  for (const Case& cs : gpu_cases(true)) {
    const Batch& b = cs.batch;
    const TensorDesc od = make_desc(b.desc.n, cs.out_h, cs.out_w, Layout::NCHW, DType::F32);
    std::vector<float> ref(num_elements(od), -777.0f), out(num_elements(od), -777.0f);
    ASSERT_EQ(cpu::crop_mirror_normalize(b.desc, b.data.data(), b.rois.data(), b.mirror.data(),
                                         kMean, kStd, od, ref.data()),
              Status::Ok);
    ASSERT_EQ(gpu_normalize(crop_mirror_normalize, b, od, out), Status::Ok);
    EXPECT_LE(max_abs_diff(out, ref), kTolF32) << cs.name;
  }
}

TEST(GpuBlur, BothKernelsMatchCpuOnStandardCases) {
  const struct {
    const char* name;
    BlurFn fn;
  } impls[] = {{"direct", detail::gaussian_blur_direct},
               {"lds", detail::gaussian_blur_lds},
               {"default", gaussian_blur}};
  for (const auto& impl : impls)
    for (int ksize : {3, 5})
      for (const Case& cs : gpu_cases(false)) {
        const Batch& b = cs.batch;
        // Start from inverted pixels so an unwritten pixel cannot pass by accident.
        std::vector<uint8_t> ref = inverted(b.data), out = inverted(b.data);
        ASSERT_EQ(cpu::gaussian_blur(b.desc, b.data.data(), b.rois.data(), ref.data(), ksize, 1.2f),
                  Status::Ok);
        ASSERT_EQ(gpu_blur(impl.fn, b, ksize, 1.2f, out), Status::Ok);
        EXPECT_LE(max_abs_diff(out, ref), kTolU8)
            << impl.name << " ksize=" << ksize << " " << cs.name;
      }
}

TEST(GpuBlur, DirectAndLdsKernelsAgreeExactly) {
  for (int ksize : {3, 5})
    for (const Case& cs : gpu_cases(false)) {
      const Batch& b = cs.batch;
      std::vector<uint8_t> a = inverted(b.data), c = inverted(b.data);
      ASSERT_EQ(gpu_blur(detail::gaussian_blur_direct, b, ksize, 1.2f, a), Status::Ok);
      ASSERT_EQ(gpu_blur(detail::gaussian_blur_lds, b, ksize, 1.2f, c), Status::Ok);
      EXPECT_EQ(a, c) << "ksize=" << ksize << " " << cs.name;
    }
}

TEST(GpuFused, MatchesCpuOnStandardCases) {
  for (const Case& cs : gpu_cases(false)) {
    const Batch& b = cs.batch;
    const TensorDesc od = make_desc(b.desc.n, cs.out_h, cs.out_w, Layout::NCHW, DType::F32);
    std::vector<float> ref(num_elements(od), -777.0f), out(num_elements(od), -777.0f);
    ASSERT_EQ(cpu::fused_rcmn(b.desc, b.data.data(), b.rois.data(), b.mirror.data(), kMean, kStd,
                              od, ref.data()),
              Status::Ok);
    ASSERT_EQ(gpu_normalize(fused_rcmn, b, od, out), Status::Ok);
    EXPECT_LE(max_abs_diff(out, ref), kTolF32) << cs.name;
  }
}

TEST(GpuFused, MatchesSeparateGpuPath) {
  // The separate path rounds to u8 between resize and normalize and the fused path
  // does not, so the tolerance is half a u8 step in normalized units: 0.5 / std.
  for (const Case& cs : gpu_cases(false)) {
    const Batch& b = cs.batch;
    const TensorDesc od = make_desc(b.desc.n, cs.out_h, cs.out_w, Layout::NCHW, DType::F32);
    std::vector<float> fused(num_elements(od), -777.0f), separate(num_elements(od), -777.0f);
    ASSERT_EQ(gpu_normalize(fused_rcmn, b, od, fused), Status::Ok);
    ASSERT_EQ(gpu_separate(b, od, separate), Status::Ok);
    for (int n = 0; n < od.n; ++n)
      for (int c = 0; c < 3; ++c) {
        const float tol = 0.5f / kStd[c] + kTolF32;
        float worst = 0.0f;
        for (size_t i = 0; i < od.stride_c; ++i) {
          const size_t idx = n * od.stride_n + c * od.stride_c + i;
          worst = std::max(worst, std::fabs(fused[idx] - separate[idx]));
        }
        EXPECT_LE(worst, tol) << cs.name << " n=" << n << " c=" << c;
      }
  }
}

TEST(GpuOps, InvalidArgumentsAreRejectedBeforeAnyLaunch) {
  const TensorDesc in = make_desc(1, 8, 8, Layout::NHWC, DType::U8);
  const TensorDesc out_u8 = make_desc(1, 4, 4, Layout::NHWC, DType::U8);
  const TensorDesc out_f32 = make_desc(1, 4, 4, Layout::NCHW, DType::F32);
  DeviceBuffer<uint8_t> src, dst;
  DeviceBuffer<Roi> roi;
  DeviceBuffer<float> fdst;
  ASSERT_EQ(src.alloc(num_elements(in)), Status::Ok);
  ASSERT_EQ(dst.alloc(num_elements(in)), Status::Ok);
  ASSERT_EQ(fdst.alloc(num_elements(out_f32)), Status::Ok);
  ASSERT_EQ(roi.upload({Roi{0, 0, 8, 8}}), Status::Ok);

  EXPECT_EQ(resize(in, nullptr, roi.get(), out_u8, dst.get(), nullptr), Status::InvalidArg);
  EXPECT_EQ(resize(in, src.get(), roi.get(), out_f32, dst.get(), nullptr), Status::InvalidArg);
  EXPECT_EQ(gaussian_blur(in, src.get(), roi.get(), dst.get(), 7, 1.0f, nullptr),
            Status::InvalidArg);
  EXPECT_EQ(gaussian_blur(in, src.get(), roi.get(), src.get(), 3, 1.0f, nullptr),
            Status::InvalidArg);
  const float zero_std[3] = {0, 1, 1};
  EXPECT_EQ(crop_mirror_normalize(in, src.get(), roi.get(), src.get(), kMean, zero_std, out_f32,
                                  fdst.get(), nullptr),
            Status::InvalidArg);
  EXPECT_EQ(
      fused_rcmn(in, src.get(), roi.get(), nullptr, kMean, kStd, out_f32, fdst.get(), nullptr),
      Status::InvalidArg);
}
