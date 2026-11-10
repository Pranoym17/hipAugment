// Shared helpers for the test suite: seeded random batches, the five standard
// test cases, and buffer comparison.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "hipaug/hipaug.hpp"

namespace testutil {

using hipaug::Roi;
using hipaug::TensorDesc;

// A padded u8 NHWC batch on the host. Every byte, padding included, is random,
// so an op that reads outside its ROI produces a visible difference.
struct Batch {
  TensorDesc desc{};
  std::vector<uint8_t> data;
  std::vector<Roi> rois;       // defaults to each image's full size
  std::vector<uint8_t> mirror; // random 0 or 1 per image
};

// sizes holds (w, h) per image. The batch is padded to the largest w and h.
inline Batch make_batch(const std::vector<std::pair<int, int>>& sizes, uint32_t seed) {
  Batch b;
  int w_max = 0, h_max = 0;
  for (const auto& s : sizes) {
    w_max = std::max(w_max, s.first);
    h_max = std::max(h_max, s.second);
  }
  const int n = static_cast<int>(sizes.size());
  b.desc = hipaug::make_desc(n, h_max, w_max, hipaug::Layout::NHWC, hipaug::DType::U8);
  b.data.resize(hipaug::num_elements(b.desc));
  std::mt19937 rng(seed);
  for (auto& v : b.data)
    v = static_cast<uint8_t>(rng() & 0xFF);
  for (const auto& s : sizes) {
    b.rois.push_back(Roi{0, 0, s.first, s.second});
    b.mirror.push_back(static_cast<uint8_t>(rng() & 1));
  }
  return b;
}

struct Case {
  std::string name;
  Batch batch;
  int out_w, out_h;
};

// The five cases every op is tested on: a single image, a batch of 16 with mixed
// sizes, ROIs touching the border, a 1x1 ROI, and ROIs covering the full image.
// With roi_equals_out, every ROI has exactly the output size, as
// crop_mirror_normalize requires.
inline std::vector<Case> standard_cases(bool roi_equals_out) {
  std::vector<Case> cases;
  std::mt19937 rng(1234);
  auto rnd = [&rng](int lo, int hi) { // inclusive range
    return lo + static_cast<int>(rng() % static_cast<uint32_t>(hi - lo + 1));
  };

  { // single image, ROI strictly inside
    Case c{"single", make_batch({{37, 29}}, 1), 24, 16};
    c.batch.rois[0] = roi_equals_out ? Roi{3, 2, 24, 16} : Roi{3, 2, 30, 21};
    cases.push_back(std::move(c));
  }
  { // 16 images of mixed sizes, random ROI in each
    std::vector<std::pair<int, int>> sizes;
    for (int i = 0; i < 16; ++i)
      sizes.push_back({rnd(24, 64), rnd(24, 64)});
    Case c{"batch16_mixed", make_batch(sizes, 2), 20, 12};
    for (int i = 0; i < 16; ++i) {
      const int w = roi_equals_out ? 20 : rnd(4, sizes[i].first);
      const int h = roi_equals_out ? 12 : rnd(4, sizes[i].second);
      c.batch.rois[i] = Roi{rnd(0, sizes[i].first - w), rnd(0, sizes[i].second - h), w, h};
    }
    cases.push_back(std::move(c));
  }
  { // ROIs touching each corner of the image
    Case c{"roi_border", make_batch({{40, 30}, {40, 30}, {40, 30}, {40, 30}}, 3), 10, 10};
    if (!roi_equals_out) {
      c.out_w = 15;
      c.out_h = 13;
    }
    c.batch.rois = {Roi{0, 0, 10, 10}, Roi{30, 20, 10, 10}, Roi{0, 20, 10, 10},
                    Roi{30, 0, 10, 10}};
    cases.push_back(std::move(c));
  }
  { // 1x1 ROI
    Case c{"roi_1x1", make_batch({{16, 12}, {16, 12}}, 4), 8, 8};
    if (roi_equals_out) {
      c.out_w = 1;
      c.out_h = 1;
    }
    c.batch.rois = {Roi{5, 7, 1, 1}, Roi{15, 11, 1, 1}};
    cases.push_back(std::move(c));
  }
  { // ROI covers the full image
    Case c{"roi_full", make_batch({{32, 24}, {32, 24}, {32, 24}}, 5), 32, 24};
    if (!roi_equals_out) {
      c.out_w = 16;
      c.out_h = 16;
    }
    cases.push_back(std::move(c));
  }
  return cases;
}

// Copy each image's ROI into its own tightly sized single-image batch. An op run on
// the crop with a full ROI must give the same result as the op run on the ROI in place.
inline Batch crop_image(const Batch& b, int n) {
  const Roi r = b.rois[n];
  Batch out;
  out.desc = hipaug::make_desc(1, r.h, r.w, hipaug::Layout::NHWC, hipaug::DType::U8);
  out.data.resize(hipaug::num_elements(out.desc));
  for (int y = 0; y < r.h; ++y)
    for (int x = 0; x < r.w; ++x)
      for (int c = 0; c < 3; ++c)
        out.data[y * out.desc.stride_h + x * out.desc.stride_w + c] =
            b.data[n * b.desc.stride_n + (r.y + y) * b.desc.stride_h + (r.x + x) * b.desc.stride_w +
                   c];
  out.rois = {Roi{0, 0, r.w, r.h}};
  out.mirror = {b.mirror[n]};
  return out;
}

// Invert every byte that lies outside its image's ROI. An op that only reads inside
// the ROI must give the same output before and after.
inline void scramble_outside_roi(Batch& b) {
  for (int n = 0; n < b.desc.n; ++n) {
    const Roi r = b.rois[n];
    for (int y = 0; y < b.desc.h_max; ++y)
      for (int x = 0; x < b.desc.w_max; ++x) {
        if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h)
          continue;
        for (int c = 0; c < 3; ++c)
          b.data[n * b.desc.stride_n + y * b.desc.stride_h + x * b.desc.stride_w + c] ^= 0xFF;
      }
  }
}

// The elements of image n, as their own vector.
template <typename T>
std::vector<T> image_slice(const std::vector<T>& buf, const TensorDesc& d, int n) {
  return std::vector<T>(buf.begin() + n * d.stride_n, buf.begin() + (n + 1) * d.stride_n);
}

inline int max_abs_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  if (a.size() != b.size())
    return 1 << 30;
  int m = 0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
  return m;
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size())
    return 1e30f;
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

// Tolerances from the PRD: 1 for u8, 1e-4 absolute for f32.
constexpr int kTolU8 = 1;
constexpr float kTolF32 = 1e-4f;

constexpr float kMean[3] = {123.675f, 116.28f, 103.53f}; // ImageNet, 0 to 255 units
constexpr float kStd[3] = {58.395f, 57.12f, 57.375f};

} // namespace testutil
