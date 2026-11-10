// Benchmark harness behind `hipaug bench`.
//
// For each batch size it builds one synthetic batch of mixed-size images and times:
//   resize                  full-image ROI -> output size
//   crop_mirror_normalize   centre crop of the output size
//   gaussian_blur_3x3/5x5   full-image ROI (direct and LDS kernels on the GPU)
//   resize_cmn              the training chain: resize + crop_mirror_normalize as two
//                           ops ("separate"), and fused_rcmn ("fused")
//
// GPU runs are timed with hipEvents: warmup runs first, then timed runs, and the
// median is reported. CPU runs use a steady clock.
//
// Bandwidth is (bytes read + bytes written) / median time. Bytes are counted from
// the pixels an op has to touch, not from the padded buffer sizes:
//   resize read    min(ROI pixels, 4 taps per output pixel) * 3 bytes
//   blur read      ROI pixels * 3 bytes (each source pixel counted once)
//   u8 write       output pixels * 3 bytes
//   f32 write      output pixels * 12 bytes
#include "bench.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "hipaug/hipaug.hpp"

#if defined(HIPAUG_HAS_HIP)
#include "common/device_buffer.hpp"
#endif

namespace bench {
namespace {

using namespace hipaug;

const float kMean[3] = {123.675f, 116.28f, 103.53f};
const float kStd[3] = {58.395f, 57.12f, 57.375f};
const float kSigma = 1.0f;

struct Config {
  std::vector<int> batches = {1, 16, 64, 256};
  int out_w = 224, out_h = 224;
  int src_min = 256, src_max = 512;
  int warmup = 10, runs = 100, cpu_runs = 10;
  double peak_gbps = 5300.0; // MI300X HBM3 peak, in GB/s
  bool run_cpu = true, run_gpu = true;
  std::vector<std::string> pipelines; // empty means all
  std::string out = "results.csv";

  bool want(const std::string& pipeline) const {
    return pipelines.empty() ||
           std::find(pipelines.begin(), pipelines.end(), pipeline) != pipelines.end();
  }
};

struct Row {
  std::string pipeline, impl, device;
  int batch = 0, runs = 0;
  double median_ms = 0, min_ms = 0, p90_ms = 0;
  double bytes_read = 0, bytes_written = 0;
};

// One synthetic batch on the host, with the ROI sets the pipelines need.
struct Workload {
  TensorDesc in{}, out_u8{}, out_f32{};
  std::vector<uint8_t> src;
  std::vector<Roi> roi_full; // each image's full size
  std::vector<Roi> roi_crop; // centre crop of the output size
  std::vector<Roi> roi_out;  // the full intermediate image, for the separate path
  std::vector<uint8_t> mirror;
  // Byte counts for the bandwidth column.
  double resize_read = 0, blur_bytes = 0, out_u8_bytes = 0, out_f32_bytes = 0;
};

Workload make_workload(const Config& cfg, int n) {
  Workload w;
  uint64_t s = 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(n + 1); // fixed seed per batch size
  auto next = [&s]() { // xorshift64
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  };
  const int span = cfg.src_max - cfg.src_min + 1;
  std::vector<int> ws(n), hs(n);
  int w_max = 0, h_max = 0;
  for (int i = 0; i < n; ++i) {
    ws[i] = cfg.src_min + static_cast<int>(next() % span);
    hs[i] = cfg.src_min + static_cast<int>(next() % span);
    w_max = std::max(w_max, ws[i]);
    h_max = std::max(h_max, hs[i]);
  }
  w.in = make_desc(n, h_max, w_max, Layout::NHWC, DType::U8);
  w.out_u8 = make_desc(n, cfg.out_h, cfg.out_w, Layout::NHWC, DType::U8);
  w.out_f32 = make_desc(n, cfg.out_h, cfg.out_w, Layout::NCHW, DType::F32);
  w.src.resize(num_elements(w.in));
  for (size_t i = 0; i < w.src.size(); ++i) {
    if ((i & 7) == 0)
      next();
    w.src[i] = static_cast<uint8_t>(s >> (8 * (i & 7)));
  }
  const double out_px = static_cast<double>(cfg.out_w) * cfg.out_h;
  for (int i = 0; i < n; ++i) {
    w.roi_full.push_back(Roi{0, 0, ws[i], hs[i]});
    const int cw = std::min(cfg.out_w, ws[i]), ch = std::min(cfg.out_h, hs[i]);
    w.roi_crop.push_back(Roi{(ws[i] - cw) / 2, (hs[i] - ch) / 2, cw, ch});
    w.roi_out.push_back(Roi{0, 0, cfg.out_w, cfg.out_h});
    w.mirror.push_back(static_cast<uint8_t>(next() & 1));
    const double roi_px = static_cast<double>(ws[i]) * hs[i];
    w.resize_read += std::min(roi_px, 4.0 * out_px) * 3.0;
    w.blur_bytes += roi_px * 3.0;
  }
  w.out_u8_bytes = n * out_px * 3.0;
  w.out_f32_bytes = n * out_px * 12.0;
  return w;
}

Row make_row(const std::string& pipeline, const std::string& impl, const std::string& device,
             int batch, std::vector<double> ms, double bytes_read, double bytes_written) {
  std::sort(ms.begin(), ms.end());
  Row r;
  r.pipeline = pipeline;
  r.impl = impl;
  r.device = device;
  r.batch = batch;
  r.runs = static_cast<int>(ms.size());
  r.median_ms = ms.size() % 2 ? ms[ms.size() / 2] : 0.5 * (ms[ms.size() / 2 - 1] + ms[ms.size() / 2]);
  r.min_ms = ms.front();
  r.p90_ms = ms[std::min(ms.size() - 1, static_cast<size_t>(0.9 * ms.size()))];
  r.bytes_read = bytes_read;
  r.bytes_written = bytes_written;
  return r;
}

// ---- CPU reference ------------------------------------------------------------

template <typename Fn> std::vector<double> time_cpu(const Config& cfg, Fn&& fn) {
  std::vector<double> ms;
  for (int i = 0; i < 2 + cfg.cpu_runs; ++i) { // two warmup runs
    const auto t0 = std::chrono::steady_clock::now();
    const Status st = fn();
    const auto t1 = std::chrono::steady_clock::now();
    if (st != Status::Ok)
      return {};
    if (i >= 2)
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return ms;
}

bool bench_cpu(const Config& cfg, const Workload& w, std::vector<Row>& rows) {
  const int n = w.in.n;
  const std::string dev = "cpu, " + std::to_string(std::thread::hardware_concurrency()) + " threads";
  std::vector<uint8_t> dst_u8(num_elements(w.out_u8)), blur_dst(w.src.size());
  std::vector<float> dst_f32(num_elements(w.out_f32));
  auto add = [&](const char* pipeline, const char* impl, std::vector<double> ms, double rd,
                 double wr) {
    if (ms.empty())
      return false;
    rows.push_back(make_row(pipeline, impl, dev, n, std::move(ms), rd, wr));
    return true;
  };

  bool ok = true;
  if (cfg.want("resize"))
    ok = add("resize", "cpu_ref", time_cpu(cfg, [&] {
               return cpu::resize(w.in, w.src.data(), w.roi_full.data(), w.out_u8, dst_u8.data());
             }),
             w.resize_read, w.out_u8_bytes);
  if (cfg.want("crop_mirror_normalize"))
    ok = ok && add("crop_mirror_normalize", "cpu_ref", time_cpu(cfg, [&] {
                   return cpu::crop_mirror_normalize(w.in, w.src.data(), w.roi_crop.data(),
                                                     w.mirror.data(), kMean, kStd, w.out_f32,
                                                     dst_f32.data());
                 }),
                 w.out_u8_bytes, w.out_f32_bytes);
  for (int ksize : {3, 5}) {
    const std::string name = ksize == 3 ? "gaussian_blur_3x3" : "gaussian_blur_5x5";
    if (!cfg.want(name))
      continue;
    ok = ok && add(name.c_str(), "cpu_ref", time_cpu(cfg, [&] {
                     return cpu::gaussian_blur(w.in, w.src.data(), w.roi_full.data(),
                                               blur_dst.data(), ksize, kSigma);
                   }),
                   w.blur_bytes, w.blur_bytes);
  }
  if (!cfg.want("resize_cmn"))
    return ok;
  ok = ok && add("resize_cmn", "cpu_ref_separate", time_cpu(cfg, [&] {
                   const Status st = cpu::resize(w.in, w.src.data(), w.roi_full.data(), w.out_u8,
                                                 dst_u8.data());
                   if (st != Status::Ok)
                     return st;
                   return cpu::crop_mirror_normalize(w.out_u8, dst_u8.data(), w.roi_out.data(),
                                                     w.mirror.data(), kMean, kStd, w.out_f32,
                                                     dst_f32.data());
                 }),
                 w.resize_read + w.out_u8_bytes, w.out_u8_bytes + w.out_f32_bytes);
  ok = ok && add("resize_cmn", "cpu_ref_fused", time_cpu(cfg, [&] {
                   return cpu::fused_rcmn(w.in, w.src.data(), w.roi_full.data(), w.mirror.data(),
                                          kMean, kStd, w.out_f32, dst_f32.data());
                 }),
                 w.resize_read, w.out_f32_bytes);
  return ok;
}

// ---- HIP ------------------------------------------------------------------------

#if defined(HIPAUG_HAS_HIP)

// Time `launch` with hipEvents on the default stream. The stop event is recorded
// after the launch, so the elapsed time covers every kernel `launch` queued.
template <typename Fn> Status time_gpu(const Config& cfg, Fn&& launch, std::vector<double>& ms) {
  hipEvent_t start = nullptr, stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  for (int i = 0; i < cfg.warmup + cfg.runs; ++i) {
    HIP_CHECK(hipEventRecord(start, nullptr));
    const Status st = launch();
    if (st != Status::Ok)
      return st;
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
    if (i >= cfg.warmup)
      ms.push_back(elapsed);
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return Status::Ok;
}

Status device_name(std::string& name) {
  int id = 0;
  HIP_CHECK(hipGetDevice(&id));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, id));
  name = props.name;
  return Status::Ok;
}

Status bench_gpu(const Config& cfg, const Workload& w, std::vector<Row>& rows) {
  const int n = w.in.n;
  std::string dev;
  Status st = device_name(dev);
  if (st != Status::Ok)
    return st;

  // All buffers are allocated once, before any timing.
  DeviceBuffer<uint8_t> src, mirror, dst_u8, blur_dst;
  DeviceBuffer<Roi> roi_full, roi_crop, roi_out;
  DeviceBuffer<float> dst_f32;
  st = src.upload(w.src);
  if (st == Status::Ok)
    st = mirror.upload(w.mirror);
  if (st == Status::Ok)
    st = roi_full.upload(w.roi_full);
  if (st == Status::Ok)
    st = roi_crop.upload(w.roi_crop);
  if (st == Status::Ok)
    st = roi_out.upload(w.roi_out);
  if (st == Status::Ok)
    st = dst_u8.alloc(num_elements(w.out_u8));
  if (st == Status::Ok)
    st = blur_dst.alloc(w.src.size());
  if (st == Status::Ok)
    st = dst_f32.alloc(num_elements(w.out_f32));
  if (st != Status::Ok)
    return st;

  std::vector<double> ms;
  auto add = [&](const std::string& pipeline, const char* impl, double rd, double wr) {
    rows.push_back(make_row(pipeline, impl, dev, n, ms, rd, wr));
    ms.clear();
  };

  if (cfg.want("resize")) {
    st = time_gpu(cfg, [&] {
      return resize(w.in, src.get(), roi_full.get(), w.out_u8, dst_u8.get(), nullptr);
    }, ms);
    if (st != Status::Ok)
      return st;
    add("resize", "hip", w.resize_read, w.out_u8_bytes);
  }

  if (cfg.want("crop_mirror_normalize")) {
    st = time_gpu(cfg, [&] {
      return crop_mirror_normalize(w.in, src.get(), roi_crop.get(), mirror.get(), kMean, kStd,
                                   w.out_f32, dst_f32.get(), nullptr);
    }, ms);
    if (st != Status::Ok)
      return st;
    add("crop_mirror_normalize", "hip", w.out_u8_bytes, w.out_f32_bytes);
  }

  for (int ksize : {3, 5}) {
    const std::string name = ksize == 3 ? "gaussian_blur_3x3" : "gaussian_blur_5x5";
    if (!cfg.want(name))
      continue;
    st = time_gpu(cfg, [&] {
      return detail::gaussian_blur_direct(w.in, src.get(), roi_full.get(), blur_dst.get(), ksize,
                                          kSigma, nullptr);
    }, ms);
    if (st != Status::Ok)
      return st;
    add(name, "hip_direct", w.blur_bytes, w.blur_bytes);

    st = time_gpu(cfg, [&] {
      return detail::gaussian_blur_lds(w.in, src.get(), roi_full.get(), blur_dst.get(), ksize,
                                       kSigma, nullptr);
    }, ms);
    if (st != Status::Ok)
      return st;
    add(name, "hip_lds", w.blur_bytes, w.blur_bytes);
  }

  if (!cfg.want("resize_cmn"))
    return Status::Ok;

  // Separate path: resize into a u8 intermediate, then normalize the whole intermediate.
  st = time_gpu(cfg, [&] {
    const Status s = resize(w.in, src.get(), roi_full.get(), w.out_u8, dst_u8.get(), nullptr);
    if (s != Status::Ok)
      return s;
    return crop_mirror_normalize(w.out_u8, dst_u8.get(), roi_out.get(), mirror.get(), kMean, kStd,
                                 w.out_f32, dst_f32.get(), nullptr);
  }, ms);
  if (st != Status::Ok)
    return st;
  add("resize_cmn", "hip_separate", w.resize_read + w.out_u8_bytes,
      w.out_u8_bytes + w.out_f32_bytes);

  st = time_gpu(cfg, [&] {
    return fused_rcmn(w.in, src.get(), roi_full.get(), mirror.get(), kMean, kStd, w.out_f32,
                      dst_f32.get(), nullptr);
  }, ms);
  if (st != Status::Ok)
    return st;
  add("resize_cmn", "hip_fused", w.resize_read, w.out_f32_bytes);
  return Status::Ok;
}

#endif // HIPAUG_HAS_HIP

// ---- output ---------------------------------------------------------------------

double gb_per_s(const Row& r) {
  return (r.bytes_read + r.bytes_written) / (r.median_ms * 1e-3) / 1e9;
}

bool write_csv(const Config& cfg, const Workload& shape, const std::vector<Row>& rows) {
  std::ofstream f(cfg.out);
  if (!f)
    return false;
  f << "pipeline,impl,batch,out_w,out_h,src_min,src_max,runs,median_ms,min_ms,p90_ms,"
       "images_per_s,bytes_read,bytes_written,gb_per_s,peak_share_pct,device\n";
  char line[512];
  for (const Row& r : rows) {
    std::snprintf(line, sizeof(line),
                  "%s,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.1f,%.0f,%.0f,%.3f,%.3f,\"%s\"\n",
                  r.pipeline.c_str(), r.impl.c_str(), r.batch, shape.out_u8.w_max,
                  shape.out_u8.h_max, cfg.src_min, cfg.src_max, r.runs, r.median_ms, r.min_ms,
                  r.p90_ms, r.batch / (r.median_ms * 1e-3), r.bytes_read, r.bytes_written,
                  gb_per_s(r), 100.0 * gb_per_s(r) / cfg.peak_gbps, r.device.c_str());
    f << line;
  }
  return static_cast<bool>(f);
}

void print_table(const Config& cfg, const std::vector<Row>& rows) {
  std::printf("%-22s %-17s %6s %11s %12s %9s %7s\n", "pipeline", "impl", "batch", "median ms",
              "images/s", "GB/s", "peak %");
  for (const Row& r : rows)
    std::printf("%-22s %-17s %6d %11.4f %12.0f %9.2f %7.2f\n", r.pipeline.c_str(), r.impl.c_str(),
                r.batch, r.median_ms, r.batch / (r.median_ms * 1e-3), gb_per_s(r),
                100.0 * gb_per_s(r) / cfg.peak_gbps);
}

} // namespace

int cmd_bench(const cli::Args& args) {
  Config cfg;
  if (args.has("batches"))
    cfg.batches = cli::parse_int_list(args.get("batches", ""));
  if (args.has("size") && !cli::parse_size(args.get("size", ""), &cfg.out_w, &cfg.out_h)) {
    std::fprintf(stderr, "hipaug: --size must look like 224x224\n");
    return 2;
  }
  cfg.src_min = args.get_int("src-min", cfg.src_min);
  cfg.src_max = args.get_int("src-max", cfg.src_max);
  cfg.warmup = args.get_int("warmup", cfg.warmup);
  cfg.runs = args.get_int("runs", cfg.runs);
  cfg.cpu_runs = args.get_int("cpu-runs", cfg.cpu_runs);
  cfg.peak_gbps = args.get_double("peak-gbps", cfg.peak_gbps);
  cfg.run_cpu = !args.has("no-cpu");
  cfg.run_gpu = !args.has("no-gpu");
  cfg.out = args.get("out", cfg.out);
  if (args.has("pipelines")) {
    std::string item;
    for (char ch : args.get("pipelines", "") + ",") {
      if (ch != ',') {
        item += ch;
        continue;
      }
      static const char* known[] = {"resize", "crop_mirror_normalize", "gaussian_blur_3x3",
                                    "gaussian_blur_5x5", "resize_cmn"};
      if (item.empty())
        continue;
      if (std::find(std::begin(known), std::end(known), item) == std::end(known)) {
        std::fprintf(stderr, "hipaug: unknown pipeline '%s'\n", item.c_str());
        return 2;
      }
      cfg.pipelines.push_back(item);
      item.clear();
    }
  }
  if (cfg.batches.empty() || cfg.src_min <= 0 || cfg.src_max < cfg.src_min || cfg.runs <= 0 ||
      cfg.cpu_runs <= 0 || cfg.warmup < 0 || cfg.peak_gbps <= 0) {
    std::fprintf(stderr, "hipaug: invalid bench options\n");
    return 2;
  }
  for (int n : cfg.batches)
    if (n <= 0) {
      std::fprintf(stderr, "hipaug: batch sizes must be positive\n");
      return 2;
    }
#if !defined(HIPAUG_HAS_HIP)
  if (cfg.run_gpu)
    std::fprintf(stderr, "hipaug: this build has no HIP backend, timing the CPU reference only\n");
  cfg.run_gpu = false;
#endif

  std::vector<Row> rows;
  Workload shape;
  for (int n : cfg.batches) {
    std::fprintf(stderr, "hipaug bench: batch %d\n", n);
    const Workload w = make_workload(cfg, n);
    shape = w;
    shape.src.clear();
    if (cfg.run_cpu && !bench_cpu(cfg, w, rows)) {
      std::fprintf(stderr, "hipaug: CPU benchmark failed\n");
      return 1;
    }
#if defined(HIPAUG_HAS_HIP)
    if (cfg.run_gpu) {
      const Status st = bench_gpu(cfg, w, rows);
      if (st != Status::Ok) {
        std::fprintf(stderr, "hipaug: GPU benchmark failed: %s\n", to_string(st));
        return 1;
      }
    }
#endif
  }
  print_table(cfg, rows);
  if (!write_csv(cfg, shape, rows)) {
    std::fprintf(stderr, "hipaug: cannot write %s\n", cfg.out.c_str());
    return 1;
  }
  std::printf("\nwrote %s\n", cfg.out.c_str());
  return 0;
}

} // namespace bench
