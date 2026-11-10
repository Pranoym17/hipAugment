// `hipaug run`: load a directory of images, run one op batch by batch, save PNGs.
#include "run.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "hipaug/hipaug.hpp"
#include "image_io.hpp"

#if defined(HIPAUG_HAS_HIP)
#include "common/device_buffer.hpp"
#endif

namespace cli {
namespace {

using namespace hipaug;
namespace fs = std::filesystem;

enum class Op { Resize, Blur, Cmn, Fused };

const float kMean[3] = {123.675f, 116.28f, 103.53f}; // ImageNet, 0 to 255 units
const float kStd[3] = {58.395f, 57.12f, 57.375f};

struct Options {
  Op op = Op::Resize;
  int batch = 16;
  int out_w = 224, out_h = 224;
  int ksize = 5;
  float sigma = 1.0f;
  bool mirror = false;
  bool use_hip = false;
};

// One padded batch on the host, plus the names and sizes of the images in it.
struct HostBatch {
  TensorDesc desc{};
  std::vector<uint8_t> data;
  std::vector<Roi> rois;
  std::vector<uint8_t> mirror;
  std::vector<std::string> names;
  std::vector<Image> images;
};

// Pad every image to the largest width and height in the batch.
HostBatch make_batch(std::vector<Image> images, std::vector<std::string> names) {
  HostBatch b;
  int w_max = 0, h_max = 0;
  for (const Image& im : images) {
    w_max = std::max(w_max, im.w);
    h_max = std::max(h_max, im.h);
  }
  const int n = static_cast<int>(images.size());
  b.desc = make_desc(n, h_max, w_max, Layout::NHWC, DType::U8);
  b.data.assign(num_elements(b.desc), 0);
  for (int i = 0; i < n; ++i) {
    const Image& im = images[i];
    for (int y = 0; y < im.h; ++y)
      std::copy_n(&im.rgb[static_cast<size_t>(y) * im.w * 3], static_cast<size_t>(im.w) * 3,
                  &b.data[i * b.desc.stride_n + y * b.desc.stride_h]);
    b.rois.push_back(Roi{0, 0, im.w, im.h});
  }
  b.mirror.assign(n, 0);
  b.names = std::move(names);
  b.images = std::move(images);
  return b;
}

// Run a u8 -> u8 op (resize or blur) on host buffers. With the HIP backend this
// uploads the batch, runs the kernel and downloads the result.
Status exec_u8(const Options& o, const HostBatch& b, const TensorDesc& od,
               std::vector<uint8_t>& out) {
  if (!o.use_hip) {
    if (o.op == Op::Resize)
      return cpu::resize(b.desc, b.data.data(), b.rois.data(), od, out.data());
    return cpu::gaussian_blur(b.desc, b.data.data(), b.rois.data(), out.data(), o.ksize, o.sigma);
  }
#if defined(HIPAUG_HAS_HIP)
  DeviceBuffer<uint8_t> src, dst;
  DeviceBuffer<Roi> roi;
  Status st = src.upload(b.data);
  if (st == Status::Ok)
    st = roi.upload(b.rois);
  if (st == Status::Ok)
    st = dst.upload(out);
  if (st != Status::Ok)
    return st;
  st = o.op == Op::Resize
           ? resize(b.desc, src.get(), roi.get(), od, dst.get(), nullptr)
           : gaussian_blur(b.desc, src.get(), roi.get(), dst.get(), o.ksize, o.sigma, nullptr);
  if (st != Status::Ok)
    return st;
  return dst.download(out);
#else
  (void)od;
  return Status::InvalidArg;
#endif
}

// Run a u8 -> f32 op (crop_mirror_normalize or fused_rcmn) on host buffers.
Status exec_f32(const Options& o, const HostBatch& b, const TensorDesc& od,
                std::vector<float>& out) {
  if (!o.use_hip) {
    if (o.op == Op::Cmn)
      return cpu::crop_mirror_normalize(b.desc, b.data.data(), b.rois.data(), b.mirror.data(),
                                        kMean, kStd, od, out.data());
    return cpu::fused_rcmn(b.desc, b.data.data(), b.rois.data(), b.mirror.data(), kMean, kStd, od,
                           out.data());
  }
#if defined(HIPAUG_HAS_HIP)
  DeviceBuffer<uint8_t> src, mirror;
  DeviceBuffer<Roi> roi;
  DeviceBuffer<float> dst;
  Status st = src.upload(b.data);
  if (st == Status::Ok)
    st = roi.upload(b.rois);
  if (st == Status::Ok)
    st = mirror.upload(b.mirror);
  if (st == Status::Ok)
    st = dst.upload(out);
  if (st != Status::Ok)
    return st;
  st = o.op == Op::Cmn ? crop_mirror_normalize(b.desc, src.get(), roi.get(), mirror.get(), kMean,
                                               kStd, od, dst.get(), nullptr)
                       : fused_rcmn(b.desc, src.get(), roi.get(), mirror.get(), kMean, kStd, od,
                                    dst.get(), nullptr);
  if (st != Status::Ok)
    return st;
  return dst.download(out);
#else
  (void)od;
  return Status::InvalidArg;
#endif
}

// Undo the normalization so a float NCHW image can be saved as an 8-bit PNG.
std::vector<uint8_t> denormalize(const std::vector<float>& planar, const TensorDesc& od, int n) {
  std::vector<uint8_t> rgb(static_cast<size_t>(od.w_max) * od.h_max * 3);
  for (int y = 0; y < od.h_max; ++y)
    for (int x = 0; x < od.w_max; ++x)
      for (int c = 0; c < 3; ++c) {
        const float v =
            planar[n * od.stride_n + c * od.stride_c + y * od.stride_h + x] * kStd[c] + kMean[c];
        rgb[(static_cast<size_t>(y) * od.w_max + x) * 3 + c] =
            static_cast<uint8_t>(std::min(std::max(v, 0.0f), 255.0f) + 0.5f);
      }
  return rgb;
}

// Run the op on one batch and write one PNG per image.
Status process_batch(const Options& o, HostBatch& b, const fs::path& out_dir, const char* op_name) {
  const int n = b.desc.n;
  auto out_path = [&](int i) {
    return (out_dir / (fs::path(b.names[i]).stem().string() + "_" + op_name + ".png")).string();
  };

  if (o.op == Op::Resize) {
    const TensorDesc od = make_desc(n, o.out_h, o.out_w, Layout::NHWC, DType::U8);
    std::vector<uint8_t> out(num_elements(od), 0);
    const Status st = exec_u8(o, b, od, out);
    if (st != Status::Ok)
      return st;
    for (int i = 0; i < n; ++i)
      save_png(out_path(i), o.out_w, o.out_h, &out[i * od.stride_n], static_cast<int>(od.stride_h));
    return Status::Ok;
  }

  if (o.op == Op::Blur) {
    // The op only writes inside the ROI, so start from a copy of the source.
    std::vector<uint8_t> out = b.data;
    const Status st = exec_u8(o, b, b.desc, out);
    if (st != Status::Ok)
      return st;
    for (int i = 0; i < n; ++i) // save the image without its padding
      save_png(out_path(i), b.images[i].w, b.images[i].h, &out[i * b.desc.stride_n],
               static_cast<int>(b.desc.stride_h));
    return Status::Ok;
  }

  // Float ops. crop_mirror_normalize takes a centre crop of the output size;
  // fused_rcmn resizes the whole image.
  if (o.op == Op::Cmn) {
    for (int i = 0; i < n; ++i) {
      const Image& im = b.images[i];
      if (im.w < o.out_w || im.h < o.out_h)
        std::fprintf(stderr, "hipaug: %s is smaller than %dx%d, edges will be replicated\n",
                     b.names[i].c_str(), o.out_w, o.out_h);
      b.rois[i] = Roi{std::max((im.w - o.out_w) / 2, 0), std::max((im.h - o.out_h) / 2, 0),
                      std::min(o.out_w, im.w), std::min(o.out_h, im.h)};
    }
  }
  std::fill(b.mirror.begin(), b.mirror.end(), o.mirror ? 1 : 0);
  const TensorDesc od = make_desc(n, o.out_h, o.out_w, Layout::NCHW, DType::F32);
  std::vector<float> out(num_elements(od), 0.0f);
  const Status st = exec_f32(o, b, od, out);
  if (st != Status::Ok)
    return st;
  for (int i = 0; i < n; ++i) {
    const std::vector<uint8_t> rgb = denormalize(out, od, i);
    save_png(out_path(i), o.out_w, o.out_h, rgb.data(), o.out_w * 3);
  }
  return Status::Ok;
}

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

} // namespace

int cmd_run(const Args& args) {
  if (args.pos.size() != 2 || !args.has("op")) {
    std::fprintf(stderr, "usage: hipaug run --op <name> --batch <N> --size <WxH> "
                         "<input_dir> <output_dir>\n");
    return 2;
  }
  Options o;
  const std::string op = args.get("op", "");
  if (op == "resize")
    o.op = Op::Resize;
  else if (op == "gaussian_blur")
    o.op = Op::Blur;
  else if (op == "crop_mirror_normalize")
    o.op = Op::Cmn;
  else if (op == "fused_rcmn")
    o.op = Op::Fused;
  else {
    std::fprintf(stderr, "hipaug: unknown op '%s'\n", op.c_str());
    return 2;
  }
  o.batch = args.get_int("batch", 16);
  o.ksize = args.get_int("ksize", 5);
  o.sigma = static_cast<float>(args.get_double("sigma", 1.0));
  o.mirror = args.has("mirror");
  if (o.batch <= 0) {
    std::fprintf(stderr, "hipaug: --batch must be positive\n");
    return 2;
  }
  if (args.has("size") && !parse_size(args.get("size", ""), &o.out_w, &o.out_h)) {
    std::fprintf(stderr, "hipaug: --size must look like 224x224\n");
    return 2;
  }
#if defined(HIPAUG_HAS_HIP)
  const std::string backend = args.get("backend", "hip");
#else
  const std::string backend = args.get("backend", "cpu");
#endif
  if (backend == "hip") {
#if defined(HIPAUG_HAS_HIP)
    o.use_hip = true;
#else
    std::fprintf(stderr, "hipaug: this build has no HIP backend, use --backend cpu\n");
    return 2;
#endif
  } else if (backend != "cpu") {
    std::fprintf(stderr, "hipaug: unknown backend '%s'\n", backend.c_str());
    return 2;
  }

  const fs::path in_dir = args.pos[0], out_dir = args.pos[1];
  std::error_code ec;
  if (!fs::is_directory(in_dir, ec)) {
    std::fprintf(stderr, "hipaug: %s is not a directory\n", in_dir.string().c_str());
    return 1;
  }
  fs::create_directories(out_dir, ec);
  if (ec) {
    std::fprintf(stderr, "hipaug: cannot create %s\n", out_dir.string().c_str());
    return 1;
  }
  std::vector<fs::path> files;
  for (const auto& e : fs::directory_iterator(in_dir))
    if (e.is_regular_file() && is_image(e.path()))
      files.push_back(e.path());
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::fprintf(stderr, "hipaug: no images found in %s\n", in_dir.string().c_str());
    return 1;
  }

  size_t done = 0;
  for (size_t first = 0; first < files.size(); first += o.batch) {
    std::vector<Image> images;
    std::vector<std::string> names;
    for (size_t i = first; i < std::min(files.size(), first + o.batch); ++i) {
      Image im;
      if (!load_image(files[i].string(), &im)) {
        std::fprintf(stderr, "hipaug: skipping unreadable file %s\n", files[i].string().c_str());
        continue;
      }
      images.push_back(std::move(im));
      names.push_back(files[i].filename().string());
    }
    if (images.empty())
      continue;
    HostBatch b = make_batch(std::move(images), std::move(names));
    const Status st = process_batch(o, b, out_dir, op.c_str());
    if (st != Status::Ok) {
      std::fprintf(stderr, "hipaug: %s failed: %s\n", op.c_str(), to_string(st));
      return 1;
    }
    done += b.desc.n;
  }
  std::printf("hipaug: %s on %zu images (%s backend), output in %s\n", op.c_str(), done,
              backend.c_str(), out_dir.string().c_str());
  return 0;
}

} // namespace cli
