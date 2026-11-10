// Thin pybind11 bindings. Every argument is a GPU tensor that exposes
// __cuda_array_interface__ (PyTorch on ROCm does, for tensors on the "cuda" device).
// The bindings read the device pointer and shape, build the TensorDesc and call the op.
// Output tensors are allocated by the caller (hipaugment/__init__.py does it with torch),
// so the library still never allocates.
//
// Ops are queued on the default stream, which is the stream PyTorch uses, so they
// stay ordered with the PyTorch operations around them.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "hipaug/hipaug.hpp"

namespace py = pybind11;
using namespace hipaug;

namespace {

struct DevArray {
  void* ptr = nullptr;
  std::vector<long> shape;
};

// Read a contiguous device array of the expected element type and rank.
DevArray dev_array(const py::handle& obj, const char* name, const char* typestr, size_t rank) {
  if (!py::hasattr(obj, "__cuda_array_interface__"))
    throw std::invalid_argument(std::string(name) + " must be a GPU tensor");
  const py::dict iface = obj.attr("__cuda_array_interface__");
  if (iface.contains("strides") && !iface["strides"].is_none())
    throw std::invalid_argument(std::string(name) + " must be contiguous");
  // typestr is byte order + kind + size, for example "|u1" or "<f4". Compare kind and size.
  const std::string ts = iface["typestr"].cast<std::string>();
  if (ts.size() != 3 || ts.substr(1) != typestr)
    throw std::invalid_argument(std::string(name) + " has dtype " + ts + ", expected " + typestr);
  DevArray a;
  a.shape = iface["shape"].cast<std::vector<long>>();
  if (a.shape.size() != rank)
    throw std::invalid_argument(std::string(name) + " has the wrong number of dimensions");
  a.ptr = reinterpret_cast<void*>(iface["data"].cast<py::tuple>()[0].cast<uintptr_t>());
  return a;
}

void check(Status st, const char* op) {
  if (st == Status::InvalidArg)
    throw std::invalid_argument(std::string(op) + ": invalid argument");
  if (st != Status::Ok)
    throw std::runtime_error(std::string(op) + ": HIP error");
}

// src is u8 [N, H, W, 3]; roi is i32 [N, 4] holding x, y, w, h per image.
struct Inputs {
  TensorDesc desc;
  const uint8_t* src;
  const Roi* roi;
};

Inputs inputs(const py::handle& src, const py::handle& roi) {
  const DevArray s = dev_array(src, "src", "u1", 4);
  const DevArray r = dev_array(roi, "roi", "i4", 2);
  if (s.shape[3] != 3 || r.shape[0] != s.shape[0] || r.shape[1] != 4)
    throw std::invalid_argument("src must be [N, H, W, 3] and roi must be [N, 4]");
  return {make_desc(static_cast<int>(s.shape[0]), static_cast<int>(s.shape[1]),
                    static_cast<int>(s.shape[2]), Layout::NHWC, DType::U8),
          static_cast<const uint8_t*>(s.ptr), static_cast<const Roi*>(r.ptr)};
}

void py_resize(py::handle src, py::handle roi, py::handle out) {
  const Inputs in = inputs(src, roi);
  const DevArray o = dev_array(out, "out", "u1", 4);
  if (o.shape[0] != in.desc.n || o.shape[3] != 3)
    throw std::invalid_argument("out must be [N, H, W, 3]");
  const TensorDesc od = make_desc(in.desc.n, static_cast<int>(o.shape[1]),
                                  static_cast<int>(o.shape[2]), Layout::NHWC, DType::U8);
  check(resize(in.desc, in.src, in.roi, od, static_cast<uint8_t*>(o.ptr), nullptr), "resize");
}

void py_gaussian_blur(py::handle src, py::handle roi, py::handle out, int ksize, float sigma) {
  const Inputs in = inputs(src, roi);
  const DevArray o = dev_array(out, "out", "u1", 4);
  if (o.shape != dev_array(src, "src", "u1", 4).shape)
    throw std::invalid_argument("out must have the same shape as src");
  check(gaussian_blur(in.desc, in.src, in.roi, static_cast<uint8_t*>(o.ptr), ksize, sigma, nullptr),
        "gaussian_blur");
}

using NormFn = Status (*)(const TensorDesc&, const uint8_t*, const Roi*, const uint8_t*,
                          const float*, const float*, const TensorDesc&, float*, hipStream_t);

// Shared by crop_mirror_normalize and fused_rcmn. out is f32 [N, 3, H, W].
void normalize_op(NormFn fn, const char* name, py::handle src, py::handle roi, py::handle mirror,
                  const std::array<float, 3>& mean, const std::array<float, 3>& std,
                  py::handle out) {
  const Inputs in = inputs(src, roi);
  const DevArray m = dev_array(mirror, "mirror", "u1", 1);
  const DevArray o = dev_array(out, "out", "f4", 4);
  if (m.shape[0] != in.desc.n || o.shape[0] != in.desc.n || o.shape[1] != 3)
    throw std::invalid_argument("mirror must be [N] and out must be [N, 3, H, W]");
  const TensorDesc od = make_desc(in.desc.n, static_cast<int>(o.shape[2]),
                                  static_cast<int>(o.shape[3]), Layout::NCHW, DType::F32);
  check(fn(in.desc, in.src, in.roi, static_cast<const uint8_t*>(m.ptr), mean.data(), std.data(), od,
           static_cast<float*>(o.ptr), nullptr),
        name);
}

} // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "hipAugment ops on caller-allocated GPU tensors";
  m.def("resize", &py_resize, py::arg("src"), py::arg("roi"), py::arg("out"));
  m.def("gaussian_blur", &py_gaussian_blur, py::arg("src"), py::arg("roi"), py::arg("out"),
        py::arg("ksize"), py::arg("sigma"));
  m.def(
      "crop_mirror_normalize",
      [](py::handle src, py::handle roi, py::handle mirror, std::array<float, 3> mean,
         std::array<float, 3> std, py::handle out) {
        normalize_op(crop_mirror_normalize, "crop_mirror_normalize", src, roi, mirror, mean, std,
                     out);
      },
      py::arg("src"), py::arg("roi"), py::arg("mirror"), py::arg("mean"), py::arg("std"),
      py::arg("out"));
  m.def(
      "fused_rcmn",
      [](py::handle src, py::handle roi, py::handle mirror, std::array<float, 3> mean,
         std::array<float, 3> std, py::handle out) {
        normalize_op(fused_rcmn, "fused_rcmn", src, roi, mirror, mean, std, out);
      },
      py::arg("src"), py::arg("roi"), py::arg("mirror"), py::arg("mean"), py::arg("std"),
      py::arg("out"));
}
