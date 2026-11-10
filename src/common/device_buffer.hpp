// Owning device buffer for the tests, the CLI and the benchmarks. The library ops
// themselves never allocate; this is caller-side code.
#pragma once

#include <cstddef>
#include <vector>

#include "hipaug/hip_check.hpp"

namespace hipaug {

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { (void)release(); }

  Status alloc(size_t count) {
    const Status st = release();
    if (st != Status::Ok)
      return st;
    HIP_CHECK(hipMalloc(&ptr_, count * sizeof(T)));
    count_ = count;
    return Status::Ok;
  }

  Status release() {
    if (ptr_) {
      T* p = ptr_;
      ptr_ = nullptr;
      count_ = 0;
      HIP_CHECK(hipFree(p));
    }
    return Status::Ok;
  }

  // Allocate to fit and copy a host vector to the device.
  Status upload(const std::vector<T>& host) {
    if (count_ != host.size()) {
      const Status st = alloc(host.size());
      if (st != Status::Ok)
        return st;
    }
    HIP_CHECK(hipMemcpy(ptr_, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    return Status::Ok;
  }

  // Copy the whole buffer back to the host. Blocks until queued kernels finish.
  Status download(std::vector<T>& host) const {
    host.resize(count_);
    HIP_CHECK(hipMemcpy(host.data(), ptr_, count_ * sizeof(T), hipMemcpyDeviceToHost));
    return Status::Ok;
  }

  T* get() const { return ptr_; }
  size_t size() const { return count_; }

private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

} // namespace hipaug
