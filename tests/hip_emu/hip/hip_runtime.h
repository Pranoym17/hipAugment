// CPU emulation of the small part of the HIP runtime that hipAugment uses.
//
// This is a test tool, not a backend. With -DHIPAUG_HIP_EMULATION=ON the .hip files
// are compiled as plain C++ against this header, and the GPU test suite runs the
// exact kernel source on the CPU. That checks the kernel logic (indexing, ROI
// handling, LDS tiling, the barrier) on machines without an AMD GPU. It says nothing
// about performance or about real device behaviour.
//
// Each block runs its threads as fibers on one OS thread. __syncthreads() suspends a
// fiber until every other fiber in the block has reached the barrier or finished.
#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>

#define __global__
#define __device__
#define __host__
// LDS becomes a static array. Blocks run one after another, so they can share it.
#define __shared__ static

struct dim3 {
  uint32_t x, y, z;
  constexpr dim3(uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1) : x(x_), y(y_), z(z_) {}
};

// Set by the emulator before each fiber runs or resumes.
extern dim3 threadIdx, blockIdx, blockDim, gridDim;

enum hipError_t { hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorOutOfMemory = 2 };
enum hipMemcpyKind { hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost };

typedef struct ihipStream_t* hipStream_t;
struct hipEmuEvent {
  std::chrono::steady_clock::time_point t;
};
typedef hipEmuEvent* hipEvent_t;
struct hipDeviceProp_t {
  char name[256];
};

namespace hipemu {
// Run `body` once for every thread of every block in the grid.
void launch(dim3 grid, dim3 block, const std::function<void()>& body);
void syncthreads();
} // namespace hipemu

inline void __syncthreads() { hipemu::syncthreads(); }

// Arguments are captured by value, as a real launch copies them to the device.
#define hipLaunchKernelGGL(kernel, grid, block, shared_bytes, stream, ...)                        \
  do {                                                                                            \
    (void)(stream);                                                                               \
    hipemu::launch((grid), (block), [=]() { kernel(__VA_ARGS__); });                              \
  } while (0)

inline const char* hipGetErrorString(hipError_t e) {
  return e == hipSuccess ? "hipSuccess" : "emulated HIP error";
}
inline hipError_t hipGetLastError() { return hipSuccess; }

template <typename T> hipError_t hipMalloc(T** ptr, size_t bytes) {
  *ptr = static_cast<T*>(std::malloc(bytes ? bytes : 1));
  return *ptr ? hipSuccess : hipErrorOutOfMemory;
}
inline hipError_t hipFree(void* ptr) {
  std::free(ptr);
  return hipSuccess;
}
inline hipError_t hipMemcpy(void* dst, const void* src, size_t bytes, hipMemcpyKind) {
  std::memcpy(dst, src, bytes);
  return hipSuccess;
}

// Launches run to completion before they return, so there is nothing to wait for.
inline hipError_t hipStreamCreate(hipStream_t* s) {
  *s = nullptr;
  return hipSuccess;
}
inline hipError_t hipStreamDestroy(hipStream_t) { return hipSuccess; }
inline hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }

inline hipError_t hipEventCreate(hipEvent_t* e) {
  *e = new hipEmuEvent{};
  return hipSuccess;
}
inline hipError_t hipEventDestroy(hipEvent_t e) {
  delete e;
  return hipSuccess;
}
inline hipError_t hipEventRecord(hipEvent_t e, hipStream_t) {
  e->t = std::chrono::steady_clock::now();
  return hipSuccess;
}
inline hipError_t hipEventSynchronize(hipEvent_t) { return hipSuccess; }
inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  *ms = std::chrono::duration<float, std::milli>(stop->t - start->t).count();
  return hipSuccess;
}

inline hipError_t hipGetDevice(int* id) {
  *id = 0;
  return hipSuccess;
}
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int) {
  std::strncpy(p->name, "HIP emulation on CPU (not a GPU)", sizeof(p->name));
  return hipSuccess;
}
