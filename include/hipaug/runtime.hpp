// Runtime selection for the shared kernels. CUDA support maps only the HIP calls
// used by this project; it is not a general HIP implementation.
#pragma once

#if defined(HIPAUG_USE_CUDA)
#include <cuda_runtime.h>

using hipError_t = cudaError_t;
using hipStream_t = cudaStream_t;
using hipEvent_t = cudaEvent_t;
using hipDeviceProp_t = cudaDeviceProp;
using hipMemcpyKind = cudaMemcpyKind;
constexpr auto hipSuccess = cudaSuccess;
constexpr auto hipMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr auto hipMemcpyDeviceToHost = cudaMemcpyDeviceToHost;

inline const char* hipGetErrorString(hipError_t error) { return cudaGetErrorString(error); }
inline hipError_t hipGetLastError() { return cudaGetLastError(); }
template <typename T> inline hipError_t hipMalloc(T** ptr, size_t bytes) {
  return cudaMalloc(reinterpret_cast<void**>(ptr), bytes);
}
inline hipError_t hipFree(void* ptr) { return cudaFree(ptr); }
inline hipError_t hipMemcpy(void* dst, const void* src, size_t bytes, hipMemcpyKind kind) {
  return cudaMemcpy(dst, src, bytes, kind);
}
inline hipError_t hipStreamCreate(hipStream_t* stream) { return cudaStreamCreate(stream); }
inline hipError_t hipStreamDestroy(hipStream_t stream) { return cudaStreamDestroy(stream); }
inline hipError_t hipStreamSynchronize(hipStream_t stream) { return cudaStreamSynchronize(stream); }
inline hipError_t hipDeviceSynchronize() { return cudaDeviceSynchronize(); }
inline hipError_t hipGetDevice(int* device) { return cudaGetDevice(device); }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, int device) {
  return cudaGetDeviceProperties(props, device);
}
inline hipError_t hipEventCreate(hipEvent_t* event) { return cudaEventCreate(event); }
inline hipError_t hipEventDestroy(hipEvent_t event) { return cudaEventDestroy(event); }
inline hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  return cudaEventRecord(event, stream);
}
inline hipError_t hipEventSynchronize(hipEvent_t event) { return cudaEventSynchronize(event); }
inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  return cudaEventElapsedTime(ms, start, stop);
}

#if defined(__CUDACC__)
#define hipLaunchKernelGGL(kernel, grid, block, shared_bytes, stream, ...) \
  kernel<<<grid, block, shared_bytes, stream>>>(__VA_ARGS__)
#endif

#else
#include <hip/hip_runtime.h>
#endif
