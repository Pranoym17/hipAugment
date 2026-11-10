// HIP_CHECK: wrap every HIP runtime call. On failure it logs the call and returns
// Status::HipError from the enclosing function, so the library never crashes its caller.
#pragma once

#include <cstdio>

#include <hip/hip_runtime.h>

#include "hipaug/hipaug.hpp"

#define HIP_CHECK(call)                                                                            \
  do {                                                                                             \
    hipError_t hipaug_err_ = (call);                                                               \
    if (hipaug_err_ != hipSuccess) {                                                               \
      std::fprintf(stderr, "hipaug: %s failed: %s (%s:%d)\n", #call,                               \
                   hipGetErrorString(hipaug_err_), __FILE__, __LINE__);                            \
      return ::hipaug::Status::HipError;                                                           \
    }                                                                                              \
  } while (0)
