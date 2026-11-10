cmake_minimum_required(VERSION 3.21)
project(hipAugment VERSION 0.1.0 LANGUAGES CXX)

option(HIPAUG_BUILD_HIP "Build the HIP (GPU) backend. Needs ROCm." ON)
option(HIPAUG_BUILD_PYTHON "Build the pybind11 Python module. Needs the HIP backend." OFF)
option(HIPAUG_BUILD_TESTS "Build the GoogleTest suite." ON)
option(HIPAUG_BUILD_TOOLS "Build the hipaug command line tool." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CPU and GPU must evaluate the same float expressions the same way, so fused
# multiply-add contraction is off in both builds.
set(HIPAUG_FP_FLAGS -ffp-contract=off)

find_package(OpenMP)

# ---- library ---------------------------------------------------------------
add_library(hipaug STATIC
  src/common/validate.cpp)
target_include_directories(hipaug PUBLIC include PRIVATE src)
target_compile_options(hipaug PRIVATE -Wall -Wextra ${HIPAUG_FP_FLAGS})
if(OpenMP_CXX_FOUND)
  target_link_libraries(hipaug PRIVATE OpenMP::OpenMP_CXX)
endif()

if(HIPAUG_BUILD_HIP)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH})
    else()
      set(ROCM_PATH /opt/rocm)
    endif()
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
  find_package(hip REQUIRED)
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)

  target_compile_definitions(hipaug PUBLIC HIPAUG_HAS_HIP=1)
  target_link_libraries(hipaug PUBLIC hip::host)
endif()

# ---- command line tool and benchmark harness --------------------------------
if(HIPAUG_BUILD_TOOLS AND EXISTS ${PROJECT_SOURCE_DIR}/tools/cli/main.cpp)
  add_executable(hipaug_cli
    tools/cli/main.cpp
    tools/cli/run.cpp
    tools/cli/image_io.cpp
    bench/bench.cpp)
  set_target_properties(hipaug_cli PROPERTIES OUTPUT_NAME hipaug)
  target_include_directories(hipaug_cli PRIVATE src bench tools/cli third_party)
  target_link_libraries(hipaug_cli PRIVATE hipaug)
endif()

# ---- tests -------------------------------------------------------------------
if(HIPAUG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- python --------------------------------------------------------------------
if(HIPAUG_BUILD_PYTHON)
  if(NOT HIPAUG_BUILD_HIP)
    message(FATAL_ERROR "HIPAUG_BUILD_PYTHON needs HIPAUG_BUILD_HIP=ON")
  endif()
  add_subdirectory(python)
endif()
