include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# CPU reference tests. These build and run on any machine.
add_executable(hipaug_cpu_tests
  test_core.cpp
  test_cpu_resize.cpp
  test_cpu_cmn.cpp)
target_include_directories(hipaug_cpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(hipaug_cpu_tests PRIVATE hipaug GTest::gtest_main)
gtest_discover_tests(hipaug_cpu_tests)

# GPU vs CPU tests. Need a GPU at run time, so they are added as one ctest entry
# instead of being discovered at build time.
if(HIPAUG_BUILD_HIP AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu.cpp)
  add_executable(hipaug_gpu_tests test_gpu.cpp)
  target_include_directories(hipaug_gpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(hipaug_gpu_tests PRIVATE hipaug GTest::gtest_main)
  add_test(NAME hipaug_gpu_tests COMMAND hipaug_gpu_tests)
endif()
