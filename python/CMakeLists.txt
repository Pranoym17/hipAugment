# pybind11 module. Builds python/hipaugment/_core.*.so next to __init__.py, so
# `PYTHONPATH=python` (or `pip install -e python`) makes `import hipaugment` work.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(hipaugment_core hipaugment_module.cpp)
set_target_properties(hipaugment_core PROPERTIES
  OUTPUT_NAME _core
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hipaugment)
target_link_libraries(hipaugment_core PRIVATE hipaug)
