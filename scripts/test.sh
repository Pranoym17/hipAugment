#!/usr/bin/env bash
# Run the test suite. On a HIP build this includes the GPU vs CPU tests,
# which need a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
ctest --test-dir "$BUILD_DIR" --output-on-failure "$@"
