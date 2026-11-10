#!/usr/bin/env bash
# Configure and build hipAugment.
#
#   scripts/build.sh              HIP backend if ROCm is installed, otherwise CPU only
#   HIPAUG_HIP=OFF scripts/build.sh   force the CPU-only build
#   HIPAUG_PYTHON=ON scripts/build.sh also build the Python module
#
# Other environment variables: BUILD_DIR (default build), BUILD_TYPE (default Release),
# ROCM_PATH (default /opt/rocm).
set -euo pipefail
cd "$(dirname "$0")/.."

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [[ -z "${HIPAUG_HIP:-}" ]]; then
  if [[ -d "$ROCM_PATH/lib/cmake/hip" ]]; then HIPAUG_HIP=ON; else HIPAUG_HIP=OFF; fi
fi
echo "hipAugment: HIP backend $HIPAUG_HIP, build type $BUILD_TYPE, build dir $BUILD_DIR"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DHIPAUG_BUILD_HIP="$HIPAUG_HIP" \
  -DHIPAUG_BUILD_PYTHON="${HIPAUG_PYTHON:-OFF}" \
  -DROCM_PATH="$ROCM_PATH"
cmake --build "$BUILD_DIR" -j "$(nproc)"
