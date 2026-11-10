#!/usr/bin/env bash
# Configure and build hipAugment.
#
#   scripts/build.sh              HIP backend if ROCm is installed, otherwise CPU only
#   HIPAUG_HIP=OFF scripts/build.sh   force the CPU-only build
#   HIPAUG_PYTHON=ON scripts/build.sh also build the Python module
#   HIPAUG_EMULATION=ON HIPAUG_HIP=OFF scripts/build.sh
#                                 run the HIP kernels on the CPU emulation (tests only)
#
# Other environment variables: BUILD_DIR (default build), BUILD_TYPE (default Release),
# ROCM_PATH (default /opt/rocm). Extra arguments are passed to the CMake configure step,
# for example: scripts/build.sh -DCMAKE_HIP_ARCHITECTURES=gfx942
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
  -DHIPAUG_HIP_EMULATION="${HIPAUG_EMULATION:-OFF}" \
  -DROCM_PATH="$ROCM_PATH" \
  "$@"
cmake --build "$BUILD_DIR" -j "$(nproc)"
