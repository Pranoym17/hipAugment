#!/usr/bin/env bash
# Run the whole benchmark suite and write docs/results.csv.
#
#   1. hipaug bench: CPU reference and HIP kernels (hipEvents, 10 warmup, 100 timed, median)
#   2. bench/baselines.py: OpenCV and torchvision on the CPU, if they are installed
#   3. docs/make_charts.py: charts for the README, if matplotlib is installed
#
# Extra arguments go to `hipaug bench`, for example: scripts/bench.sh --batches 1,16
# Environment: BUILD_DIR (default build), OUT (default docs/results.csv), PYTHON (default python3).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
OUT="${OUT:-docs/results.csv}"
PYTHON="${PYTHON:-python3}"

"$BUILD_DIR/hipaug" bench --out "$OUT" "$@"

if "$PYTHON" -c "import numpy, cv2" 2>/dev/null; then
  "$PYTHON" bench/baselines.py --append "$OUT"
else
  echo "bench.sh: numpy or opencv-python not installed, skipping the OpenCV and torchvision baselines"
fi

if "$PYTHON" -c "import matplotlib" 2>/dev/null; then
  "$PYTHON" docs/make_charts.py "$OUT"
else
  echo "bench.sh: matplotlib not installed, skipping charts"
fi
