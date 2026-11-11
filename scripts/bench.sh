#!/usr/bin/env bash
# Run the whole benchmark suite and write docs/results.csv.
#
#   1. hipaug bench: CPU reference and HIP kernels (hipEvents, 10 warmup, 100 timed, median)
#   2. bench/baselines.py: OpenCV and torchvision on the CPU, if they are installed
#   3. docs/make_charts.py: charts for the README, if matplotlib is installed
#
# Extra arguments go to `hipaug bench`; shared workload options also reach the CPU baselines.
# --cpu-runs controls baseline runs; --no-cpu skips all CPU implementations.
# Environment: BUILD_DIR (default build), OUT (default docs/results.csv), PYTHON (default python3).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
OUT="${OUT:-docs/results.csv}"
PYTHON="${PYTHON:-python3}"

# Extract only options understood by the Python baselines. Keep the original
# argument list intact for the C++ CLI (including its validation).
baseline_args=()
run_baselines=1
args=("$@")
for ((i = 0; i < ${#args[@]}; ++i)); do
  case "${args[i]}" in
    --batches|--size|--src-min|--src-max|--peak-gbps|--pipelines|--warmup)
      baseline_args+=("${args[i]}" "${args[i+1]:-}")
      i=$((i + 1))
      ;;
    --cpu-runs)
      baseline_args+=(--runs "${args[i+1]:-}")
      i=$((i + 1))
      ;;
    --out)
      OUT="${args[i+1]:-}"
      i=$((i + 1))
      ;;
    --no-cpu) run_baselines=0 ;;
  esac
done

"$BUILD_DIR/hipaug" bench --out "$OUT" "$@"

if (( ! run_baselines )); then
  echo "bench.sh: --no-cpu supplied, skipping CPU baselines"
elif "$PYTHON" -c "import numpy, cv2" 2>/dev/null; then
  "$PYTHON" bench/baselines.py --append "$OUT" "${baseline_args[@]}"
else
  echo "bench.sh: numpy or opencv-python not installed, skipping the OpenCV and torchvision baselines"
fi

if "$PYTHON" -c "import matplotlib" 2>/dev/null; then
  "$PYTHON" docs/make_charts.py "$OUT"
else
  echo "bench.sh: matplotlib not installed, skipping charts"
fi
