#!/usr/bin/env bash
# Profile every HIP kernel with rocprofv3 and write docs/profile.md.
#
# Each (batch size, pipeline) pair runs in its own rocprofv3 session, so the kernel
# statistics of one session belong to one workload. scripts/profile_report.py then joins
# the kernel times with the byte counts from `hipaug bench` to get bandwidth per kernel.
#
# Environment: BUILD_DIR (default build), BATCHES (default "1 16 64 256"),
# PROFILE_DIR (default profile_out), PYTHON (default python3).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
BATCHES="${BATCHES:-1 16 64 256}"
PROFILE_DIR="${PROFILE_DIR:-profile_out}"
PYTHON="${PYTHON:-python3}"
PIPELINES="resize crop_mirror_normalize gaussian_blur_3x3 gaussian_blur_5x5 resize_cmn"

command -v rocprofv3 >/dev/null || { echo "profile.sh: rocprofv3 not found (is ROCm on PATH?)"; exit 1; }

rm -rf "$PROFILE_DIR"
for batch in $BATCHES; do
  for pipeline in $PIPELINES; do
    dir="$PROFILE_DIR/b$batch/$pipeline"
    mkdir -p "$dir"
    echo "profile.sh: batch $batch, $pipeline"
    # --kernel-trace  record every kernel dispatch
    # --stats         write per-kernel statistics (<name>_kernel_stats.csv)
    # -S              print a summary; -T  short kernel names
    rocprofv3 --kernel-trace --stats -T -S --summary-output-file "$dir/summary.txt" \
      -d "$dir" -o hipaug -f csv -- \
      "$BUILD_DIR/hipaug" bench --no-cpu --batches "$batch" --pipelines "$pipeline" \
      --out "$dir/results.csv" >"$dir/bench.log" 2>&1
  done
done

"$PYTHON" scripts/profile_report.py "$PROFILE_DIR" --md docs/profile.md --csv docs/profile.csv
echo "profile.sh: wrote docs/profile.md and docs/profile.csv"
