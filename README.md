# hipAugment

Batched RGB image preprocessing in C++17 and HIP, with CPU references and a fused
crop/resize/mirror/normalize kernel. The project explores how kernel fusion and
shared-memory tiling affect the preprocessing stage of an image-training pipeline.
Its batch and ROI model is inspired by ROCm Performance Primitives; it is an
independent implementation.

**Status:** CPU functionality and kernel emulation were tested during initial
development. Real GPU correctness and performance are still unverified. An
experimental CUDA build reuses the kernel sources for NVIDIA GPUs; its first
compilation and device run are pending. No GPU speedup is claimed.

## Operations

| Operation | Input → output | Behavior |
| --- | --- | --- |
| `resize` | u8 NHWC → u8 NHWC | Bilinear resize of each image's ROI |
| `gaussian_blur` | u8 NHWC → u8 NHWC | 3×3 or 5×5 Gaussian blur, with ROI-edge replication |
| `crop_mirror_normalize` | u8 NHWC → f32 NCHW | Crop, optional horizontal mirror, per-channel normalization |
| `fused_rcmn` | u8 NHWC → f32 NCHW | ROI crop, resize, mirror, normalize and layout conversion in one pass |

Mixed-size images share a padded batch buffer and carry individual regions of
interest. Library operations write to caller-allocated buffers. GPU operations
accept a caller stream and enqueue work without synchronizing.

![CPU reference outputs for each operation](docs/grid.png)

The preview uses synthetic images and the CPU reference backend. Float outputs
are mapped back to RGB for display; this image is not evidence of GPU execution.

## Try it on the CPU

Use Linux or WSL 2 with a C++17 compiler and CMake 3.21+. The test dependency is
downloaded from GitHub during configuration.

```bash
HIPAUG_HIP=OFF BUILD_DIR=build-cpu bash scripts/build.sh
BUILD_DIR=build-cpu bash scripts/test.sh

# Process the included samples; outputs are written as PNGs.
./build-cpu/hipaug run --op fused_rcmn --backend cpu \
  --batch 3 --size 224x224 docs/samples demo-output
```

```bash
./build-cpu/hipaug run --op gaussian_blur --backend cpu \
  --ksize 5 --sigma 2.0 docs/samples demo-output
./build-cpu/hipaug --help
```

The CLI decodes images on the CPU with stb_image. Float operations are
denormalized when saving PNGs; use the C++ API to consume the normalized tensors.

## GPU builds

**AMD / ROCm**

```bash
HIPAUG_HIP=ON BUILD_DIR=build-hip bash scripts/build.sh
BUILD_DIR=build-hip bash scripts/test.sh
```

An AMD GPU is needed to run the tests. For compilation on a ROCm machine without
a GPU, specify the target, e.g. `-DCMAKE_HIP_ARCHITECTURES=gfx942` for MI300X.

**NVIDIA / CUDA — experimental**

```bash
HIPAUG_CUDA=ON BUILD_DIR=build-cuda bash scripts/build.sh \
  -DCMAKE_CUDA_ARCHITECTURES=120
BUILD_DIR=build-cuda bash scripts/test.sh
```

Architecture `120` targets RTX Blackwell and requires a compatible CUDA toolkit.
Use your GPU's architecture for other devices. For Ubuntu 22.04/24.04 on WSL 2,
`bash scripts/setup_cuda_wsl.sh` installs CUDA 12.8 and build dependencies, then
builds/tests CPU, emulation and CUDA targets. It needs sudo and several GB of
disk space. It compiles the Python extension but does not validate PyTorch.
The CLI retains `--backend hip` for the compiled GPU implementation, including CUDA.

| CMake option | Default | Purpose |
| --- | --- | --- |
| `HIPAUG_BUILD_HIP` | ON | AMD HIP backend; requires ROCm |
| `HIPAUG_BUILD_CUDA` | OFF | NVIDIA backend; requires CUDA and `HIPAUG_BUILD_HIP=OFF` |
| `HIPAUG_HIP_EMULATION` | OFF | Run the shared kernel source on CPU fibers for tests |
| `HIPAUG_BUILD_PYTHON` | OFF | Build the experimental pybind11 extension |
| `HIPAUG_BUILD_TESTS` | ON | Build GoogleTest suites |
| `HIPAUG_BUILD_TOOLS` | ON | Build the CLI and benchmark harness |

For GPU C++ calls, include `hipaug/hipaug.hpp` and link the CMake `hipaug` target.
Host implementations use the same operation names in `hipaug::cpu` without a
stream argument. Python bindings and the DataLoader demo are experimental;
actual PyTorch interoperability, stream ordering and tensor lifetimes still need
device validation. The wrapper currently launches on the default stream.

## Design

- Images are padded to the largest dimensions in the batch, with individual ROIs.
  Descriptors address elements as `n*stride_n + y*stride_h + x*stride_w + c*stride_c`.
- Each thread computes one RGB output pixel. Blocks are 16x16; grid z selects
  the image. Block geometry has not been tuned on hardware.
- Resize uses half-pixel bilinear sampling and ROI-edge clamping. Blur has direct
  and shared-memory/LDS variants; their relative performance is unmeasured.
- Fusion removes one launch and the intermediate u8 RGB image. At 224x224 it
  eliminates 294 KiB/image of logical intermediate read/write traffic. This is
  an analytical saving, not a measured speedup or physical DRAM traffic result.
- The separate path rounds resize output to u8; fusion retains the float.
  Fused/separate tests allow `0.5/std[c] + 1e-4` for positive ImageNet standard
  deviations. Same-op CPU/GPU tests use 1 for u8 and `1e-4` absolute for f32.
- Empty ROIs leave outputs untouched; blur writes only inside the ROI. Callers
  must initialize destinations when they need defined values outside those writes.

## Tests and measurements

```bash
# Exercise kernel indexing and barriers without a GPU.
HIPAUG_HIP=OFF HIPAUG_EMULATION=ON BUILD_DIR=build-emu bash scripts/build.sh
BUILD_DIR=build-emu bash scripts/test.sh

# Dependency-free checks for benchmark orchestration and profiler reports.
python3 -m unittest discover -s tests -p test_scripts.py -v
```

CPU tests include hand-computed results and ROI invariants. The GPU suite compares
each kernel with its CPU reference, compares direct and tiled blur, and checks
fused versus separate processing. CPU emulation checks logic; it does not model
GPU timing, occupancy or hardware races. CI workflows cover CPU and emulation,
but a successful hosted CI run has not yet been recorded.

Initial development recorded 31 passing CPU tests and nine GPU-style tests under
emulation. These predate the experimental CUDA changes; the current C++ revision
still needs a fresh build/test run. Five script regression checks pass locally.

**There are no measured GPU speedups or GPU benchmark results to report yet.**
The harness supports warmups, median latency, images/s, estimated effective
bandwidth, CPU/OpenCV/torchvision comparisons, and a rocprofv3 reporting path.

On a validated AMD GPU build:

```bash
BUILD_DIR=build-hip bash scripts/bench.sh
BUILD_DIR=build-hip bash scripts/profile.sh
```

The benchmark writes `docs/results.csv` and optional charts; profiling writes
`docs/profile.md` and `docs/profile.csv`. NumPy/OpenCV, Matplotlib and a compatible
torch/torchvision pair enable the optional CPU baselines and charts. Custom
`--batches`, `--size`, source ranges and pipeline filters reach the baselines.
`--cpu-runs` controls CPU samples; `--runs` controls GPU samples; `--no-cpu` skips
all CPU implementations.

Reported bandwidth is an estimate of logical bytes divided by latency, not a
hardware-counter measurement. Verify the actual device's peak bandwidth before
reporting utilization. CUDA benchmarks require an explicit `--peak-gbps` value;
CSV implementation labels retain their historical `hip*` spelling, so record
backend/device and keep NVIDIA results separate. The rocprofv3 schema still needs
verification on hardware; NVIDIA profiling needs an Nsight workflow.

## Source layout

```text
include/hipaug/  Public API, error checking, optional CUDA runtime adapter
src/cpu/        CPU reference implementations
src/hip/        Shared GPU kernel source, direct and tiled blur
src/common/     Validation, Gaussian weights, caller-side device buffers
tests/          CPU/GPU suites, CPU kernel emulation, script regression tests
tools/cli/      Image loading, batching and command-line interface
bench/          C++ benchmark harness and Python CPU baselines
python/         Experimental bindings and DataLoader demo
scripts/        Build, test, setup, benchmark and AMD profiling commands
docs/           CPU preview assets and chart/grid generation scripts
```

## Next steps

Complete device correctness testing, validate PyTorch interoperability, measure
fusion and blur variants, and publish reproducible results. Further extensions
could include separable blur, additional transforms and GPU image decoding.

## License

MIT. Vendored stb headers retain their own license notices.
