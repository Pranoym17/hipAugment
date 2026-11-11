#!/usr/bin/env bash
# Install a driver-free CUDA 12.8 toolchain on Ubuntu WSL, then build and test.
# Run as your normal Ubuntu user; sudo is used only for system packages.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
trap 'echo "Setup stopped at line $LINENO. Share the last console output." >&2' ERR
echo "Starting hipAugment CUDA setup in $repo"

source /etc/os-release
if [[ "$ID" != ubuntu || ! "$VERSION_ID" =~ ^(22\.04|24\.04)$ ]]; then
  echo "This installer supports Ubuntu 22.04/24.04; detected $PRETTY_NAME."
  echo "Share this output so the package setup can be adapted."
  exit 1
fi
if ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  echo "This installer is intended for Ubuntu on WSL 2."
  exit 1
fi
if [[ "$(id -u)" == 0 ]]; then
  echo "Run this script as your regular Ubuntu user, without sudo."
  exit 1
fi
nvidia-smi -L

echo "Installing Ubuntu build dependencies (sudo may ask for your Ubuntu password)."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential g++-12 ninja-build python3-venv python3-dev curl ca-certificates

cuda_root=/usr/local/cuda-12.8
if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
  echo "Installing CUDA Toolkit 12.8 from NVIDIA; the Windows GPU driver stays in use."
  keyring_file="$(mktemp --suffix=.deb)"
  curl --fail --location --retry 3 \
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb" \
    --output "$keyring_file"
  sudo dpkg -i "$keyring_file"
  rm -f -- "$keyring_file"
  sudo apt-get update
  # Unlike cuda or cuda-runtime, this package does not install a Linux GPU driver.
  sudo apt-get install -y --no-install-recommends cuda-toolkit-12-8
fi
export PATH="$cuda_root/bin:$PATH"
export CUDACXX="$cuda_root/bin/nvcc"
export CXX=/usr/bin/g++-12
nvcc --version

# Keep Linux executables/builds on the Linux filesystem, not /mnt/c.
venv_dir="${HIPAUG_CUDA_VENV:-$HOME/hipaugment-cuda-venv}"
build_root="${HIPAUG_CUDA_BUILD_ROOT:-$HOME/hipaugment-cuda-builds}"
jobs="${HIPAUG_JOBS:-2}"
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
python -m pip install 'cmake==3.31.6' numpy opencv-python-headless matplotlib pybind11
python -m unittest discover -s tests -p test_scripts.py -v

for backend in cpu emulation cuda; do
  build_dir="$build_root/$backend"
  options=(-DHIPAUG_BUILD_HIP=OFF -DHIPAUG_BUILD_CUDA=OFF -DHIPAUG_HIP_EMULATION=OFF)
  if [[ "$backend" == emulation ]]; then
    options+=(-DHIPAUG_HIP_EMULATION=ON)
  elif [[ "$backend" == cuda ]]; then
    options+=(-DHIPAUG_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
              -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
              -DHIPAUG_BUILD_PYTHON=ON
              "-Dpybind11_DIR=$(python -m pybind11 --cmakedir)"
              "-DPYTHON_EXECUTABLE=$venv_dir/bin/python")
  fi
  echo "Building and testing $backend."
  cmake -S "$repo" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release "${options[@]}"
  cmake --build "$build_dir" -j "$jobs"
  ctest --test-dir "$build_dir" --output-on-failure
done

echo "CPU, emulation, and CUDA C++ tests passed. Python extension compiled."
echo "CUDA executable: $build_root/cuda/hipaug"
echo "Python environment: $venv_dir"
echo "Next: validate the PyTorch wrapper, then collect RTX 5060 benchmarks/profiles."
