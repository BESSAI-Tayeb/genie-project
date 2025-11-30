#!/bin/bash

set -e  # Exit on error

# If the machine doesn't have CUDA/NVidia, let the user skip this build.
# The script will still work if you rely on the Python CPU fallback (knn_algorithms.py).
# To force a CUDA build even if the checks fail, set the environment variable FORCE_CUDA=1.
if [ -z "$FORCE_CUDA" ]; then
  if ! command -v nvcc >/dev/null 2>&1 || ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvcc or nvidia-smi not found. Skipping OptiX/CUDA build in this environment."
    echo "To force a build attempt, set FORCE_CUDA=1 and re-run the script (not recommended without CUDA)."
    exit 0
  fi
fi

# === Find GPU Compute Capability ===
echo "🔍 Detecting GPU Compute Capability..."
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
CUDA_ARCH="sm_${CC/./}"
echo "Detected CUDA_ARCH=$CUDA_ARCH"

# === Find CUDA Include Path ===
echo "🔍 Detecting CUDA include path..."
CUDA_INCLUDE=$(dirname $(dirname $(which nvcc)))/include
echo "Detected CUDA include path: $CUDA_INCLUDE"

# === Find Thrust Include Path ===
echo "🔍 Detecting Thrust include path..."
CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
THRUST_INCLUDE="$CUDA_ROOT/include"
# Check if thrust headers exist in CUDA include, otherwise try targets path
if [ ! -d "$THRUST_INCLUDE/thrust" ]; then
    THRUST_INCLUDE="$CUDA_ROOT/targets/x86_64-linux/include"
fi
echo "Detected Thrust include path: $THRUST_INCLUDE"

LIBCUDA_PATH=$(find /usr -name 'libcuda.so*' 2>/dev/null | head -n 1)

if [ -n "$LIBCUDA_PATH" ]; then
  CUDA_LIB_DIR=$(dirname "$LIBCUDA_PATH")
  echo "Found libcuda.so at $CUDA_LIB_DIR"
else
  echo "libcuda.so not found in system paths!"
  # fallback or error handling here
  CUDA_LIB_DIR=""
fi

OPTIX_INCLUDE="NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64/include"

# Set Python path for includes and Torch
PYTHON_BIN=python3

# Safe fallback (no cpp_extension required)
PYTORCH_DIR=$($PYTHON_BIN -c "import torch, os; print(os.path.join(torch.__path__[0], 'include'))")
PYTORCH_API_DIR="$PYTORCH_DIR/torch/csrc/api/include"

PYTHON_SITE_PACKAGES=$($PYTHON_BIN -c "import site; print(site.getsitepackages()[0])")
TORCH_LIB_DIR="$PYTHON_SITE_PACKAGES/torch/lib"
PYBIND11_INCLUDES=$($PYTHON_BIN -m pybind11 --includes)

# Get Python include path
if command -v python3-config &> /dev/null; then
    PYTHON_INCLUDE=$(python3-config --includes)
    PYTHON_LDFLAGS=$(python3-config --ldflags)
else
    PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_INCLUDE_DIR=$($PYTHON_BIN -c "from sysconfig import get_paths; print(get_paths()['include'])")
    PYTHON_INCLUDE="-I${PYTHON_INCLUDE_DIR}"
    PYTHON_LDFLAGS="-lpython${PYTHON_VERSION}"
fi
echo "Python includes: $PYTHON_INCLUDE"

# === Auto-detect ABI Flag ===
echo "🔍 Detecting PyTorch ABI flag..."
TORCH_CXX11_ABI=$($PYTHON_BIN -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
if [ "$TORCH_CXX11_ABI" = "1" ]; then
    ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=1"
    echo "Detected ABI: CXX11 ABI enabled"
else
    ABI_FLAG="-D_GLIBCXX_USE_CXX11_ABI=0"
    echo "Detected ABI: CXX11 ABI disabled (pre-CXX11)"
fi
CXX_STD="-std=c++17"

# === Build Directories ===
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# === 1. Compile OptiX Shader to PTX ===
echo "📦 Compiling shaders.cu to PTX..."
nvcc -ptx -arch=${CUDA_ARCH} -o ${BUILD_DIR}/shaders.ptx shaders.cu \
  -I${OPTIX_INCLUDE} -I${CUDA_INCLUDE} ${CXX_STD}

# === 2. Compile CUDA Source ===
echo "🔧 Compiling KNN.cu..."
nvcc -Xcompiler -fPIC -c KNN.cu -o ${BUILD_DIR}/KNN.o \
  --gpu-architecture=compute_${CC/./} --gpu-code=${CUDA_ARCH} \
  -I${OPTIX_INCLUDE} -I${CUDA_INCLUDE} -I${THRUST_INCLUDE} \
  ${ABI_FLAG} ${CXX_STD}

# === Compile bindings.cpp (host + CUDA-aware using nvcc) ===
echo "🔗 Compiling bindings.cpp with nvcc..."
nvcc -c -Xcompiler -fPIC bindings.cpp -o ${BUILD_DIR}/bindings.o \
  -std=c++17 ${ABI_FLAG} ${PYBIND11_INCLUDES} ${PYTHON_INCLUDE} \
  -I${CUDA_INCLUDE} -I${OPTIX_INCLUDE} -I${THRUST_INCLUDE} -I${PYTORCH_DIR} -I${PYTORCH_API_DIR} \
  -I${TORCH_LIB_DIR} \
  -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}

# === Link Everything Together using nvcc (handles CUDA runtime libraries) ===
echo "🔗 Linking shared library with nvcc..."
nvcc -shared -Xcompiler -fPIC ${BUILD_DIR}/bindings.o ${BUILD_DIR}/KNN.o -o optix_knn.so \
  ${CXX_STD} ${ABI_FLAG} ${PYBIND11_INCLUDES} ${PYTHON_LDFLAGS} \
  -I${CUDA_INCLUDE} -I${OPTIX_INCLUDE} -I${PYTORCH_DIR} -I${PYTORCH_API_DIR} \
  -L${TORCH_LIB_DIR} -L${CUDA_LIB_DIR} \
  -ltorch -ltorch_cpu -ltorch_python -lc10 -lcudart -lcuda \
  -Wl,-rpath,${TORCH_LIB_DIR}

echo "✅ Build complete! Output: optix_knn.so"
