#!/bin/bash
# ============================================================================
#  HEonGPU local installation script (conda-based, no HPC modules)
#  Uses nvcc / CUDA headers / GMP / NTL / host compiler from a conda env.
#  Installs HEonGPU into $HOME/local/heongpu-conda (user-local, no root).
# ============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-chehabEnv}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${HOME}/local/heongpu-conda}"
BUILD_DIR="/tmp/${USER}_heongpu_build"
HEONGPU_REPO="https://github.com/Alisah-Ozcan/HEonGPU.git"
HEONGPU_BRANCH="main"
CUDA_ARCH="${CUDA_ARCH:-80}"      # 80 = A100; override: CUDA_ARCH=90 ./script.sh
JOBS="${JOBS:-$(nproc)}"

echo "==========================================================="
echo "  HEonGPU Installation Script (conda)"
echo "  Conda env:      ${ENV_NAME}"
echo "  Install prefix: ${INSTALL_PREFIX}"
echo "==========================================================="

# ---------------------------------------------------------------------------
# 1. Activate the conda env (no `module load` anywhere)
# ---------------------------------------------------------------------------
if [ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]; then
    if ! command -v conda &>/dev/null; then
        echo "ERROR: conda not found in PATH. Activate it first (or load your conda module)."
        exit 1
    fi
    set +u   # conda.sh / activate scripts are not nounset-safe
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    set -u
fi

: "${CONDA_PREFIX:?CONDA_PREFIX is not set - is the env activated?}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"

# Make sure the env wins over anything the HPC session may have loaded,
# and drop variables that would drag in the system/module CUDA.
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset CUDA_HOME CUDA_PATH CUDA_ROOT CUDA_DIR CUDACXX CUDAHOSTCXX CPATH \
      C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Locate everything inside the env and verify it is complete
# ---------------------------------------------------------------------------
NVCC="${CONDA_PREFIX}/bin/nvcc"
HOST_CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
HOST_CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

# CUDA headers: conda puts them under targets/x86_64-linux/include
CUDA_INC=""
for c in "${CONDA_PREFIX}/targets/x86_64-linux/include" "${CONDA_PREFIX}/include"; do
    if [ -f "${c}/cuda_runtime.h" ]; then CUDA_INC="${c}"; break; fi
done

# Thrust (shipped by cuda-cccl)
THRUST_DIR=""
for c in "${CUDA_INC}" "${CUDA_INC}/cccl" "${CONDA_PREFIX}/include" "${CONDA_PREFIX}/include/cccl"; do
    if [ -n "${c}" ] && [ -d "${c}/thrust" ]; then THRUST_DIR="${c}"; break; fi
done

missing=()
[ -x "${NVCC}" ]                          || missing+=("cuda-nvcc")
[ -n "${CUDA_INC}" ]                      || missing+=("cuda-cudart-dev")
[ -n "${THRUST_DIR}" ]                    || missing+=("cuda-cccl")
[ -f "${CONDA_PREFIX}/include/gmp.h" ]    || missing+=("gmp")
[ -f "${CONDA_PREFIX}/include/NTL/ZZ.h" ] || missing+=("ntl")
[ -x "${HOST_CXX}" ]                      || missing+=("gxx_linux-64=12")

if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: missing packages in env '${ENV_NAME}': ${missing[*]}"
    echo ""
    echo "Install them with (adjust the CUDA label/version to match your env):"
    echo "  conda install -n ${ENV_NAME} -c \"nvidia/label/cuda-12.4.1\" \\"
    echo "      cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-cccl=12.4"
    echo "  conda install -n ${ENV_NAME} -c conda-forge gmp ntl cmake \"gxx_linux-64=12\""
    exit 1
fi

# Sanity: nvcc must be the conda one, not an HPC leftover
if [ "$(command -v nvcc)" != "${NVCC}" ]; then
    echo "ERROR: 'nvcc' resolves to $(command -v nvcc), expected ${NVCC}"
    exit 1
fi

echo ""
echo "nvcc:        $(nvcc --version | grep release)"
echo "CMake:       $(cmake --version | head -1)"
echo "Host g++:    $(${HOST_CXX} --version | head -1)"
echo "CUDA headers: ${CUDA_INC}"
echo "Thrust dir:   ${THRUST_DIR}"
echo "CUDA arch:    ${CUDA_ARCH}"
echo ""

# ---------------------------------------------------------------------------
# 3. Clone
# ---------------------------------------------------------------------------
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Cloning HEonGPU..."
git clone --depth 1 --branch "${HEONGPU_BRANCH}" "${HEONGPU_REPO}" heongpu
cd heongpu

# ---------------------------------------------------------------------------
# 4. Configure
# ---------------------------------------------------------------------------
echo ""
echo "Configuring HEonGPU..."
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${HOST_CC}" \
    -DCMAKE_CXX_COMPILER="${HOST_CXX}" \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_CUDA_HOST_COMPILER="${HOST_CXX}" \
    -DCUDAToolkit_ROOT="${CONDA_PREFIX}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DTHRUST_INCLUDE_DIR="${THRUST_DIR}" \
    -DGMP_ROOT="${CONDA_PREFIX}" \
    -DGMP_DIR="${CONDA_PREFIX}" \
    -DGMP_INCLUDE_DIR="${CONDA_PREFIX}/include" \
    -DGMP_LIBRARIES="${CONDA_PREFIX}/lib/libgmp.so" \
    -DNTL_ROOT="${CONDA_PREFIX}" \
    -DNTL_DIR="${CONDA_PREFIX}" \
    -DNTL_INCLUDE_DIR="${CONDA_PREFIX}/include" \
    -DNTL_LIBRARIES="${CONDA_PREFIX}/lib/libntl.so" \
    -DCMAKE_CXX_FLAGS="-I${CONDA_PREFIX}/include" \
    -DCMAKE_CUDA_FLAGS="-I${CONDA_PREFIX}/include -I${CUDA_INC}" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${CONDA_PREFIX}/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${CONDA_PREFIX}/lib" \
    -DCMAKE_BUILD_RPATH="${CONDA_PREFIX}/lib" \
    -DCMAKE_INSTALL_RPATH="${CONDA_PREFIX}/lib;${INSTALL_PREFIX}/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DHEONGPU_BUILD_EXAMPLES=OFF \
    -DHEONGPU_BUILD_TESTS=OFF

# ---------------------------------------------------------------------------
# 5. Build + install
# ---------------------------------------------------------------------------
echo ""
echo "Building HEonGPU with ${JOBS} jobs (this may take 10-20 minutes)..."
cmake --build build -j"${JOBS}"

echo ""
echo "Installing to ${INSTALL_PREFIX}..."
cmake --install build

echo ""
echo "Cleaning up build directory..."
cd /
rm -rf "${BUILD_DIR}"

echo ""
echo "==========================================================="
echo "  HEonGPU installed successfully!"
echo "  Location: ${INSTALL_PREFIX}"
echo ""
echo "  To use with CMake (inside the '${ENV_NAME}' env), add:"
echo "    -DCMAKE_PREFIX_PATH=\"${INSTALL_PREFIX};\${CONDA_PREFIX}\""
echo "    -DCMAKE_CUDA_COMPILER=\${CONDA_PREFIX}/bin/nvcc"
echo "==========================================================="