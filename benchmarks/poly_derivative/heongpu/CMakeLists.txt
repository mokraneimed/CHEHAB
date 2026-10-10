cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Toolchain: use the conda env (nvcc 12.4 + conda GCC 13) when it is active.
# Must be set before project().
# ---------------------------------------------------------------------------
if(DEFINED ENV{CONDA_PREFIX})
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc")
    endif()
    set(_conda_cxx "$ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++")
    set(_conda_cc  "$ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc")
    if(EXISTS "${_conda_cxx}" AND NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "${_conda_cxx}")
        set(CMAKE_C_COMPILER   "${_conda_cc}")
        set(CMAKE_CUDA_HOST_COMPILER "${_conda_cxx}")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)   # A100; override with -DCMAKE_CUDA_ARCHITECTURES=...
endif()

project(generated_fhe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# HEonGPU install location (default: the conda-built install)
#   override with:  -DHEONGPU_DIR=/path   or   export HEONGPU_DIR=/path
# ---------------------------------------------------------------------------
if(NOT DEFINED HEONGPU_DIR)
    if(DEFINED ENV{HEONGPU_DIR})
        set(HEONGPU_DIR "$ENV{HEONGPU_DIR}")
    else()
        set(HEONGPU_DIR "$ENV{HOME}/local/heongpu-conda")
    endif()
endif()
message(STATUS "Using HEonGPU from: ${HEONGPU_DIR}")

if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${HEONGPU_DIR}")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

# libheongpu.a uses GMP and NTL (big-integer / RR arithmetic): take them from conda
set(_conda_lib "")
if(DEFINED ENV{CONDA_PREFIX})
    set(_conda_lib "$ENV{CONDA_PREFIX}/lib")
endif()
find_library(NTL_LIB NAMES ntl HINTS "${_conda_lib}" REQUIRED)
find_library(GMP_LIB NAMES gmp HINTS "${_conda_lib}" REQUIRED)
message(STATUS "NTL: ${NTL_LIB}")
message(STATUS "GMP: ${GMP_LIB}")

# ---------------------------------------------------------------------------
# HEonGPU headers and libraries (layout of the heongpu-conda install)
# ---------------------------------------------------------------------------
set(HEONGPU_INCLUDES
    "${HEONGPU_DIR}/include"                      # rmm/, rapids_logger/, nvtx3/
    "${HEONGPU_DIR}/include/HEonGPU-1.1"          # <heongpu/heongpu.hpp>
    "${HEONGPU_DIR}/include/GPUFFT-1.0"
    "${HEONGPU_DIR}/include/GPUNTT-1.0"
    "${HEONGPU_DIR}/include/RNGonGPU-1.0"
)

set(HEONGPU_LIBS
    "${HEONGPU_DIR}/lib/libheongpu.a"
    "${HEONGPU_DIR}/lib64/libfft-1.0.a"
    "${HEONGPU_DIR}/lib64/libntt-1.0.a"
    "${HEONGPU_DIR}/lib64/librngongpu-1.0.a"
    "${HEONGPU_DIR}/lib64/librmm.so"
    "${HEONGPU_DIR}/lib64/librapids_logger.so"
)

# Fail early with a clear message if something is missing
foreach(_f ${HEONGPU_LIBS} "${HEONGPU_DIR}/include/HEonGPU-1.1/heongpu/heongpu.hpp")
    if(NOT EXISTS "${_f}")
        message(FATAL_ERROR "Missing HEonGPU file: ${_f}\n"
                            "Check HEONGPU_DIR (currently ${HEONGPU_DIR}).")
    endif()
endforeach()

add_executable(generated_fhe generated_fhe.cu)

# HEonGPU's bundled CCCL (thrust/cub/libcudacxx 3.0.3) must come first on the
# include path, so it matches librmm.so and shadows the toolkit's older CCCL.
target_include_directories(generated_fhe BEFORE PRIVATE "${HEONGPU_DIR}/include/rapids")
target_include_directories(generated_fhe PRIVATE ${HEONGPU_INCLUDES})

target_link_libraries(generated_fhe PRIVATE
    ${HEONGPU_LIBS}
    ${NTL_LIB}
    ${GMP_LIB}
    CUDA::cudart
    CUDA::curand
    Threads::Threads
    OpenSSL::Crypto
    OpenSSL::SSL
    ZLIB::ZLIB
)

target_compile_definitions(generated_fhe PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

set_target_properties(generated_fhe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Let the executable find librmm.so / librapids_logger.so / conda libs at runtime
set(_rpath "${HEONGPU_DIR}/lib64")
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _rpath "$ENV{CONDA_PREFIX}/lib")
endif()
set_target_properties(generated_fhe PROPERTIES
    BUILD_RPATH "${_rpath}"
    INSTALL_RPATH "${_rpath}"
)