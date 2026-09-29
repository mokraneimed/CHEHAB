import os
import subprocess

build_path = "."

cmake_content = '''cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")
project(generated_fhe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(DEFINED ENV{HEONGPU_DIR})
    set(HEONGPU_DIR $ENV{HEONGPU_DIR})
elseif(DEFINED ENV{HOME})
    set(HEONGPU_DIR "$ENV{HOME}/local/heongpu")
endif()
set(NTL_DIR "$ENV{HOME}/local/ntl")
set(GMP_DIR "/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gmp-6.3.0-zims4vx7m6ggtn3ava2r2ksidghaly5v")

list(APPEND CMAKE_PREFIX_PATH "${HEONGPU_DIR}")
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(rmm QUIET CONFIG)

set(HEONGPU_INCLUDES
    "${HEONGPU_DIR}/include"
    "${HEONGPU_DIR}/include/HEonGPU-1.1"
    "${HEONGPU_DIR}/include/GPUFFT-1.0"
    "${HEONGPU_DIR}/include/GPUNTT-1.0"
    "${HEONGPU_DIR}/include/RNGonGPU-1.0"
    "${NTL_DIR}/include"
    "${GMP_DIR}/include"
)

set(HEONGPU_LIBS
    "${HEONGPU_DIR}/lib/libheongpu.a"
    "${HEONGPU_DIR}/lib64/libfft-1.0.a"
    "${HEONGPU_DIR}/lib64/libntt-1.0.a"
    "${HEONGPU_DIR}/lib64/librngongpu-1.0.a"
)

set(COMMON_LIBS "${NTL_DIR}/lib/libntl.so" "${GMP_DIR}/lib/libgmp.so" CUDA::cudart Threads::Threads OpenSSL::Crypto OpenSSL::SSL ZLIB::ZLIB)

add_executable(generated_fhe generated_fhe.cu)
target_include_directories(generated_fhe BEFORE PRIVATE "${HEONGPU_DIR}/include")
target_include_directories(generated_fhe PRIVATE ${HEONGPU_INCLUDES})
target_link_libraries(generated_fhe PRIVATE ${HEONGPU_LIBS})
if(rmm_FOUND)
    target_link_libraries(generated_fhe PRIVATE rmm::rmm)
else()
    target_link_libraries(generated_fhe PRIVATE "${HEONGPU_DIR}/lib64/librmm.so" "${HEONGPU_DIR}/lib64/librapids_logger.so")
endif()
target_link_libraries(generated_fhe PRIVATE ${COMMON_LIBS})
target_compile_definitions(generated_fhe PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
set_target_properties(generated_fhe PROPERTIES CUDA_ARCHITECTURES "70" CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
'''

with open("CMakeLists.txt", "w") as f:
    f.write(cmake_content)

print("Running cmake...")
env = os.environ.copy()
# Strip conda paths from PATH so CMake uses the cluster's module gcc instead of conda's gcc
env["PATH"] = os.pathsep.join([p for p in env.get("PATH", "").split(os.pathsep) if "conda" not in p.lower()])

# Override Conda's CC and CXX variables to force use of the cluster's module gcc
env["CC"] = "gcc"
env["CXX"] = "g++"

subprocess.run(['cmake', '-S', '.', '-B', 'build_cu', '-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler'], cwd=build_path, universal_newlines=True, env=env)
print("Building...")
subprocess.run(['cmake', '--build', 'build_cu'], cwd=build_path, universal_newlines=True, env=env)

print("\nTo run the compiled CUDA file, execute:")
print("  ./build_cu/generated_fhe")
