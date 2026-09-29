add_executable(deep_network deep_network.cpp)
target_link_libraries(deep_network fheco)

add_executable(test_orion_comparison test_orion_comparison.cpp)
target_link_libraries(test_orion_comparison fheco)

add_executable(benchmark_detailed benchmark_detailed.cpp)
target_link_libraries(benchmark_detailed fheco)

set(SOURCE_FILE_COMPILE_CUDA_deep_network ${CMAKE_SOURCE_DIR}/benchmarks/deep_network/compile_cuda.py)
set(DESTINATION_FILE_COMPILE_CUDA_deep_network ${CMAKE_SOURCE_DIR}/build/benchmarks/deep_network/compile_cuda.py)
add_custom_target(copy_compile_cuda_deep_network ALL 
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_COMPILE_CUDA_deep_network} ${DESTINATION_FILE_COMPILE_CUDA_deep_network}
    DEPENDS ${SOURCE_FILE_COMPILE_CUDA_deep_network}
    COMMENT "Copying ${SOURCE_FILE_COMPILE_CUDA_deep_network} to ${DESTINATION_FILE_COMPILE_CUDA_deep_network}"
)

