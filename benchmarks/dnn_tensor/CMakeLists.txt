file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/he)
add_executable(dnn_tensor_benchmarks_runner dnn_tensor_benchmarks.cpp)
target_link_libraries(dnn_tensor_benchmarks_runner PRIVATE fheco)

set(SOURCE_FILE_HEONGPU_CMAKE_dnn_tensor ${CMAKE_SOURCE_DIR}/benchmarks/dnn_tensor/heongpu/CMakeLists.txt)
set(DESTINATION_FILE_HEONGPU_CMAKE_dnn_tensor ${CMAKE_SOURCE_DIR}/build/benchmarks/dnn_tensor/heongpu/CMakeLists.txt)
add_custom_target(copy_heongpu_cmake_dnn_tensor ALL 
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_HEONGPU_CMAKE_dnn_tensor} ${DESTINATION_FILE_HEONGPU_CMAKE_dnn_tensor}
    DEPENDS ${SOURCE_FILE_HEONGPU_CMAKE_dnn_tensor}
    COMMENT "Copying ${SOURCE_FILE_HEONGPU_CMAKE_dnn_tensor} to ${DESTINATION_FILE_HEONGPU_CMAKE_dnn_tensor}"
)
