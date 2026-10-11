
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/matrix_mul/he)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/matrix_mul/heongpu)
add_executable(matrix_mul matrix_mul.cpp)
target_link_libraries(matrix_mul PRIVATE fheco)

# Define the source file and destination file 

set(SOURCE_FILE_2 ${CMAKE_SOURCE_DIR}/benchmarks/utils.cpp)
set(DESTINATION_FILE_2 ${CMAKE_SOURCE_DIR}/build/benchmarks/matrix_mul/he/utils.cpp)

set(SOURCE_FILE_3 ${CMAKE_SOURCE_DIR}/benchmarks/utils.hpp) 
set(DESTINATION_FILE_3 ${CMAKE_SOURCE_DIR}/build/benchmarks/matrix_mul/he/utils.hpp)

set(SOURCE_FILE_4 ${CMAKE_SOURCE_DIR}/benchmarks/matrix_mul/he/CMakeLists.txt)
set(DESTINATION_FILE_4 ${CMAKE_SOURCE_DIR}/build/benchmarks/matrix_mul/he/CMakeLists.txt)

set(SOURCE_FILE_5 ${CMAKE_SOURCE_DIR}/benchmarks/matrix_mul/fhe_io_example.txt)
set(DESTINATION_FILE_5 ${CMAKE_SOURCE_DIR}/build/benchmarks/matrix_mul/fhe_io_example.txt)

set(SOURCE_FILE_6 ${CMAKE_SOURCE_DIR}/benchmarks/matrix_mul/generate_matrix_mul.py)
set(DESTINATION_FILE_6 ${CMAKE_SOURCE_DIR}/build/benchmarks/matrix_mul/generate_matrix_mul.py)

add_custom_target(copy_my_file_6matrix_mul ALL 
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_6} ${DESTINATION_FILE_6}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_6} to ${DESTINATION_FILE_6}"
)

add_custom_target(copy_my_file_2matrix_mul ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_2} ${DESTINATION_FILE_2}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_2} to ${DESTINATION_FILE_2}"
)
add_custom_target(copy_my_file_3matrix_mul ALL 
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_3} ${DESTINATION_FILE_3}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_3} to ${DESTINATION_FILE_3}"
)

add_custom_target(copy_my_file_5matrix_mul ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_5} ${DESTINATION_FILE_5}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_5} to ${DESTINATION_FILE_5}"
)

set(SOURCE_FILE_4 ${CMAKE_SOURCE_DIR}/benchmarks/matrix_mul/he/CMakeLists.txt)
set(DESTINATION_FILE_4 ${CMAKE_SOURCE_DIR}/build/benchmarks/matrix_mul/he/CMakeLists.txt)
add_custom_target(copy_my_file_4matrix_mul ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_4} ${DESTINATION_FILE_4}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_4} to ${DESTINATION_FILE_4}"
)

set(SOURCE_FILE_HEONGPU_CMAKE_matrix_mul ${CMAKE_SOURCE_DIR}/benchmarks/matrix_mul/heongpu/CMakeLists.txt)
set(DESTINATION_FILE_HEONGPU_CMAKE_matrix_mul ${CMAKE_SOURCE_DIR}/build/benchmarks/matrix_mul/heongpu/CMakeLists.txt)
add_custom_target(copy_heongpu_cmake_matrix_mul ALL 
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_HEONGPU_CMAKE_matrix_mul} ${DESTINATION_FILE_HEONGPU_CMAKE_matrix_mul}
    DEPENDS ${SOURCE_FILE_HEONGPU_CMAKE_matrix_mul}
    COMMENT "Copying ${SOURCE_FILE_HEONGPU_CMAKE_matrix_mul} to ${DESTINATION_FILE_HEONGPU_CMAKE_matrix_mul}"
)
