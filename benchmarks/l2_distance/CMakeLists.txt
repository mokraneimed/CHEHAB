

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/l2_distance/he)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/l2_distance/heongpu)
add_executable(l2_distance l2_distance.cpp)
target_link_libraries(l2_distance PRIVATE fheco)
 
# Define the source file and destination file

set(SOURCE_FILE_2 ${CMAKE_SOURCE_DIR}/benchmarks/utils.cpp)
set(DESTINATION_FILE_2 ${CMAKE_SOURCE_DIR}/build/benchmarks/l2_distance/he/utils.cpp)

set(SOURCE_FILE_3 ${CMAKE_SOURCE_DIR}/benchmarks/utils.hpp)
set(DESTINATION_FILE_3 ${CMAKE_SOURCE_DIR}/build/benchmarks/l2_distance/he/utils.hpp)

set(SOURCE_FILE_4 ${CMAKE_SOURCE_DIR}/benchmarks/l2_distance/he/CMakeLists.txt)
set(DESTINATION_FILE_4 ${CMAKE_SOURCE_DIR}/build/benchmarks/l2_distance/he/CMakeLists.txt)
 
set(SOURCE_FILE_5 ${CMAKE_SOURCE_DIR}/benchmarks/l2_distance/fhe_io_example.txt)
set(DESTINATION_FILE_5 ${CMAKE_SOURCE_DIR}/build/benchmarks/l2_distance/fhe_io_example.txt)

set(SOURCE_FILE_6 ${CMAKE_SOURCE_DIR}/benchmarks/l2_distance/generate_l2_distance.py)
set(DESTINATION_FILE_6 ${CMAKE_SOURCE_DIR}/build/benchmarks/l2_distance/generate_l2_distance.py)

add_custom_target(copy_my_file_6l2_distance ALL 
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_6} ${DESTINATION_FILE_6}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_6} to ${DESTINATION_FILE_6}"
)
add_custom_target(copy_my_file_2l2_distance ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_2} ${DESTINATION_FILE_2}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_2} to ${DESTINATION_FILE_2}"
)
add_custom_target(copy_my_file_3l2_distance ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_3} ${DESTINATION_FILE_3}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_3} to ${DESTINATION_FILE_3}"
)
add_custom_target(copy_my_file_5l2_distance ALL 
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_5} ${DESTINATION_FILE_5}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_5} to ${DESTINATION_FILE_5}"
)
set(SOURCE_FILE_4 ${CMAKE_SOURCE_DIR}/benchmarks/l2_distance/he/CMakeLists.txt)
set(DESTINATION_FILE_4 ${CMAKE_SOURCE_DIR}/build/benchmarks/l2_distance/he/CMakeLists.txt)
add_custom_target(copy_my_file_4l2_distance ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_4} ${DESTINATION_FILE_4}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying ${SOURCE_FILE_4} to ${DESTINATION_FILE_4}"
)

set(SOURCE_FILE_HEONGPU_CMAKE_l2_distance ${CMAKE_SOURCE_DIR}/benchmarks/l2_distance/heongpu/CMakeLists.txt)
set(DESTINATION_FILE_HEONGPU_CMAKE_l2_distance ${CMAKE_SOURCE_DIR}/build/benchmarks/l2_distance/heongpu/CMakeLists.txt)
add_custom_target(copy_heongpu_cmake_l2_distance ALL 
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE_HEONGPU_CMAKE_l2_distance} ${DESTINATION_FILE_HEONGPU_CMAKE_l2_distance}
    DEPENDS ${SOURCE_FILE_HEONGPU_CMAKE_l2_distance}
    COMMENT "Copying ${SOURCE_FILE_HEONGPU_CMAKE_l2_distance} to ${DESTINATION_FILE_HEONGPU_CMAKE_l2_distance}"
)
