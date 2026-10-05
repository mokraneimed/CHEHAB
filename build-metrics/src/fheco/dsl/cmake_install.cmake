# Install script for directory: C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/FHECo")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/c_files/mingw64/bin/objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/FHECO/fheco/dsl" TYPE FILE FILES
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/ciphertext.hpp"
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/common.hpp"
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/compiler_helper_functions.hpp"
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/compiler.hpp"
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/ops_overloads.hpp"
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/plaintext.hpp"
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/compiler_simplification.hpp"
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/tensor.hpp"
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/plaintext_matrix.hpp"
    "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/src/fheco/dsl/matmul.hpp"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/Nazne/PFE/CHEHAB/Merged_Framework_2/CHEHAB/build-metrics/src/fheco/dsl/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
