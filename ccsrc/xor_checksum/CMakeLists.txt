cmake_minimum_required(VERSION 3.19.3)

set(PKG_NAME xor_checksum_ext)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend.cmake)
find_package(ASC REQUIRED)
project(${PKG_NAME} VERSION 1.0.0 LANGUAGES C CXX ASC)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (TORCH_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX_PATH}")
endif()

find_package(Torch REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    RESULT_VARIABLE TORCH_NPU_RESULT
    OUTPUT_VARIABLE TORCH_NPU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT TORCH_NPU_RESULT EQUAL 0)
    message(FATAL_ERROR "torch_npu is required to build xor_checksum_ext.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH TRUE)

set(SUPPORTED_NPU_ARCHS dav-2201 dav-3510)
if(DEFINED ENV{NPU_ARCH} AND NOT "$ENV{NPU_ARCH}" STREQUAL "")
    set(NPU_ARCH "$ENV{NPU_ARCH}" CACHE STRING "NPU architecture" FORCE)
elseif(NOT DEFINED NPU_ARCH OR "${NPU_ARCH}" STREQUAL "")
    set(NPU_ARCH "dav-2201" CACHE STRING "NPU architecture" FORCE)
endif()

if(NOT "${NPU_ARCH}" IN_LIST SUPPORTED_NPU_ARCHS)
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}. Supported values: ${SUPPORTED_NPU_ARCHS}")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/func.cmake)

set(TORCH_NPU_INCLUDE_PATH "${TORCH_NPU_ROOT}/include")
set(TORCH_NPU_LIB_PATH "${TORCH_NPU_ROOT}/lib")

set(INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
)

set(LINK_DIRECTORIES
    ${TORCH_NPU_LIB_PATH}
)

set(LINK_LIBRARIES
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    torch_npu
)

set(COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O3
    -fdiagnostics-color=always
    -w
    -DEXTENSION_MODULE_NAME=msprobe
)

add_subdirectory(csrc)
get_property(OPERATOR_TARGETS GLOBAL PROPERTY ASCEND_OPS_OPERATOR_TARGETS)
if(NOT OPERATOR_TARGETS)
    message(FATAL_ERROR "No operator targets found for NPU_ARCH=${NPU_ARCH}.")
endif()

set(OPERATOR_OBJECTS)
foreach(OPERATOR_TARGET IN LISTS OPERATOR_TARGETS)
    list(APPEND OPERATOR_OBJECTS $<TARGET_OBJECTS:${OPERATOR_TARGET}>)
endforeach()

add_library(xor_checksum_ext SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp
    ${OPERATOR_OBJECTS}
)
set_target_properties(xor_checksum_ext PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    OUTPUT_NAME "xor_checksum_ext"
)
target_compile_options(xor_checksum_ext PRIVATE ${COMPILE_OPTIONS})
target_include_directories(xor_checksum_ext PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(xor_checksum_ext PRIVATE ${LINK_DIRECTORIES})
target_link_libraries(xor_checksum_ext PRIVATE ${LINK_LIBRARIES})
if(OPERATOR_TARGETS)
    target_link_libraries(xor_checksum_ext PRIVATE ${OPERATOR_TARGETS})
endif()
