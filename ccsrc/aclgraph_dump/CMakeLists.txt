cmake_minimum_required(VERSION 3.14)
project(aclgraph_dump_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH TRUE)

find_package(cpython MODULE REQUIRED)

if (NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python3_EXECUTABLE not found.")
endif()

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
    message(FATAL_ERROR "torch_npu is required to build aclgraph_dump_ext.")
endif()

if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME_PATH $ENV{ASCEND_TOOLKIT_HOME})
else()
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(ACL_INC "${ASCEND_HOME_PATH}/include")
set(ACL_LIB "${ASCEND_HOME_PATH}/lib64")
set(TORCH_NPU_INCLUDE "${TORCH_NPU_ROOT}/include")
set(TORCH_NPU_LIB_DIR "${TORCH_NPU_ROOT}/lib")

add_library(aclgraph_dump_ext SHARED)

set_target_properties(aclgraph_dump_ext PROPERTIES PREFIX "" OUTPUT_NAME "aclgraph_dump_ext")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(aclgraph_dump_ext PRIVATE -O0 -g)
else()
    target_compile_options(aclgraph_dump_ext PRIVATE -O3)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_compile_definitions(aclgraph_dump_ext PRIVATE TORCH_EXTENSION_NAME=aclgraph_dump_ext)

target_include_directories(aclgraph_dump_ext PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${ACL_INC}
    ${TORCH_NPU_INCLUDE}
)

target_link_directories(aclgraph_dump_ext PRIVATE
    ${ACL_LIB}
    ${TORCH_NPU_LIB_DIR}
)

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
target_sources(aclgraph_dump_ext PRIVATE ${SOURCES})

target_link_libraries(aclgraph_dump_ext PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
)

target_link_options(aclgraph_dump_ext PRIVATE
    "-Wl,-rpath,${ACL_LIB}"
    "-Wl,-rpath,${TORCH_NPU_LIB_DIR}"
)

set(ACLGRAPH_DUMP_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../python/msprobe/lib/)
install(TARGETS aclgraph_dump_ext LIBRARY DESTINATION ${ACLGRAPH_DUMP_LIB_DIR})
