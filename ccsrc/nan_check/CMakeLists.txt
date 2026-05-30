cmake_minimum_required(VERSION 3.14)
project(nan_check_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH TRUE)

if (PYTHON_VERSION)
    find_package(Python3 ${PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter)
else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

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
    message(FATAL_ERROR "torch_npu is required to build nan_check_ext.")
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

add_library(nan_check_ext SHARED)

set_target_properties(nan_check_ext PROPERTIES PREFIX "" OUTPUT_NAME "nan_check_ext")

if(DEFINED BUILD_TYPE AND "${BUILD_TYPE}" STREQUAL "debug")
    target_compile_options(nan_check_ext PRIVATE -O0 -g)
else()
    target_compile_options(nan_check_ext PRIVATE -O3)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_compile_definitions(nan_check_ext PRIVATE TORCH_EXTENSION_NAME=nan_check_ext)
target_compile_definitions(nan_check_ext PRIVATE
    MSPROBE_CUST_OPAPI_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../vendors/customize/op_api/lib/libcust_opapi.so"
)

target_include_directories(nan_check_ext PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${ACL_INC}
    ${TORCH_NPU_INCLUDE}
)

target_link_directories(nan_check_ext PRIVATE
    ${ACL_LIB}
    ${TORCH_NPU_LIB_DIR}
)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nan_check.cpp
)
target_sources(nan_check_ext PRIVATE ${SOURCES})

target_link_libraries(nan_check_ext PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
)

target_link_options(nan_check_ext PRIVATE
    "-Wl,-rpath,${ACL_LIB}"
    "-Wl,-rpath,${TORCH_NPU_LIB_DIR}"
)

set(NAN_CHECK_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../python/msprobe/lib/)
install(TARGETS nan_check_ext LIBRARY DESTINATION ${NAN_CHECK_LIB_DIR})
