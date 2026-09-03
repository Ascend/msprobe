cmake_minimum_required(VERSION 3.19.3)
project(MindStudio-Probe)

if(NOT CMAKE_BUILD_TYPE)
    if(DEFINED BUILD_TYPE AND "${BUILD_TYPE}" STREQUAL "debug")
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    else()
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE machine_arch
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (DEFINED ARCH_TYPE AND NOT "${ARCH_TYPE}" STREQUAL "${machine_arch}")
    message(FATAL_ERROR "Cross-compilation is not supported currently. (compile ${ARCH_TYPE} on ${machine_arch})")
endif()


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(ENV{PROJECT_ROOT_PATH} "${CMAKE_SOURCE_DIR}")
include(utils)
add_subdirectory(ccsrc)
