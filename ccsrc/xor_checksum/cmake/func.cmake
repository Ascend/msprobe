# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

function(recursive_add_subdirectory)
    file(GLOB OP_CMAKE_FILES
        CONFIGURE_DEPENDS
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/*/${NPU_ARCH}/CMakeLists.txt"
    )
    list(SORT OP_CMAKE_FILES)

    if(OP_CMAKE_FILES STREQUAL "")
        message(WARNING "No operator CMakeLists.txt found for NPU_ARCH=${NPU_ARCH} under ${CMAKE_CURRENT_SOURCE_DIR}")
        return()
    endif()

    foreach(OP_CMAKE_FILE IN LISTS OP_CMAKE_FILES)
        get_filename_component(OP_ARCH_DIR "${OP_CMAKE_FILE}" DIRECTORY)
        add_subdirectory("${OP_ARCH_DIR}")
    endforeach()
endfunction()

function(msprobe_add_current_op OUT_TARGET)
    get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
    get_filename_component(OP_NAME "${PARENT_DIR}" NAME)

    file(GLOB_RECURSE SOURCE_FILES
        CONFIGURE_DEPENDS
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.asc"
    )
    list(SORT SOURCE_FILES)
    if(SOURCE_FILES STREQUAL "")
        message(FATAL_ERROR "No .asc source files found in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    set(TARGET_NAME "${OP_NAME}_${NPU_ARCH}_obj")
    string(REPLACE "-" "_" TARGET_NAME "${TARGET_NAME}")

    add_library(${TARGET_NAME} OBJECT ${SOURCE_FILES})
    target_compile_options(${TARGET_NAME} PRIVATE
        ${COMPILE_OPTIONS}
        "--npu-arch=${NPU_ARCH}"
    )
    target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        ${INCLUDE_DIRECTORIES}
    )

    set_property(GLOBAL APPEND PROPERTY ASCEND_OPS_OPERATOR_TARGETS ${TARGET_NAME})
    set(${OUT_TARGET} ${TARGET_NAME} PARENT_SCOPE)
endfunction()
