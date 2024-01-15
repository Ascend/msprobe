cmake_minimum_required(VERSION 3.16.5)
project(dump)

# 设置编译选项，可以根据需要进行修改
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (DEFINED ENV{AIT_LLM_ABI})
    if ($ENV{AIT_LLM_ABI} STREQUAL "0")
        set(LIB_NAME atb_probe_abi0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        message("ABI is set to 0. Compiling with cxx11_abi=0.")
    else()
        set(LIB_NAME atb_probe_abi1)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
        message("ABI is not 0. Compiling with cxx11_abi=1.")
    endif()
else()
    set(LIB_NAME atb_probe_abi0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    message("ABI is not set. Compiling with cxx11_abi=1.")
endif()

add_library(${LIB_NAME} SHARED atb_probe.cpp binfile.cpp)

# 设置头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nlohmannJson/include)

#target_include_directories(dump PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 设置共享库输出目录
#set_target_properties(dump PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

set(ATB_PROBE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/output/ait_backend/dump/)
install(
    TARGETS ${LIB_NAME}
    DESTINATION ${ATB_PROBE_LIB_DIR}
)
