cmake_minimum_required(VERSION 3.19.3)
project(atb_probe)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH TRUE)
set(AIT_LOGGER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ait_logger)
set(UTILS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/utils)

find_package(nlohmannjson MODULE REQUIRED)
find_package(securec MODULE REQUIRED)

if (DEFINED ENV{ATB_PROBE_ABI})
    if ($ENV{ATB_PROBE_ABI} STREQUAL "0")
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
    message("ABI is not set. Compiling with cxx11_abi=0.")
endif()

# 设置头文件目录
include_directories(${AIT_LOGGER_PATH})
include_directories(${UTILS_PATH})

file(GLOB_RECURSE LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE SECUREC_SRC ${SECUREC_DIR}/src/*.c)
add_library(${LIB_NAME} SHARED ${LIB_SRC} ${SECUREC_SRC})

target_compile_options(${LIB_NAME} PRIVATE
    -fPIC
    -pie
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2
    -ftrapv
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${LIB_NAME} PRIVATE -O0 -g)
else()
    target_compile_options(${LIB_NAME} PRIVATE -O2)
endif()
add_compile_options(-fstack-check)
target_link_options(${LIB_NAME} PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options(${LIB_NAME} PRIVATE -s)
endif()
target_link_libraries(${LIB_NAME} PRIVATE
    libstdc++fs.a
    -lpthread
)

set(ATB_PROBE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../python/msprobe/lib/)
install(TARGETS ${LIB_NAME} LIBRARY DESTINATION ${ATB_PROBE_LIB_DIR})
