cmake_minimum_required(VERSION 3.16.5)
project(dump)

# 设置编译选项，可以根据需要进行修改
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH TRUE)
set(AIT_LOGGER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../ait_logger)
set(AIT_UTILS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../utils)

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

# 设置头文件目录
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
include_directories(${AIT_LOGGER_PATH})
include_directories(${AIT_UTILS_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nlohmannJson/include)
include_directories(${TOP_DIR}/platform/securec/include)  # 编译环境中 git clone 的路径
include_directories($ENV{ASCEND_HOME_PATH}/runtime/include/)  # 保留 CANN 包中的路径，避免本地编译不通过

set(SECUREC_DIR ${TOP_DIR}/platform/securec)
file(GLOB_RECURSE SECUREC_SRC ${SECUREC_DIR}/src/*.c)
add_library(${LIB_NAME} SHARED
    atb_probe.cpp bin_file.cpp ${AIT_LOGGER_PATH}/ait_logger.cpp
    ${AIT_UTILS_PATH}/utils.cpp
    ${AIT_UTILS_PATH}/safety_guard.cpp
    ${AIT_UTILS_PATH}/file.cpp
    ${SECUREC_SRC}
)

target_compile_options(${LIB_NAME} PRIVATE
    -fPIC
    -pie
    -fstack-protector-all
    -O2
    -D_FORTIFY_SOURCE=2
    -ftrapv
)
add_compile_options(-fstack-check)
target_link_options(${LIB_NAME} PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -s
)
target_link_libraries(${LIB_NAME} PRIVATE
    libstdc++fs.a
    -lpthread
)

set(ATB_PROBE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/output/ait_backend/dump/)
install(
    TARGETS ${LIB_NAME}
    DESTINATION ${ATB_PROBE_LIB_DIR}
)
