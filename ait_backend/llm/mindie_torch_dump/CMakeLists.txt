CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(mindiedump)

# 设置安全编译选项
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH TRUE)
set(AIT_LOGGER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../ait_logger)

add_compile_options(-fPIE -fstack-protector-all -ftrapv -fPIC -Wall -Wfatal-errors -O2)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

SET(ASCEND_HOME     /usr/local/Ascend     CACHE STRING "")
SET(ASCEND_TOOLKIT_HOME $ENV{ASCEND_HOME_PATH}/ascend-toolkit/latest)
SET(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# 设置头文件目录
include_directories(${AIT_LOGGER_PATH})

add_library(mindiedump SHARED ${ROOT_DIR}/dump_utils.cpp ${AIT_LOGGER_PATH}/ait_logger.cpp)
set_property(TARGET mindiedump PROPERTY CXX_STANDARD 11)

target_link_options(mindiedump PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -s
)

set(MINDIE_TORCH_LIB_DIR ${ROOT_DIR}/../../../build/output/ait_backend/mindie_torch_dump/)
message(${MINDIE_TORCH_LIB_DIR})
install(
    TARGETS mindiedump
    DESTINATION ${MINDIE_TORCH_LIB_DIR}
)