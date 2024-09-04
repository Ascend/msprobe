PROJECT(mindiedump)
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

add_compile_options(-fPIE -fstack-protector-all -fPIC -Wall -Wfatal-errors -O2)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

SET(ASCEND_HOME     /usr/local/Ascend     CACHE STRING "")
SET(ASCEND_TOOLKIT_HOME $ENV{ASCEND_HOME_PATH}/ascend-toolkit/latest)
SET(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

add_library(mindiedump SHARED ${ROOT_DIR}/dump_utils.cpp)
set_property(TARGET mindiedump PROPERTY CXX_STANDARD 11)

set(MINDIE_TORCH_LIB_DIR ${ROOT_DIR}/../../../build/output/ait_backend/mindie_torch_dump/)
message(${MINDIE_TORCH_LIB_DIR})
install(
    TARGETS mindiedump
    DESTINATION ${MINDIE_TORCH_LIB_DIR}
)