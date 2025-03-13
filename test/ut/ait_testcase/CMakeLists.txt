cmake_minimum_required (VERSION 3.16.5)

project(ait_backend_ut)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -lgcov -g -std=c++14 -Wall -fno-access-control")

find_package(Threads REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/llm/dump/
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/llm/mindie_torch_dump/
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/utils/
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/ait_logger/
    ${PROJECT_SOURCE_DIR}/common/
    ${PROJECT_SOURCE_DIR}/common/stub/
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/llm/dump/nlohmannJson/include/
)

FILE(GLOB_RECURSE GTEST_CXX_SOURCE
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/llm/dump/atb_probe.cpp
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/llm/dump/bin_file.cpp
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/llm/mindie_torch_dump/dump_utils.cpp
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/utils/*.cpp
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/ait_logger/*.cpp
    ${PROJECT_SOURCE_DIR}/cases/test*.cpp
    ${PROJECT_SOURCE_DIR}/common/*.cpp
)

add_executable(${PROJECT_NAME} ${GTEST_CXX_SOURCE})

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include
    ${PROJECT_SOURCE_DIR}/3rdparty/securec/include
)
target_link_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib64
    ${PROJECT_SOURCE_DIR}/3rdparty/securec/lib
    ${PROJECT_SOURCE_DIR}/../../../build/output/ait_backend/dump
    ${PROJECT_SOURCE_DIR}/../../../build/output/ait_backend/utils/
    $ENV{ASCEND_HOME_PATH}/lib64
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    libgtest.a
    libgtest_main.a
    libgmock.a
    libgmock_main.a
    ${CMAKE_THREAD_LIBS_INIT}
    libsecurec.so
    libstdc++fs.a
    dl
)

add_test(Test ${PROJECT_NAME})
enable_testing()

