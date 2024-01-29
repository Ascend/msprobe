cmake_minimum_required (VERSION 3.16.5)

project(ait_backend_ut)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftest-coverage -g -std=c++14 -Wall -fno-access-control")

find_package(Threads REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../ait_backend/llm/dump/
    ${PROJECT_SOURCE_DIR}/common/
    ${PROJECT_SOURCE_DIR}/common/stub/
)

FILE(GLOB_RECURSE GTEST_CXX_SOURCE
    ${PROJECT_SOURCE_DIR}/cases/test*.cpp
    ${PROJECT_SOURCE_DIR}/common/*.cpp
)

add_executable(${PROJECT_NAME} ${GTEST_CXX_SOURCE})

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
target_link_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib64
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    libgtest.a
    libgtest_main.a
    libgmock.a
    libgmock_main.a
    ${CMAKE_THREAD_LIBS_INIT}
    ${PROJECT_SOURCE_DIR}/../../../build/output/ait_backend/dump/libatb_probe_abi1.so
)

add_test(Test ${PROJECT_NAME})
enable_testing()