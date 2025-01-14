cmake_minimum_required(VERSION 3.16)
project(msaccucmp_fuzz)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../ait_backend/llm/dump/
    ${PROJECT_SOURCE_DIR}/../../ait_backend/llm/mindie_torch_dump/
    ${PROJECT_SOURCE_DIR}/../../ait_backend/utils/
    ${PROJECT_SOURCE_DIR}/../../ait_backend/ait_logger/
    ${PROJECT_SOURCE_DIR}/common/
    ${PROJECT_SOURCE_DIR}/common/stub/
    ${PROJECT_SOURCE_DIR}/../../ait_backend/llm/dump/nlohmannJson/include/
    ${PROJECT_SOURCE_DIR}/case/
)

aux_source_directory(${PROJECT_SOURCE_DIR}/case FUZZ_COMPUTE_TEST)

FILE(GLOB_RECURSE GTEST_CXX_SOURCE
    ${PROJECT_SOURCE_DIR}/case/*.cpp
    ${PROJECT_SOURCE_DIR}/common/*.cpp
)

message("source_dir: ${PROJECT_SOURCE_DIR}")
message("files: ${GTEST_CXX_SOURCE}")

add_executable(msaccucmp_fuzz FuzzMain.cpp ${GTEST_CXX_SOURCE})

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include
    ${PROJECT_SOURCE_DIR}/3rdparty/secodefuzz/secodefuzz/Secodefuzz
    ${PROJECT_SOURCE_DIR}/3rdparty/securec/include
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include/gtest
    ${PROJECT_SOURCE_DIR}
)
target_link_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib64
    ${PROJECT_SOURCE_DIR}/3rdparty/securec/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/secodefuzz/secodefuzz/examples/out-bin-x64/
    ${PROJECT_SOURCE_DIR}/../../build/output/ait_backend/dump
    ${PROJECT_SOURCE_DIR}/../../build/output/ait_backend/utils
    $ENV{ASCEND_HOME_PATH}/lib64
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    libgtest.a
    libgtest_main.a
    libgmock.a
    libgmock_main.a
    ${CMAKE_THREAD_LIBS_INIT}
    ${PROJECT_SOURCE_DIR}/../../build/output/ait_backend/dump/libatb_probe_abi1.so
    libsecurec.so
    libstdc++fs.a
)

add_compile_options(
        -g
        -fPIC
        -Wall
        -Winvalid-pch
        -fms-extensions
        -std=c++11
        -O0
        -fno-omit-frame-pointer
        -ftest-coverage
        -fprofile-arcs
        -fdump-rtl-expand
        -Wno-pmf-conversions
        -D_GLIBCXX_USE_CXX11_ABI=0
        # for fsannitize
        -fsanitize=address
        -fsanitize=undefined
        -fsanitize-coverage=trace-cmp
        -fsanitize-coverage=trace-pc
)
 
add_link_options(
        -fsanitize=address
        -fsanitize=undefined
        -fsanitize-coverage=trace-pc,trace-cmp
)

target_link_libraries(msaccucmp_fuzz PRIVATE gtest gtest_main gmock_main
		pthread dl z Secodefuzz stdc++fs)

install(TARGETS msaccucmp_fuzz
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/build)
