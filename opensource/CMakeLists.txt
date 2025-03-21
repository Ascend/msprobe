project(protobuf)

add_subdirectory(protobuf/cmake)

target_compile_options(libprotobuf PRIVATE
    -std=c++11
    -fPIC
    -fstack-protector-all
    -fno-common
    -fno-strict-aliasing
    -Wfloat-equal
    -Wextra
    )
