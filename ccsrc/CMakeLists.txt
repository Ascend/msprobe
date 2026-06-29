cmake_minimum_required(VERSION 3.19.3)
project(ccsrc)

if ($ENV{MSPROBE_INCLUDE_MOD} STREQUAL "atb_probe")
    add_subdirectory(atb_probe)
endif()

if ($ENV{MSPROBE_INCLUDE_MOD} STREQUAL "aclgraph_dump")
    add_subdirectory(aclgraph_dump)
endif()

if ($ENV{MSPROBE_INCLUDE_MOD} STREQUAL "nan_check")
    add_subdirectory(nan_check)
endif()
