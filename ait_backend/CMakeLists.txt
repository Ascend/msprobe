cmake_minimum_required(VERSION 3.16.5)
project(ait_backend)

if (DEFINED HITEST AND HITEST STREQUAL "on")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE hitestwrapper)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK hitestwrapper)
endif()

add_subdirectory(llm)
