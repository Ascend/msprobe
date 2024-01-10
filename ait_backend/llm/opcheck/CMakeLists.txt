cmake_minimum_required(VERSION 3.16.5)
project(ait_llm_opcheck)
set(LIB_NAME opcheck_test_framework)

IF (CMAKE_BUILD_TYPE MATCHES Debug)
    set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)
    
ELSEIF (CMAKE_BUILD_TYPE MATCHES Release)
    set(DEFAULT_COMPILE -std=c++14 -O3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)
ENDIF()

FILE(GLOB_RECURSE  CXX_SOURCE ./*.cpp)
# add_library(${LIB_NAME} SHARED ${CXX_SOURCE})
add_executable(${LIB_NAME} ${CXX_SOURCE})

install(
    TARGETS ${LIB_NAME} RUNTIME
    DESTINATION ${PROJECT_SOURCE_DIR}/../../../build/output/ait_backend
)

#install(
#    TARGETS ${LIB_NAME}
#    DESTINATION ${CMAKE_INSTALL_PREFIX}/${APP_NAME}/lib
#)