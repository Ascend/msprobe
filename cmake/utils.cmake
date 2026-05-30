
function(download_opensource_pkg pkg_name)
    message("start to download ${pkg_name}...")
    set(options)
    set(oneValueArgs SHA256 GIT_TAG DOWNLOAD_PATH)
    set(multiValueArgs PATCHES)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT PKG_DOWNLOAD_PATH)
        set(PKG_DOWNLOAD_PATH "${CMAKE_SOURCE_DIR}/../third_party")
    endif()
    file(MAKE_DIRECTORY ${PKG_DOWNLOAD_PATH})

    execute_process(
        WORKING_DIRECTORY $ENV{PROJECT_ROOT_PATH}/cmake
        COMMAND bash download_opensource.sh ${pkg_name} ${PKG_DOWNLOAD_PATH} ${PKG_SHA256} ${PKG_GIT_TAG}
        RESULT_VARIABLE RESULT
    )
    if (NOT RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to download ${pkg_name}(${RESULT}).")
    endif()
endfunction()

function(compile_protobuf_file output_path)
    if (NOT PROTOC_EXECUTABLE)
        message(FATAL_ERROR "You shall install protobuf first.")
    endif()
    file(MAKE_DIRECTORY ${output_path})
    foreach(file ${ARGN})
        get_filename_component(abs_file_path ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file_path} PATH)
        file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${file_dir})
        execute_process(
            COMMAND ${PROTOC_EXECUTABLE} -I${file_dir} --cpp_out=${output_path} ${abs_file_path}
        )
        message("Compile protobuf file ${file}")
    endforeach()
endfunction()
