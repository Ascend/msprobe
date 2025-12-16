/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */


#ifndef CONST_H
#define CONST_H

#include <string>
#include <vector>
#include <map>
#include <regex>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

namespace MsConst {
    constexpr size_t MAX_BIN_SIZE = 10ULL * 1024 * 1024 * 1024;
    constexpr size_t MAX_JSON_SIZE = 1024ULL * 1024 * 1024;
    constexpr size_t MAX_CSV_SIZE = 1024ULL * 1024 * 1024;
    constexpr size_t MAX_FILE_SIZE_DEFAULT = 10ULL * 1024 * 1024 * 1024;

    constexpr int DIR_CHECK_MODE = R_OK | W_OK | X_OK;
    constexpr const char PATH_SEPARATOR = '/';
    constexpr const char* FILE_VALID_PATTERN = "^[a-zA-Z0-9_./-]+$";
    constexpr const char* NORMAL_STRING_VALID_PATTERN = "^[_A-Za-z0-9\"'><=\\[\\])(,}{: /.~-]+$";

    constexpr const uint32_t FULL_PATH_LENGTH_MAX = 4096;
    constexpr const uint32_t FILE_NAME_LENGTH_MAX = 255;
    constexpr const size_t PATH_DEPTH_MAX = 32;

    constexpr mode_t NORMAL_FILE_MODE_DEFAULT = 0640;
    constexpr mode_t READONLY_FILE_MODE_DEFAULT = 0440;
    constexpr mode_t SCRIPT_FILE_MODE_DEFAULT = 0550;
    constexpr mode_t NORMAL_DIR_MODE_DEFAULT = 0750;
    constexpr mode_t MAX_PERMISSION = 0777;
    constexpr mode_t READ_FILE_NOT_PERMITTED = S_IWGRP | S_IWOTH;
    constexpr mode_t WRITE_FILE_NOT_PERMITTED = S_IWGRP | S_IWOTH | S_IROTH | S_IXOTH;
    constexpr mode_t CREATE_FILE_MODE_DEFAULT = O_EXCL | O_CREAT;

    const size_t DEFAULT_STRING_MAX_LEN = 4096;

    enum class SAFETY_RET {
        SAFE_ERR_NONE,
        SAFE_ERR_PATH_IS_EXIST,
        SAFE_ERR_FILE_TO_READ_ILLEGAL,
        SAFE_ERR_FILE_TO_WRITE_ILLEGAL,
        SAFE_ERR_EXIST_DIR_ILLEGAL,
        SAFE_ERR_CREATE_DIR_FAILED,
        SAFE_ERR_STR_OVER_MAX_LEN,
        SAFE_ERR_STR_CONTAIN_ILLEGAL_CHAR
    };

    enum class OPERATE_MODE {
        READ,
        WRITE,
    };

    enum class SUFFIX {
        NONE,
        BIN,
        JSON,
        CSV,
    };

    const std::map<SUFFIX, std::pair<std::string, size_t>> SUFFIX_TYPE_TABLE = {
        {SUFFIX::BIN, {"bin", MAX_BIN_SIZE}},
        {SUFFIX::JSON, {"json", MAX_JSON_SIZE}},
        {SUFFIX::CSV, {"csv", MAX_CSV_SIZE}},
    };

}

#endif // CONST_H