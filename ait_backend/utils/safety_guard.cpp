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


#include <regex>
#include "ait_logger.h"
#include "file.h"
#include "safety_guard.h"
using MsConst::SAFETY_RET;
using MsConst::OPERATE_MODE;
using MsConst::SUFFIX;

namespace Utils {
SAFETY_RET SafetyGuard::CheckFileLegality(
    const std::string originPath,
    OPERATE_MODE operateMode,
    const size_t maxSize,
    SUFFIX fileSuffix
)
{
    if (operateMode == OPERATE_MODE::READ) {
        if (!File::CheckFileBeforeRead(originPath, fileSuffix, maxSize)) {
            return SAFETY_RET::SAFE_ERR_FILE_TO_READ_ILLEGAL;
        }
    } else if (operateMode == OPERATE_MODE::WRITE) {
        if (!File::CheckFileBeforeCreateOrWrite(originPath, false)) {
            return SAFETY_RET::SAFE_ERR_FILE_TO_WRITE_ILLEGAL;
        }
    }
    return SAFETY_RET::SAFE_ERR_NONE;
}

SAFETY_RET SafetyGuard::CheckNormalStr(
    const std::string str,
    const char* whiteList,
    const size_t maxLen
)
{
    if (str.size() > maxLen) {
        return SAFETY_RET::SAFE_ERR_STR_OVER_MAX_LEN;
    }
    if (!std::regex_match(str, std::regex(whiteList))) {
        return SAFETY_RET::SAFE_ERR_STR_CONTAIN_ILLEGAL_CHAR;
    }
    return SAFETY_RET::SAFE_ERR_NONE;
}

SAFETY_RET SafetyGuard::CreateDir(
    std::string originPath,
    mode_t mode,
    bool existOK
)
{
    if (existOK && File::IsPathExist(originPath)) {
        if (!File::CheckDir(originPath)) {
            return SAFETY_RET::SAFE_ERR_EXIST_DIR_ILLEGAL;
        }
        return SAFETY_RET::SAFE_ERR_NONE;
    }
    if (!existOK && File::IsPathExist(originPath)) {
        return SAFETY_RET::SAFE_ERR_PATH_IS_EXIST;
    }
    if (!File::CreateDir(originPath, true, mode)) {
        return SAFETY_RET::SAFE_ERR_CREATE_DIR_FAILED;
    }
    return SAFETY_RET::SAFE_ERR_NONE;
}
}