/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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