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
#ifndef SAFETY_GUARD_H
#define SAFETY_GUARD_H

#include <sys/stat.h>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include "const.h"
using MsConst::SAFETY_RET;
using MsConst::OPERATE_MODE;
using MsConst::SUFFIX;


class SafetyGuard {
public:
// function for checking
    // read write exec
    static SAFETY_RET CheckFileLegality(
        const std::string originPath,
        OPERATE_MODE operateMode = OPERATE_MODE::READ,
        const size_t maxSize = MsConst::MAX_FILE_SIZE_DEFAULT,
        SUFFIX fileSuffix = SUFFIX::NONE
    );

    static SAFETY_RET CheckNormalStr(
        const std::string str,
        const char* whiteList = MsConst::NORMAL_STRING_VALID_PATTERN,
        const size_t maxLen = MsConst::DEFAULT_STRING_MAX_LEN
    );

public:
// function for doing something
    static SAFETY_RET CreateDir(
        std::string originPath,
        mode_t mode = MsConst::NORMAL_DIR_MODE_DEFAULT,
        bool existOK = false
    );
};

#endif // SAFETY_GUARD_H