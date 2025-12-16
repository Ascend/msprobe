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


#ifndef SAFETY_GUARD_H
#define SAFETY_GUARD_H

#include <sys/stat.h>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include "const.h"

namespace Utils {
class SafetyGuard {
public:
// function for checking
    // read write exec
    static MsConst::SAFETY_RET CheckFileLegality(
        const std::string originPath,
        MsConst::OPERATE_MODE operateMode = MsConst::OPERATE_MODE::READ,
        const size_t maxSize = MsConst::MAX_FILE_SIZE_DEFAULT,
        MsConst::SUFFIX fileSuffix = MsConst::SUFFIX::NONE
    );

    static MsConst::SAFETY_RET CheckNormalStr(
        const std::string str,
        const char* whiteList = MsConst::NORMAL_STRING_VALID_PATTERN,
        const size_t maxLen = MsConst::DEFAULT_STRING_MAX_LEN
    );

public:
// function for doing something
    static MsConst::SAFETY_RET CreateDir(
        std::string originPath,
        mode_t mode = MsConst::NORMAL_DIR_MODE_DEFAULT,
        bool existOK = false
    );
};
}
#endif // SAFETY_GUARD_H