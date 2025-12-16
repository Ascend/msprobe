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
#include <string>
#include <cstring>
#include <cerrno>
#include "safety_guard.h"
#include "const.h"
#include "utils.h"

std::string GetRealPath(const std::string &outPath)
{
    std::experimental::filesystem::path realOutPath = std::experimental::filesystem::is_symlink(outPath.c_str()) ? \
std::experimental::filesystem::read_symlink(outPath.c_str()) : std::experimental::filesystem::path(outPath.c_str());
    return std::string(realOutPath.c_str());
}

std::vector<std::string> SplitString(const std::string &ss, const char &tar)
{
    std::vector<std::string> tokens;
    std::stringstream input(ss);
    std::string token;
    while (std::getline(input, token, tar)) {
        tokens.emplace_back(token);
    }

    return tokens;
}

bool Exists(const std::string &path)
{
    struct stat fileStatus;
    int ret = stat(path.c_str(), &fileStatus);

    return ret == 0;
}

bool DirectoryExists(const std::string &path)
{
    struct stat info;
    return (stat(path.c_str(), &info) == 0) && (S_ISDIR(info.st_mode));
}

bool Utils::CheckDirectory(const std::string &directory, bool existOK)
{
    MsConst::SAFETY_RET ret = SafetyGuard::CreateDir(directory, MsConst::NORMAL_DIR_MODE_DEFAULT, existOK);
    if (ret != MsConst::SAFETY_RET::SAFE_ERR_NONE) {
        return false;
    }
    return true;
}

bool Utils::ValidateCsvString(const std::string& str)
{
    if (str.empty()) {
        return true;  // 字符串为空
    }

    char firstChar = str[0];
    if (firstChar == '-') {
        std::regex pattern("[0-9,-;]+");
        if (!std::regex_match(str, pattern)) {
            return false;
        }
    }

    return !(firstChar == '+' || firstChar == '=' || firstChar == '@' || firstChar == '%');
}

std::string Utils::GetLastErrorStr()
{
    const int savedErrno = errno;
    // 使用线程局部存储（thread_local）确保线程安全
    thread_local char buffer[1024] = {};  // 缓冲区建议 >= 256 字节

#if (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && !_GNU_SOURCE
    const int ret = ::strerror_r(savedErrno, buffer, sizeof(buffer));
    if (ret == 0) {
        return std::string(buffer);
    } else {
        return "strerror_r failed with code " + std::to_string(ret) +
            " (original errno=" + std::to_string(savedErrno) + ").";
    }
#else
    const char* const msg = ::strerror_r(savedErrno, buffer, sizeof(buffer));
    return std::string(msg);
#endif
}