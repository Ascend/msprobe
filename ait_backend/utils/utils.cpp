/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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

bool CheckDirectory(const std::string &directory)
{
    std::vector<std::string> dirs = SplitString(directory, '/');
    std::string curDir = "";
    for (auto &dir : dirs) {
        curDir += dir + "/";
        curDir = GetRealPath(curDir);
        if (!DirectoryExists(curDir)) {
            int status = mkdir(curDir.c_str(), 0750);
            if (status) {
                AIT_LOG_ERROR("cannot create directory: " + curDir);
                AIT_LOG_ERROR("mkdir: " + std::string(std::strerror(errno)));
            }
        }
    }
    if (!DirectoryExists(directory)) {
        AIT_LOG_ERROR("cannot create directory: " + directory);
        AIT_LOG_ERROR("mkdir: " + std::string(std::strerror(errno)));
        return false;
    }
    return true;
}
