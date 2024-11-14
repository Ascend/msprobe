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


#include <fstream>
#include <iostream>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include "tools.h"

const size_t CMD_BUFFER_LEN = 1024;

int32_t GetCurrentProcessId()
{
    int32_t pid = getpid();
    if (pid == -1) {
        std::cout << "get pid failed " << std::endl;
    }
    return pid;
}

bool IfFileExists(const std::string &outPath)
{
    std::ifstream f(outPath);
    if (f.good()) {
        return true;
    } else {
        return false;
    }
}

void DeleteFile(const std::string &outPath)
{
    if (IfFileExists(outPath)) {
        if (std::remove(outPath.c_str()) == 0) {
            std::cout << "File deleted sucessfully! outPath: " << outPath << std::endl;
        } else {
            std::cout << "Failed to delete file! outPath: " << outPath << std::endl;
        }
    } else {
        std::cout << "File is not existed! outPath: " << outPath << std::endl;
    }
    return;
}

bool CheckFileContainsString(const std::string& filePath, const std::string& targetString)
{
    std::ifstream file(filePath);
    std::string line;

    while (std::getline(file, line)) {
        if (line.find(targetString) != std::string::npos) {
            return true;
        }
    }

    return false;
}

bool IsPathExist(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::string ExecShellCommand(const std::string& cmd)
{
    std::array<char, CMD_BUFFER_LEN> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}