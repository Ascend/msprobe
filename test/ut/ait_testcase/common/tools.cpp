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
#include <unistd.h>
#include "tools.h"


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