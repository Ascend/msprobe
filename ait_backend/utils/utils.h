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
#ifndef UTILS_H
#define UTILS_H

#include <sys/stat.h>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <unistd.h>
#include <cstdio>
#include <cerrno>
#include <experimental/filesystem>
#include "ait_logger.h"


extern std::vector<std::string> SplitString(const std::string &ss, const char &tar);

extern bool Exists(const std::string &path);

std::string GetRealPath(const std::string &outPath);

extern bool DirectoryExists(const std::string &path);

extern bool CheckDirectory(const std::string &directory, bool existOK = true);

bool ValidateCsvString(const std::string& str);

#endif