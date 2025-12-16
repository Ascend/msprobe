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


std::vector<std::string> SplitString(const std::string &ss, const char &tar);

bool Exists(const std::string &path);

std::string GetRealPath(const std::string &outPath);

extern bool DirectoryExists(const std::string &path);

namespace Utils {
extern bool CheckDirectory(const std::string &directory, bool existOK = true);

bool ValidateCsvString(const std::string& str);

std::string GetLastErrorStr();
}
#endif