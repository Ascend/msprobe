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


#ifndef AIT_TEST_COMMON_TOOLS_H
#define AIT_TEST_COMMON_TOOLS_H

#include <string>
#include <fstream>
#include <chrono>
#include <vector>

const int TIMEOUT = 5;
const int CHECK_INTERVAL = 50;

int32_t GetCurrentProcessId();
bool CheckFileContainsString(const std::string& filePath, const std::string& targetString);
bool IsPathExist(const std::string& path);
void DeletePath(const std::string& path);
std::string ExecShellCommand(const std::string& cmd);
std::string RoundStrNum(std::string numberStr, uint8_t decimalPlaces, bool enableRound = true);
std::string ExtractValue(std::ifstream& file, const std::string& prefix, uint8_t decimalPlaces);
std::string ExtractValueComplex64(std::ifstream& file, const std::string& prefix, uint8_t decimalPlaces);
bool AreLastDigitsWithinRangeStr(const std::string& numStr1, const std::string& numStr2, uint8_t decimalPlaces);
bool WaitUntilFileReady(const std::string& path, std::chrono::milliseconds timeout = std::chrono::seconds(TIMEOUT),
                        std::chrono::milliseconds checkBaseInterval = std::chrono::milliseconds(CHECK_INTERVAL));
bool CompareBinaryFiles(std::ifstream& file1, std::ifstream& file2);
#endif