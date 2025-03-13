/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
 * Create Date: 2023
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


#ifndef AIT_TEST_COMMON_TOOLS_H
#define AIT_TEST_COMMON_TOOLS_H

#include <string>
#include <fstream>
#include <chrono>

const int TIMEOUT = 5;
const int CHECK_INTERVAL = 50;

int32_t GetCurrentProcessId();
bool IfFileExists(const std::string &outPath);
void DeleteFile(const std::string &outPath);
bool CheckFileContainsString(const std::string& filePath, const std::string& targetString);
bool IsPathExist(const std::string& path);
std::string ExecShellCommand(const std::string& cmd);
std::string RoundStrNum(std::string numberStr, uint8_t decimalPlaces);
std::string ExtractValue(std::ifstream& file, const std::string& prefix, uint8_t decimalPlaces);
std::string ExtractValueComplex64(std::ifstream& file, const std::string& prefix, uint8_t decimalPlaces);
bool WaitUntilFileReady(const std::string& path, std::chrono::milliseconds timeout = std::chrono::seconds(TIMEOUT),
                        std::chrono::milliseconds checkBaseInterval = std::chrono::milliseconds(CHECK_INTERVAL));
#endif