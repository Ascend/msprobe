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


#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <experimental/filesystem>
#include <cstdio>
#include <cstdlib>
#include <poll.h>
#include <sys/stat.h>
#include <sys/inotify.h>
#include <unistd.h>
#include "tools.h"

namespace fs = std::experimental::filesystem;
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
    return (lstat(path.c_str(), &buffer) == 0);
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

std::string RoundStrNum(std::string numberStr, uint8_t decimalPlaces)
{
    std::string result = "N/A";
    try {
        if (numberStr != "N/A") {
            double value = std::stod(numberStr);
            std::stringstream stream;
            stream << std::fixed << std::setprecision(decimalPlaces) << value;
            stream >> result;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error converting number: " << e.what() << std::endl;
    }
    return result;
};

std::string ExtractValue(std::ifstream& file, const std::string& prefix, uint8_t decimalPlaces)
{
    std::string line;
    std::string value = "N/A";
    auto originalPosition = file.tellg();
    const size_t tagLength = prefix.length() + 1; // 1 = "=".length()
    while (std::getline(file, line)) {
        if (line.find(prefix) == 0) {
            if (prefix.length() + 1 > line.length()) {
                std::cerr << "Prefix with wrong length: " << tagLength << std::endl;
                break;
            }
            std::istringstream iss(line.substr(tagLength));
            std::string numberStr;
            if (std::getline(iss, numberStr)) {
                value = RoundStrNum(numberStr, decimalPlaces);
            }
            break;
        }
    }
    file.seekg(originalPosition); // 重置文件读写指针到原始位置
    return value;
}

std::string ExtractValueComplex64(std::ifstream& file, const std::string& prefix, uint8_t decimalPlaces)
{
    std::string line;
    std::string value = "N/A";
    auto originalPosition = file.tellg();
    const size_t tagLength = prefix.length() + 2; // 2 = ("=" + "(").length()
    while (std::getline(file, line)) {
        if (line.find(prefix) == 0) {
            value="";
            if (tagLength > line.length()) {
                std::cerr << "Prefix with wrong length: " << tagLength << std::endl;
                break;
            }
            std::istringstream iss(line.substr(tagLength));
            std::string numberStr;
            if (std::getline(iss, numberStr, ',')) {
                value += "(" + RoundStrNum(numberStr, decimalPlaces);
            }
            if (std::getline(iss, numberStr, ')')) {
                value += "," + RoundStrNum(numberStr, decimalPlaces) + ")";
            }
            break;
        }
    }
    file.seekg(originalPosition); // 重置文件读写指针到原始位置
    return value;
}

bool WaitUntilFileReady(const std::string& path,
                        std::chrono::milliseconds timeout,
                        std::chrono::milliseconds checkBaseInterval)
{
    using Clock = std::chrono::steady_clock;
    constexpr int requiredStableChecks = 3;
    const int backoffFactor = 2;
    constexpr auto maxCheckInterval = std::chrono::seconds(2);  // 保持为seconds类型
    std::string checkPath = path;
    auto start = Clock::now();
    struct stat prevAttr;
    while (!IsPathExist(checkPath)) { // 阶段1：等待文件出现
        if (Clock::now() - start > timeout) { return false; }
        std::this_thread::sleep_for(checkBaseInterval);
    }
    if (fs::is_symlink(checkPath)) { checkPath = fs::read_symlink(checkPath); } // 符号链接检查指向的文件
    int stableCount = 0; // 阶段2：检测稳定性
    auto checkInterval = checkBaseInterval;
    while (true) {
        std::this_thread::sleep_for(checkInterval);
        struct stat currAttr;
        if (stat(checkPath.c_str(), &currAttr) != 0) { return false; }
        checkInterval = std::min(
            checkInterval * backoffFactor,
            std::chrono::duration_cast<std::chrono::milliseconds>(maxCheckInterval)
        );
        if (currAttr.st_mtime == prevAttr.st_mtime &&
            currAttr.st_size == prevAttr.st_size) {
            if (++stableCount >= requiredStableChecks) { return true; }
        } else {
            stableCount = 0;
            checkInterval = checkBaseInterval;
            prevAttr = currAttr;
        }
        if (Clock::now() - start > timeout) { return false; }
    }
}