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


#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <experimental/filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::string RoundStrNum(std::string numberStr, uint8_t decimalPlaces, bool enableRound)
{
    std::string result = "N/A";
    try {
        if (numberStr != "N/A") {
            double value = std::stod(numberStr);
            std::stringstream stream;
            
            if (enableRound) {
                // 启用四舍五入模式
                stream << std::fixed << std::setprecision(decimalPlaces) << value;
            } else {
                // 禁用四舍五入模式（截断小数位）
                double factor = std::pow(10, decimalPlaces);
                double truncated = std::trunc(value * factor) / factor;
                stream << std::fixed << std::setprecision(decimalPlaces) << truncated;
            }
            
            // 优化显示：移除尾部多余的零和小数点（可选）
            std::string s = stream.str();
            s.erase(s.find_last_not_of('0') + 1, std::string::npos);
            if (s.back() == '.') { s.pop_back(); }
            
            return s.empty() ? "0" : s;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error converting number: " << e.what() << std::endl;
    }
    return result;
}

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

bool AreLastDigitsWithinRangeStr(const std::string& numStr1,
    const std::string& numStr2,
    uint8_t decimalPlaces)
{
    // 处理特殊值N/A
    if (numStr1 == "N/A" && numStr2 == "N/A") { return true; }
    if (numStr1 == "N/A" || numStr2 == "N/A") { return false; }

    try {
        // 字符串转数值
        const double num1 = std::stod(numStr1);
        const double num2 = std::stod(numStr2);

        // 缩放因子
        const double factor = std::pow(10, decimalPlaces);

        // 四舍五入并取整（精确到指定位数）
        const long long scaled1 = std::llround(num1 * factor);
        const long long scaled2 = std::llround(num2 * factor);

        // 计算绝对差值
        const long long diff = std::abs(scaled1 - scaled2);
        return (diff <= 1);
    } catch (const std::exception& e) {
        std::cerr << "数值转换错误: " << e.what() << std::endl;
        return false;
    }
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
    while (fs::is_symlink(checkPath)) { checkPath = fs::read_symlink(checkPath); } // 符号链接检查指向的文件
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

void DeletePath(const std::string& path)
{
    try {
        if (!IsPathExist(path)) {
            return;
        }
        // 处理文件
        if (fs::is_regular_file(path) || fs::is_symlink(path)) {
            if (fs::remove(path)) {
            } else {
                std::cerr << "Delete file failed: " << path << std::endl;
            }
            return;
        }
        // 处理目录（递归删除）
        if (fs::is_directory(path)) {
            fs::remove_all(path);
            return;
        }
        // 处理特殊文件类型
        std::cerr << "Unsupported file type: " << path << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error deleting " << path << ": " << e.what() << std::endl;
    }
}

bool CompareBinaryFiles(std::ifstream& file1, std::ifstream& file2)
{
    auto resetStreams = [](std::ifstream& file1, std::ifstream& file2) {
        file1.clear();
        file2.clear();
        file1.seekg(0);
        file2.seekg(0);
    };
    resetStreams(file1, file2);
    if (!file1.is_open() || !file2.is_open()) {
        resetStreams(file1, file2);
        return false;
    }
    file1.seekg(0, std::ios::end);
    file2.seekg(0, std::ios::end);
    const auto size1 = file1.tellg();
    const auto size2 = file2.tellg();
    if (size1 != size2) {
        resetStreams(file1, file2);  // 关键修正：在返回前重置
        return false;
    }
    resetStreams(file1, file2);  // 回到文件开头
    const size_t bufferSize = 4096;
    std::vector<char> buffer1(bufferSize);
    std::vector<char> buffer2(bufferSize);
    while (true) {
        file1.read(buffer1.data(), bufferSize);
        file2.read(buffer2.data(), bufferSize);
        const auto bytesRead1 = file1.gcount();
        const auto bytesRead2 = file2.gcount();
        if (file1.fail() && !file1.eof()) {
            resetStreams(file1, file2);
            return false;
        }
        if (bytesRead1 != bytesRead2) {
            resetStreams(file1, file2);
            return false;
        }
        if (bytesRead1 == 0) { break; }
        if (std::memcmp(buffer1.data(), buffer2.data(), bytesRead1) != 0) {
            resetStreams(file1, file2);
            return false;
        }
    }
    resetStreams(file1, file2);
    return true;
}
