/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

#ifndef FILE_H
#define FILE_H

#include <string>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include "ait_logger.h"
#include "const.h"

// File 类主要处理文件相关操作
namespace File {
    // 安全的创建目录
    bool CreateDir(const std::string &path, bool recursion = false, mode_t mode = MsConst::NORMAL_DIR_MODE_DEFAULT);
    // 文件夹校验：包括路径长度，文件存在性，软链接，属组，权限
    bool CheckDir(const std::string &path);
    // 文件权限校验
    bool IsFileReadable(const std::string& path);
    bool IsFileWritable(const std::string& path);
    bool IsOtherWritable(const std::string& path);
    // 文件存在性校验
    bool IsPathExist(const std::string& path);
    // 软链接校验
    bool IsSoftLink(const std::string &path);
    // 校验是否是文件夹
    bool IsDir(const std::string& path);
    // 校验文件属组
    bool CheckOwner(const std::string &path);
    // 获取文件父目录
    std::string GetParentDir(const std::string& path);
    // 获取绝对路径
    std::string GetFullPath(const std::string &originPath);
    std::string GetAbsPath(const std::string &originPath);
    // 获取文件大小
    size_t GetFileSize(const std::string &path);
    // 路径长度校验
    bool IsPathLengthLegal(const std::string& path);
    // 路径字符校验
    bool IsPathCharactersValid(const std::string& path);
    // 路径深度校验
    bool IsPathDepthValid(const std::string& path);
    // 常规文件校验
    bool IsRegularFile(const std::string& path);
    // 获取文件名
    std::string GetFileName(const std::string& path);
    // 获取文件权限
    mode_t GetPathPermissions(const std::string& path);
    // 获取文件后缀
    std::string GetFileSuffix(const std::string& path);
    // 校验文件后缀和内容长度
    bool CheckFileSuffixAndSize(const std::string &path, MsConst::SUFFIX type, const size_t maxSize);
    // 读文件前的校验
    bool CheckFileBeforeRead(const std::string &path, MsConst::SUFFIX type, const size_t maxSize);
    bool CheckFileBeforeCreateOrWrite(const std::string &path, bool overwrite = false);
    bool CheckConfigFile(const std::string &absPath, const size_t maxSize = MsConst::MAX_JSON_SIZE);
    bool WriteTextToFile(const std::string &filePath, const std::string &textContent);
};
#endif // FILE_H
