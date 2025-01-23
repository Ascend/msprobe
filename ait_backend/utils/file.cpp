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

#include <map>
#include <iostream>
#include <cstring>
#include <regex>
#include <unordered_map>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "const.h"
#include "file.h"
using MsConst::SUFFIX;
using MsConst::SUFFIX_TYPE_TABLE;

std::string File::GetFullPath(const std::string &originPath)
{
    if (originPath.empty()) {
        return "";
    }
    if (originPath[0] == MsConst::PATH_SEPARATOR) {
        return originPath;
    }

    char* cwd = nullptr;
    char* cwdBuf = nullptr;
    try {
        cwdBuf = new char[PATH_MAX];
    } catch (const std::bad_alloc& e) {
        AIT_LOG_ERROR("create buffer failed: "+ std::string(e.what()));
        throw std::runtime_error("No memory.");
    }
    cwd = getcwd(cwdBuf, PATH_MAX);
    if (cwd == nullptr) {
        delete[] cwdBuf;
        return "";
    }

    std::string fullPath = std::move(std::string(cwd) + MsConst::PATH_SEPARATOR + originPath);
    delete[] cwdBuf;
    cwdBuf = nullptr;

    return fullPath;
}

static std::vector<std::string> SplitPath(const std::string &path)
{
    std::vector<std::string> tokens;
    size_t len = path.length();
    size_t start = 0;

    while (start < len) {
        size_t end = path.find(MsConst::PATH_SEPARATOR, start);
        if (end == std::string::npos) {
            end = len;
        }
        if (start != end) {
            tokens.push_back(path.substr(start, end - start));
        }
        start = end + 1;
    }
    return tokens;
}

std::string File::GetAbsPath(const std::string &originPath)
{
    std::string fullPath = GetFullPath(originPath);
    if (fullPath.empty()) {
        return "";
    }

    std::vector<std::string> tokens = SplitPath(fullPath);
    std::vector<std::string> tokensRefined;

    for (std::string& token : tokens) {
        if (token.empty() || token == ".") {
            continue;
        } else if (token == "..") {
            if (tokensRefined.empty()) {
                return "";
            }
            tokensRefined.pop_back();
        } else {
            tokensRefined.emplace_back(token);
        }
    }

    if (tokensRefined.empty()) {
        return "/";
    }
    std::string resolvedPath("");
    for (std::string& token : tokensRefined) {
        resolvedPath.append("/").append(token);
    }
    return resolvedPath;
}

bool File::IsFileReadable(const std::string& path)
{
    return access(path.c_str(), R_OK) == 0;
}

bool File::IsFileWritable(const std::string& path)
{
    return access(path.c_str(), W_OK) == 0;
}

bool File::IsOtherWritable(const std::string& path)
{
    return ((GetPathPermissions(path) & MsConst::READ_FILE_NOT_PERMITTED) > 0);
}

bool File::IsPathExist(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

bool File::IsPathLengthLegal(const std::string& path)
{
    if (path.length() > MsConst::FULL_PATH_LENGTH_MAX || path.length() == 0) {
        return false;
    }
    std::vector<std::string> tokens = SplitPath(path);
    for (std::string& token : tokens) {
        if (token.length() > MsConst::FILE_NAME_LENGTH_MAX) {
            return false;
        }
    }
    return true;
}

bool File::IsPathCharactersValid(const std::string& path)
{
    return std::regex_match(path, std::regex(MsConst::FILE_VALID_PATTERN));
}

bool File::IsPathDepthValid(const std::string& path)
{
    return std::count(path.begin(), path.end(), MsConst::PATH_SEPARATOR) <= MsConst::PATH_DEPTH_MAX;
}

bool File::IsRegularFile(const std::string& path)
{
    struct stat pathStat;
    if (stat(path.c_str(), &pathStat) == 0) {
        return S_ISREG(pathStat.st_mode);
    }
    return false;
}

bool File::IsDir(const std::string& path)
{
    struct stat buffer;
    if (stat(path.c_str(), &buffer) == 0) {
        return (buffer.st_mode & S_IFDIR) != 0;
    }
    return false;
}

size_t File::GetFileSize(const std::string &path)
{
    struct stat pathStat;
    if (stat(path.c_str(), &pathStat) != 0) {
        AIT_LOG_ERROR("file not exists");
        return 0;
    }
    return static_cast<size_t>(pathStat.st_size);
}

std::string File::GetParentDir(const std::string& path)
{
    size_t found = path.find_last_of('/');
    if (found != std::string::npos) {
        return path.substr(0, found);
    }
    return ".";
}

std::string File::GetFileName(const std::string& path)
{
    size_t found = path.find_last_of('/');
    if (found != std::string::npos) {
        return path.substr(found + 1);
    }
    return path;
}

mode_t File::GetPathPermissions(const std::string& path)
{
    struct stat pathStat;
    if (stat(path.c_str(), &pathStat) != 0) {
        AIT_LOG_ERROR("path is not exists");
        return MsConst::MAX_PERMISSION;
    }
    mode_t permissions = pathStat.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    return permissions;
}

std::string File::GetFileSuffix(const std::string& path)
{
    std::string fileName = GetFileName(path);
    size_t dotPos = fileName.find_last_of('.');
    if (dotPos != std::string::npos && dotPos + 1 < fileName.size()) {
        return fileName.substr(dotPos + 1);
    }
    return "";
}

bool File::CheckFileSuffixAndSize(const std::string &path, SUFFIX type, const size_t maxSize)
{
    struct stat pathStat;
    if (stat(path.c_str(), &pathStat) != 0) {
        AIT_LOG_ERROR("file not exists");
        return false;
    }
    size_t size = GetFileSize(path);

    if (type == SUFFIX::NONE) {
        if (size > maxSize) {
            AIT_LOG_ERROR("file size invalid");
            return false;
        }
        return true;
    }

    auto iter = SUFFIX_TYPE_TABLE.find(type);
    if (iter == SUFFIX_TYPE_TABLE.end()) {
        AIT_LOG_ERROR("unknown file suffix");
        return false;
    }

    std::string suffix = GetFileSuffix(path);
    if (suffix != iter->second.first) {
        AIT_LOG_ERROR("unknown file suffix");
        return false;
    }
    if (size > iter->second.second && size > maxSize) {
        AIT_LOG_ERROR("file size is invalid");
        return false;
    }
    return true;
}

bool File::IsSoftLink(const std::string &path)
{
    std::string absPath = GetAbsPath(path);
    struct stat fileStat;
    if (lstat(absPath.c_str(), &fileStat) != 0) {
        AIT_LOG_ERROR("the file lstat failed");
        return false;
    }
    return S_ISLNK(fileStat.st_mode);
}

/****************** 文件操作函数库，会对入参做基本检查 ************************/

static bool CreateDirAux(const std::string& path, bool recursion, mode_t mode)
{
    std::string parent = File::GetParentDir(path);
    if (!File::IsPathExist(parent)) {
        if (!recursion) {
            AIT_LOG_ERROR("dir path not exist");
            return false;
        }
        /* 递归创建父目录，由于前面已经判断过目录深度，此处递归是安全的 */
        if (!CreateDirAux(parent, recursion, mode)) {
            AIT_LOG_ERROR("recursive creation of parent directory failed");
            return false;
        }
    }
    if (!File::CheckDir(parent)) { // 每当需要创建文件夹时都要校验父目录
        AIT_LOG_ERROR("parent directory is illegal");
        return false;
    }
    if (mkdir(path.c_str(), mode) != 0) {
        if (errno == EACCES || errno == EROFS) {
            AIT_LOG_ERROR("mkdir permission denined");
            return false;
        } else {
            AIT_LOG_ERROR("syscall failed");
            return false;
        }
    }
    return true;
}

bool File::CreateDir(const std::string &path, bool recursion, mode_t mode)
{
    if (IsPathExist(path)) {
        AIT_LOG_INFO("dir already exist, no need to create");
        return true;
    }
    std::string absPath = GetAbsPath(path);
    if (absPath.empty()) {
        AIT_LOG_ERROR("path is empty");
        return false;
    }
    if (!IsPathLengthLegal(absPath)) {
        AIT_LOG_ERROR("path length illegal");
        return false;
    }
    if (!IsPathCharactersValid(absPath)) {
        AIT_LOG_ERROR("path characters invalid");
        return false;
    }
    if (!IsPathDepthValid(absPath)) {
        AIT_LOG_ERROR("path depth invalid");
        return false;
    }
    return CreateDirAux(absPath, recursion, mode);
}

/****************************** 通用检查函数 ********************************/
bool File::CheckOwner(const std::string &path)
{
    std::string absPath = GetAbsPath(path);
    struct stat buf;
    if (stat(absPath.c_str(), &buf)) {
        AIT_LOG_ERROR("get file stat failed");
        return false;
    }
    if (buf.st_uid != getuid()) {
        AIT_LOG_ERROR("file owner is not process usr");
        return false;
    }
    return true;
}

bool File::CheckDir(const std::string &path)
{
    std::string absPath = GetAbsPath(path);
    if (absPath.empty()) {
        AIT_LOG_ERROR("path is empty");
        return false;
    }
    if (!IsPathLengthLegal(absPath)) {
        AIT_LOG_ERROR("path length illegal");
        return false;
    }
    if (!IsPathCharactersValid(absPath)) {
        AIT_LOG_ERROR("path characters invalid");
        return false;
    }
    if (!IsPathDepthValid(absPath)) {
        AIT_LOG_ERROR("path depth invalid");
        return false;
    }
    if (!IsPathExist(absPath)) {
        AIT_LOG_ERROR("path is not exist");
        return false;
    }
    if (IsSoftLink(absPath)) {
        AIT_LOG_ERROR("path is soft link");
        return false;
    }
    if (!CheckOwner(absPath)) {
        return false;
    }
    if (!IsDir(absPath)) {
        AIT_LOG_ERROR("path is not a dir");
        return false;
    }
    if (IsOtherWritable(absPath)) {
        AIT_LOG_ERROR("dir permission should not be over 0o755(rwxr-xr-x)");
        return false;
    }
    return true;
}

bool File::CheckFileBeforeRead(const std::string &path, SUFFIX type, const size_t maxSize)
{
    std::string absPath = GetAbsPath(path);
    if (absPath.empty()) {
        AIT_LOG_ERROR("path is empty");
        return false;
    }
    if (!IsPathLengthLegal(absPath)) {
        AIT_LOG_ERROR("path length illegal");
        return false;
    }
    if (!IsPathCharactersValid(absPath)) {
        AIT_LOG_ERROR("path characters invalid");
        return false;
    }
    if (!IsPathDepthValid(absPath)) {
        AIT_LOG_ERROR("path depth invalid");
        return false;
    }
    if (!IsRegularFile(absPath)) {
        AIT_LOG_ERROR("path is not regular file");
        return false;
    }
    if (IsSoftLink(absPath)) {
        AIT_LOG_ERROR("path is a soft link");
        return false;
    }
    if (!CheckOwner(absPath)) {
        return false;
    }
    if (IsOtherWritable(absPath)) {
        AIT_LOG_ERROR("file permission should not be over 0o755(rwxr-xr-x)");
        return false;
    }
    if (!IsFileReadable(absPath) || (GetPathPermissions(absPath) & S_IRUSR) == 0) {
        AIT_LOG_ERROR("file permission should be at least 0o400(r--------)");
        return false;
    }
    /* 如果是/dev/random之类的无法计算size的文件，不要用本函数check */
    if (!CheckFileSuffixAndSize(path, type, maxSize)) {
        AIT_LOG_ERROR("file suffix and size invalid");
        return false;
    }
    return CheckDir(GetParentDir(absPath));
}

bool File::CheckFileBeforeCreateOrWrite(const std::string &path, bool overwrite)
{
    std::string absPath = GetAbsPath(path);
    if (absPath.empty()) {
        AIT_LOG_ERROR("path is empty");
        return false;
    }
    if (!IsPathLengthLegal(absPath)) {
        AIT_LOG_ERROR("path length illegal");
        return false;
    }
    if (!IsPathCharactersValid(absPath)) {
        AIT_LOG_ERROR("path characters invalid");
        return false;
    }
    if (!IsPathDepthValid(absPath)) {
        AIT_LOG_ERROR("path depth invalid");
        return false;
    }
    if (IsPathExist(absPath)) {
        if (!overwrite) {
            AIT_LOG_ERROR("path already exist and not allow to overwrite");
            return false;
        }
        if (!IsRegularFile(absPath)) {
            AIT_LOG_ERROR("path is not regular file");
            return false;
        }
        if (IsSoftLink(absPath)) {
            AIT_LOG_ERROR("path is soft link");
            return false;
        }
        if ((GetPathPermissions(absPath) & MsConst::WRITE_FILE_NOT_PERMITTED) > 0) {
            AIT_LOG_ERROR("path permission should not be over 0o750(rwxr-x---)");
            return false;
        }
        /* 默认不允许覆盖其他用户创建的文件，若有特殊需求（如多用户通信管道等）由业务自行校验 */
        if (!IsFileWritable(absPath) || !CheckOwner(absPath)) {
            AIT_LOG_ERROR("path already create by other owner");
            return false;
        }
    }
    return CheckDir(GetParentDir(absPath));
}
