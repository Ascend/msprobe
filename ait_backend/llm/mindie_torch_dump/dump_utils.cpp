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


#include "dump_utils.h"

#include <unistd.h>
#include <iostream>
#include <string>
#include <climits>
#include <dlfcn.h>
#include <sys/stat.h>
#include "ait_logger.h"

using FuncPtr1 = int (*)();
using FuncPtr2 = int (*)(const char *);

namespace {
constexpr char const * MINDIE_RT_DUMP_CONFIG_PATH = "MINDIE_RT_DUMP_CONFIG_PATH";
constexpr size_t ENV_MAX_LENGTH = 1024;
std::string GetEnv(const std::string &name, size_t maxLen = ENV_MAX_LENGTH)
{
    auto env = std::getenv(name.c_str());
    if (env != nullptr) {
        return std::string(env).size() > maxLen ? "" : env;
    }
    return "";
}

const std::string LIBASCENDCL_SO = "libascendcl.so";
constexpr char const * ASCEND_TOOLKIT_HOME = "ASCEND_TOOLKIT_HOME";
bool FileExists(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}
std::string GetSoInCANNPackagePath(const std::string& libPath)
{
    std::string absoluteSoPath;
    std::string cannPath = GetEnv(ASCEND_TOOLKIT_HOME);
    if (cannPath.empty() || !FileExists(cannPath)) {
        AIT_LOG_ERROR("[mindie-dump]cann_path is invalid, please install " +
                      std::string("cann-toolkit and set the environment variables."));
        absoluteSoPath.clear();
        return absoluteSoPath;
    }
    absoluteSoPath = cannPath + "/lib64/" + libPath;
    if (!FileExists(absoluteSoPath)) {
        AIT_LOG_ERROR("[mindie-dump]" + libPath + "is not found in " + cannPath +
                      ". Try installing the latest cann-toolkit");
        absoluteSoPath.clear();
        return absoluteSoPath;
    }
    return absoluteSoPath;
}
}

namespace AscendIE {
bool DumpUtils::IsDumpEnabled()
{
    std::string dumpConfigFile = GetEnv(MINDIE_RT_DUMP_CONFIG_PATH);
    if (dumpConfigFile.empty()) {
        AIT_LOG_ERROR("[mindie-dump]Dump config file path is not set in env. Disable dump.");
        return false;
    }

    char path[PATH_MAX] = { 0 };
    if (realpath(dumpConfigFile.c_str(), path) == nullptr) {
        AIT_LOG_ERROR("[mindie-dump]Dump config file path is not exist. Disable dump.");
        return false;
    }

    return true;
}

void DumpUtils::SetDump()
{
    std::string dumpConfigFile = GetEnv(MINDIE_RT_DUMP_CONFIG_PATH);
    std::string libAscendclsoPath = GetSoInCANNPackagePath(LIBASCENDCL_SO);
    if (libAscendclsoPath.empty()) {
        AIT_LOG_ERROR("[mindie-dump]Library absolute path got failed.");
        return;
    }

    struct stat fileStat;
    if (stat(libAscendclsoPath.c_str(), &fileStat) != 0) { return; }
    if (getuid() != 0 && fileStat.st_uid != getuid()) { return; }
    mode_t permissions = fileStat.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    if ((permissions & (S_IWGRP | S_IWOTH)) > 0) { return; }

    void* handle = dlopen(libAscendclsoPath.c_str(), RTLD_LAZY);
    if (!handle) {
        AIT_LOG_ERROR("[mindie-dump]Load library failed.");
        return;
    }

    void *func1 = dlsym(handle, "aclmdlInitDump");
    void *func2 = dlsym(handle, "aclmdlSetDump");
    if (func1 == nullptr || func2 == nullptr) {
        AIT_LOG_ERROR("[mindie-dump]Dynamic linking symbol failed. ");
        dlclose(handle);
        return;
    }

    FuncPtr1 aclImitFunc = reinterpret_cast<FuncPtr1>(func1);
    FuncPtr2 aclSetDumpFunc = reinterpret_cast<FuncPtr2>(func2);
    auto ret = aclImitFunc();
    if (ret != 0) {
        AIT_LOG_ERROR("[mindie-dump]Failed to init acl dump. Acl ret code = " + std::to_string(ret));
        dlclose(handle);
        return;
    }

    ret = aclSetDumpFunc(dumpConfigFile.c_str());
    if (ret != 0) {
        AIT_LOG_ERROR("[mindie-dump]Failed to set acl dump info. Acl ret code = " + std::to_string(ret));
        dlclose(handle);
        return;
    }
    dlclose(handle);
    AIT_LOG_INFO("[mindie-dump]Init acl dump succeed.");
}

void DumpUtils::FinalizeDump()
{
    std::string libAscendclsoPath = GetSoInCANNPackagePath(LIBASCENDCL_SO);
    if (libAscendclsoPath.empty()) {
        AIT_LOG_ERROR("[mindie-dump]Library path got failed.");
        return;
    }

    struct stat fileStat;
    if (stat(libAscendclsoPath.c_str(), &fileStat) != 0) { return; }
    if (getuid() != 0 && fileStat.st_uid != getuid()) { return; }
    mode_t permissions = fileStat.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    if ((permissions & (S_IWGRP | S_IWOTH)) > 0) { return; }

    void* handle = dlopen(libAscendclsoPath.c_str(), RTLD_LAZY);
    if (!handle) {
        AIT_LOG_ERROR("[mindie-dump]Load library failed.");
        return;
    }
    void *func1 = dlsym(handle, "aclmdlFinalizeDump");
    if (func1 == nullptr) {
        AIT_LOG_ERROR("[mindie-dump]Dynamic linking symbol failed. ");
        dlclose(handle);
        return;
    }
    FuncPtr1 aclFinalizeDumpFunc = reinterpret_cast<FuncPtr1>(func1);
    auto ret = aclFinalizeDumpFunc();
    if (ret != 0) {
        AIT_LOG_ERROR("[mindie-dump]Failed to finalize acl dump. Acl ret code = " + std::to_string(ret));
        dlclose(handle);
        return;
    }
    dlclose(handle);
    AIT_LOG_INFO("[mindie-dump]Finalize acl dump succeed.");
}
}