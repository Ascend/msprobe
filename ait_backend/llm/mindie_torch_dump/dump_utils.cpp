/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dump_utils.h"
#include <iostream>
#include <string>
#include <climits>
#include <dlfcn.h>
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
    void *handle = dlopen("libascendcl.so", RTLD_LAZY);
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
    void *handle = dlopen("libascendcl.so", RTLD_LAZY);
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