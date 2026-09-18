// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the BSD 3-Clause License.

#ifndef MSPROBE_CUST_OP_API_LOADER_H
#define MSPROBE_CUST_OP_API_LOADER_H

#include <dlfcn.h>
#include <limits.h>
#include <sys/stat.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include <cstdlib>
#include <string>

inline void *LoadCheckedCustOpApiLib(const std::string &libPath)
{
    if (libPath.empty() || libPath.size() >= PATH_MAX)
    {
        ASCEND_LOGW("Invalid custom op API library path length: %zu.", libPath.size());
        return nullptr;
    }
    char resolvedPath[PATH_MAX];
    if (realpath(libPath.c_str(), resolvedPath) == nullptr)
    {
        return nullptr;
    }
    struct stat fileStat;
    if (stat(resolvedPath, &fileStat) != 0 || !S_ISREG(fileStat.st_mode))
    {
        return nullptr;
    }
    if ((fileStat.st_mode & (S_IWGRP | S_IWOTH)) != 0)
    {
        ASCEND_LOGW("Refusing to load %s: library must not be writable by group or others.", resolvedPath);
        return nullptr;
    }
    void *handler = dlopen(resolvedPath, RTLD_LAZY);
    if (handler == nullptr)
    {
        ASCEND_LOGW("dlopen %s failed, error:%s.", resolvedPath, dlerror());
    }
    return handler;
}

#endif  // MSPROBE_CUST_OP_API_LOADER_H
