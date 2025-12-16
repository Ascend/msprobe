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


#include <cstdlib>
#include <iostream>
#include <fstream>
#include <new>
#include <experimental/filesystem>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "dump_utils.h"
#include "env_var_guard.h"

using namespace AscendIE;
namespace fs = std::experimental::filesystem;

/****************​ 文件系统操作工具函数 ​****************/
void CreateFile(const std::string& path)
{
    const int mode = 0644;
    int fd = open(path.c_str(), O_CREAT | O_WRONLY, mode);
    close(fd);
}

void CreateDir(const std::string& path)
{
    const int mode = 0755;
    if (mkdir(path.c_str(), mode) == -1 && errno != EEXIST) {
        throw std::system_error(errno, std::system_category(), "mkdir failed");
    }
}

/******************​ IsDumpEnabled 测试 ​******************/
TEST(MindIEDumpUtils_Func, IsDumpEnabled_EnvNotSet)
{
    EnvVarGuard configGuard("MINDIE_RT_DUMP_CONFIG_PATH");
    unsetenv("MINDIE_RT_DUMP_CONFIG_PATH");
    EXPECT_FALSE(DumpUtils::IsDumpEnabled());
}

TEST(MindIEDumpUtils_Func, IsDumpEnabled_FileNotExist)
{
    EnvVarGuard configGuard("MINDIE_RT_DUMP_CONFIG_PATH");
    setenv("MINDIE_RT_DUMP_CONFIG_PATH", "/invalid/path", 1);
    EXPECT_FALSE(DumpUtils::IsDumpEnabled());
}

/******************​ SetDump 全路径测试 ​******************/
TEST(MindIEDumpUtils_Func, SetDump_EmptyLibPath)
{
    char tmpDir[] = "/tmp/cannXXXXXX";
    char* dir = mkdtemp(tmpDir);
    ASSERT_NE(dir, nullptr) << "临时目录创建失败: " << strerror(errno);
    EnvVarGuard guard("ASCEND_TOOLKIT_HOME");
    setenv("ASCEND_TOOLKIT_HOME", dir, 1);
    CreateDir(std::string(dir) + "/lib64");
    std::string soPath = std::string(dir) + "/lib64/libascendcl.so";
    testing::internal::CaptureStdout(); // 执行测试, 捕捉标准输出并验证
    DumpUtils::SetDump();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("Library absolute path got failed."), std::string::npos);  // 验证存在性
    void* handle = dlopen(soPath.c_str(), RTLD_NOLOAD);
    EXPECT_EQ(handle, nullptr);
    if (handle) { dlclose(handle); }  // 安全释放
    unlink(soPath.c_str());
    rmdir((std::string(dir) + "/lib64").c_str());
    rmdir(dir);
}

TEST(MindIEDumpUtils_Func, SetDump_DlopenFailed)
{
    char tmpDir[] = "/tmp/cannXXXXXX";
    char* dir = mkdtemp(tmpDir);
    ASSERT_NE(dir, nullptr) << "临时目录创建失败: " << strerror(errno);
    EnvVarGuard guard("ASCEND_TOOLKIT_HOME");
    setenv("ASCEND_TOOLKIT_HOME", dir, 1);
    CreateDir(std::string(dir) + "/lib64");
    std::string soPath = std::string(dir) + "/lib64/libascendcl.so";
    CreateFile(soPath);
    testing::internal::CaptureStdout();
    DumpUtils::SetDump();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("Load library failed."), std::string::npos);
    void* handle = dlopen(soPath.c_str(), RTLD_NOLOAD);
    EXPECT_EQ(handle, nullptr);
    if (handle) { dlclose(handle); }
    unlink((std::string(dir) + "/lib64/libascendcl.so").c_str());
    rmdir((std::string(dir) + "/lib64").c_str());
    rmdir(dir);
}

TEST(MindIEDumpUtils_Func, SetDump_PartialSymbolMissing)
{
    const std::string mockCode = R"(
        extern "C" {
            int aclmdlInitDump() { return 0; }
            // 故意缺失aclmdlSetDump
        }
    )";
    char tmpDir[] = "/tmp/cannXXXXXX";
    char* dir = mkdtemp(tmpDir);
    EnvVarGuard guard("ASCEND_TOOLKIT_HOME");
    setenv("ASCEND_TOOLKIT_HOME", dir, 1);
    const std::string libPath = std::string(dir) + "/lib64";
    CreateDir(libPath);
    const std::string soPath = libPath + "/libascendcl.so";
    int ret = system(("echo '" + mockCode + "' | g++ -shared -fPIC -x c++ -o " + soPath + " -").c_str());
    ASSERT_EQ(ret, 0) << "编译失败，退出码: " << WEXITSTATUS(ret) << " 错误: " << strerror(errno);
    testing::internal::CaptureStdout();
    DumpUtils::SetDump();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("Dynamic linking symbol failed."), std::string::npos);
    void* handle = dlopen(soPath.c_str(), RTLD_NOLOAD);
    EXPECT_EQ(handle, nullptr);
    if (handle) { dlclose(handle); }
    unlink(soPath.c_str());
    rmdir(libPath.c_str());
    rmdir(dir);
}

TEST(MindIEDumpUtils_Func, SetDump_InitFunctionFailed)
{
    const std::string mockCode = R"(
        extern "C" {
            int aclmdlInitDump() { return 1; }  // 返回错误码
            int aclmdlSetDump(const char*) { return 0; }
        }
    )";
    std::cout << "" << std::endl;
    char tmpDir[] = "/tmp/cannXXXXXX";
    char* dir = mkdtemp(tmpDir);
    EnvVarGuard guard("ASCEND_TOOLKIT_HOME");
    setenv("ASCEND_TOOLKIT_HOME", dir, 1);
    const std::string libPath = std::string(dir) + "/lib64";
    CreateDir(libPath);
    const std::string soPath = libPath + "/libascendcl.so";
    int ret = system(("echo '" + mockCode + "' | g++ -shared -fPIC -x c++ -o " + soPath + " -").c_str());
    ASSERT_EQ(ret, 0) << "编译失败，退出码: " << WEXITSTATUS(ret) << " 错误: " << strerror(errno);
    testing::internal::CaptureStdout();
    DumpUtils::SetDump();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("Failed to init acl dump."), std::string::npos);
    void* handle = dlopen(soPath.c_str(), RTLD_NOLOAD);
    EXPECT_EQ(handle, nullptr);
    if (handle) { dlclose(handle); }
    unlink(soPath.c_str());
    rmdir(libPath.c_str());
    rmdir(dir);
}

TEST(MindIEDumpUtils_Func, SetDump_SetFunctionFailed)
{
    const std::string mockCode = R"(
        extern "C" {
            int aclmdlInitDump() { return 0; }
            int aclmdlSetDump(const char*) { return -1; }  // 返回错误码
        }
    )";
    char tmpDir[] = "/tmp/cannXXXXXX";
    char* dir = mkdtemp(tmpDir);
    EnvVarGuard guard("ASCEND_TOOLKIT_HOME");
    setenv("ASCEND_TOOLKIT_HOME", dir, 1);
    const std::string libPath = std::string(dir) + "/lib64";
    CreateDir(libPath);
    const std::string soPath = libPath + "/libascendcl.so";
    int ret = system(("echo '" + mockCode + "' | g++ -shared -fPIC -x c++ -o " + soPath + " -").c_str());
    ASSERT_EQ(ret, 0) << "编译失败，退出码: " << WEXITSTATUS(ret) << " 错误: " << strerror(errno);
    testing::internal::CaptureStdout();
    DumpUtils::SetDump();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("Failed to set acl dump info."), std::string::npos);
    void* handle = dlopen(soPath.c_str(), RTLD_NOLOAD);
    EXPECT_EQ(handle, nullptr);
    if (handle) { dlclose(handle); }
    unlink(soPath.c_str());
    rmdir(libPath.c_str());
    rmdir(dir);
}

TEST(MindIEDumpUtils_Func, SetDump_FullSuccessPath)
{
    const std::string mockCode = R"(
        extern "C" {
            int aclmdlInitDump() { return 0; }
            int aclmdlSetDump(const char*) { return 0; }
        }
    )";
    char tmpDir[] = "/tmp/cannXXXXXXXX";
    char* dir = mkdtemp(tmpDir);
    EnvVarGuard guard("ASCEND_TOOLKIT_HOME");
    setenv("ASCEND_TOOLKIT_HOME", dir, 1);
    const std::string libPath = std::string(dir) + "/lib64";
    CreateDir(libPath);
    const std::string soPath = libPath + "/libascendcl.so";
    int ret = system(("echo '" + mockCode + "' | g++ -shared -fPIC -x c++ -o " + soPath + " -").c_str());
    ASSERT_EQ(ret, 0) << "编译失败，退出码: " << WEXITSTATUS(ret) << " 错误: " << strerror(errno);
    char confFile[] = "/tmp/configXXXXXX";
    int fd = mkstemp(confFile);
    close(fd);
    EnvVarGuard configGuard("MINDIE_RT_DUMP_CONFIG_PATH");
    setenv("MINDIE_RT_DUMP_CONFIG_PATH", confFile, 1);
    testing::internal::CaptureStdout();
    DumpUtils::SetDump();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("Init acl dump succeed."), std::string::npos);
    void* handle = dlopen(soPath.c_str(), RTLD_NOLOAD);
    EXPECT_EQ(handle, nullptr);
    if (handle) { dlclose(handle); }
    unlink(soPath.c_str());
    rmdir(libPath.c_str());
    rmdir(dir);
    unlink(confFile);
}

/******************​ FinalizeDump 测试 ​******************/
TEST(MindIEDumpUtils_Func, FinalizeDump_EmptyLibPath)
{
    char tmpDir[] = "/tmp/cannXXXXXX";
    char* dir = mkdtemp(tmpDir);
    EnvVarGuard guard("ASCEND_TOOLKIT_HOME");
    setenv("ASCEND_TOOLKIT_HOME", dir, 1);
    CreateDir(std::string(dir) + "/lib64");
    std::string soPath = std::string(dir) + "/lib64/libascendcl.so";
    testing::internal::CaptureStdout();
    DumpUtils::FinalizeDump();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("Library path got failed."), std::string::npos);
    void* handle = dlopen(soPath.c_str(), RTLD_NOLOAD);
    EXPECT_EQ(handle, nullptr);
    if (handle) { dlclose(handle); }
    unlink(soPath.c_str());
    rmdir((std::string(dir) + "/lib64").c_str());
    rmdir(dir);
}

TEST(MindIEDumpUtils_Func, FinalizeDump_Failed)
{
    char tmpDir[] = "/tmp/cannXXXXXX";
    char* dir = mkdtemp(tmpDir);
    EnvVarGuard guard("ASCEND_TOOLKIT_HOME");
    setenv("ASCEND_TOOLKIT_HOME", dir, 1);
    CreateDir(std::string(dir) + "/lib64");
    std::string soPath = std::string(dir) + "/lib64/libascendcl.so";
    int ret = system(
                ("echo 'int aclmdlFinalizeDump(){return -1;}' | gcc -shared -fPIC -x c -o " + soPath + " -").c_str()
            );
    ASSERT_EQ(ret, 0) << "编译失败，退出码: " << WEXITSTATUS(ret) << " 错误: " << strerror(errno);
    testing::internal::CaptureStdout();
    DumpUtils::FinalizeDump();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("Failed to finalize acl dump."), std::string::npos);
    void* handle = dlopen(soPath.c_str(), RTLD_NOLOAD);
    EXPECT_EQ(handle, nullptr);
    if (handle) { dlclose(handle); }
    unlink(soPath.c_str());
    rmdir((std::string(dir) + "/lib64").c_str());
    rmdir(dir);
}

TEST(MindIEDumpUtils_Func, FinalizeDump_Success)
{
    char tmpDir[] = "/tmp/cannXXXXXX";
    char* dir = mkdtemp(tmpDir);
    EnvVarGuard guard("ASCEND_TOOLKIT_HOME");
    setenv("ASCEND_TOOLKIT_HOME", dir, 1);
    CreateDir(std::string(dir) + "/lib64");
    std::string soPath = std::string(dir) + "/lib64/libascendcl.so";
    int ret = system(
                ("echo 'int aclmdlFinalizeDump(){return 0;}' | gcc -shared -fPIC -x c -o " + soPath + " -").c_str()
            );
    ASSERT_EQ(ret, 0) << "编译失败，退出码: " << WEXITSTATUS(ret) << " 错误: " << strerror(errno);
    testing::internal::CaptureStdout();
    DumpUtils::FinalizeDump();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("Finalize acl dump succeed."), std::string::npos);
    void* handle = dlopen(soPath.c_str(), RTLD_NOLOAD);
    EXPECT_EQ(handle, nullptr);
    if (handle) { dlclose(handle); }
    unlink(soPath.c_str());
    rmdir((std::string(dir) + "/lib64").c_str());
    rmdir(dir);
}