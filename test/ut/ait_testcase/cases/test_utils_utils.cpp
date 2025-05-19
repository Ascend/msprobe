/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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

#include <string>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "utils.h"
#include "tools.h"

TEST(Utils_Func, ValidateCsvString_success_opname)
{
    const std::string testString = "TransdataOperation_01";
    EXPECT_TRUE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_success_dtype)
{
    const std::string testString = "float16;float16;float16;float16;int32";
    EXPECT_TRUE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_success_format)
{
    const std::string testString = "nd;nd;fractal_nz;fractal_nz;nd";
    EXPECT_TRUE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_success_shape)
{
    const std::string testString = "1,16,128;9,8,128,16;9,8,128,16;1,1;1";
    EXPECT_TRUE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_success_shape_with_minus)
{
    const std::string testString = "-1,16,128;9,8,128,16;9,8,128,16;1,1;1";
    EXPECT_TRUE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_success_path)
{
    const std::string testString = "msit_dump_20241010_082402/tensors/0_1563353/0/3_Decoder_layer/after/intensor0.bin";
    EXPECT_TRUE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_failed_with_minus)
{
    const std::string testString = "-1a,16,128;9,8,128,16;";
    EXPECT_FALSE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_failed_with_plus)
{
    const std::string testString = "+1,16,128;9,8,128,16;";
    EXPECT_FALSE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_failed_with_equal)
{
    const std::string testString = "=1,16,128;9,8,128,16;";
    EXPECT_FALSE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_failed3_with_at)
{
    const std::string testString = "@1,16,128;9,8,128,16;";
    EXPECT_FALSE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, ValidateCsvString_failed_with_percent)
{
    const std::string testString = "%1,16,128;9,8,128,16;";
    EXPECT_FALSE(Utils::ValidateCsvString(testString));
}

TEST(Utils_Func, Exists_file_exists)
{
    char tmpFile[] = "/tmp/testfileXXXXXX";
    int fd = mkstemp(tmpFile);
    ASSERT_NE(fd, -1);
    close(fd);
    
    EXPECT_TRUE(Exists(tmpFile));
    unlink(tmpFile);
}

TEST(Utils_Func, Exists_directory_exists)
{
    char tmpDir[] = "/tmp/testdirXXXXXX";
    char* dirName = mkdtemp(tmpDir);
    ASSERT_NE(dirName, nullptr);
    
    EXPECT_TRUE(Exists(dirName));
    rmdir(dirName);
}

TEST(Utils_Func, Exists_path_not_exist)
{
    std::string invalidPath = "/tmp/nonexistent_" + std::to_string(getpid());
    EXPECT_FALSE(Exists(invalidPath));
}

TEST(Utils_Func, DirectoryExists_normal_directory)
{
    char tmpDir[] = "/tmp/testdirXXXXXX";
    char* dirName = mkdtemp(tmpDir);
    ASSERT_NE(dirName, nullptr);
    
    EXPECT_TRUE(DirectoryExists(dirName));
    rmdir(dirName);
}

TEST(Utils_Func, DirectoryExists_file_instead_of_dir)
{
    char tmpFile[] = "/tmp/testfileXXXXXX";
    int fd = mkstemp(tmpFile);
    ASSERT_NE(fd, -1);
    close(fd);
    
    EXPECT_FALSE(DirectoryExists(tmpFile));
    unlink(tmpFile);
}

TEST(Utils_Func, DirectoryExists_symlink_to_dir)
{
    char tmpDir[] = "/tmp/testdirXXXXXX";
    char* dirName = mkdtemp(tmpDir);
    ASSERT_NE(dirName, nullptr);
    
    const std::string symlinkPath = std::string(dirName) + "_symlink";
    ASSERT_EQ(symlink(dirName, symlinkPath.c_str()), 0);
    
    EXPECT_TRUE(DirectoryExists(symlinkPath));
    
    unlink(symlinkPath.c_str());
    rmdir(dirName);
}

TEST(Utils_Func, DirectoryExists_invalid_path)
{
    std::string invalidPath = "/tmp/nonexistent_" + std::to_string(getpid());
    EXPECT_FALSE(DirectoryExists(invalidPath));
}

TEST(Utils_Func, CheckDirectory_path_exists_when_exist_OK_false)
{
    char tmpDir[] = "/tmp/testdirXXXXXX";
    char* dirName = mkdtemp(tmpDir);
    ASSERT_NE(dirName, nullptr);
    bool result = Utils::CheckDirectory(dirName, false);
    EXPECT_FALSE(result);
    rmdir(dirName);
}

TEST(Utils_Func, ValidateCsvString_empty_input)
{
    const std::string emptyStr("");
    bool result = Utils::ValidateCsvString(emptyStr);
    EXPECT_TRUE(result);
}