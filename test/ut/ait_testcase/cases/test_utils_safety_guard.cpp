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


#include <string>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "safety_guard.h"
#include "tools.h"

const std::string INVALID_PATH = "./invalid_path";
const std::string TEST_FILE = "./test_file.bin";
const std::string TEST_DIR = "./test_dir/";
const std::string TEST_SUB_DIR = "./test_dir/subdir";
const std::string LINK_FILE = "./link_file.bin";
const std::string ILLEGAL_CHAR_FILE = "./&AS.bin";
const std::string NOT_SUPPORT_SUFFIX_FILE = "./test.ccs";
using MsConst::SAFETY_RET;
using MsConst::OPERATE_MODE;
using MsConst::SUFFIX;

TEST(SafetyGuard_Func, CheckFileLegality_read_failed_sofelink)
{
    if (IsPathExist(TEST_FILE)) {
        ExecShellCommand("rm -f " + TEST_FILE);
    }
    if (IsPathExist(LINK_FILE)) {
        ExecShellCommand("rm -f " + LINK_FILE);
    }
    ExecShellCommand("rm -f " + LINK_FILE);
    ExecShellCommand("touch " + TEST_FILE);
    ExecShellCommand("chmod 640 " + TEST_FILE);
    ExecShellCommand("ln -s " + TEST_FILE + " " + LINK_FILE);
    SAFETY_RET ret = Utils::SafetyGuard::CheckFileLegality(LINK_FILE, OPERATE_MODE::READ);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_FILE_TO_READ_ILLEGAL);
    ExecShellCommand("rm -f " + TEST_FILE);
    ExecShellCommand("rm -f " + LINK_FILE);
}

TEST(SafetyGuard_Func, CheckFileLegality_read_failed_permission_high)
{
    if (IsPathExist(TEST_FILE)) {
        ExecShellCommand("rm -f " + TEST_FILE);
    }
    ExecShellCommand("touch " + TEST_FILE);
    ExecShellCommand("chmod 770 " + TEST_FILE);
    SAFETY_RET ret = Utils::SafetyGuard::CheckFileLegality(TEST_FILE, OPERATE_MODE::READ);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_FILE_TO_READ_ILLEGAL);
    ExecShellCommand("rm -f " + TEST_FILE);
}

TEST(SafetyGuard_Func, CheckFileLegality_read_failed_not_exist)
{
    if (IsPathExist(INVALID_PATH)) {
        ExecShellCommand("rm -f " + INVALID_PATH);
    }
    ExecShellCommand("rm -f " + INVALID_PATH);
    SAFETY_RET ret = Utils::SafetyGuard::CheckFileLegality(INVALID_PATH, OPERATE_MODE::READ);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_FILE_TO_READ_ILLEGAL);
}

TEST(SafetyGuard_Func, CheckFileLegality_read_failed_illegal_chars)
{
    SAFETY_RET ret = Utils::SafetyGuard::CheckFileLegality(ILLEGAL_CHAR_FILE, OPERATE_MODE::READ);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_FILE_TO_READ_ILLEGAL);
}

TEST(SafetyGuard_Func, CheckFileLegality_read_failed_not_a_file)
{
    if (IsPathExist(TEST_DIR)) {
        ExecShellCommand("rm -rf " + TEST_DIR);
    }
    ExecShellCommand("mkdir " + TEST_DIR);
    ExecShellCommand("chmod 750 " + TEST_DIR);
    SAFETY_RET ret = Utils::SafetyGuard::CheckFileLegality(TEST_DIR, OPERATE_MODE::READ);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_FILE_TO_READ_ILLEGAL);
    ExecShellCommand("rm -rf " + TEST_DIR);
}

TEST(SafetyGuard_Func, CheckFileLegality_read_failed_size_over_max)
{
    if (IsPathExist(TEST_FILE)) {
        ExecShellCommand("rm -f " + TEST_FILE);
    }
    ExecShellCommand("dd if=/dev/zero of=" + TEST_FILE + " bs=1024 count=1"); // create a 1MB file
    ExecShellCommand("chmod 640 " + TEST_FILE);
    SAFETY_RET ret = Utils::SafetyGuard::CheckFileLegality(TEST_FILE, OPERATE_MODE::READ, 1023);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_FILE_TO_READ_ILLEGAL);
    ExecShellCommand("rm -f " + TEST_FILE);
}

TEST(SafetyGuard_Func, CheckFileLegality_read_failed_suffix_wrong)
{
    if (IsPathExist(NOT_SUPPORT_SUFFIX_FILE)) {
        ExecShellCommand("rm -f " + NOT_SUPPORT_SUFFIX_FILE);
    }
    ExecShellCommand("touch " + NOT_SUPPORT_SUFFIX_FILE);
    ExecShellCommand("chmod 640 " + NOT_SUPPORT_SUFFIX_FILE);
    SAFETY_RET ret = Utils::SafetyGuard::CheckFileLegality(
        NOT_SUPPORT_SUFFIX_FILE, OPERATE_MODE::READ, 1023, SUFFIX::JSON);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_FILE_TO_READ_ILLEGAL);
    ExecShellCommand("rm -f " + NOT_SUPPORT_SUFFIX_FILE);
}

TEST(SafetyGuard_Func, CheckFileLegality_write_failed_exist)
{
    if (IsPathExist(TEST_FILE)) {
        ExecShellCommand("rm -f " + TEST_FILE);
    }
    ExecShellCommand("touch " + TEST_FILE);
    ExecShellCommand("chmod 640 " + TEST_FILE);
    SAFETY_RET ret = Utils::SafetyGuard::CheckFileLegality(TEST_FILE, OPERATE_MODE::WRITE);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_FILE_TO_WRITE_ILLEGAL);
    ExecShellCommand("rm -f " + TEST_FILE);
}

TEST(SafetyGuard_Func, CheckNormalStr_failed_over_len)
{
    std::string str(4097, 'a');
    SAFETY_RET ret = Utils::SafetyGuard::CheckNormalStr(str);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_STR_OVER_MAX_LEN);
}

TEST(SafetyGuard_Func, CheckNormalStr_failed_illegal_char)
{
    std::string str = "a&";
    SAFETY_RET ret = Utils::SafetyGuard::CheckNormalStr(str);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_STR_CONTAIN_ILLEGAL_CHAR);
}

TEST(SafetyGuard_Func, CreateDir_failed_create_failed)
{
    if (IsPathExist(TEST_DIR)) {
        ExecShellCommand("rm -rf " + TEST_DIR);
    }
    ExecShellCommand("mkdir " + TEST_DIR);
    ExecShellCommand("chmod 200 " + TEST_DIR);
    SAFETY_RET ret = Utils::SafetyGuard::CreateDir(TEST_SUB_DIR);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_CREATE_DIR_FAILED);
    ExecShellCommand("rm -rf " + TEST_DIR);
}

TEST(SafetyGuard_Func, CreateDir_failed_is_exist)
{
    if (IsPathExist(TEST_DIR)) {
        ExecShellCommand("rm -rf " + TEST_DIR);
    }
    ExecShellCommand("mkdir " + TEST_DIR);
    ExecShellCommand("chmod 640 " + TEST_DIR);
    SAFETY_RET ret = Utils::SafetyGuard::CreateDir(TEST_DIR);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_PATH_IS_EXIST);
    ExecShellCommand("rm -rf " + TEST_DIR);
}

TEST(SafetyGuard_Func, CreateDir_failed_check_failed)
{
    if (IsPathExist(TEST_DIR)) {
        ExecShellCommand("rm -rf " + TEST_DIR);
    }
    ExecShellCommand("mkdir " + TEST_DIR);
    ExecShellCommand("chmod 770 " + TEST_DIR);
    SAFETY_RET ret = Utils::SafetyGuard::CreateDir(TEST_DIR, MsConst::NORMAL_DIR_MODE_DEFAULT, true);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_EXIST_DIR_ILLEGAL);
    ExecShellCommand("rm -rf " + TEST_DIR);
}

TEST(SafetyGuard_Func, CreateDir_parent_check_failed)
{
    if (IsPathExist(TEST_DIR)) {
        ExecShellCommand("rm -rf " + TEST_DIR);
    }
    ExecShellCommand("mkdir " + TEST_DIR);
    ExecShellCommand("chmod 770 " + TEST_DIR);
    SAFETY_RET ret = Utils::SafetyGuard::CreateDir(TEST_SUB_DIR);
    EXPECT_TRUE(ret == SAFETY_RET::SAFE_ERR_CREATE_DIR_FAILED);
    ExecShellCommand("rm -rf " + TEST_DIR);
}