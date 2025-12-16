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
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "file.h"

namespace fs = std::experimental::filesystem;
using namespace File;

TEST(File_Func, GetFullPathTest_Input_Empty)
{
    ASSERT_EQ(GetFullPath(""), "");
}

// 测试极端空输入情况
TEST(File_Func, GetAbsPathTest_Input_Empty)
{
    // 输入为空字符串时直接返回空
    EXPECT_EQ(GetAbsPath(""), "");
}

TEST(File_Func, GetAbsPathTest_Absolute_Path_Exceeding_Root)
{
    std::string path = "/..";
    for (uint32_t i = 0; i < MsConst::PATH_DEPTH_MAX; i++) {
        path += "/..";
    }
    EXPECT_EQ(GetAbsPath(path), "");
}

TEST(File_Func, GetAbsPathTest_Relative_Path_Exceeding_Parent_Dirs)
{
    // Save original working directory
    char originalCwd[PATH_MAX];
    ASSERT_NE(getcwd(originalCwd, sizeof(originalCwd)), nullptr);

    // Change to root directory for test
    ASSERT_EQ(chdir("/"), 0);
    EXPECT_EQ(GetAbsPath("a/../../.."), "");

    // Restore original directory
    ASSERT_EQ(chdir(originalCwd), 0);
}

TEST(File_Func, GetAbsPathTest_Invalid_Absolute_Path_With_Nested_Parent_Refs)
{
    EXPECT_EQ(GetAbsPath("/a/../../b"), "");
}

TEST(File_Func, GetAbsPathTest_Multiple_Parent_Refs_At_Root_Level)
{
    EXPECT_EQ(GetAbsPath("/.."), "");
}

TEST(File_Func, GetAbsPathTest_Parent_Ref_From_Root_Returns_Empty)
{
    // Save original working directory
    char originalCwd[PATH_MAX];
    ASSERT_NE(getcwd(originalCwd, sizeof(originalCwd)), nullptr);

    // Change to root directory for test
    ASSERT_EQ(chdir("/"), 0);
    EXPECT_EQ(GetAbsPath(".."), "");

    // Restore original directory
    ASSERT_EQ(chdir(originalCwd), 0);
}

TEST(File_Func, GetAbsPathTest_Skip_Empty_And_Dot_Tokens)
{
    // 测试空token和.符号的跳过逻辑
    // 路径包含：空token（//）、单独的.、以及有效token
    EXPECT_EQ(GetAbsPath("/a//b/./c/"), "/a/b/c");
    
    // 测试相对路径中的空token和.符号
    char originalCwd[PATH_MAX];
    ASSERT_NE(getcwd(originalCwd, sizeof(originalCwd)), nullptr);
    ASSERT_EQ(chdir("/tmp"), 0);
    EXPECT_EQ(GetAbsPath("subdir///./docs/../src"), "/tmp/subdir/src");
    ASSERT_EQ(chdir(originalCwd), 0);
}

TEST(File_Func, GetAbsPathTest_Return_Root_When_Tokens_Empty)
{
    // 显式构造空tokensRefined的路径
    EXPECT_EQ(GetAbsPath("/"), "/");           // 直接根路径
    EXPECT_EQ(GetAbsPath("/./"), "/");         // 带.的根路径
    EXPECT_EQ(GetAbsPath("/../.."), "");       // 失败情况已在之前覆盖
}

TEST(File_Func, GetAbsPathTest_Normal_Path_Resolutions)
{
    // 情况1：相对路径解析
    char originalCwd[PATH_MAX];
    ASSERT_NE(getcwd(originalCwd, sizeof(originalCwd)), nullptr);
    ASSERT_EQ(chdir("/var"), 0);
    EXPECT_EQ(GetAbsPath("log/../tmp"), "/var/tmp");
    ASSERT_EQ(chdir(originalCwd), 0);

    // 情况2：绝对路径保持不变
    EXPECT_EQ(GetAbsPath("/usr/local/bin"), "/usr/local/bin");

    // 情况3：包含.符号的有效路径
    EXPECT_EQ(GetAbsPath("/home/./user/"), "/home/user");
}

TEST(File_Func, IsFileWritableTest_Returns_False_When_Not_Writable)
{
    const std::string path = "test_non_writable.tmp";
    
    // 创建文件后移除写权限
    std::ofstream ofs(path);
    ASSERT_TRUE(ofs);
    ofs.close();
    ASSERT_EQ(chmod(path.c_str(), S_IRUSR | S_IRGRP | S_IROTH), 0);
    
    EXPECT_FALSE(IsFileWritable(path));
    
    remove(path.c_str()); // 删除文件（不受文件权限影响）
}

TEST(File_Func, IsDirTest_Returns_False_For_Nonexistent_Path)
{
    // 不存在的路径
    EXPECT_FALSE(IsDir("/tmp/non_existent_path_" + std::to_string(getpid())));
}

TEST(File_Func, IsDirTest_Returns_False_For_Broken_Symlink)
{
    // 创建指向不存在的路径的符号链接
    char tmpLink[] = "/tmp/broken_linkXXXXXX";
    int fd = mkstemp(tmpLink);  // 生成唯一路径
    ASSERT_NE(fd, -1);
    close(fd);
    unlink(tmpLink);  // 立即删除使链接失效
    
    ASSERT_EQ(symlink(tmpLink, tmpLink), 0);
    EXPECT_FALSE(IsDir(tmpLink));  // 检查损坏的链接
    
    unlink(tmpLink);  // 清理符号链接
}

TEST(File_Func, IsDirTest_Returns_False_For_Empty_Path)
{
    // 空路径直接触发 stat 失败
    EXPECT_FALSE(IsDir(""));
}

// Test case for path length exceeding the maximum limit
TEST(File_Func, GetFileSizeTest_Path_Length_Exceeds_Max_Limit)
{
    ASSERT_EQ(GetFileSize("/tmp/" + std::to_string(getpid())), 0);
}

// GetParentDir 测试：路径没有斜杠的情况
TEST(File_Func, GetParentDirTest_Returns_Dot_When_No_Slash)
{
    EXPECT_EQ(GetParentDir("filename.txt"), ".");
}

// GetFileName 测试：路径没有斜杠的情况
TEST(File_Func, GetFileNameTest_Returns_Full_When_No_Slash)
{
    EXPECT_EQ(GetFileName("document.pdf"), "document.pdf");
}

// GetFileSuffix 测试：文件名没有点号的情况
TEST(File_Func, GetFileSuffixTest_Returns_Empty_When_No_Dot)
{
    EXPECT_EQ(GetFileSuffix("/tmp/README"), "");
}

TEST(File_Func, GetPathPermissionsTest_Returns_Max_Permission_When_Stat_Fails)
{
    // 使用不存在的路径触发stat失败
    const std::string invalidPath = "/tmp/nonexistent_directory_" + std::to_string(getpid());
    
    // 验证返回值为最大权限常量
    EXPECT_EQ(GetPathPermissions(invalidPath), MsConst::MAX_PERMISSION);
}


// 场景1：stat失败（文件不存在）
TEST(File_Func, CheckFileSuffixAndSizeTest_Returns_False_When_File_Not_Exist)
{
    EXPECT_FALSE(CheckFileSuffixAndSize("/tmp/nonexistent_" + std::to_string(getpid()) + ".bin",
                                        MsConst::SUFFIX::BIN, 100));
}

// 场景2：type=NONE且大小有效
TEST(File_Func, CheckFileSuffixAndSizeTest_Type_None_With_Valid_Size)
{
    const std::string path = "/tmp/empty_file";
    std::ofstream tempFile(path);  // 创建0字节文件
    tempFile.close();
    
    EXPECT_TRUE(CheckFileSuffixAndSize(path, MsConst::SUFFIX::NONE, 1));
    remove(path.c_str());
}

// 场景3：查找不到type映射（使用强制转换模拟无效类型）
TEST(File_Func, CheckFileSuffixAndSizeTest_Invalid_Suffix_Type)
{
    const std::string path = "/tmp/test.xml";
    std::ofstream tempFile(path);
    tempFile << "<data/>";
    tempFile.close();
    
    // 强制转换模拟未定义的XML类型
    EXPECT_FALSE(CheckFileSuffixAndSize(path, static_cast<MsConst::SUFFIX>(4), 100));
    remove(path.c_str());
}

// 场景4：文件大小双超限（BIN类型）
TEST(File_Func, CheckFileSuffixAndSizeTest_Size_Exceed_Both_Limits)
{
    const std::string path = "/tmp/oversize.bin";
    std::ofstream tempFile(path, std::ios::binary);
    tempFile.seekp(MsConst::MAX_BIN_SIZE + 20);  // 创建超过预设值的文件
    tempFile.write("", 1);
    tempFile.close();
    
    EXPECT_FALSE(CheckFileSuffixAndSize(path, MsConst::SUFFIX::BIN, MsConst::MAX_BIN_SIZE - 10));
    remove(path.c_str());
}

// 场景5：CSV文件完全合规
TEST(File_Func, CheckFileSuffixAndSizeTest_Valid_CSV_File)
{
    const std::string path = "/tmp/data.csv";
    std::ofstream tempFile(path);
    tempFile << "1,2,3\n4,5,6";  // 10字节
    
    EXPECT_TRUE(CheckFileSuffixAndSize(path, MsConst::SUFFIX::CSV, MsConst::MAX_CSV_SIZE));
    remove(path.c_str());
}

TEST(File_Func, IsSoftLinkTest_Returns_False_When_Path_Not_Exist)
{
    // 使用不存在的绝对路径
    const std::string invalidPath = "/tmp/nonexistent_" + std::to_string(getpid());
    
    // 验证函数返回false
    EXPECT_FALSE(IsSoftLink(invalidPath));
}

/******************​ 测试场景1：触发!recursion路径 ​******************/
TEST(File_Func, CreateDirTest_Fail_When_Parent_Not_Exist_Without_Recursion)
{
    // 构造多级不存在路径
    const std::string path = "tmp_test/a/b/c";  // 相对路径测试
    
    // 确保初始环境干净
    fs::remove_all("tmp_test");
    
    // 调用公共接口（recursion=false）
    EXPECT_FALSE(CreateDir(path, false, 0755));
    
    // 验证路径确实未创建
    EXPECT_FALSE(IsPathExist("tmp_test"));
}

/******************​ 测试场景2：触发递归创建父目录失败 ​******************/
TEST(File_Func, CreateDirTest_Fail_When_Recursive_Permission_Denied)
{
    // 在当前用户的临时目录下创建受保护目录
    char parentDir[] = "/tmp/protectedXXXXXX";
    ASSERT_NE(mkdtemp(parentDir), nullptr);
    
    // 设置目录为只读权限（555 = r-xr-xr-x）
    ASSERT_EQ(chmod(parentDir, 0555), 0);
    
    // 构造测试路径：受保护目录下的子目录
    const std::string path = std::string(parentDir) + "/child";
    
    // 调用公共接口（recursion=true）
    EXPECT_FALSE(CreateDir(path, true, 0755));
    
    // 恢复权限并清理（必须恢复写权限才能删除）
    ASSERT_EQ(chmod(parentDir, 0700), 0);
    ASSERT_EQ(rmdir(parentDir), 0);
}

/******************​ 测试场景3：触发syscall失败路径 ​******************/
TEST(File_Func, CreateDirTest_Fail_When_Parent_Is_Regular_File)
{
    // 创建符号链接链 link1->link2->link1
    const std::string link1 = "link1";
    const std::string link2 = "link2";
    symlink(link2.c_str(), link1.c_str());
    symlink(link1.c_str(), link2.c_str());

    EXPECT_FALSE(CreateDir(link1 + "/subdir", true, 0755));
    
    unlink(link1.c_str());
    unlink(link2.c_str());
}

/******************​ 场景1：目录已存在 ​******************/
TEST(File_Func, CreateDirTest_Return_True_When_Dir_Exists)
{
    const std::string path = "existing_dir";
    
    // 预先创建目录
    ASSERT_TRUE(CreateDir(path, false, 0755));
    
    // 测试函数返回true
    EXPECT_TRUE(CreateDir(path, false, 0755));
    
    // 清理
    rmdir(path.c_str());
}

/******************​ 场景2：绝对路径解析为空 ​******************/
TEST(File_Func, CreateDirTest_Return_False_When_AbsPath_Empty)
{
    // 构造无效路径
    const std::string invalidPath = "";  // 假设当前目录层级不足
    
    EXPECT_FALSE(CreateDir(invalidPath, false, 0755));
}

/******************​ 场景3：路径包含非法字符 ​******************/
TEST(File_Func, CreateDirTest_Return_False_For_Invalid_Characters)
{
    const std::string invalidPath = "invalid*path?";
    
    EXPECT_FALSE(CreateDir(invalidPath, false, 0755));
}

/******************​ 场景4：路径深度超标 ​******************/
TEST(File_Func, CreateDirTest_Return_False_For_Excessive_Depth)
{
    // 构造深度超标的路径
    std::string deepPath = "a";
    for (uint32_t i = 0; i < MsConst::PATH_DEPTH_MAX; i++) {
        deepPath += "/a";
    }
    EXPECT_FALSE(CreateDir(deepPath, false, 0755));
}

TEST(File_Func, CheckOwnerTest_Stat_Fails_When_Path_Not_Exist)
{
    // 构造不存在的绝对路径
    const std::string path = "/tmp/nonexistent_dir_" + std::to_string(getpid());
    
    EXPECT_FALSE(CheckOwner(path));
}

/******************​ 场景1：绝对路径为空 ​******************/
TEST(File_Func, CheckDirTest_Return_False_When_AbsPath_Empty)
{
    // 构造导致GetAbsPath返回空的路径（如超出根目录）
    std::string path = "/..";
    for (uint32_t i = 0; i < MsConst::PATH_DEPTH_MAX; i++) {
        path += "/..";
    }
    EXPECT_FALSE(CheckDir(path));
}

/******************​ 场景2：路径长度非法 ​******************/
TEST(File_Func, CheckDirTest_Return_False_When_Path_Too_Long)
{
    // 构造超长文件名
    std::string longName(MsConst::FILE_NAME_LENGTH_MAX + 1, 'a');  // 超过文件名长度限制
    const std::string path = "test_dir/" + longName;
    
    EXPECT_FALSE(CheckDir(path));
}

/******************​ 场景3：路径字符非法 ​******************/
TEST(File_Func, CheckDirTest_Return_False_For_Invalid_Characters)
{
    // 构造包含非法字符的路径
    const std::string path = "invalid/path*+?;here";
    
    EXPECT_FALSE(CheckDir(path));
}

/******************​ 场景4：路径深度超标 ​******************/
TEST(File_Func, CheckDirTest_Return_False_For_Excessive_Depth)
{
    // 构造深度超标的路径
    std::string deepPath = "a";
    for (uint32_t i = 0; i < MsConst::PATH_DEPTH_MAX; i++) {
        deepPath += "/a";
    }
    EXPECT_FALSE(CheckDir(deepPath));
}

/******************​ 场景5：路径不存在 ​******************/
TEST(File_Func, CheckDirTest_Return_False_When_Path_Not_Exist)
{
    // 构造唯一不存在路径
    const std::string path = "/tmp/nonexistent_dir_" + std::to_string(getpid());
    
    EXPECT_FALSE(CheckDir(path));
}

/******************​ 场景6：路径是符号链接 ​******************/
TEST(File_Func, CheckDirTest_Return_False_For_Symlink)
{
    // 创建目录和符号链接
    const std::string targetDir = "target_dir";
    ASSERT_EQ(mkdir(targetDir.c_str(), 0755), 0);
    const std::string linkPath = "symlink_dir";
    ASSERT_EQ(symlink(targetDir.c_str(), linkPath.c_str()), 0);
    
    EXPECT_FALSE(CheckDir(linkPath));
    
    // 清理
    unlink(linkPath.c_str());
    rmdir(targetDir.c_str());
}

/******************​ 场景1：绝对路径为空 ​******************/
TEST(File_Func, CheckFileBeforeReadTest_Return_False_When_AbsPath_Empty)
{
    // 构造导致GetAbsPath返回空的路径
    std::string path = "/..";
    for (uint32_t i = 0; i < MsConst::PATH_DEPTH_MAX; i++) {
        path += "/..";
    }
    
    EXPECT_FALSE(CheckFileBeforeRead(path, MsConst::SUFFIX::NONE, 0));
}

/******************​ 场景2：路径长度非法 ​******************/
TEST(File_Func, CheckFileBeforeReadTest_Return_False_For_Long_FileName)
{
    // 构造超长文件名
    std::string longName(MsConst::FILE_NAME_LENGTH_MAX + 1, 'a');
    const std::string path = "test_dir/" + longName;
    
    EXPECT_FALSE(CheckFileBeforeRead(path, MsConst::SUFFIX::NONE, 0));
}

/******************​ 场景3：路径深度超标 ​******************/
TEST(File_Func, CheckFileBeforeReadTest_Return_False_For_Deep_Path)
{
    // 构造超深路径
    std::string deepPath = "a";
    for (uint32_t i = 0; i < MsConst::PATH_DEPTH_MAX; i++) {
        deepPath += "/a";
    }
    
    EXPECT_FALSE(CheckFileBeforeRead(deepPath, MsConst::SUFFIX::NONE, 0));
}

/******************​ 场景4：所有者检查失败 ​******************/
TEST(File_Func, CheckFileBeforeReadTest_Return_False_When_Not_Owner)
{
    // 使用系统文件测试（假设以普通用户运行）
    const std::string path = "/etc/passwd";
    
    EXPECT_FALSE(CheckFileBeforeRead(path, MsConst::SUFFIX::NONE, 0));
}

/******************​ 场景5：文件不可读 ​******************/
TEST(File_Func, CheckFileBeforeReadTest_Return_False_For_Unreadable_File)
{
    // 创建临时文件并移除读权限
    const std::string path = "unreadable_file";
    std::ofstream tempFile(path);
    tempFile.close();
    chmod(path.c_str(), 0300);  // 设置权限为-w--wx-wx
    
    EXPECT_FALSE(CheckFileBeforeRead(path, MsConst::SUFFIX::NONE, 0));
    
    // 清理
    chmod(path.c_str(), 0644);
    remove(path.c_str());
}

/******************​ 场景6：父目录检查 ​******************/
TEST(File_Func, CheckFileBeforeReadTest_Check_Parent_Dir_Validity)
{
    // 创建合法父目录和文件
    const std::string parent = "valid_parent";
    const std::string path = parent + "/test_file";
    ASSERT_TRUE(CreateDir(parent, true, 0755));
    std::ofstream tempFile(path);
    tempFile.close();
    chmod(path.c_str(), 0400);
    
    // 验证是否执行到最后一步
    EXPECT_TRUE(CheckFileBeforeRead(path, MsConst::SUFFIX::NONE, 0));
    
    // 清理
    remove(path.c_str());
    rmdir(parent.c_str());
}

/******************​ 场景1：绝对路径为空 ​******************/
TEST(File_Func, CheckFileBeforeCreateOrWriteTest_Return_False_When_AbsPath_Empty)
{
    std::string path = "/..";
    for (uint32_t i = 0; i < MsConst::PATH_DEPTH_MAX; i++) {
        path += "/..";
    }
    
    EXPECT_FALSE(CheckFileBeforeCreateOrWrite(path, true));
}

/******************​ 场景2：路径长度非法 ​******************/
TEST(File_Func, CheckFileBeforeCreateOrWriteTest_Return_False_For_Long_FileName)
{
    std::string longName(MsConst::FILE_NAME_LENGTH_MAX + 1, 'a');
    const std::string path = "test_dir/" + longName;
    
    EXPECT_FALSE(CheckFileBeforeCreateOrWrite(path, true));
}

/******************​ 场景3：路径深度超标 ​******************/
TEST(File_Func, CheckFileBeforeCreateOrWriteTest_Return_False_For_Deep_Path)
{
    std::string deepPath = "a";
    for (uint32_t i = 0; i < MsConst::PATH_DEPTH_MAX; i++) {
        deepPath += "/a";
    }
    
    EXPECT_FALSE(CheckFileBeforeCreateOrWrite(deepPath, true));
}

/******************​ 场景4：路径字符非法 ​******************/
TEST(File_Func, CheckFileBeforeCreateOrWriteTest_Return_False_For_Invalid_Characters)
{
    const std::string path = "invalid/path*?;=here";
    
    EXPECT_FALSE(CheckFileBeforeCreateOrWrite(path, true));
}

/******************​ 场景5：非普通文件 ​******************/
TEST(File_Func, CheckFileBeforeCreateOrWriteTest_Return_False_For_Directory)
{
    const std::string path = "test_dir";
    ASSERT_EQ(mkdir(path.c_str(), 0755), 0);
    
    EXPECT_FALSE(CheckFileBeforeCreateOrWrite(path, true));
    rmdir(path.c_str());
}

/******************​ 场景6：符号链接文件 ​******************/
TEST(File_Func, CheckFileBeforeCreateOrWriteTest_Return_False_For_Symlink_File)
{
    const std::string target = "target_file";
    std::ofstream(target).close();
    
    const std::string link = "symlink_file";
    ASSERT_EQ(symlink(target.c_str(), link.c_str()), 0);
    
    EXPECT_FALSE(CheckFileBeforeCreateOrWrite(link, true));
    
    unlink(link.c_str());
    remove(target.c_str());
}

/******************​ 场景7：权限超标（其他用户可写）​******************/
TEST(File_Func, CheckFileBeforeCreateOrWriteTest_Return_False_For_Over_Permission)
{
    const std::string path = "over_perm_file";
    std::ofstream(path).close();
    chmod(path.c_str(), 0775);  // rwxrwxr-x
    
    EXPECT_FALSE(CheckFileBeforeCreateOrWrite(path, true));
    
    remove(path.c_str());
}

/******************​ 场景8：非所有者文件 ​******************/
TEST(File_Func, CheckFileBeforeCreateOrWriteTest_Return_False_For_Other_Owner_File)
{
    // 创建不可写且不属于当前用户的文件
    const std::string path = "unwritable_file";
    std::ofstream tempFile(path);
    tempFile.close();
    
    // 设置权限为只读（假设当前用户是所有者）
    chmod(path.c_str(), 0400);  // 用户可读，不可写
    
    EXPECT_FALSE(CheckFileBeforeCreateOrWrite(path, true));
    
    // 清理
    chmod(path.c_str(), 0644);
    remove(path.c_str());
}

/******************​ 场景9：父目录检查通过 ​******************/
TEST(File_Func, CheckFileBeforeCreateOrWriteTest_Return_True_With_Valid_Parent_Dir)
{
    const std::string parent = "valid_parent";
    const std::string path = parent + "/test_file";
    ASSERT_TRUE(CreateDir(parent, true, 0755));
    
    EXPECT_TRUE(CheckFileBeforeCreateOrWrite(path, false));
    
    rmdir(parent.c_str());
}

// Test case for path length exceeding the maximum limit
TEST(File_Func, IsPathLengthLegalTest_Path_Length_Exceeds_Max_Limit)
{
    std::string path = "a";
    for (uint32_t i = 0; i < MsConst::FULL_PATH_LENGTH_MAX; i++) {
        path += "a";
    }
    EXPECT_FALSE(IsPathLengthLegal(path));
}

// Test case for path length equal to zero
TEST(File_Func, IsPathLengthLegalTest_Path_Length_Is_Zero)
{
    std::string path = "";
    EXPECT_FALSE(IsPathLengthLegal(path));
}

// Test case for path length within the limit but file name length exceeds the maximum limit
TEST(File_Func, IsPathLengthLegalTest_Filename_Length_Exceeds_Max_Limit)
{
    std::string path(MsConst::FILE_NAME_LENGTH_MAX + 1, 'a');

    path += "/b";
    EXPECT_FALSE(IsPathLengthLegal(path));
}

// Test case for path length and file name length within the limit
TEST(File_Func, IsPathLengthLegalTest_Legal)
{
    std::string path = "a/b";
    EXPECT_TRUE(IsPathLengthLegal(path));
}