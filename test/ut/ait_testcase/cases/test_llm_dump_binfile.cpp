/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
 * Create Date: 2023
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

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "bin_file.h"

// 测试基类
namespace {
class TestLLMDumpBinFile : public testing::Test {
protected:
    void SetUp() override
    {
        // 创建临时测试文件
        tmp_file_ = "/tmp/test_bin_file";
    }

    void TearDown() override
    {
        // 清理临时文件
        std::remove(tmp_file_.c_str());
    }

    std::string tmp_file_;
    FileSystem::BinFile bin_file_;
};
}

/******************​ AddAttr 测试 ​******************/
TEST_F(TestLLMDumpBinFile, AddAttr_DuplicateName)
{
    bin_file_.AddAttr("test", "value");
    EXPECT_FALSE(bin_file_.AddAttr("test", "new_value"));
}

TEST_F(TestLLMDumpBinFile, AddAttr_NewAttribute)
{
    EXPECT_TRUE(bin_file_.AddAttr("new_attr", "value"));
}

/******************​ Write 测试 ​******************/
TEST_F(TestLLMDumpBinFile, Write_FileOpenFailed)
{
    EXPECT_FALSE(bin_file_.Write("/invalid/path/file.bin"));
}

TEST_F(TestLLMDumpBinFile, Write_EmptyDataSuccess)
{
    EXPECT_TRUE(bin_file_.Write(tmp_file_));
}

TEST_F(TestLLMDumpBinFile, Write_WithBinaryData)
{
    const char data[] = {0x01, 0x02, 0x03};
    bin_file_.AddObject("obj1", data, sizeof(data));
    EXPECT_TRUE(bin_file_.Write(tmp_file_));
}

/******************​ AddObject 测试 ​******************/
TEST_F(TestLLMDumpBinFile, AddObject_NullBuffer)
{
    const int binarySize = 10;
    EXPECT_FALSE(bin_file_.AddObject("null_obj", nullptr, binarySize));
}

TEST_F(TestLLMDumpBinFile, AddObject_DuplicateName)
{
    const char data[] = {0x01};
    bin_file_.AddObject("dup_obj", data, sizeof(data));
    EXPECT_FALSE(bin_file_.AddObject("dup_obj", data, sizeof(data)));
}

/******************​ CalcHash 测试 ​******************/
TEST_F(TestLLMDumpBinFile, CalcHash_EmptyData)
{
    auto hash = bin_file_.CalcHash();
    EXPECT_NE(hash, 0);
}

TEST_F(TestLLMDumpBinFile, CalcHash_AlignedData)
{
    const uint8_t data[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    bin_file_.AddObject("aligned", data, sizeof(data));
    auto hash = bin_file_.CalcHash();
    EXPECT_NE(hash, 0);
}

TEST_F(TestLLMDumpBinFile, CalcHash_UnalignedData)
{
    const uint8_t data[5] = {0x01, 0x02, 0x03, 0x04, 0x05};
    bin_file_.AddObject("unaligned", data, sizeof(data));
    auto hash = bin_file_.CalcHash();
    EXPECT_NE(hash, 0);
}

TEST_F(TestLLMDumpBinFile, CalcHash_VerifyRestoreBuffer)
{
    const uint8_t data[5] = {0x01};
    bin_file_.AddObject("temp", data, sizeof(data));
    size_t original_size = bin_file_.binariesBuffer_.size();
    bin_file_.CalcHash();
    EXPECT_EQ(bin_file_.binariesBuffer_.size(), original_size);
}

/******************​ WriteAttr 测试 ​******************/
TEST_F(TestLLMDumpBinFile, WriteAttr_ContentCheck)
{
    std::ofstream test_stream(tmp_file_);
    bin_file_.WriteAttr(test_stream, "test_key", "test_value");
    test_stream.close();

    std::ifstream in(tmp_file_);
    std::string line;
    std::getline(in, line);
    EXPECT_EQ(line, "test_key=test_value");
}
