/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Create Date: 2025
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

#include <gtest/gtest.h>
#include <cstdlib>
#include <string>

#include "ait_logger.h"
#include "env_var_guard.h"

using namespace ait;

TEST(ait_logger_Func, Logger_EnvNotSet_DefaultToINFO)
{
    testing::internal::CaptureStdout();
    EnvVarGuard envGuard("ATB_AIT_LOG_LEVEL");
    unsetenv("ATB_AIT_LOG_LEVEL");

    // DEBUG不应输出，INFO应输出
    AIT_LOG_DEBUG("Debug message");
    AIT_LOG_INFO("Info message");

    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output.find("[DEBUG]"), std::string::npos);
    ASSERT_NE(output.find("[INFO]"), std::string::npos);
}

TEST(ait_logger_Func, Logger_AllLevelOutputWhenLogLevelDEBUG)
{
    testing::internal::CaptureStdout();
    EnvVarGuard envGuard("ATB_AIT_LOG_LEVEL");
    setenv("ATB_AIT_LOG_LEVEL", "0", 1);

    AIT_LOG_DEBUG("Debug");
    AIT_LOG_CRITICAL("Critical");

    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("[DEBUG]"), std::string::npos);
    ASSERT_NE(output.find("[CRITICAL]"), std::string::npos);
}

TEST(ait_logger_Func, Logger_LevelFiltering)
{
    testing::internal::CaptureStdout();
    EnvVarGuard envGuard("ATB_AIT_LOG_LEVEL");
    setenv("ATB_AIT_LOG_LEVEL", "3", 1); // ERROR级别

    AIT_LOG_WARNING("Warn");  // 不应输出（级别2 < 3）
    AIT_LOG_ERROR("Error");    // 应输出（级别3 == 3）
    AIT_LOG_FATAL("Fatal");    // 应输出（级别4 > 3）

    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output.find("[WARNING]"), std::string::npos);
    ASSERT_NE(output.find("[ERROR]"), std::string::npos);
    ASSERT_NE(output.find("[FATAL]"), std::string::npos);
}

TEST(ait_logger_Func, Logger_FilenameFormat)
{
    testing::internal::CaptureStdout();
    EnvVarGuard envGuard("ATB_AIT_LOG_LEVEL");
    setenv("ATB_AIT_LOG_LEVEL", "0", 1);

    const int line = __LINE__ + 1; // 记录下一行行号
    AIT_LOG_INFO("Check filename");

    std::string output = testing::internal::GetCapturedStdout();
    std::string expectedFile = FILENAME; // 验证宏逻辑

    // 验证输出包含正确文件名和行号
    std::string expectedStr = "[" + std::string(FILENAME) +
                               "+" + std::to_string(line) + "]";
    ASSERT_NE(output.find(expectedStr), std::string::npos);
}

TEST(ait_logger_Func, Logger_AllLevelNamesMapped)
{
    testing::internal::CaptureStdout();
    EnvVarGuard envGuard("ATB_AIT_LOG_LEVEL");
    setenv("ATB_AIT_LOG_LEVEL", "0", 1);

    AIT_LOG_DEBUG("0");
    AIT_LOG_INFO("1");
    AIT_LOG_WARNING("2");
    AIT_LOG_ERROR("3");
    AIT_LOG_FATAL("4");
    AIT_LOG_CRITICAL("5");

    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("[DEBUG]"), std::string::npos);
    ASSERT_NE(output.find("[INFO]"), std::string::npos);
    ASSERT_NE(output.find("[WARNING]"), std::string::npos);
    ASSERT_NE(output.find("[ERROR]"), std::string::npos);
    ASSERT_NE(output.find("[FATAL]"), std::string::npos);
    ASSERT_NE(output.find("[CRITICAL]"), std::string::npos);
}

TEST(ait_logger_Func, Logger_BoundaryCondition_CRITICAL)
{
    testing::internal::CaptureStdout();
    EnvVarGuard envGuard("ATB_AIT_LOG_LEVEL");
    setenv("ATB_AIT_LOG_LEVEL", "5", 1);

    AIT_LOG_CRITICAL("Critical");  // 应输出（5 == 5）
    AIT_LOG_FATAL("Fatal");        // 不应输出（4 < 5）

    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_NE(output.find("[CRITICAL]"), std::string::npos);
    ASSERT_EQ(output.find("[FATAL]"), std::string::npos);
}

TEST(ait_logger_Func, Logger_BoundaryCondition_ERRORINPUT)
{
    testing::internal::CaptureStdout();
    EnvVarGuard envGuard("ATB_AIT_LOG_LEVEL");
    setenv("ATB_AIT_LOG_LEVEL", "aaa", 1);

    AIT_LOG_DEBUG("0");

    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output.find("[DEBUG]"), std::string::npos);
    ASSERT_NE(output.find("[WARNING]"), std::string::npos);
}