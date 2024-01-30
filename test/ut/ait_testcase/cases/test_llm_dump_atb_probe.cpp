/*
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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


#include <fstream>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "atb_probe.h"
#include "tools.h"


static void ReportIOTensorTest(const size_t &executeCount, const std::string &testType)
{
    const std::string opName = "TestOperation";
    const std::string opParam = "{\"testParam1\":0,\"testParam2\":[0,0,0],\"testParam3\":{\"testParam4\":9.99}}";

    atb::Probe::Tensor testInTentor1;
    testInTentor1.dype = "ACL_FLOAT16";
    testInTentor1.format = "ACL_FORMAT_ND";
    testInTentor1.shape = "65024,4096";
    atb::Probe::Tensor testInTentor2;
    testInTentor2.dype = "ACL_INT64";
    testInTentor2.format = "ACL_FORMAT_ND";
    testInTentor2.shape = "8,1024";
    atb::Probe::Tensor testOutTentor;
    testOutTentor.dype = "ACL_FLOAT16";
    testOutTentor.format = "ACL_FORMAT_ND";
    testOutTentor.shape = "8,1024,4096";

    std::vector<atb::Probe::Tensor> inTensors;
    inTensors.push_back(testInTentor1);
    inTensors.push_back(testInTentor2);

    std::vector<atb::Probe::Tensor> outTensors;
    outTensors.push_back(testOutTentor);

    if (testType == "op") {
        atb::Probe::ReportOperationIOTensor(executeCount, opName, opParam, inTensors, outTensors);
    } else {
        atb::Probe::ReportKernelIOTensor(executeCount, opName, opParam, inTensors, outTensors);
    }

    return;
}

static void ReportOperationStatisticTest(const size_t &executeCount)
{
    const std::string opName = "TestOperation";
    std::string setupSt = "totalTime:535801, streamSyncTime:0, \
    tillingCopyTime:69, kernelExecuteTime:416";
    std::string executeSt = "totalTime:188, runnerSetupTime:115, \
    runnerFillHostTilingTime:40, setupTotalCount:1, setupCacheHitCount:0, setupCacheMissCount:1, \
    opsInitLuanchBufferTime:0, tilingCacheHitCount:1, tilingCacheMissCount:0, tilingLocalCacheHitCount:1, \
    tilingGlobalCacheHitCount:0, kernelCacheGetTilingTime:20, kernelCacheAddTilingTime:0, \
    kernelCacheCompareRunInfoTime:10, kernelCacheGetRunInfoTime:6";
    
    atb::Probe::ReportOperationSetupStatistic(executeCount, opName, setupSt);
    atb::Probe::ReportOperationExecuteStatistic(executeCount, opName, executeSt);

    return;
}

TEST(atb_Probe, IsSaveChild_001)
{
    EXPECT_FALSE(atb::Probe::IsSaveChild());
}

TEST(atb_Probe, ReportOperationIOTensorEnable_001)
{
    setenv("ATB_DUMP_TYPE", "tensor", 1);
    EXPECT_FALSE(atb::Probe::ReportOperationIOTensorEnable());
}


TEST(atb_Probe, ReportOperationIOTensorEnable_002)
{
    setenv("ATB_DUMP_TYPE", "op", 1);
    EXPECT_TRUE(atb::Probe::ReportOperationIOTensorEnable());
}

TEST(atb_Probe, ReportOperationIOTensor_001)
{
    setenv("ATB_OUTPUT_DIR", "./tmp/", 0);
    const size_t executeCount = 0;
    const std::string pid = std::to_string(GetCurrentProcessId());
    const std::string fPath =
        "ait_dump/operation_io_tensors/" + pid + "/operation_tensors_" + std::to_string(executeCount) + ".csv";
    const std::string outPath = "./tmp/" + fPath;
    const std::string testType = "op";

    DeleteFile(outPath);
    ReportIOTensorTest(executeCount, testType);
    EXPECT_TRUE(IfFileExists(outPath));
}

TEST(atb_Probe, ReportKernelIOTensorEnable_001)
{
    setenv("ATB_DUMP_TYPE", "tensor", 1);
    EXPECT_FALSE(atb::Probe::ReportKernelIOTensorEnable());
}


TEST(atb_Probe, ReportKernelIOTensorEnable_002)
{
    setenv("ATB_DUMP_TYPE", "kernel", 1);
    EXPECT_TRUE(atb::Probe::ReportKernelIOTensorEnable());
}

TEST(atb_Probe, ReportKernelIOTensor_001)
{
    setenv("ATB_OUTPUT_DIR", "./tmp/", 0);
    const size_t executeCount = 0;
    const std::string pid = std::to_string(GetCurrentProcessId());
    const std::string fPath =
        "ait_dump/kernel_io_tensors/" + pid + "/kernel_tensors_" + std::to_string(executeCount) + ".csv";
    const std::string outPath = "./tmp/" + fPath;
    const std::string testType = "kernel";

    DeleteFile(outPath);
    ReportIOTensorTest(executeCount, testType);
    EXPECT_TRUE(IfFileExists(outPath));
}

TEST(atb_Probe, ReportOperationStatisticEnable_001)
{
    setenv("ATB_DUMP_TYPE", "tensor", 1);
    EXPECT_FALSE(atb::Probe::ReportOperationStatisticEnable());
}

TEST(atb_Probe, ReportOperationStatisticEnable_002)
{
    setenv("ATB_DUMP_TYPE", "cpu_profiling", 1);
    EXPECT_TRUE(atb::Probe::ReportOperationStatisticEnable());
}

TEST(atb_Probe, ReportOperationStatisticTest_001)
{
    const size_t executeCount = 0;
    const std::string pid = std::to_string(GetCurrentProcessId());
    setenv("ATB_CUR_PID", pid.c_str(), 0);
    setenv("ATB_OUTPUT_DIR", "./tmp/", 0);
    const std::string fPath =
        "ait_dump/cpu_profiling/" + pid + "/operation_statistic_" + std::to_string(executeCount) + ".txt";
    const std::string outPath = "./tmp/" + fPath;
    const std::string testType = "cpu_profiling";
 
    DeleteFile(outPath);
    ReportOperationStatisticTest(executeCount);
    EXPECT_TRUE(IfFileExists(outPath));
    EXPECT_TRUE(CheckFileContainsString(outPath, "[TestOperation]:totalTime:535801, streamSyncTime:0, \
    tillingCopyTime:69, kernelExecuteTime:416"));
    EXPECT_TRUE(CheckFileContainsString(outPath, "[TestOperation]:totalTime:188, runnerSetupTime:115, \
    runnerFillHostTilingTime:40, setupTotalCount:1, setupCacheHitCount:0, setupCacheMissCount:1, \
    opsInitLuanchBufferTime:0, tilingCacheHitCount:1, tilingCacheMissCount:0, tilingLocalCacheHitCount:1, \
    tilingGlobalCacheHitCount:0, kernelCacheGetTilingTime:20, kernelCacheAddTilingTime:0, \
    kernelCacheCompareRunInfoTime:10, kernelCacheGetRunInfoTime:6"));
}