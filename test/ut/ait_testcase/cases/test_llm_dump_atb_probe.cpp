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

#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "atb_probe.h"
#include "nlohmann/json.hpp"
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

TEST(atb_Probe, ReportOperationGraphEnable_TRUE)
{
    const char *value = "layer|model";
    setenv("ATB_DUMP_TYPE", value, 1);
    EXPECT_TRUE(atb::Probe::ReportOperationGraphEnable());

    value = "model";
    setenv("ATB_DUMP_TYPE", value, 1);
    EXPECT_TRUE(atb::Probe::ReportOperationGraphEnable());

    value = "layer";
    setenv("ATB_DUMP_TYPE", value, 1);
    EXPECT_TRUE(atb::Probe::ReportOperationGraphEnable());
}

using ordered_json = nlohmann::ordered_json;
TEST(atb_Probe, ReportOperationGraph)
{
    std::ifstream file("../layer_test.json");
    EXPECT_TRUE(file.is_open());
    unsetenv("ATB_OUTPUT_DIR");

    ordered_json graphNodeJson = ordered_json::parse(file);
    std::string opName = "EncoderLayer_2";
    atb::Probe::ReportOperationGraph(opName, graphNodeJson.dump());

    int32_t pid = getpid();
    std::string pidDir = "./ait_dump/layer/" + std::to_string(pid) + "/EncoderLayer_2.json";
    std::ifstream dumpFile(pidDir);
    EXPECT_TRUE(dumpFile.is_open());

    ordered_json dumpJson = ordered_json::parse(dumpFile);
    EXPECT_EQ(dumpJson["opName"].get<std::string>(), opName);

    EXPECT_EQ(dumpJson["inTensors"].size(), 12);
    EXPECT_EQ(dumpJson["internalTensors"].size(), 8);
    EXPECT_EQ(dumpJson["outTensors"].size(), 3);
    EXPECT_EQ(dumpJson["nodes"].size(), 9);
}

ordered_json g_model = {{"modelName", "EncoderModel"},
    {"inTensors", {"EncoderModel_input_0", "EncoderModel_input_1", "EncoderModel_input_2"}},
    {"outTensors", {"EncoderModel_output_0", "EncoderModel_output_1", "EncoderModel_output_2"}},
    {"internalTensors", {"EncoderModel_internal_0", "EncoderModel_internal_1", "EncoderModel_internal_2"}},
    {"weightTensors", {"EncoderModel_weight_0", "EncoderModel_weight_1", "EncoderModel_weight_2"}},
    {"nodes",
        {{
            {"opName", "EncoderLayer"},
            {"inTensors", {"EncoderModel_input_0", "EncoderModel_weight_0"}},
            {"outTensors", {"EncoderModel_output_0", "EncoderModel_internal_0", "EncoderModel_internal_1"}},
            },
            {
                {"opName", "EncoderLayer"},
                {"inTensors", {"EncoderModel_output_0", "EncoderModel_internal_0", "EncoderModel_internal_1"}},
                {"outTensors", {"EncoderModel_output_1", "EncoderModel_output_2"}},
            }}}};

TEST(atb_speed_Probe, ReportModelTopoInfo)
{
    const char *value = "layer|model";
    setenv("ATB_DUMP_TYPE", value, 1);
    unsetenv("ATB_OUTPUT_DIR");

    std::string modelName = "EncoderModel";

    EXPECT_TRUE(atb_speed::SpeedProbe::IsReportModelTopoInfo(modelName));

    std::string opName0 = "EncoderLayer_0";
    ordered_json node0 = {{"inTensorNum", 2},
        {"outTensorNum", 3},
        {"internalTensorNum", 2},
        {"opName", opName0},
        {"opType", "EncoderLayer"},
        {"outTensorNum", 3},
        {"param", ""}};

    atb::Probe::ReportOperationGraph(opName0, node0.dump());
    int32_t pid = getpid();
    std::string node0Json = "./ait_dump/layer/" + std::to_string(pid) + "/EncoderLayer_0.json";
    std::ifstream node0File(node0Json);
    EXPECT_TRUE(node0File.is_open());

    std::string opName1 = "EncoderLayer_1";
    ordered_json node1 = {{"inTensorNum", 3},
        {"outTensorNum", 2},
        {"internalTensorNum", 3},
        {"opName", opName1},
        {"opType", "EncoderLayer"},
        {"param", ""}};
    atb::Probe::ReportOperationGraph(opName1, node1.dump());
    std::string node1Json = "./ait_dump/layer/" + std::to_string(pid) + "/EncoderLayer_1.json";
    std::ifstream node1File(node1Json);
    EXPECT_TRUE(node1File.is_open());

    ordered_json model = g_model;

    atb_speed::SpeedProbe::ReportModelTopoInfo(modelName, model.dump());

    std::string pidDir = "./ait_dump/model/" + std::to_string(pid) + "/EncoderModel.json";
    std::ifstream dumpFile(pidDir);
    EXPECT_TRUE(dumpFile.is_open());

    ordered_json dumpJson = ordered_json::parse(dumpFile);
    EXPECT_EQ(dumpJson["modelName"].get<std::string>(), modelName);

    // node0
    EXPECT_EQ(dumpJson["nodes"][0]["opName"].get<std::string>(), "EncoderLayer_0");
    EXPECT_EQ(dumpJson["nodes"][0]["inTensors"][1].get<std::string>(), "EncoderModel_weight_0");
    EXPECT_EQ(dumpJson["nodes"][0]["outTensors"][2].get<std::string>(), "EncoderModel_internal_1");
    EXPECT_EQ(dumpJson["nodes"][0]["internalTensors"][0].get<std::string>(), "EncoderLayer_0_internal_0");
    EXPECT_EQ(dumpJson["nodes"][0]["internalTensors"][1].get<std::string>(), "EncoderLayer_0_internal_1");

    // node1
    EXPECT_EQ(dumpJson["nodes"][1]["opName"].get<std::string>(), "EncoderLayer_1");
    EXPECT_EQ(dumpJson["nodes"][1]["internalTensors"][2].get<std::string>(), "EncoderLayer_1_internal_2");
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