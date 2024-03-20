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

    atb::Probe::Tensor testInTensor1;
    testInTensor1.dype = "ACL_FLOAT16";
    testInTensor1.format = "ACL_FORMAT_ND";
    testInTensor1.shape = "65024,4096";
    testInTensor1.path = "";
    atb::Probe::Tensor testInTensor2;
    testInTensor2.dype = "ACL_INT64";
    testInTensor2.format = "ACL_FORMAT_ND";
    testInTensor2.shape = "8,1024";
    testInTensor2.path = "";
    atb::Probe::Tensor testOutTensor;
    testOutTensor.dype = "ACL_FLOAT16";
    testOutTensor.format = "ACL_FORMAT_ND";
    testOutTensor.shape = "8,1024,4096";
    testOutTensor.path = "";

    std::vector<atb::Probe::Tensor> inTensors;
    inTensors.push_back(testInTensor1);
    inTensors.push_back(testInTensor2);

    std::vector<atb::Probe::Tensor> outTensors;
    outTensors.push_back(testOutTensor);

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
    const std::string headStr = "CaseNum|CaseName|OpName|OpParam|InNum|InDType|InFormat|InShape|\
OutNum|OutDType|OutFormat|OutShape|DataGenType|DataGenRange|InTensorFile|OutTensorFile|TestType|TestLevel|FromModel|\
SocVersion|ExpectedError";
    const std::string contentStr = "1|TestOperation1|TestOperation|{\"testParam1\":0,\"testParam2\":\
[0,0,0],\"testParam3\":{\"testParam4\":9.99}}|2|ACL_FLOAT16;ACL_INT64|ACL_FORMAT_ND;ACL_FORMAT_ND|65024,4096;8,1024|1|\
ACL_FLOAT16|ACL_FORMAT_ND|8,1024,4096|customize";

    DeleteFile(outPath);
    ReportIOTensorTest(executeCount, testType);
    EXPECT_TRUE(IfFileExists(outPath));
    EXPECT_TRUE(CheckFileContainsString(outPath, headStr));
    EXPECT_TRUE(CheckFileContainsString(outPath, contentStr));
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
    const std::string headStr = "CaseNum|CaseName|OpName|OpParam|InNum|InDType|InFormat|InShape|\
OutNum|OutDType|OutFormat|OutShape|DataGenType|DataGenRange|InTensorFile|OutTensorFile|TestType|TestLevel|FromModel|\
SocVersion|ExpectedError";
    const std::string contentStr = "1|TestOperation1|TestOperation|{\"testParam1\":0,\"testParam2\":\
[0,0,0],\"testParam3\":{\"testParam4\":9.99}}|2|ACL_FLOAT16;ACL_INT64|ACL_FORMAT_ND;ACL_FORMAT_ND|65024,4096;8,1024|1|\
ACL_FLOAT16|ACL_FORMAT_ND|8,1024,4096|customize";

    DeleteFile(outPath);
    ReportIOTensorTest(executeCount, testType);
    EXPECT_TRUE(IfFileExists(outPath));
    EXPECT_TRUE(CheckFileContainsString(outPath, headStr));
    EXPECT_TRUE(CheckFileContainsString(outPath, contentStr));
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

TEST(atb_Probe, IsSaveParam_001)
{
    EXPECT_TRUE(atb::Probe::IsSaveParam());
}

TEST(atb_Probe, SaveParamTest)
{
    std::string paramJson = "fake_param";
    std::string fileName = "fake_param.json";

    atb::Probe::SaveParam(paramJson, fileName);
    setenv("ATB_OUTPUT_DIR", "./tmp/", 0);

    std::string paramDir = "./tmp/ait_dump/tensors/fake_param.json";
    std::ifstream paramFile(paramDir);
    EXPECT_TRUE(paramFile.is_open());

    DeleteFile(paramDir);
}
/*
    ait llm --type
*/
TEST(atb_probe, IsOverflowEnabled)
{
    setenv("ATB_CHECK_TYPE", "1", 1);
    EXPECT_TRUE(atb::Probe::IsOverflowCheck()) << "ATB_CHECK_TYPE (which is '1') should result in TRUE, got FALSE.";
    setenv("ATB_CHECK_TYPE", "123", 1);
    EXPECT_TRUE(atb::Probe::IsOverflowCheck()) << "ATB_CHECK_TYPE (which is '123') should result in TRUE, got FALSE.";
    setenv("ATB_CHECK_TYPE", "231", 1);
    EXPECT_TRUE(atb::Probe::IsOverflowCheck()) << "ATB_CHECK_TYPE (which is '231') should result in TRUE, got FALSE.";

    unsetenv("ATB_CHECK_TYPE");
    EXPECT_FALSE(atb::Probe::IsOverflowCheck()) << "ATB_CHECK_TYPE (which is '') should result in FALSE, got TRUE.";
    setenv("ATB_CHECK_TYPE", "23", 1);
    EXPECT_FALSE(atb::Probe::IsOverflowCheck()) << "ATB_CHECK_TYPE (which is '23') should result in FALSE, got TRUE.";
}
/*
    ait llm --exit
*/
TEST(atb_probe, HandlesExitFlag)
{
    setenv("ATB_EXIT", "1", 1);
    EXPECT_TRUE(atb::Probe::IsOverflowStop()) << "ATB_EXIT (which is '1') should result in TRUE, got FALSE.";

    unsetenv("ATB_EXIT");
    EXPECT_FALSE(atb::Probe::IsOverflowStop()) << "ATB_EXIT (which is nullptr) should result in TRUE, got FALSE.";
    setenv("ATB_EXIT", "0", 1);
    EXPECT_FALSE(atb::Probe::IsOverflowStop()) << "ATB_EXIT (which is '0') should result in FALSE, got TRUE.";
    setenv("ATB_EXIT", "", 1);
    EXPECT_FALSE(atb::Probe::IsOverflowStop()) << "ATB_EXIT (which is '') should result in FALSE, got TRUE.";
    setenv("ATB_EXIT", "2", 1);
    EXPECT_FALSE(atb::Probe::IsOverflowStop()) << "ATB_EXIT (which is '2') should result in FALSE, got TRUE.";
}
/*
    ait llm --output
*/
TEST(atb_probe, HandlesEmptyKernelPath)
{
    EXPECT_NO_THROW(atb::Probe::ReportOverflowKernel("")) << "Empty Kernel Path should not result in any error.";
}

TEST(atb_probe, HandlesEmptyOutputDir)
{
    const std::string kernelPath("Some operators got some errors.");
    unsetenv("ATB_OUTPUT_DIR");
    EXPECT_NO_THROW(atb::Probe::ReportOverflowKernel(kernelPath)) << "AIT_OUTPUT_DIR (which is nullptr)" \
                                                                     "should not result in any error.";
}

TEST(atb_probe, HandlesOutputFile)
{
    const std::string kernelPath("a");
    setenv("ATB_OUTPUT_DIR", "./", 1);
    ASSERT_NO_THROW(atb::Probe::ReportOverflowKernel(kernelPath)) << "invoking ReportOverflowKernel normally" \
                                                                     "should not return any error.";

    const std::string pidID(std::to_string(getpid()));
    const std::string fileName("ait_overflow_res_" + pidID + ".txt");
    const std::string outPath("./" + fileName);

    std::ifstream ifs(outPath);
    ASSERT_TRUE(ifs.is_open()) << "ReportOverflowKernel should have created a file, but not found.";

    std::string content;
    while (getline(ifs, content)) {
        EXPECT_EQ(content, "Overflow detected! Operator name: a") << "The error information should be the same, " \
                                                                     "but got different.";
    }

    ifs.close();
    DeleteFile(outPath);
}