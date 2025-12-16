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
#include <vector>
#include <sys/statvfs.h>
#include "const.h"
#include "FuzzDefs.h"
#include "atb_probe.h"
#include "dump_utils.h"
#include "nlohmann/json.hpp"


TEST(test_ReportOperationGraph, fuzz_test)
{
    char testApi[] = "test_ReportOperationGraph";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *opname = DT_SetGetString(&g_Element[0], 28, 4096, "/conv1/Convfunction_graph_0");
        char *graph = DT_SetGetString(&g_Element[1], 10, 4096, "aaaaaaaaa");
        atb::Probe::ReportOperationGraph(opname, graph);
    }
    DT_FUZZ_END()
}

TEST(test_IsSaveTensorData, fuzz_test)
{
    char testApi[] = "test_IsSaveTensorData";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *tensors = DT_SetGetString(&g_Element[0], 2, 4096, "1");
        setenv("ATB_SAVE_TENSOR", tensors, 1);
        atb::Probe::IsSaveTensorData();
    }
    DT_FUZZ_END()
}

TEST(test_IsSaveChild, fuzz_test)
{
    char testApi[] = "test_IsSaveChild";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *child = DT_SetGetString(&g_Element[0], 2, 4096, "1");
        setenv("ATB_SAVE_CHILD", child, 1);
        atb::Probe::IsSaveChild();
    }
    DT_FUZZ_END()
}

TEST(test_IsExecuteCountInRange, fuzz_test)
{
    char testApi[] = "test_IsExecuteCountInRange";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *range = DT_SetGetString(&g_Element[0], 18, 4096, "1,10,20,50,90,100");
        setenv("ATB_SAVE_TENSOR_RANGE", range, 1);
        u64 count = *(u64 *)DT_SetGetS64(&g_Element[1], 0x12);
        atb::Probe::IsExecuteCountInRange(count);
    }
    DT_FUZZ_END()
}

TEST(test_IsSaveTensorBefore, fuzz_test)
{
    char testApi[] = "test_IsSaveTensorBefore";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *time = DT_SetGetString(&g_Element[0], 2, 4096, "1");
        setenv("ATB_SAVE_TENSOR_TIME", time, 1);
        char *out = DT_SetGetString(&g_Element[1], 2, 4096, "1");
        setenv("ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER", out, 1);
        atb::Probe::IsSaveTensorBefore();
    }
    DT_FUZZ_END()
}

TEST(test_IsSaveTensorAfter, fuzz_test)
{
    char testApi[] = "test_IsSaveTensorAfter";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *time = DT_SetGetString(&g_Element[0], 2, 4096, "1");
        setenv("ATB_SAVE_TENSOR_TIME", time, 1);
        char *out = DT_SetGetString(&g_Element[1], 2, 4096, "1");
        setenv("ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER", out, 1);
        atb::Probe::IsSaveTensorAfter();
    }
    DT_FUZZ_END()
}

TEST(test_SaveTensor, fuzz_test)
{
    char testApi[] = "test_SaveTensor";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *format = DT_SetGetString(&g_Element[0], 4, 50, "bin");
        char *dtype = DT_SetGetString(&g_Element[1], 4, 50, "int");
        char *dims = DT_SetGetString(&g_Element[2], 10, 50, "1,3,20,20");
        u64 dataSize = *(u64 *)DT_SetGetNumberRange(&g_Element[3], 10, 1, 1000);
        char *filePath = DT_SetGetString(&g_Element[4], 7, 50, "./save");
        char *out = DT_SetGetString(&g_Element[5], 2, 4096, "1");
        setenv("ATB_SAVE_TENSOR_IN_BEFORE_OUT_AFTER", out, 1);
        std::vector<uint8_t> data = {};
        int size = dataSize / sizeof(uint8_t);
        std::cout << "size: " << size << std::endl;
        for (int i = 0; i < size; i++) {
            u8 num = *(u8 *)DT_SetGetU8(&g_Element[6], 1);
            data.push_back(num);
        }
        uint8_t *hostData = &data[0];
        atb::Probe::SaveTensor(format, dtype, dims, hostData, dataSize, filePath);
    }
    DT_FUZZ_END()
}

TEST(test_IsTensorNeedSave, fuzz_test)
{
    char testApi[] = "test_IsTensorNeedSave";
    std::vector<int64_t> ids;
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *tensor_ids = DT_SetGetString(&g_Element[0], 19, 4096, "20_1_9,1_23,5_29_1");
        char *tensor_runner = DT_SetGetString(&g_Element[1], 10, 4096, "LinearOps");
        setenv("ATB_DUMP_TYPE", "tensor", 1);
        setenv("ATB_SAVE_TENSOR_IDS", tensor_ids, 1);
        setenv("ATB_SAVE_TENSOR_RUNNER", tensor_runner, 1);
        ids = {};
        uint32_t size = *(u32 *)DT_SetGetNumberRange(&g_Element[2], 10, 0, 100000);
        for (uint32_t i = 0; i < size; i++) {
            int64_t num = *(s64 *)DT_SetGetS64(&g_Element[3], 1);
            ids.push_back(num);
        }
        char *opType = DT_SetGetString(&g_Element[4], 7, 10000, "optype");
        atb::Probe::IsTensorNeedSave(ids, opType);
    }
    DT_FUZZ_END()
}

TEST(test_ReportOperationSetupStatistic, fuzz_test)
{
    char testApi[] = "test_ReportOperationSetupStatistic";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        u64 executeCount = *(u64 *)DT_SetGetNumberRange(&g_Element[0], 10, 1, 1000);
        char *opname = DT_SetGetString(&g_Element[1], 28, 4096, "/conv1/Convfunction_graph_0");
        char *st = DT_SetGetString(&g_Element[2], 3, 4096, "30");
        char *outputDir = DT_SetGetString(&g_Element[3], 15, 4096, "/tmp/fuzz_out/");
        setenv("ATB_OUTPUT_DIR", outputDir, 1);
        atb::Probe::ReportOperationSetupStatistic(executeCount, opname, st);
    }
    DT_FUZZ_END()
}

TEST(test_ReportOperationExecuteStatistic, fuzz_test)
{
    char testApi[] = "test_ReportOperationExecuteStatistic";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        u64 executeCount = *(u64 *)DT_SetGetNumberRange(&g_Element[0], 10, 1, 1000);
        char *opname = DT_SetGetString(&g_Element[1], 28, 4096, "/conv1/Convfunction_graph_0");
        char *st = DT_SetGetString(&g_Element[2], 3, 4096, "30");
        char *outputDir = DT_SetGetString(&g_Element[3], 15, 4096, "/tmp/fuzz_out/");
        setenv("ATB_OUTPUT_DIR", outputDir, 1);
        atb::Probe::ReportOperationExecuteStatistic(executeCount, opname, st);
    }
    DT_FUZZ_END()
}

TEST(test_SaveTiling, fuzz_test)
{
    char testApi[] = "test_SaveTiling";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        u64 dataSize = *(u64 *)DT_SetGetNumberRange(&g_Element[0], 10, 1, 1000);
        char *filePath = DT_SetGetString(&g_Element[1], 7, 50, "./save");
        std::vector<uint8_t> data = {};
        int size = dataSize / sizeof(uint8_t);
        std::cout << "size: " << size << std::endl;
        for (int i = 0; i < size; i++) {
            u8 num = *(u8 *)DT_SetGetU8(&g_Element[2], 1);
            data.push_back(num);
        }
        uint8_t *hostData = &data[0];
        char *outputDir = DT_SetGetString(&g_Element[3], 15, 4096, "/tmp/fuzz_out/");
        setenv("ATB_OUTPUT_DIR", outputDir, 1);
        atb::Probe::SaveTiling(hostData, dataSize, filePath);
    }
    DT_FUZZ_END()
}

TEST(test_ReportOperationIOTensor, fuzz_test)
{
    char testApi[] = "test_ReportOperationIOTensor";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        u64 executeCount = *(u64 *)DT_SetGetNumberRange(&g_Element[0], 10, 1, 1000);
        char *opName = DT_SetGetString(&g_Element[1], 28, 4096, "/conv1/Convfunction_graph_0");
        char *opParam = DT_SetGetString(&g_Element[2], 8, 4096, "Param:1");
        u64 dataSize = *(u64 *)DT_SetGetNumberRange(&g_Element[3], 10, 1, 1000);
        int size = dataSize / sizeof(uint8_t);
        atb::Probe::Tensor inTensor;
        inTensor.dype = "ACL_FLOAT16";
        inTensor.format = "ACL_FORMAT_ND";
        inTensor.shape = "65024,4096";
        inTensor.path = "";
        std::vector<atb::Probe::Tensor> inTensors;
        for (int i = 0; i < size; i++) {
            inTensors.push_back(inTensor);
        }

        atb::Probe::Tensor outTensor;
        outTensor.dype = "ACL_FLOAT16";
        outTensor.format = "ACL_FORMAT_ND";
        outTensor.shape = "8,1024,4096";
        outTensor.path = "";
        std::vector<atb::Probe::Tensor> outTensors;
        for (int i = 0; i < size; i++) {
            outTensors.push_back(outTensor);
        }
        char *outputDir = DT_SetGetString(&g_Element[4], 15, 4096, "/tmp/fuzz_out/");
        setenv("ATB_OUTPUT_DIR", outputDir, 1);
        atb::Probe::ReportOperationIOTensor(executeCount, opName, opParam, inTensors, outTensors);
    }
    DT_FUZZ_END()
}

TEST(test_ReportKernelIOTensor, fuzz_test)
{
    char testApi[] = "test_ReportKernelIOTensor";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        u64 executeCount = *(u64 *)DT_SetGetNumberRange(&g_Element[0], 10, 1, 1000);
        char *opName = DT_SetGetString(&g_Element[1], 28, 4096, "/conv1/Convfunction_graph_0");
        char *opParam = DT_SetGetString(&g_Element[2], 8, 4096, "Param:1");
        u64 dataSize = *(u64 *)DT_SetGetNumberRange(&g_Element[3], 10, 1, 1000);
        int size = dataSize / sizeof(uint8_t);
        atb::Probe::Tensor inTensor;
        inTensor.dype = "ACL_FLOAT16";
        inTensor.format = "ACL_FORMAT_ND";
        inTensor.shape = "65024,4096";
        inTensor.path = "";
        std::vector<atb::Probe::Tensor> inTensors;
        for (int i = 0; i < size; i++) {
            inTensors.push_back(inTensor);
        }
        atb::Probe::Tensor outTensor;
        outTensor.dype = "ACL_FLOAT16";
        outTensor.format = "ACL_FORMAT_ND";
        outTensor.shape = "8,1024,4096";
        outTensor.path = "";
        std::vector<atb::Probe::Tensor> outTensors;
        for (int i = 0; i < size; i++) {
            outTensors.push_back(outTensor);
        }
        char *outputDir = DT_SetGetString(&g_Element[4], 15, 4096, "/tmp/fuzz_out/");
        setenv("ATB_OUTPUT_DIR", outputDir, 1);
        atb::Probe::ReportKernelIOTensor(executeCount, opName, opParam, inTensors, outTensors);
    }
    DT_FUZZ_END()
}

TEST(test_SaveParam, fuzz_test)
{
    char testApi[] = "test_SaveParam";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *param = DT_SetGetString(&g_Element[0], 8, 4096, "Param:1");
        char *filePath = DT_SetGetString(&g_Element[1], 11, 100, "./save_param");
        char *outputDir = DT_SetGetString(&g_Element[4], 15, 4096, "/tmp/fuzz_out/");
        setenv("ATB_OUTPUT_DIR", outputDir, 1);
        atb::Probe::SaveParam(param, filePath);
    }
    DT_FUZZ_END()
}

TEST(test_ReportOverflowKernel, fuzz_test)
{
    char testApi[] = "test_ReportOverflowKernel";
    DT_FUZZ_START(0, g_fuzzRunTime, testApi, 0)
    {
        printf("\r%d", fuzzSeed + fuzzi);
        char *filePath = DT_SetGetString(&g_Element[0], 16, 100, "./save_overflow");
        char *outputDir = DT_SetGetString(&g_Element[4], 15, 4096, "/tmp/fuzz_out/");
        setenv("ATB_OUTPUT_DIR", outputDir, 1);
        atb::Probe::ReportOverflowKernel(filePath);
    }
    DT_FUZZ_END()
}
