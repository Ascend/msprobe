/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
