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

#include <iostream>
#include <string>
#include "securec.h"
#include "FuzzDefs.h"

int g_fuzzRunTime = 1000000;

GTEST_API_ int main(int argc, char **argv)
{
    int singleCaseTimeout = 60; // second
    // 设置报告路径
    DT_Set_Report_Path(REPORT_PATH.c_str());
    // 设置使能fork模式，每个测试用例单独在子进程运行
    DT_SetEnableFork(1);
    // 检测大内存使用，超过2048M使用或者1024M分配则当做bug报错
    DT_SetCheckOutOfMemory(1024, 2048);
    // 是能内存泄漏单次执行检测，默认也开启
    DT_Enable_Leak_Check(1, 0);
    // 设置用例单次执行多久超时
    DT_Set_TimeOut_Second(singleCaseTimeout);
    if (argc == 2) { // 2 input with fuzz run time
        if (sscanf_s(argv[1], "%d", &g_fuzzRunTime) == -1) {
            std::cout << "failed to get fuzz run time" << std::endl;
            return -1;
        }
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
