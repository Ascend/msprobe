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
