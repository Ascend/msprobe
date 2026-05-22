/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd. 
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

#ifndef _NAN_TEST_H_
#define _NAN_TEST_H_
#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class KernelNanTest {
public:
  __aicore__ inline KernelNanTest() {}
  __aicore__ inline void Init(GM_ADDR in, GM_ADDR tensorlist, GM_ADDR out, GM_ADDR workspace, const NanTestTilingData* __restrict tiling, TPipe* pipe)
  {
    // init input global gm buffer
    inTensorGm.SetGlobalBuffer((__gm__ T*)in);
    outTensorGm.SetGlobalBuffer((__gm__ T*)out);
  }

  __aicore__ inline void Process() {
    T retValue = inTensorGm.GetValue(0);
    outTensorGm.SetValue(0, 1);
    if(retValue != 0) {
      trap();
    }

    outTensorGm.SetValue(0, 0);
    return;
  }

private:
  // define input global input gm buffer
  GlobalTensor<T> inTensorGm;
  GlobalTensor<T> outTensorGm;

};
#endif // _NAN_TEST_H_