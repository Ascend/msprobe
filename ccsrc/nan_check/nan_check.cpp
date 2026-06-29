// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/framework/utils/RandomOpAdapter.h>
#include <torch_npu/csrc/include/ops.h>

#include "acl/acl_rt.h"
#include "aclnn_common.h"

extern "C"
{
    extern aclError aclrtNpuGetFloatOverFlowStatus(void* outputAddr, uint64_t outputSize, uint32_t checkMode,
                                                   aclrtStream stream);
    extern aclError aclrtNpuClearFloatOverFlowStatus(uint32_t checkMode, aclrtStream stream);
    extern aclError aclrtGetFloatOverflowStatus(void* outputAddr, uint64_t outputSize, aclrtStream stream);
    extern aclError aclrtResetFloatOverflowStatus(aclrtStream stream);
}

/**
 * @brief 获取NPU硬件寄存器溢出状态
 * @param out_tensor 输入张量：存储了「结果缓冲区地址」的张量（二级指针/指针的指针）
 * @return 原样返回out_tensor张量
 */
at::Tensor npu_over_flow(const at::Tensor& out_tensor)
{
    void* descBuf;
    uint64_t descBufLen = sizeof(uint64_t) * 8;
    uint32_t checkMode = 0;

    c10::DeviceGuard guard(out_tensor.device());
    auto stream = c10_npu::getCurrentNPUStream();
    descBuf = out_tensor.data_ptr();

    auto ret = aclrtNpuGetFloatOverFlowStatus(descBuf, descBufLen, checkMode, stream);
    if (ret != ACL_SUCCESS)
    {
        std::cout << "aclrtNpuGetFloatOverFlowStatus ret is not ACL_SUCCESS" << std::endl;
    }

    return out_tensor;
}

/**
 * @brief 清空NPU硬件寄存器溢出状态
 */
void npu_clear_over_flow(const at::Device& device)
{
    c10::DeviceGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream();

    uint32_t checkMode = 0;
    auto ret = aclrtNpuClearFloatOverFlowStatus(checkMode, stream);
    if (ret != ACL_SUCCESS)
    {
        std::cout << "aclrtNpuClearFloatOverFlowStatus ret is not ACL_SUCCESS" << std::endl;
    }
    return;
}

/**
 * @brief 溢出时，对指定tensorList进行数据落盘
 * @param in_tensor 输入溢出状态张量
 * @param tensorList 要采集的tensor列表
 * @return 原样返回in_tensor张量
 */
at::Tensor npu_nan_test(const at::Tensor& in_tensor, const std::vector<at::Tensor>& tensorList)
{
    auto in_shape = in_tensor.sizes();
    at::TensorList tensorListIn = at::TensorList(tensorList);
    auto out_tensor = at::empty(in_shape, in_tensor.options());
    ACLNN_CMD(aclnnNanTest, in_tensor, tensorListIn, out_tensor);
    return out_tensor;
}

TORCH_LIBRARY_FRAGMENT(my_ns, m)
{
    m.def("npu_over_flow(Tensor out_tensor) -> Tensor");
    m.def("npu_clear_over_flow(Device device) -> ()");
    m.def("npu_nan_test(Tensor in_tensor, Tensor[] tensorList) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ns, PrivateUse1, m)
{
    m.impl("npu_over_flow", npu_over_flow);
    m.impl("npu_nan_test", npu_nan_test);
}

TORCH_LIBRARY_IMPL(my_ns, CatchAll, m) { m.impl("npu_clear_over_flow", npu_clear_over_flow); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.doc() = "nan_check_ext: Check NaN/Inf state after compute"; }
