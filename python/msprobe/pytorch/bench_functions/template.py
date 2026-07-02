# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""
融合算子标杆函数模板（Template for Fusion Operator Benchmark Functions）

============================================
 使用说明（Instructions）
============================================

  Step 1: 在本目录下创建 Python 文件，实现前向标杆函数（及可选的反向标杆函数）
           复制本文件并重命名，按实际需求修改函数体。

  Step 2: 在 fusion_operator_config.yaml 中注册算子信息，
           指定前向/反向函数名及可选的输出处理配置。

  Step 3: 运行 acc_check，预检工具会自动加载本文件中的函数并执行精度比对。

============================================
 标杆函数规范（Specification）
============================================

  1. 函数名称以 npu_ 开头，与 NPU 算子名保持一致。
  2. 前向函数接受与 NPU 算子相同的参数（args 和 kwargs）。
  3. 反向函数接受 grad 和前向输入作为参数。
  4. 输出必须在 CPU 上（返回前调用 .cpu()）。
  5. 使用纯 PyTorch 算子实现，不依赖任何 NPU 特有算子。

============================================
 示例：npu_example_op 融合算子标杆函数
============================================
"""

import torch


def template_npu_example_op_forward(x, weight, bias=None):
    """
    前向标杆函数（Forward Benchmark Function）

    参数:
        x (torch.Tensor): 输入张量
        weight (torch.Tensor): 权重张量
        bias (torch.Tensor, optional): 偏置张量

    返回:
        torch.Tensor: 计算结果（必须在 CPU 上）
    """
    # 使用纯 PyTorch 算子实现与 NPU 融合算子等价的计算
    result = torch.matmul(x, weight.t())
    if bias is not None:
        result = result + bias
    return result


def template_npu_example_op_backward(grad_output, x, weight):
    """
    反向标杆函数（Backward Benchmark Function，可选）

    参数:
        grad_output (torch.Tensor): 上游梯度
        x (torch.Tensor): 前向输入
        weight (torch.Tensor): 权重张量

    返回:
        tuple or torch.Tensor: 对输入的梯度
    """
    grad_x = torch.matmul(grad_output, weight)
    grad_weight = torch.matmul(grad_output.t(), x)
    return grad_x, grad_weight


# ============================================================================
# 测试方法：取消下方注释后运行 python template.py
# 注意：复制模板后需将 template_ 前缀替换为自定义算子名
# ============================================================================
# if __name__ == '__main__':
#     x = torch.randn(4, 16)
#     w = torch.randn(8, 16)
#     out = template_npu_example_op_forward(x, w)
#     print(f"Output shape: {out.shape}")
