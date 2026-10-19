#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

from unittest import TestCase

from msprobe.core.common.const import CompareConst
from msprobe.core.compare.indicator_analysis.calculator import (
    BENCH_NAME_MISSING_VALUES,
    LINEAR_BACKWARD_VIEW_HINT,
    _mark_linear_backward_weight_missing_as_pass,
    _shape_last_dim,
    _shape_numels,
)

# 被测函数只通过 head.index() 取下列列，用最小列集构造行即可覆盖全部逻辑分支
HEAD = [
    CompareConst.NPU_NAME,
    CompareConst.BENCH_NAME,
    CompareConst.NPU_SHAPE,
    CompareConst.BENCH_SHAPE,
    CompareConst.RESULT,
    CompareConst.ERROR_MESSAGE,
]

INITIAL_RESULT = CompareConst.ERROR  # 未触发时保持原有比对结果（此处用 error 表示原本比对失败）


def make_row(npu_name, bench_name=None, npu_shape=None, bench_shape=None, result=INITIAL_RESULT, err_msg=''):
    row = [''] * len(HEAD)
    row[HEAD.index(CompareConst.NPU_NAME)] = npu_name
    row[HEAD.index(CompareConst.BENCH_NAME)] = bench_name
    row[HEAD.index(CompareConst.NPU_SHAPE)] = npu_shape
    row[HEAD.index(CompareConst.BENCH_SHAPE)] = bench_shape
    row[HEAD.index(CompareConst.RESULT)] = result
    row[HEAD.index(CompareConst.ERROR_MESSAGE)] = err_msg
    return row


# 表驱动用例：(用例名, data_lists, 期望被置 pass 的 npu_name 集合)
CASES = [
    # ---- 触发：3D 输入 + output.0 维度对不上 + output.1 bench 缺失（小写 nan）----
    (
        'linear_3d_input_both_outputs_missing',
        [
            make_row('Functional.linear.0.backward.input.0', npu_shape='[1, 976, 151936]'),
            make_row('Functional.linear.0.backward.output.0', npu_shape='[1, 976, 5120]', bench_shape='[976, 5120]'),
            make_row(
                'Functional.linear.0.backward.output.1',
                npu_shape='[5120, 151936]',
                bench_shape='[5120, 151936]',
                bench_name='nan',
            ),
        ],
        {'Functional.linear.0.backward.output.0', 'Functional.linear.0.backward.output.1'},
    ),
    # 触发：matmul 3D + output.1 bench 显式 N/A
    (
        'matmul_3d_input_bench_na',
        [
            make_row('Torch.matmul.0.backward.input.0', npu_shape='[1, 976, 5120]'),
            make_row('Torch.matmul.0.backward.output.0', npu_shape='[1, 976, 151936]', bench_shape='[976, 151936]'),
            make_row(
                'Torch.matmul.0.backward.output.1',
                npu_shape='[5120, 151936]',
                bench_shape='[5120, 151936]',
                bench_name=CompareConst.N_A,
            ),
        ],
        {'Torch.matmul.0.backward.output.0', 'Torch.matmul.0.backward.output.1'},
    ),
    # 触发：output.1 bench 为项目统一缺失标记 Nan（CompareConst.NAN）
    (
        'matmul_3d_input_bench_nan_const',
        [
            make_row('Torch.matmul.0.backward.input.0', npu_shape='[1, 976, 5120]'),
            make_row('Torch.matmul.0.backward.output.0', npu_shape='[1, 976, 151936]', bench_shape='[976, 151936]'),
            make_row('Torch.matmul.0.backward.output.1', bench_name=CompareConst.NAN),
        ],
        {'Torch.matmul.0.backward.output.0', 'Torch.matmul.0.backward.output.1'},
    ),
    # 触发：3D 输入 + output.0 维度相同（不触发 output.0） + output.1 缺失 → 仅 output.1
    (
        'output0_same_ndims_not_pass_output1_missing_pass',
        [
            make_row('Functional.linear.0.backward.input.0', npu_shape='[1, 976, 151936]'),
            make_row('Functional.linear.0.backward.output.0', npu_shape='[1, 976, 5120]', bench_shape='[1, 976, 5120]'),
            make_row('Functional.linear.0.backward.output.1', npu_shape='[5120, 151936]'),
        ],
        {'Functional.linear.0.backward.output.1'},
    ),
    # 触发：3D 输入 + output.0 npu shape 缺失（_shape_ndims 返回 None） + output.1 缺失 → 仅 output.1
    (
        'output0_shape_none_not_pass_output1_missing_pass',
        [
            make_row('Functional.linear.0.backward.input.0', npu_shape='[1, 976, 151936]'),
            make_row('Functional.linear.0.backward.output.0', npu_shape=None, bench_shape='[976, 5120]'),
            make_row('Functional.linear.0.backward.output.1', bench_name=None),
        ],
        {'Functional.linear.0.backward.output.1'},
    ),
    # 触发：3D 输入 + output.0 维度对不上 + output.1 bench 存在 → 仅 output.0
    (
        'output0_pass_output1_bench_exists_not_pass',
        [
            make_row('Functional.linear.0.backward.input.0', npu_shape='[1, 976, 151936]'),
            make_row('Functional.linear.0.backward.output.0', npu_shape='[1, 976, 5120]', bench_shape='[976, 5120]'),
            make_row(
                'Functional.linear.0.backward.output.1',
                npu_shape='[5120, 151936]',
                bench_shape='[5120, 151936]',
                bench_name='some_bench_op',
            ),
        ],
        {'Functional.linear.0.backward.output.0'},
    ),
    # 触发（反例，不置 pass）：output.0 维度不同但元素总数也不一致（[2, 3, 5]=30 vs [7, 5]=35）
    # → 不是 view 展平，保留 error；output.1 bench 缺失 → 仍单独置 pass
    (
        'output0_numels_mismatch_keep_error_output1_missing_pass',
        [
            make_row('Functional.linear.0.backward.input.0', npu_shape='[2, 3, 5]'),
            make_row('Functional.linear.0.backward.output.0', npu_shape='[2, 3, 5]', bench_shape='[7, 5]'),
            make_row('Functional.linear.0.backward.output.1', bench_name=None),
        ],
        {'Functional.linear.0.backward.output.1'},
    ),
    # 触发（反例，不置 pass）：output.0 元素总数一致（30 == 30）但最后一维不一致（5 vs 15）
    # → 非展平关系，保留 error；output.1 bench 存在 → 也不置 pass
    (
        'output0_last_dim_mismatch_keep_error',
        [
            make_row('Functional.linear.0.backward.input.0', npu_shape='[2, 3, 5]'),
            make_row('Functional.linear.0.backward.output.0', npu_shape='[2, 3, 5]', bench_shape='[2, 15]'),
            make_row(
                'Functional.linear.0.backward.output.1',
                npu_shape='[5, 7]',
                bench_shape='[5, 7]',
                bench_name='some_bench_op',
            ),
        ],
        set(),
    ),
    # 触发（合法展平，置 pass）：output.0 元素总数一致且最后一维一致（[1, 6, 5] vs [6, 5]）
    (
        'output0_valid_flatten_pass',
        [
            make_row('Functional.linear.0.backward.input.0', npu_shape='[1, 6, 5]'),
            make_row('Functional.linear.0.backward.output.0', npu_shape='[1, 6, 5]', bench_shape='[6, 5]'),
            make_row('Functional.linear.0.backward.output.1', bench_name=None),
        ],
        {'Functional.linear.0.backward.output.0', 'Functional.linear.0.backward.output.1'},
    ),
    # ---- 不触发 ----
    # 2D 输入：任何输出行都保持原结果
    (
        'linear_2d_input_no_trigger',
        [
            make_row('Functional.linear.1.backward.input.0', npu_shape='[976, 151936]'),
            make_row('Functional.linear.1.backward.output.0', npu_shape='[976, 5120]', bench_shape='[976, 5120]'),
            make_row(
                'Functional.linear.1.backward.output.1',
                npu_shape='[5120, 151936]',
                bench_shape='[5120, 151936]',
                bench_name=None,
            ),
        ],
        set(),
    ),
    # 非目标算子（embedding）不处理
    (
        'non_target_api_no_trigger',
        [
            make_row('Functional.embedding.0.backward.input.0', npu_shape='[4, 128]'),
            make_row('Functional.embedding.0.backward.output.0', npu_shape='[4, 128, 5120]', bench_shape='[4, 128, 5120]'),
        ],
        set(),
    ),
    # 非 output.0 / output.1 行（如 input 行本身）不处理
    (
        'input_row_only_no_trigger',
        [
            make_row('Torch.matmul.0.backward.input.0', npu_shape='[1, 976, 5120]'),
        ],
        set(),
    ),
    # 同 api 无 input 行 → 判定不到 3D 输入 → 不触发
    (
        'no_input_row_no_trigger',
        [
            make_row('Torch.matmul.0.backward.output.0', npu_shape='[1, 976, 151936]', bench_shape='[976, 151936]'),
            make_row('Torch.matmul.0.backward.output.1', bench_name=None),
        ],
        set(),
    ),
    # input shape 为 nan → _shape_ndims 返回 None → 不触发
    (
        'input_shape_nan_no_trigger',
        [
            make_row('Torch.matmul.0.backward.input.0', npu_shape='nan'),
            make_row('Torch.matmul.0.backward.output.0', npu_shape='[1, 976, 151936]', bench_shape='[976, 151936]'),
            make_row('Torch.matmul.0.backward.output.1', bench_name=None),
        ],
        set(),
    ),
]


class TestMarkLinearBackwardWeightMissingAsPass(TestCase):
    """表驱动验证 _mark_linear_backward_weight_missing_as_pass 的触发与不触发分支。"""

    def test_trigger_and_no_trigger_branches(self):
        for name, rows, expected_pass in CASES:
            with self.subTest(name=name):
                data_lists = [row[:] for row in rows]
                _mark_linear_backward_weight_missing_as_pass(data_lists, HEAD)
                for npu_name, _bench, _shape, _bench_shape, result, err_msg in data_lists:
                    if npu_name in expected_pass:
                        self.assertEqual(result, CompareConst.PASS, msg=f'{npu_name} should be pass')
                        self.assertEqual(err_msg, LINEAR_BACKWARD_VIEW_HINT, msg=f'{npu_name} should carry hint')
                    else:
                        self.assertEqual(result, INITIAL_RESULT, msg=f'{npu_name} should keep original result')
                        self.assertEqual(err_msg, '', msg=f'{npu_name} should keep empty err_msg')

    def test_missing_values_cover_const_nan_and_na(self):
        self.assertIn(CompareConst.N_A, BENCH_NAME_MISSING_VALUES)
        self.assertIn(CompareConst.NAN, BENCH_NAME_MISSING_VALUES)
        self.assertIn('nan', BENCH_NAME_MISSING_VALUES)

    def test_shape_numels_and_last_dim(self):
        """元素总数与最后一维解析：view 展平保持两者不变，反例能据此被拦截。"""
        self.assertEqual(_shape_numels('[2, 3, 5]'), 30)
        self.assertEqual(_shape_numels('[7, 5]'), 35)
        self.assertEqual(_shape_numels('[976, 5120]'), _shape_numels('[1, 976, 5120]'))
        self.assertEqual(_shape_last_dim('[2, 3, 5]'), 5)
        self.assertEqual(_shape_last_dim('[7, 5]'), 5)
        self.assertIsNone(_shape_numels('nan'))
        self.assertIsNone(_shape_numels(None))
        self.assertIsNone(_shape_last_dim('[]'), '空 shape 最后一维取不到')
        self.assertIsNone(_shape_last_dim('nan'))
