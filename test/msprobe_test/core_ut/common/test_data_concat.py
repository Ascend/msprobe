# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
#  MindStudio is licensed under Mulan PSL v2.
#  You can use this software according to the terms and conditions of the Mulan PSL v2.
#  You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
#  EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
#  MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#  See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest

from msprobe.core.common.const import Data2DBConst
from msprobe.core.common.data_concat import (
    ConcatPlan,
    DatasetAxis,
    StepMapper,
    build_concat_plan,
    build_micro_prefix,
    build_single_plan,
)


def make_axis(steps, ranks, micro_counts=None):
    """构造一个用于测试的 DatasetAxis"""
    return DatasetAxis(
        step_keys=list(steps),
        micro_counts=micro_counts or {},
        rank_ids=list(ranks),
        max_rank=max(ranks) if ranks else 0,
    )


class TestDatasetAxis(unittest.TestCase):
    """测试 DatasetAxis 的属性计算"""

    def test_step_count_and_index(self):
        """迭代步数量与序号映射"""
        axis = make_axis([3, 5], [0, 1])
        self.assertEqual(axis.step_count, 2)
        self.assertEqual(axis.step_index, {3: 0, 5: 1})

    def test_empty_axis(self):
        """空数据源的轴长度为零, 微步信息未知"""
        axis = DatasetAxis()
        self.assertEqual(axis.step_count, 0)
        self.assertEqual(axis.step_index, {})
        self.assertIsNone(axis.micro_total)
        self.assertFalse(axis.micro_known)
        self.assertEqual(axis.rank_count, 0)

    def test_micro_total(self):
        """微步时间线总长度"""
        axis = make_axis([0, 1], [0], {0: 2, 1: 3})
        self.assertEqual(axis.micro_total, 5)

    def test_rank_count(self):
        """rank 轴长度"""
        axis = make_axis([0, 1], [0, 1, 2])
        self.assertEqual(axis.rank_count, 3)


class TestStepMapper(unittest.TestCase):
    """测试 StepMapper 的映射逻辑"""

    def test_map_without_micro_step(self):
        """关闭微步计数时按归一化后的迭代步号映射"""
        mapper = StepMapper(prefix={}, step_index={3: 0, 5: 1}, offset=0, use_micro_step=False)
        self.assertEqual(mapper.map(3), 0)
        self.assertEqual(mapper.map(5), 1)

    def test_map_with_micro_step(self):
        """开启微步计数时按全局微步计数器映射"""
        mapper = StepMapper(prefix={0: 0, 1: 2}, step_index={0: 0, 1: 1}, offset=4, use_micro_step=True)
        self.assertEqual(mapper.map(0, 0), 4)
        self.assertEqual(mapper.map(0, 1), 5)
        self.assertEqual(mapper.map(1, 0), 6)

    def test_map_unknown_step_falls_back_to_zero(self):
        """未登记的迭代步在微步模式下按前缀 0 处理"""
        mapper = StepMapper(prefix={0: 0}, step_index={0: 0}, offset=0, use_micro_step=True)
        self.assertEqual(mapper.map(99, 0), 0)


class TestBuildMicroPrefix(unittest.TestCase):
    """测试微步前缀和计算"""

    def test_prefix_is_cumulative(self):
        """按迭代步升序做前缀和"""
        prefix = build_micro_prefix({1: 2, 0: 3})
        self.assertEqual(prefix, {0: 0, 1: 3})

    def test_empty_counts(self):
        """没有微步数据时前缀为空"""
        self.assertEqual(build_micro_prefix({}), {})


class TestBuildSinglePlan(unittest.TestCase):
    """测试单数据源场景的拼接计划"""

    def test_single_plan_without_micro_step(self):
        """单数据源保持原始迭代步号, 不做归一化"""
        axis = make_axis([3, 5], [0, 1])
        plan = build_single_plan(axis, use_micro_step=False)
        self.assertEqual(plan.concat_dim, Data2DBConst.CONCAT_DIM_NONE)
        self.assertEqual(plan.min_step, 3)
        self.assertEqual(plan.max_step, 5)
        self.assertEqual(plan.step_mapper_a.map(3), 3)
        self.assertEqual(plan.step_mapper_a.map(5), 5)
        self.assertEqual(plan.max_rank, 1)

    def test_single_plan_with_micro_step(self):
        """单数据源开启微步计数时 step 轴为全局微步计数器"""
        axis = make_axis([3, 4], [0], {3: 2, 4: 1})
        plan = build_single_plan(axis, use_micro_step=True)
        self.assertEqual(plan.min_step, 0)
        self.assertEqual(plan.max_step, 2)
        self.assertEqual(plan.step_mapper_a.map(3, 0), 0)
        self.assertEqual(plan.step_mapper_a.map(3, 1), 1)
        self.assertEqual(plan.step_mapper_a.map(4, 0), 2)


class TestBuildConcatPlanStepDim(unittest.TestCase):
    """测试 step 维度拼接"""

    def test_step_concat_without_micro_step(self):
        """关闭微步计数时两份数据的迭代步首尾相接"""
        axis_a = make_axis([0, 1], [0, 1])
        axis_b = make_axis([0, 1], [0, 1])
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_STEP, use_micro_step=False)
        self.assertEqual(plan.concat_dim, Data2DBConst.CONCAT_DIM_STEP)
        self.assertEqual(plan.span_a, 2)
        self.assertEqual(plan.span_b, 2)
        self.assertEqual(plan.boundary, 2)
        self.assertEqual(plan.min_step, 0)
        self.assertEqual(plan.max_step, 3)
        self.assertEqual(plan.rank_offset_b, 0)
        # 被测数据占 0..1, 对比数据占 2..3
        self.assertEqual(plan.step_mapper_a.map(0), 0)
        self.assertEqual(plan.step_mapper_a.map(1), 1)
        self.assertEqual(plan.step_mapper_b.map(0), 2)
        self.assertEqual(plan.step_mapper_b.map(1), 3)

    def test_step_concat_with_micro_step(self):
        """开启微步计数时对比数据的微步时间线整体后移"""
        axis_a = make_axis([0, 1], [0, 1], {0: 2, 1: 2})
        axis_b = make_axis([0, 1], [0, 1], {0: 2, 1: 2})
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_STEP, use_micro_step=True)
        self.assertEqual(plan.span_a, 4)
        self.assertEqual(plan.span_b, 4)
        self.assertEqual(plan.boundary, 4)
        self.assertEqual(plan.max_step, 7)
        self.assertEqual(plan.step_mapper_a.map(0, 0), 0)
        self.assertEqual(plan.step_mapper_a.map(1, 0), 2)
        self.assertEqual(plan.step_mapper_b.map(0, 0), 4)
        self.assertEqual(plan.step_mapper_b.map(1, 0), 6)

    def test_step_concat_normalizes_different_start(self):
        """两份数据的迭代步起点不同时按各自起点归一化"""
        axis_a = make_axis([5, 6], [0])
        axis_b = make_axis([0, 1], [0])
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_STEP, use_micro_step=False)
        self.assertEqual(plan.step_mapper_a.map(5), 0)
        self.assertEqual(plan.step_mapper_a.map(6), 1)
        self.assertEqual(plan.step_mapper_b.map(0), 2)
        self.assertEqual(plan.step_mapper_b.map(1), 3)

    def test_step_concat_warns_on_different_step_count(self):
        """两份数据迭代步数不同时给出提示"""
        axis_a = make_axis([0, 1, 2], [0])
        axis_b = make_axis([0, 1], [0])
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_STEP, use_micro_step=False)
        self.assertTrue(any("different step counts" in w for w in plan.warnings))


class TestBuildConcatPlanRankDim(unittest.TestCase):
    """测试 rank 维度拼接"""

    def test_rank_concat_without_micro_step(self):
        """关闭微步计数时两份数据共享同一条迭代步轴, rank 整体后移"""
        axis_a = make_axis([0, 1, 2], [0, 1])
        axis_b = make_axis([0, 1, 2], [0, 1])
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_RANK, use_micro_step=False)
        self.assertEqual(plan.concat_dim, Data2DBConst.CONCAT_DIM_RANK)
        self.assertEqual(plan.rank_offset_b, 2)
        self.assertEqual(plan.boundary, 2)
        self.assertEqual(plan.span_a, 2)
        self.assertEqual(plan.span_b, 2)
        self.assertEqual(plan.step_axis_len, 3)
        self.assertEqual(plan.min_step, 0)
        self.assertEqual(plan.max_step, 2)
        self.assertEqual(plan.max_rank, 3)
        # 同一迭代步在两份数据中映射到同一个库内 step
        self.assertEqual(plan.step_mapper_a.map(1), plan.step_mapper_b.map(1))
        self.assertEqual(plan.step_mapper_a.map(2), plan.step_mapper_b.map(2))

    def test_rank_concat_with_micro_step(self):
        """开启微步计数时两份数据共用一条公共微步时间线"""
        axis_a = make_axis([0, 1], [0, 1], {0: 2, 1: 2})
        axis_b = make_axis([0, 1], [0, 1], {0: 2, 1: 2})
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_RANK, use_micro_step=True)
        self.assertEqual(plan.rank_offset_b, 2)
        self.assertEqual(plan.step_axis_len, 4)
        self.assertEqual(plan.max_step, 3)
        for step in (0, 1):
            for micro in (0, 1):
                self.assertEqual(
                    plan.step_mapper_a.map(step, micro),
                    plan.step_mapper_b.map(step, micro),
                )

    def test_rank_concat_shares_step_axis_when_start_differs(self):
        """两份数据迭代步起点不同时仍按同一序号对齐"""
        axis_a = make_axis([0, 1], [0])
        axis_b = make_axis([4, 5], [0])
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_RANK, use_micro_step=False)
        self.assertEqual(plan.step_mapper_a.map(0), plan.step_mapper_b.map(4))
        self.assertEqual(plan.step_mapper_a.map(1), plan.step_mapper_b.map(5))

    def test_rank_concat_warns_on_different_micro_count(self):
        """两份数据同一迭代步的微步数不同时给出提示并按较大值撑宽公共轴"""
        axis_a = make_axis([0, 1], [0], {0: 2, 1: 2})
        axis_b = make_axis([0, 1], [0], {0: 3, 1: 2})
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_RANK, use_micro_step=True)
        self.assertTrue(any("different micro step counts" in w for w in plan.warnings))
        # 公共轴按较大的微步数撑宽
        self.assertEqual(plan.step_axis_len, 5)
        self.assertEqual(plan.step_mapper_a.map(0, 0), plan.step_mapper_b.map(0, 0))
        self.assertEqual(plan.step_mapper_a.map(1, 0), plan.step_mapper_b.map(1, 0))

    def test_rank_concat_warns_on_different_rank_count(self):
        """两份数据 rank 数不同时给出提示"""
        axis_a = make_axis([0, 1], [0, 1])
        axis_b = make_axis([0, 1], [0])
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_RANK, use_micro_step=False)
        self.assertTrue(any("different rank counts" in w for w in plan.warnings))
        self.assertEqual(plan.rank_offset_b, 2)

    def test_rank_concat_shifts_by_max_rank_when_not_contiguous(self):
        """被测数据 rank 编号不连续时, 对比数据后移到最大 rank 之后, 两份数据 rank 不重叠"""
        axis_a = make_axis([0, 1], [1, 3])  # rank 编号不连续
        axis_b = make_axis([0, 1], [0, 1, 2])
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_RANK, use_micro_step=False)
        self.assertEqual(plan.rank_offset_b, 4)
        self.assertTrue(any("not contiguous" in w for w in plan.warnings))
        ranks_a = set(axis_a.rank_ids)
        ranks_b = {rank + plan.rank_offset_b for rank in axis_b.rank_ids}
        self.assertEqual(ranks_a & ranks_b, set())
        self.assertEqual(plan.max_rank, 6)

    def test_micro_step_unknown_leaves_span_unavailable(self):
        """微步信息未知时, 拼接计划的跨度与最大 step 不可用"""
        axis_a = DatasetAxis(step_keys=[0, 1], micro_counts=None, rank_ids=[0], max_rank=0)
        axis_b = DatasetAxis(step_keys=[0, 1], micro_counts=None, rank_ids=[0], max_rank=0)
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_STEP, use_micro_step=True)
        self.assertIsNone(plan.span_a)
        self.assertIsNone(plan.span_b)
        self.assertIsNone(plan.max_step)
        self.assertIsNone(plan.step_axis_len)
        self.assertTrue(any("micro step counts" in w for w in plan.warnings))
        # 计划层偏移不可用, 但不应声称对比数据完全没有平移, dump 解析阶段会按实际微步累加
        warning = next(w for w in plan.warnings if "micro step counts" in w)
        self.assertNotIn("not shifted", warning)
        self.assertIn("parsing", warning)

    def test_rank_concat_micro_unknown_leaves_axis_unavailable(self):
        """rank 维度拼接时微步信息未知, 公共 step 轴长度同样不可用"""
        axis_a = DatasetAxis(step_keys=[0, 1], micro_counts=None, rank_ids=[0], max_rank=0)
        axis_b = DatasetAxis(step_keys=[0, 1], micro_counts=None, rank_ids=[0], max_rank=0)
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_RANK, use_micro_step=True)
        self.assertIsNone(plan.step_axis_len)
        self.assertIsNone(plan.max_step)
        self.assertTrue(any("micro step counts" in w for w in plan.warnings))
        self.assertEqual(plan.rank_offset_b, 1)
        # 微步映射仍按微步号累加, 不使用伪造的迭代步前缀
        self.assertEqual(plan.step_mapper_a.map(0, 1), 1)


class TestBuildConcatPlanErrors(unittest.TestCase):
    """测试拼接计划的异常处理"""

    def test_invalid_concat_dim(self):
        """非法的拼接维度应抛出异常"""
        axis = make_axis([0, 1], [0])
        with self.assertRaises(ValueError):
            build_concat_plan(axis, axis, "invalid", use_micro_step=False)

    def test_plan_accessors(self):
        """计划按数据源返回对应的映射器与偏移"""
        axis_a = make_axis([0, 1], [0, 1])
        axis_b = make_axis([0, 1], [0, 1])
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_RANK, use_micro_step=False)
        self.assertIs(plan.mapper_for(0), plan.step_mapper_a)
        self.assertIs(plan.mapper_for(1), plan.step_mapper_b)
        self.assertEqual(plan.rank_offset_for(0), 0)
        self.assertEqual(plan.rank_offset_for(1), 2)

    def test_group_norm_step(self):
        """分组归一化序号按迭代步基准计算"""
        axis_a = make_axis([2, 3], [0])
        axis_b = make_axis([2, 3], [0])
        plan = build_concat_plan(axis_a, axis_b, Data2DBConst.CONCAT_DIM_RANK, use_micro_step=False)
        self.assertEqual(plan.group_norm_step(0, 2), 0)
        self.assertEqual(plan.group_norm_step(0, 3), 1)
        self.assertEqual(plan.group_norm_step(1, 2), 0)
        self.assertEqual(plan.group_norm_step(1, 3), 1)


if __name__ == '__main__':
    unittest.main()
