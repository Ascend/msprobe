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

"""两份精度数据的拼接计划。

趋势可视化工具一次只能解析一份数据路径，对比两组实验时需要把两份数据拼接到同一个
数据库中。本模块负责计算拼接计划：两份数据各自的 step 映射、rank 偏移以及合并后
的全局区间。dump 链路与 monitor 链路各自按计划把数据写入同一份数据库。

拼接维度：

- step：两份数据在 step 轴上首尾相接，被测数据占据前一段，对比数据占据后一段。
- rank：两份数据共享同一条 step 轴，在 rank 轴上首尾相接。

step 轴的含义随 micro_step 开关变化：关闭时是迭代步序号，开启时是全局微步计数器。
迭代步之间按序号对齐，即两份数据各自的第 k 个迭代步落在同一位置，与迭代步编号
本身是否连续无关。
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from msprobe.core.common.const import Data2DBConst


@dataclass
class DatasetAxis:
    """单份数据的 step 轴与 rank 轴信息。"""

    step_keys: List[int] = field(default_factory=list)
    # 迭代步号 -> 该步的微步数; None 表示该数据源的微步信息未知
    micro_counts: Optional[Dict[int, int]] = None
    rank_ids: List[int] = field(default_factory=list)
    max_rank: int = 0

    @property
    def step_count(self) -> int:
        """实际出现的迭代步数量。"""
        return len(self.step_keys)

    @property
    def step_index(self) -> Dict[int, int]:
        """迭代步号 -> 序号, 按序号对齐而非按数值对齐。"""
        return {step: index for index, step in enumerate(self.step_keys)}

    @property
    def micro_known(self) -> bool:
        """微步信息是否已知。"""
        return self.micro_counts is not None

    @property
    def micro_total(self) -> Optional[int]:
        """微步时间线总长度; 微步信息未知时为 None。"""
        if self.micro_counts is None:
            return None
        return sum(self.micro_counts.values())

    @property
    def rank_count(self) -> int:
        """实际出现的 rank 数量。"""
        return len(self.rank_ids)

    @property
    def rank_span(self) -> int:
        """rank 轴跨度, 即最大 rank 编号加一。

        rank 编号可能不连续(部分 rank 目录没有有效数据), 该跨度保证对比数据的 rank
        平移后与被测数据的 rank 不重叠。
        """
        return self.max_rank + 1


@dataclass
class StepMapper:
    """把原始的 (step, micro_step) 映射为库内 step。"""

    prefix: Dict[int, int] = field(default_factory=dict)
    step_index: Dict[int, int] = field(default_factory=dict)
    offset: int = 0
    use_micro_step: bool = False

    def map(self, raw_step: int, micro_step: int = 0) -> int:
        """把原始的迭代步号与微步号映射为库内 step。"""
        if self.use_micro_step:
            return self.prefix.get(raw_step, 0) + micro_step + self.offset
        return self.step_index.get(raw_step, 0) + self.offset

    def index_of(self, raw_step: int) -> int:
        """取迭代步在 step 轴上的序号。"""
        return self.step_index.get(raw_step, 0)


@dataclass
class ConcatPlan:
    """两份数据的拼接计划。

    dataset_id 为 0 表示被测数据, 1 表示对比数据。span_a 与 span_b 是两份数据在拼接
    维度上的跨度: step 维度拼接时为各自的迭代步数, rank 维度拼接时为各自的 rank 轴
    跨度。step_axis_len 是合并后 step 轴的长度。这些字段在数据源的微步信息未知时
    为 None, 表示不可用, 消费方需显式处理。
    """

    concat_dim: str = Data2DBConst.CONCAT_DIM_NONE
    step_mapper_a: Optional[StepMapper] = None
    step_mapper_b: Optional[StepMapper] = None
    rank_offset_b: int = 0
    min_step: int = 0
    max_step: Optional[int] = None
    max_rank: int = 0
    boundary: int = 0
    span_a: Optional[int] = None
    span_b: Optional[int] = None
    step_axis_len: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    def mapper_for(self, dataset_id: int) -> StepMapper:
        """取指定数据源的 step 映射器。"""
        mapper = self.step_mapper_a if dataset_id == 0 else self.step_mapper_b
        return mapper if mapper is not None else StepMapper()

    def rank_offset_for(self, dataset_id: int) -> int:
        """取指定数据源的 rank 偏移。"""
        return 0 if dataset_id == 0 else self.rank_offset_b

    def group_norm_step(self, dataset_id: int, raw_step: int) -> int:
        """取指定数据源的迭代步序号, 用于把两份数据的同一迭代步配对。"""
        return self.mapper_for(dataset_id).index_of(raw_step)


def build_micro_prefix(micro_counts: Dict[int, int]) -> Dict[int, int]:
    """按迭代步升序对每个迭代步的微步数做前缀和。

    :param micro_counts: 迭代步号 -> 该迭代步的微步数
    :return: 迭代步号 -> 该迭代步在全局微步时间线上的起始位置
    """
    prefix = {}
    cumsum = 0
    for step in sorted(micro_counts):
        prefix[step] = cumsum
        cumsum += micro_counts.get(step, 0)
    return prefix


def _merge_micro_counts(axis_a: DatasetAxis, axis_b: DatasetAxis) -> Dict[int, int]:
    """按迭代步序号合并两份数据的微步数，取较大值。

    按序号合并保证两份数据的第 k 个迭代步对齐到同一个位置。
    """
    merged = defaultdict(int)
    for axis in (axis_a, axis_b):
        counts = axis.micro_counts or {}
        for offset, raw_step in enumerate(axis.step_keys):
            merged[offset] = max(merged[offset], counts.get(raw_step, 0))
    return dict(merged)


def build_single_plan(axis: DatasetAxis, use_micro_step: bool) -> ConcatPlan:
    """构造单数据源场景的拼接计划。

    单数据源场景保持现有行为：迭代步号沿用原始编号，不做归一化。
    """
    if use_micro_step:
        prefix = build_micro_prefix(axis.micro_counts or {})
        mapper = StepMapper(prefix=prefix, step_index=axis.step_index, offset=0, use_micro_step=True)
        min_step = 0
        max_step = None if axis.micro_total is None else axis.micro_total - 1
        span = axis.micro_total
        step_axis_len = span
    else:
        # 单数据源时沿用原始迭代步编号
        identity = {step: step for step in axis.step_keys}
        mapper = StepMapper(prefix={}, step_index=identity, offset=0, use_micro_step=False)
        min_step = min(axis.step_keys) if axis.step_keys else 0
        max_step = max(axis.step_keys) if axis.step_keys else 0
        span = axis.step_count
        step_axis_len = span
    return ConcatPlan(
        concat_dim=Data2DBConst.CONCAT_DIM_NONE,
        step_mapper_a=mapper,
        step_mapper_b=mapper,
        rank_offset_b=0,
        min_step=min_step,
        max_step=max_step,
        max_rank=axis.max_rank,
        boundary=0,
        span_a=span,
        span_b=0,
        step_axis_len=step_axis_len,
    )


def build_concat_plan(
    axis_a: DatasetAxis,
    axis_b: DatasetAxis,
    concat_dim: str,
    use_micro_step: bool,
) -> ConcatPlan:
    """构造两份数据的拼接计划。

    :param axis_a: 被测数据的轴信息
    :param axis_b: 对比数据的轴信息
    :param concat_dim: 拼接维度, 取 step 或 rank
    :param use_micro_step: 是否启用微步计数
    :return: 拼接计划
    :raises ValueError: 拼接维度非法时抛出
    """
    if concat_dim not in Data2DBConst.CONCAT_DIM_CHOICES:
        raise ValueError(f"Unsupported concat_dim: {concat_dim}. Supported: {list(Data2DBConst.CONCAT_DIM_CHOICES)}")

    warnings = []
    if concat_dim == Data2DBConst.CONCAT_DIM_STEP:
        plan = _build_step_concat_plan(axis_a, axis_b, use_micro_step, warnings)
    else:
        plan = _build_rank_concat_plan(axis_a, axis_b, use_micro_step, warnings)
    plan.warnings = warnings
    return plan


def _build_step_concat_plan(
    axis_a: DatasetAxis,
    axis_b: DatasetAxis,
    use_micro_step: bool,
    warnings: List[str],
) -> ConcatPlan:
    """构造 step 维度拼接的计划。

    两份数据在 step 轴上首尾相接: 被测数据占据前一段, 对比数据整体后移。微步开启
    时, 后移量为被测数据的微步时间线总长度; 微步信息未知时 plan 内的偏移置 0 并告警,
    实际边界由解析阶段按实际微步累加确定。
    """
    if use_micro_step:
        prefix_a = build_micro_prefix(axis_a.micro_counts or {})
        prefix_b = build_micro_prefix(axis_b.micro_counts or {})
        span_a = axis_a.micro_total
        span_b = axis_b.micro_total
        offset_b = span_a if span_a is not None else 0
        if span_a is None or span_b is None:
            warnings.append(
                "The micro step counts of the data sources are unknown, the step axis length and "
                "offset of the concat plan are unavailable; the actual boundary is determined while "
                "parsing the dump data."
            )
        mapper_a = StepMapper(prefix=prefix_a, step_index=axis_a.step_index, offset=0, use_micro_step=True)
        mapper_b = StepMapper(prefix=prefix_b, step_index=axis_b.step_index, offset=offset_b, use_micro_step=True)
    else:
        span_a = axis_a.step_count
        span_b = axis_b.step_count
        mapper_a = StepMapper(prefix={}, step_index=axis_a.step_index, offset=0, use_micro_step=False)
        mapper_b = StepMapper(prefix={}, step_index=axis_b.step_index, offset=span_a, use_micro_step=False)

    if axis_a.step_count != axis_b.step_count:
        warnings.append(
            f"The two datasets have different step counts ({axis_a.step_count} vs {axis_b.step_count}), "
            f"they are concatenated along the step axis as they are."
        )
    if span_a is not None and span_b is not None:
        max_step = span_a + span_b - 1
        step_axis_len = span_a + span_b
    else:
        max_step = None
        step_axis_len = None
    return ConcatPlan(
        concat_dim=Data2DBConst.CONCAT_DIM_STEP,
        step_mapper_a=mapper_a,
        step_mapper_b=mapper_b,
        rank_offset_b=0,
        min_step=0,
        max_step=max_step,
        max_rank=max(axis_a.max_rank, axis_b.max_rank),
        boundary=span_a if span_a is not None else 0,
        span_a=span_a,
        span_b=span_b,
        step_axis_len=step_axis_len,
    )


def _build_rank_concat_plan(
    axis_a: DatasetAxis,
    axis_b: DatasetAxis,
    use_micro_step: bool,
    warnings: List[str],
) -> ConcatPlan:
    """构造 rank 维度拼接的计划。

    两份数据共享同一条 step 轴, 在 rank 轴上首尾相接。rank 偏移取被测数据的 rank 轴
    跨度(最大 rank 编号加一)而非 rank 数量, 保证被测数据的 rank 编号不连续时, 对比
    数据的 rank 平移后仍与之不重叠。
    """
    if use_micro_step:
        if axis_a.micro_known and axis_b.micro_known:
            merged_counts = _merge_micro_counts(axis_a, axis_b)
            common_prefix = build_micro_prefix(merged_counts)
            span = sum(merged_counts.values())
        else:
            # 微步信息未知时无法给出公共微步时间线, 不用 0 或 -1 伪装成有效值
            common_prefix = {}
            span = None
            warnings.append(
                "The micro step counts of the data sources are unknown, the shared step axis length "
                "of the concat plan is unavailable; it is determined while parsing the dump data."
            )
        mapper_a = StepMapper(
            prefix={step: common_prefix.get(offset, 0) for offset, step in enumerate(axis_a.step_keys)},
            step_index=axis_a.step_index,
            offset=0,
            use_micro_step=True,
        )
        mapper_b = StepMapper(
            prefix={step: common_prefix.get(offset, 0) for offset, step in enumerate(axis_b.step_keys)},
            step_index=axis_b.step_index,
            offset=0,
            use_micro_step=True,
        )
        _check_micro_counts_mismatch(axis_a, axis_b, warnings)
    else:
        span = max(axis_a.step_count, axis_b.step_count)
        mapper_a = StepMapper(prefix={}, step_index=axis_a.step_index, offset=0, use_micro_step=False)
        mapper_b = StepMapper(prefix={}, step_index=axis_b.step_index, offset=0, use_micro_step=False)
        if axis_a.step_count != axis_b.step_count:
            warnings.append(
                f"The two datasets have different step counts ({axis_a.step_count} vs {axis_b.step_count}), "
                f"the shared step axis covers {span} steps."
            )

    rank_offset_b = axis_a.rank_span
    if axis_a.rank_count != axis_b.rank_count:
        warnings.append(
            f"The two datasets have different rank counts ({axis_a.rank_count} vs {axis_b.rank_count}), "
            f"they are concatenated along the rank axis as they are."
        )
    if axis_a.rank_ids != list(range(axis_a.rank_span)):
        warnings.append(
            f"Rank ids of dataset1 are not contiguous ({axis_a.rank_ids}), "
            f"the compare dataset is shifted to start from rank {rank_offset_b}."
        )
    return ConcatPlan(
        concat_dim=Data2DBConst.CONCAT_DIM_RANK,
        step_mapper_a=mapper_a,
        step_mapper_b=mapper_b,
        rank_offset_b=rank_offset_b,
        min_step=0,
        max_step=span - 1 if span is not None else None,
        max_rank=max(axis_a.max_rank, axis_b.max_rank + rank_offset_b),
        boundary=rank_offset_b,
        span_a=axis_a.rank_span,
        span_b=axis_b.rank_span,
        step_axis_len=span,
    )


def _check_micro_counts_mismatch(
    axis_a: DatasetAxis,
    axis_b: DatasetAxis,
    warnings: List[str],
) -> None:
    """两份数据同一迭代步的微步数不一致时给出告警。"""
    if not axis_a.micro_known or not axis_b.micro_known:
        return
    mismatch_steps = []
    for offset in range(min(axis_a.step_count, axis_b.step_count)):
        count_a = axis_a.micro_counts.get(axis_a.step_keys[offset], 0)
        count_b = axis_b.micro_counts.get(axis_b.step_keys[offset], 0)
        if count_a != count_b:
            mismatch_steps.append(axis_a.step_keys[offset])
    if mismatch_steps:
        warnings.append(
            f"The two datasets have different micro step counts at step {mismatch_steps} of dataset1, "
            f"the shared step axis is widened by the larger micro step count."
        )
