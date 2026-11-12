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

import os
from typing import Dict, List, Optional, Callable

from msprobe.core.common.file_utils import (
    check_file_or_directory_path,
    create_directory,
    load_json,
    remove_path,
)
from msprobe.core.common.log import logger
from msprobe.core.common.utils import is_int
from msprobe.core.common.const import Data2DBConst
from msprobe.core.common.data_concat import (
    DatasetAxis,
    build_concat_plan,
    build_single_plan,
)
from msprobe.core.monitor.csv2db import import_data as monitor_import_data, CSV2DBConfig
from msprobe.core.monitor.db_utils import MonitorDB
from msprobe.core.dump.dump2db.dump2db import DumpRecordBuilder, DumpDB
from msprobe.core.dump.dump2db.dump2db import scan_files as dump_scan_files
from msprobe.core.dump.dump2db.dump2db import build_dump_step_groups, merge_dump_step_groups
from msprobe.core.monitor.utils import get_target_output_dir


def validate_micro_step(micro_step) -> bool:
    """转换micro_step参数为布尔值"""
    # 命令行已限制只能输入'(T)true'或'(F)false'，直接转换
    return micro_step.lower() == 'true'


def validate_process_num(process_num: int) -> None:
    """Validate process number parameter"""
    if not is_int(process_num) or process_num <= 0:
        raise ValueError("process_num must be a positive integer")
    if process_num > Data2DBConst.MAX_PROCESS_NUM:
        raise ValueError(f"Maximum supported process_num is {Data2DBConst.MAX_PROCESS_NUM}")


def validate_concat_dim(concat_dim: str) -> None:
    """Validate the concat dimension parameter"""
    if concat_dim not in Data2DBConst.CONCAT_DIM_CHOICES:
        raise ValueError(f"Unsupported concat_dim: {concat_dim}. Supported: {list(Data2DBConst.CONCAT_DIM_CHOICES)}")


def load_mapping(mapping_path: Optional[str]) -> dict:
    if mapping_path and isinstance(mapping_path, str):
        return load_json(mapping_path)
    return {}


def build_dump_axis(valid_ranks: Dict[int, List]) -> DatasetAxis:
    """从 dump 扫描结果中归纳出一份数据的 step 轴与 rank 轴信息。

    dump 数据的微步数需要在解析 construct.json 时才能得到, 扫描阶段无法获取,
    因此 micro_counts 为 None, 表示微步信息未知。微步计数开启时, 拼接计划中的
    span 与 max_step 同样不可用, 由 dump 解析链路按实际累加结果给出。
    """
    step_keys = sorted(valid_ranks.keys())
    rank_ids = []
    max_rank = 0
    for step in step_keys:
        for rank, _, _ in valid_ranks[step]:
            rank_ids.append(rank)
            max_rank = max(max_rank, rank)
    return DatasetAxis(
        step_keys=step_keys,
        micro_counts=None,
        rank_ids=sorted(set(rank_ids)),
        max_rank=max_rank,
    )


class DBImporter:
    """统一数据库导入器"""

    def __init__(
        self,
        db_path: str,
        data_path: str,
        format: str = 'auto',  # pylint: disable=redefined-builtin
        mapping_path: Optional[str] = None,
        micro_step: bool = True,
        process_num: int = 1,
        compare_data_path: Optional[str] = None,
        concat_dim: str = Data2DBConst.CONCAT_DIM_STEP,
    ):
        self.db_path = db_path
        self.data_path = data_path
        self.format = format
        self.mapping_path = mapping_path
        self.micro_step = micro_step
        self.process_num = process_num
        self.compare_data_path = compare_data_path
        self.concat_dim = concat_dim

        # Load and validate
        self.mapping = load_mapping(self.mapping_path)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """验证所有输入参数"""
        # Validate data path
        check_file_or_directory_path(self.data_path, isdir=True)
        if self.compare_data_path:
            check_file_or_directory_path(self.compare_data_path, isdir=True)
        create_directory(self.db_path)

        # Validate flags
        self.micro_step = validate_micro_step(self.micro_step)
        validate_process_num(self.process_num)
        validate_concat_dim(self.concat_dim)

        # Validate format
        supported_formats = {'auto', 'dump', 'monitor'}
        if self.format not in supported_formats:
            raise ValueError(f"Unsupported format: {self.format}. Supported: {supported_formats}")
        if self.compare_data_path and self.format == 'auto':
            # 混合数据目录下 auto 会优先识别 dump, 比对模式下必须由用户明确数据类型
            raise ValueError(
                "When --compare_data is specified, --format must be set to 'dump' or 'monitor' explicitly."
            )

        if not self.compare_data_path:
            logger.info("No compare data path is given, the concat_dim parameter takes no effect.")

    def _ensure_db_file_clean(self, db_file: str) -> None:
        """确保目标数据库文件不存在（若存在则删除）"""
        if os.path.exists(db_file):
            logger.warning(f"Existing database will be replaced: {db_file}")
            remove_path(db_file)

    def _write_concat_info(
        self,
        db,
        concat_dim: str,
        dataset_count: int,
        boundary: int,
        span_a: int,
        span_b: int,
    ) -> None:
        """把拼接信息写入数据库, 便于用户定位两组数据的分界点"""
        data_path = self.data_path
        compare_data_path = self.compare_data_path
        try:
            db.db_manager.insert_data(
                "concat_info",
                [
                    (
                        concat_dim,
                        dataset_count,
                        boundary,
                        span_a,
                        span_b,
                        data_path,
                        compare_data_path,
                    )
                ],
                key_list=[
                    "concat_dim",
                    "dataset_count",
                    "boundary",
                    "dataset1_span",
                    "dataset2_span",
                    "data_path",
                    "compare_data_path",
                ],
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Failed to write concat info: {e}")

    def _log_concat_info(self, concat_dim: str, boundary: int, span_a: int, span_b: int) -> None:
        """打印拼接信息"""
        if concat_dim == Data2DBConst.CONCAT_DIM_RANK:
            logger.info(
                f"[Concat] dim=rank, dataset1: rank[0, {span_a - 1}], "
                f"dataset2: rank[{span_a}, {span_a + span_b - 1}], boundary={boundary}"
            )
        elif concat_dim == Data2DBConst.CONCAT_DIM_STEP:
            logger.info(
                f"[Concat] dim=step, dataset1: step[0, {span_a - 1}], "
                f"dataset2: step[{span_a}, {span_a + span_b - 1}], boundary={boundary}"
            )
        else:
            logger.info("[Concat] single data source, no concatenation is performed.")

    def import_dump_data(self) -> None:
        """导入 dump 格式数据"""
        logger.info("Starting dump data import...")

        valid_ranks, valid_step_found = dump_scan_files(self.data_path)
        if not valid_ranks:
            msg = (
                f"No valid dump data discovered in: {self.data_path}, "
                f"dump import skipped. See preceding scan warnings for the exact reason."
            )
            # 如果指定了dump数据解析，或者扫描到step文件，那么没有数据是报ERROR
            if self.format == "dump" or valid_step_found:
                logger.error(msg)
                return
            # 否则INFO级别
            else:
                logger.info(msg)
                return

        compare_ranks = {}
        if self.compare_data_path:
            compare_ranks, _ = dump_scan_files(self.compare_data_path)
            if not compare_ranks:
                logger.warning(
                    f"No valid dump data discovered in the compare data path: {self.compare_data_path}. "
                    f"Two data sources must be of the same format. "
                    f"Fall back to the single data source in: {self.data_path}."
                )
                self.compare_data_path = None

        dump_db_file = os.path.join(self.db_path, Data2DBConst.DB_DUMP)
        self._ensure_db_file_clean(dump_db_file)

        db = DumpDB(dump_db_file)
        builder = DumpRecordBuilder(db=db, data_dir=self.data_path, mapping=self.mapping, micro_step=self.micro_step)

        axis_a = build_dump_axis(valid_ranks)
        groups_a = build_dump_step_groups(valid_ranks, self.data_path, dataset_id=0)
        if compare_ranks:
            axis_b = build_dump_axis(compare_ranks)
            groups_b = build_dump_step_groups(compare_ranks, self.compare_data_path, dataset_id=1)
            plan = build_concat_plan(axis_a, axis_b, self.concat_dim, self.micro_step)
            groups = merge_dump_step_groups(groups_a, groups_b, plan)
        else:
            plan = build_single_plan(axis_a, self.micro_step)
            groups = groups_a
        for warning in plan.warnings:
            logger.warning(warning)

        dataset_end_step = builder.import_data(groups, plan)

        dataset_count = 2 if compare_ranks else 1
        end_a = dataset_end_step.get(0)
        end_b = dataset_end_step.get(1, end_a)
        if dataset_count == 1:
            # 只有一份数据时不存在分界点, 不写入无意义的分界信息
            boundary = 0
            # 开启微步计数时 plan 的 span 不可用, 取解析阶段实际累加的微步总数
            span_a = (end_a if self.micro_step else plan.span_a) or 0
            span_b = 0
        elif plan.concat_dim == Data2DBConst.CONCAT_DIM_RANK:
            boundary = plan.rank_offset_b
            span_a = plan.span_a or 0
            span_b = plan.span_b or 0
        else:
            if end_a is None:
                boundary = 0
                span_a = plan.span_a or 0
                span_b = plan.span_b or 0
            elif self.micro_step:
                boundary = end_a
                span_a = end_a
                span_b = end_b - end_a
            else:
                boundary = end_a + 1
                span_a = plan.span_a or 0
                span_b = plan.span_b or 0
        self._log_concat_info(plan.concat_dim, boundary, span_a, span_b)
        self._write_concat_info(db, plan.concat_dim, dataset_count, boundary, span_a, span_b)
        logger.info(f"Dump data import completed. DB: {dump_db_file}")

    def import_monitor_data(self) -> None:
        """导入 monitor 格式数据"""
        logger.info("Starting monitor data import...")

        target_output_dirs = get_target_output_dir(self.data_path, None, None)
        if not target_output_dirs:
            logger.warning(f"No valid monitor directories found in: {self.data_path}")
            return

        compare_output_dirs = {}
        if self.compare_data_path:
            compare_output_dirs = get_target_output_dir(self.compare_data_path, None, None)
            if not compare_output_dirs:
                logger.warning(
                    f"No valid monitor directories found in the compare data path: {self.compare_data_path}. "
                    f"Two data sources must be of the same format. "
                    f"Fall back to the single data source in: {self.data_path}."
                )
                self.compare_data_path = None

        monitor_db_file = os.path.join(self.db_path, Data2DBConst.DB_MONITOR)
        self._ensure_db_file_clean(monitor_db_file)

        config = CSV2DBConfig(
            target_output_dirs=target_output_dirs,
            process_num=self.process_num,
            db_file=monitor_db_file,
            mapping=self.mapping,
            micro_step=self.micro_step,
            compare_output_dirs=compare_output_dirs or None,
            concat_dim=self.concat_dim,
        )

        success = monitor_import_data(config=config)
        if success:
            if self.compare_data_path and config.compare_output_dirs is None:
                # 对比数据没有可用内容, _pre_scan 已降级为单数据源
                self.compare_data_path = None
            plan = config.concat_plan
            concat_dim = plan.concat_dim if plan is not None else Data2DBConst.CONCAT_DIM_NONE
            # 以实际生效的拼接计划为准, 对比数据不可用时 _pre_scan 会降级为单数据源
            dataset_count = 2 if concat_dim != Data2DBConst.CONCAT_DIM_NONE else 1
            span_a = (plan.span_a if plan is not None else 0) or 0
            span_b = (plan.span_b if plan is not None else 0) or 0
            if dataset_count == 1:
                # 只有一份数据时不存在分界点, 不写入无意义的分界信息
                boundary = 0
            elif plan is not None and plan.concat_dim == Data2DBConst.CONCAT_DIM_RANK:
                boundary = plan.rank_offset_b
            else:
                boundary = span_a
            self._log_concat_info(concat_dim, boundary, span_a, span_b)
            self._write_concat_info(MonitorDB(monitor_db_file), concat_dim, dataset_count, boundary, span_a, span_b)
            logger.info(f"Monitor data import completed. DB: {monitor_db_file}")

    def import_data(self) -> None:
        """主导入方法"""
        converters: Dict[str, Callable[[], None]] = {"dump": self.import_dump_data, "monitor": self.import_monitor_data}

        if self.format == "auto":
            logger.info("Auto-detect mode: attempting both dump and monitor imports")
            for name, func in converters.items():
                logger.info(f"Trying {name} import...")
                func()

        elif self.format in converters:
            logger.info(f"Importing data in '{self.format}' format")
            converters[self.format]()
        else:
            raise ValueError(f"Unsupported format: {self.format}")


def _data2db_service_parser(parser):
    parser.add_argument('--db', type=str, required=True, help='Path to SQLite database output directory')
    parser.add_argument('--data', type=str, required=True, help='Path to input data directory')
    parser.add_argument(
        '--format',
        type=str,
        choices=['auto', 'dump', 'monitor'],
        default='auto',
        help='Type of data to parse: dump, monitor, or auto (default: auto). '
        'When --compare_data is specified, it must be dump or monitor.',
    )
    parser.add_argument('--mapping', type=str, default=None, help='Path to optional JSON mapping file')
    parser.add_argument(
        '--micro_step',
        type=str.lower,
        choices=['true', 'false'],
        default='true',
        help='Use micro-step counting (default: true)',
    )
    parser.add_argument(
        '--process_num', type=int, default=1, help='Number of parallel processes for monitor data (default: 1)'
    )
    parser.add_argument(
        '--compare_data',
        type=str,
        default=None,
        help='Path to the comparison data directory, used to compare two data sources (optional)',
    )
    parser.add_argument(
        '--concat_dim',
        type=str,
        choices=list(Data2DBConst.CONCAT_DIM_CHOICES),
        default=Data2DBConst.CONCAT_DIM_STEP,
        help='Dimension along which the two data sources are concatenated (default: step)',
    )


def _data2db_command(args):
    importer = DBImporter(
        db_path=args.db,
        data_path=args.data,
        format=args.format,
        mapping_path=args.mapping,
        micro_step=args.micro_step,
        process_num=args.process_num,
        compare_data_path=args.compare_data,
        concat_dim=args.concat_dim,
    )
    importer.import_data()
