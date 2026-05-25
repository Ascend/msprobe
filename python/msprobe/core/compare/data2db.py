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

import os
from typing import Dict, Optional, Callable

from msprobe.core.common.file_utils import (
    check_file_or_directory_path,
    create_directory,
    load_json,
    remove_path,
    recursive_chmod
)
from msprobe.core.common.log import logger
from msprobe.core.common.utils import is_int
from msprobe.core.common.const import Data2DBConst
from msprobe.core.monitor.csv2db import import_data as monitor_import_data, CSV2DBConfig
from msprobe.core.dump.dump2db.dump2db import DumpRecordBuilder, DumpDB
from msprobe.core.dump.dump2db.dump2db import scan_files as dump_scan_files
from msprobe.core.monitor.utils import get_target_output_dir


def validate_micro_step(micro_step) -> bool:
    """转换micro_step参数为布尔值
    """
    # 命令行已限制只能输入'(T)true'或'(F)false'，直接转换
    return micro_step.lower() == 'true'


def validate_process_num(process_num: int) -> None:
    """Validate process number parameter"""
    if not is_int(process_num) or process_num <= 0:
        raise ValueError("process_num must be a positive integer")
    if process_num > Data2DBConst.MAX_PROCESS_NUM:
        raise ValueError(
            f"Maximum supported process_num is {Data2DBConst.MAX_PROCESS_NUM}"
        )


def load_mapping(mapping_path: Optional[str]) -> dict:
    if mapping_path and isinstance(mapping_path, str):
        return load_json(mapping_path)
    return {}


class DBImporter:
    """统一数据库导入器"""

    def __init__(
        self,
        db_path: str,
        data_path: str,
        format: str = 'auto',
        mapping_path: Optional[str] = None,
        micro_step: bool = True,
        process_num: int = 1
    ):
        self.db_path = db_path
        self.data_path = data_path
        self.format = format
        self.mapping_path = mapping_path
        self.micro_step = micro_step
        self.process_num = process_num

        # Load and validate
        self.mapping = load_mapping(self.mapping_path)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """验证所有输入参数"""
        # Validate data path
        check_file_or_directory_path(self.data_path, isdir=True)
        create_directory(self.db_path)

        # Validate flags
        self.micro_step = validate_micro_step(self.micro_step)
        validate_process_num(self.process_num)

        # Validate format
        supported_formats = {'auto', 'dump', 'monitor'}
        if self.format not in supported_formats:
            raise ValueError(f"Unsupported format: {self.format}. "
                             f"Supported: {supported_formats}")

    def _ensure_db_file_clean(self, db_file: str) -> None:
        """确保目标数据库文件不存在（若存在则删除）"""
        if os.path.exists(db_file):
            logger.warning(f"Existing database will be replaced: {db_file}")
            remove_path(db_file)

    def import_dump_data(self) -> None:
        """导入 dump 格式数据"""
        logger.info("Starting dump data import...")

        valid_ranks = dump_scan_files(self.data_path)
        if not valid_ranks:
            logger.warning(
                f"No valid 'step*' directories found in: {self.data_path}"
            )
            return
        dump_db_file = os.path.join(self.db_path, Data2DBConst.DB_DUMP)
        self._ensure_db_file_clean(dump_db_file)
        
        db = DumpDB(dump_db_file)
        builder = DumpRecordBuilder(
            db=db,
            data_dir=self.data_path,
            mapping=self.mapping,
            micro_step=self.micro_step
        )

        builder.import_data(valid_ranks)
        logger.info(f"Dump data import completed. DB: {dump_db_file}")

    def import_monitor_data(self) -> None:
        """导入 monitor 格式数据"""
        logger.info("Starting monitor data import...")

        target_output_dirs = get_target_output_dir(self.data_path, None, None)
        if not target_output_dirs:
            logger.warning(f"No valid monitor directories found in: {self.data_path}")
            return

        monitor_db_file = os.path.join(self.db_path, Data2DBConst.DB_MONITOR)
        self._ensure_db_file_clean(monitor_db_file)

        config = CSV2DBConfig(
            target_output_dirs=target_output_dirs,
            process_num=self.process_num,
            db_file=monitor_db_file,
            mapping=self.mapping,
            micro_step=self.micro_step
        )

        success = monitor_import_data(config=config)
        if success:
            logger.info(f"Monitor data import completed. DB: {monitor_db_file}")

    def import_data(self) -> None:
        """主导入方法"""
        converters: Dict[str, Callable[[], None]] = {
            "dump": self.import_dump_data,
            "monitor": self.import_monitor_data
        }

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
    parser.add_argument(
        '--db', type=str, required=True,
        help='Path to SQLite database output directory'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to input data directory'
    )
    parser.add_argument(
        '--format', type=str, choices=['auto', 'dump', 'monitor'], default='auto',
        help='Data format (default: auto)'
    )
    parser.add_argument(
        '--mapping', type=str, default=None,
        help='Path to optional JSON mapping file'
    )
    parser.add_argument(
        '--micro_step', type=str, choices=['true', 'false', 'True', 'False'], default='true',
        help='Use micro-step counting (default: true)'
    )
    parser.add_argument(
        '--process_num', type=int, default=1,
        help='Number of parallel processes for monitor data (default: 1)'
    )


def _data2db_command(args):
    importer = DBImporter(
        db_path=args.db,
        data_path=args.data,
        format=args.format,
        mapping_path=args.mapping,
        micro_step=args.micro_step,
        process_num=args.process_num
    )
    importer.import_data()
    recursive_chmod(args.db)