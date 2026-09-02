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
Function:
AlgorithmScheduler class.
Manages builtin and custom compare algorithms.
"""

import importlib
import inspect
import os
import re
import sys
import threading
import numpy as np
import torch
from typing import Any, Dict, List
from msprobe.core.common.log import logger


class AlgorithmScheduler:
    """
    The class for algorithm manager
    Singleton: please use AlgorithmScheduler.get_instance() to get instance, do not call constructor directly
    """

    ALGORITHM_PATH = "msprobe.core.compare.algorithm"
    BUILT_IN_ALGORITHM_DIR_NAME = "builtin_algorithm"
    CUSTOM_ALGORITHM_DIR_NAME = "custom_algorithm"
    ALGORITHM_FILE_NAME_PATTERN = r"^alg_([a-z0-9_]+)\.py$"
    COMPARE_FUNC_NAME = "compare"
    COLUMN_NAME_FUNC_NAME = "column_name"
    ARG_COUNT = 2  # compare方法入参的个数

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Thread‑safe lazy singleton entry, only this way to get instance"""
        if cls._instance is None:
            with cls._lock:
                # double‑check lock
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.__init__()  # pylint: disable=unnecessary-dunder-call
                    cls._instance = instance
        return cls._instance

    def __new__(cls):
        raise RuntimeError("Do not instantiate AlgorithmScheduler directly, use AlgorithmScheduler.get_instance()")

    def __init__(self):
        # only invoked inside get_instance
        if os.path.dirname(__file__) not in sys.path:
            sys.path.append(os.path.dirname(__file__))
        # 懒加载：初始化不扫描算法目录、不 import 任何算法模块。
        # 算法文件名扫描、列名映射构建均延迟到首次使用时：
        # - 首次 compare（内置列名）：扫描目录 + import 内置算法
        # - 首次 get_algorithm_column_names（全量列名）：额外 import 自定义算法
        self.build_in_support_algorithm: List[str] = []
        self.custom_support_algorithm: List[str] = []
        self._module_cache: Dict[str, Any] = {}
        self._builtin_column_name_mapping: Dict[str, str] = {}  # 列名 -> 内置算法名
        self._full_column_name_mapping: Dict[str, str] = {}  # 列名 -> 算法名（内置+自定义）
        self._discovered = False
        self._builtin_mapping_loaded = False
        self._full_mapping_loaded = False
        self._mapping_lock = threading.Lock()

    def _make_support_algorithm(self) -> None:
        buildin_dir = os.path.join(os.path.dirname(__file__), self.BUILT_IN_ALGORITHM_DIR_NAME)
        self._make_support_algorithm_from_dir(buildin_dir, self.build_in_support_algorithm)
        custom_dir = os.path.join(os.path.dirname(__file__), self.CUSTOM_ALGORITHM_DIR_NAME)
        self._make_support_algorithm_from_dir(custom_dir, self.custom_support_algorithm)

    def _make_support_algorithm_from_dir(self, dir_path: str, algorithm_list: List[str]) -> None:
        if not os.path.exists(dir_path):
            logger.warning(f"The algorithm directory {dir_path} does not exist, skip.")
            return

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            self._add_algorithm_file_to_list(file_path, algorithm_list)

    def _ensure_algorithm_discovered(self) -> None:
        """按需扫描算法目录，仅收集算法文件名，不 import 模块"""
        if self._discovered:
            return
        with self._mapping_lock:
            if self._discovered:
                return
            self._make_support_algorithm()
            self._discovered = True

    def _ensure_builtin_mapping_loaded(self) -> None:
        """按需构建内置算法列名映射（首次调用时才扫描目录并 import 内置算法模块）"""
        self._ensure_algorithm_discovered()
        if self._builtin_mapping_loaded:
            return
        with self._mapping_lock:
            if self._builtin_mapping_loaded:
                return
            self._build_mapping(self.build_in_support_algorithm, self._builtin_column_name_mapping)
            self._builtin_mapping_loaded = True

    def _ensure_full_mapping_loaded(self) -> None:
        """按需构建全量列名映射（内置+自定义，首次调用时才 import 自定义算法模块）"""
        self._ensure_builtin_mapping_loaded()
        if self._full_mapping_loaded:
            return
        with self._mapping_lock:
            if self._full_mapping_loaded:
                return
            full_mapping = dict(self._builtin_column_name_mapping)
            self._build_mapping(self.custom_support_algorithm, full_mapping)
            self._full_column_name_mapping = full_mapping
            self._full_mapping_loaded = True

    def _build_mapping(self, algorithm_names: List[str], mapping: Dict[str, str]) -> None:
        """加载算法模块并构建列名 -> 算法名映射，校验保留列冲突与列名唯一性"""
        reserved_columns = self._get_reserved_column_names()
        for algo_name in algorithm_names:
            module = self._get_module(algo_name)
            col_name = self._get_column_name(algo_name, module)
            if col_name in reserved_columns:
                raise ValueError(
                    f"The column name '{col_name}' of algorithm [{algo_name}] conflicts with a reserved "
                    f"result column. Reserved columns: {sorted(reserved_columns)}. "
                    f"Please use a different column name."
                )
            if col_name in mapping:
                raise ValueError(
                    f"The column name {col_name} is duplicated for algorithm {algo_name} and {mapping[col_name]}."
                )
            mapping[col_name] = algo_name
            logger.info(f"Load algorithm [{algo_name}] success, column: {col_name}")

    @staticmethod
    def _get_reserved_column_names() -> set:
        """
        返回真实数据比对结果完整表头中由框架管理的固定列名集合。
        算法列名不得与这些固定列名冲突，否则会覆盖框架写入的固定列数据。
        """
        from msprobe.core.common.const import CompareConst

        fixed_columns = (
            CompareConst.BASIC_INFO
            + CompareConst.SUMMARY_INFO
            + CompareConst.EXTRACT_INDEX
            + [CompareConst.STACK, CompareConst.DATA_NAME, CompareConst.DIRTY_VALID_LEN]
        )
        return set(fixed_columns)

    @staticmethod
    def _add_algorithm_file_to_list(file_path: str, algorithm_list: List[str]) -> bool:
        if not os.path.isfile(file_path):
            return False
        file_name = os.path.basename(file_path)
        match_result = re.match(AlgorithmScheduler.ALGORITHM_FILE_NAME_PATTERN, file_name)
        if not match_result:
            if file_name != "__init__.py":
                logger.warning(
                    f"The algorithm file {file_name} does not match the pattern "
                    f"{AlgorithmScheduler.ALGORITHM_FILE_NAME_PATTERN}, skip."
                )
            return False
        algorithm_list.append(match_result.group(1))
        return True

    def _get_module(self, algorithm_name: str) -> Any:
        if algorithm_name in self._module_cache:
            return self._module_cache[algorithm_name]
        if algorithm_name in self.custom_support_algorithm:
            module_name = f"{self.ALGORITHM_PATH}.{self.CUSTOM_ALGORITHM_DIR_NAME}.alg_{algorithm_name}"
            algorithm_module = importlib.import_module(module_name)
        elif algorithm_name in self.build_in_support_algorithm:
            module_name = f"{self.ALGORITHM_PATH}.{self.BUILT_IN_ALGORITHM_DIR_NAME}.alg_{algorithm_name}"
            algorithm_module = importlib.import_module(module_name)
        else:
            raise ValueError(f"Algorithm {algorithm_name} is not supported.")

        if not hasattr(algorithm_module, self.COMPARE_FUNC_NAME):
            raise AttributeError(f"Algorithm {algorithm_name} must define a {self.COMPARE_FUNC_NAME} function.")

        compare_func = getattr(algorithm_module, self.COMPARE_FUNC_NAME)
        if not callable(compare_func):
            raise TypeError(f"{self.COMPARE_FUNC_NAME} in {algorithm_name} is not a callable function.")

        sig = inspect.signature(compare_func)
        required_params = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(required_params) != self.ARG_COUNT:
            raise ValueError(
                f"{self.COMPARE_FUNC_NAME} in {algorithm_name} must have {self.ARG_COUNT} positional arguments."
            )

        self._module_cache[algorithm_name] = algorithm_module
        return algorithm_module

    def _get_column_name(self, algorithm_name: str, algorithm_module: Any) -> str:
        if not hasattr(algorithm_module, self.COLUMN_NAME_FUNC_NAME):
            raise AttributeError(f"Algorithm {algorithm_name} must define a {self.COLUMN_NAME_FUNC_NAME} function.")
        column_name_func = getattr(algorithm_module, self.COLUMN_NAME_FUNC_NAME)
        if not callable(column_name_func):
            raise TypeError(f"{self.COLUMN_NAME_FUNC_NAME} in {algorithm_name} is not a callable function.")
        try:
            column_name = column_name_func()
        except Exception as e:
            raise ValueError(f"Error occurred when calling {self.COLUMN_NAME_FUNC_NAME} in {algorithm_name}: {e}")
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(f"{self.COLUMN_NAME_FUNC_NAME} in {algorithm_name} must return a non‑empty string.")
        return column_name

    def _validate_return_value(self, value: Any, algorithm_name: str) -> bool:
        """
        校验自定义比对算法返回值格式
        result_value: int | float | str
        """
        if algorithm_name not in self.build_in_support_algorithm:
            if not self._is_real_number(value) and not isinstance(value, str):
                raise ValueError(
                    f"Algorithm [{algorithm_name}] return value type invalid. "
                    f"Expected int/float/str, got {type(value).__name__}"
                )
        return True

    def _is_real_number(self, value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _call_algorithm(self, algorithm_name, n_value, b_value):
        if algorithm_name not in self.build_in_support_algorithm:
            if isinstance(n_value, np.ndarray):
                n_value = torch.from_numpy(n_value)
            if isinstance(b_value, np.ndarray):
                b_value = torch.from_numpy(b_value)
            with torch.no_grad():
                result = self._call_compare(algorithm_name, n_value, b_value)
                if not isinstance(result, tuple):
                    return result, ""
                return result
        return self._call_compare(algorithm_name, n_value, b_value)

    def _call_compare(self, algorithm_name, n_value, b_value):
        try:
            algorithm_module = self._get_module(algorithm_name)
            compare_func = getattr(algorithm_module, self.COMPARE_FUNC_NAME)
            result = compare_func(n_value, b_value)
            self._validate_return_value(result, algorithm_name)
            return result
        except Exception as e:
            logger.error(f"Call algorithm [{algorithm_name}] failed, error: {e}")
            return "unsupported", ""

    def _get_algorithm_by_column(self, column_name: str):
        self._ensure_builtin_mapping_loaded()
        if column_name in self._builtin_column_name_mapping:
            return self._builtin_column_name_mapping[column_name]
        self._ensure_full_mapping_loaded()
        return self._full_column_name_mapping.get(column_name)

    def compare(self, n_value, b_value, column_names=None):
        if not column_names:
            column_names = self.get_algorithm_column_names()
        results = []
        error_msgs = []
        for column_name in column_names:
            algo_name = self._get_algorithm_by_column(column_name)
            if algo_name is not None:
                result, error_msg = self._call_algorithm(algo_name, n_value, b_value)
            else:
                result, error_msg = "unsupported", f"No available algorithm for {column_name}"
            results.append(result)
            error_msgs.append(error_msg if error_msg else "")
        return results, error_msgs

    def get_algorithm_column_names(self) -> List[str]:
        self._ensure_full_mapping_loaded()
        return list(self._full_column_name_mapping.keys())
