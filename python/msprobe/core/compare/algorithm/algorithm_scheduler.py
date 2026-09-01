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
    RESULT_COUNT = 2  # compare方法返回值的个数

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
        self.build_in_support_algorithm: List[str] = []
        self.custom_support_algorithm: List[str] = []
        # 为了校验column_name唯一性，初始化全量加载
        self._module_cache: Dict[str, Any] = {}
        self._column_name_algorithm_mapping: Dict[str, str] = {}  # 算法名 -> 列名
        self._make_support_algorithm()
        self.algorithm_names = self.build_in_support_algorithm + self.custom_support_algorithm
        self._validate_column_names()
        self.algorithm_column_names = list(self._column_name_algorithm_mapping.keys())

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

    def _validate_column_names(self) -> None:
        reserved_columns = self._get_reserved_column_names()
        all_algorithms = self.get_algorithm_names()
        for algo_name in all_algorithms:
            module = self._get_module(algo_name)
            col_name = self._get_column_name(algo_name, module)
            if col_name in reserved_columns:
                raise ValueError(
                    f"The column name '{col_name}' of algorithm [{algo_name}] conflicts with a reserved "
                    f"result column. Reserved columns: {sorted(reserved_columns)}. "
                    f"Please use a different column name."
                )
            if col_name in self._column_name_algorithm_mapping:
                raise ValueError(
                    f"The column name {col_name} is duplicated for algorithm {algo_name} "
                    f"and {self._column_name_algorithm_mapping[col_name]}."
                )
            self._column_name_algorithm_mapping[col_name] = algo_name
        for col_name, algo_name in self._column_name_algorithm_mapping.items():
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
        要求返回二元组: (result_value, message)
            result_value: int | float | str
            message: str
        """
        if not isinstance(value, tuple) or len(value) != self.RESULT_COUNT:
            raise ValueError(
                f"Algorithm [{algorithm_name}] return value invalid. Expected {self.RESULT_COUNT}‑element tuple"
            )

        ret_val, ret_msg = value

        if not self._is_real_number(ret_val) and not isinstance(ret_val, str):
            raise ValueError(
                f"Algorithm [{algorithm_name}] first return value type invalid. "
                f"Expected int/float/str, got {type(ret_val).__name__}"
            )

        if not isinstance(ret_msg, str):
            raise ValueError(
                f"Algorithm [{algorithm_name}] second return value type invalid. "
                f"Expected str(message), got {type(ret_msg).__name__}"
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
                return self._call_compare(algorithm_name, n_value, b_value)
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
            return "unsupported", str(e)

    def compare(self, n_value, b_value, column_names=None):
        if not column_names:
            column_names = self.algorithm_column_names
        results = []
        error_msgs = []
        for column_name in column_names:
            if column_name in self._column_name_algorithm_mapping:
                result, error_msg = self._call_algorithm(
                    self._column_name_algorithm_mapping[column_name], n_value, b_value
                )
            else:
                result, error_msg = "unsupported", f"No available algorithm for {column_name}"
            results.append(result)
            error_msgs.append(error_msg if error_msg else "")
        return results, error_msgs

    def get_algorithm_names(self) -> List[str]:
        return self.algorithm_names

    def get_algorithm_column_names(self) -> List[str]:
        return self.algorithm_column_names
