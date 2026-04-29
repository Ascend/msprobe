#!/usr/bin/env python3
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
Utility helpers that support the torchair accuracy comparison flow.
"""

import datetime
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.file_utils import FileChecker, create_directory, write_df_to_csv
from msprobe.core.common.log import logger
from msprobe.infer.utils.check.rule import Rule
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE
from msprobe.infer.utils.util import safe_torch_load

GLOBAL_HISTORY_AIT_DUMP_PATH_LIST = ["msit_dump", "ait_dump"]

TOKEN_ID = "token_id"
DATA_ID = "data_id"
GOLDEN_DATA_PATH = "golden_data_path"
GOLDEN_DTYPE = "golden_dtype"
GOLDEN_SHAPE = "golden_shape"
GOLDEN_MAX_VALUE = "golden_max_value"
GOLDEN_MIN_VALUE = "golden_min_value"
GOLDEN_MEAN_VALUE = "golden_mean_value"
GOLDEN_OP_TYPE = "golden_op_type"
MY_DATA_PATH = "my_data_path"
MY_DTYPE = "my_dtype"
MY_SHAPE = "my_shape"
MY_MAX_VALUE = "my_max_value"
MY_MIN_VALUE = "my_min_value"
MY_MEAN_VALUE = "my_mean_value"
MY_OP_TYPE = "my_op_type"
CMP_FAIL_REASON = "cmp_fail_reason"
DTYPE = "dtype"
SHAPE = "shape"
MAX_VALUE = "max_value"
MIN_VALUE = "min_value"
MEAN_VALUE = "mean_value"

CSV_GOLDEN_HEADER_BASE = [
    TOKEN_ID,
    DATA_ID,
    GOLDEN_DATA_PATH,
    GOLDEN_OP_TYPE,
    GOLDEN_DTYPE,
    GOLDEN_SHAPE,
    GOLDEN_MAX_VALUE,
    GOLDEN_MIN_VALUE,
    GOLDEN_MEAN_VALUE,
    MY_DATA_PATH,
    MY_OP_TYPE,
    MY_DTYPE,
    MY_SHAPE,
    MY_MAX_VALUE,
    MY_MIN_VALUE,
    MY_MEAN_VALUE,
]


def _lazy_import_torch():
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as err:
        if err.name == "torch":
            logger.error("Missing dependency: torch. Please install torch to use torchair compare.")
        raise
    return torch


def _lazy_cmp_alg_maps():
    from msprobe.infer.utils import cmp_algorithm  # pylint: disable=import-outside-toplevel

    return cmp_algorithm.CMP_ALG_MAP, cmp_algorithm.CUSTOM_ALG_MAP


def _build_csv_golden_header():
    cmp_alg_map, _ = _lazy_cmp_alg_maps()
    header = list(CSV_GOLDEN_HEADER_BASE)
    header.extend(list(cmp_alg_map.keys()))
    header.append(CMP_FAIL_REASON)
    return header


class BasicDataInfo:
    """Metadata holder for a pair of tensors."""

    count_data_id = 0  # Count data_id, increment by 1 every time creating a new instance
    TORCH_UNSUPPORTED_D_TYPE_MAP = {"uint16": "int32", "uint32": "int64"}

    def __init__(self, golden_data_path, my_data_path, token_id=None, data_id=None, op_type=None):
        global_path = os.path.realpath(golden_data_path.split(",")[0])
        self._check_path(global_path)
        my_real_path = os.path.realpath(my_data_path.split(",")[0])
        self._check_path(my_real_path)

        self.my_data_path, self.golden_data_path = os.path.realpath(my_data_path), os.path.realpath(golden_data_path)
        self.token_id = self.get_token_id(self.my_data_path) if token_id is None else token_id
        self.data_id = self.count_data_id if data_id is None else data_id
        self.my_op_type, self.golden_op_type = self._validate_op_type(op_type)
        self._count()

    @staticmethod
    def _check_path(path):
        path_type = FileCheckConst.DIR if os.path.isdir(path) else FileCheckConst.FILE
        FileChecker(path, path_type, ability=FileCheckConst.READ_ABLE).common_check()

    @staticmethod
    def _validate_op_type(op_type):
        if op_type is None:
            return None, None

        if isinstance(op_type, (list, tuple)) and len(op_type) == 2:
            return op_type[0], op_type[1]
        raise ValueError("op_type must be a list or tuple containing two elements")

    @classmethod
    def _count(cls):
        cls.count_data_id += 1

    def to_dict(self):
        return {
            TOKEN_ID: str(self.token_id),
            DATA_ID: str(self.data_id),
            GOLDEN_DATA_PATH: self.golden_data_path,
            GOLDEN_OP_TYPE: self.golden_op_type,
            MY_DATA_PATH: self.my_data_path,
            MY_OP_TYPE: self.my_op_type,
        }

    def get_token_id(self, cur_path):
        dump_filename_idx = 4
        dump_tensor_idx = 3
        dirseg = cur_path.split(os.path.sep)
        if len(dirseg) > 16:
            raise RecursionError(f'The depth of "{cur_path}" directory is too deep.')
        if len(dirseg) < dump_filename_idx:
            return 0
        flag1 = dirseg[-dump_tensor_idx] in {"tensors", "torch_tensors"}
        flag2 = any(dirseg[-dump_filename_idx].startswith(x) for x in GLOBAL_HISTORY_AIT_DUMP_PATH_LIST)
        if flag1 and flag2:
            try:
                token_id = int(dirseg[-1])
            except (IndexError, AttributeError, TypeError, ValueError) as err:
                logger.debug(f"get_token_id error, dirseg: {dirseg}, error: {err}")
                token_id = 0
        else:
            token_id = self.get_token_id(os.path.dirname(cur_path))
        return token_id


def fill_row_data(
    data_info: BasicDataInfo,
    loaded_my_data: Any = None,
    loaded_golden_data: Any = None,
):
    golden_data_path, my_data_path = data_info.golden_data_path, data_info.my_data_path
    row_data = data_info.to_dict()
    if loaded_golden_data is None and not os.path.isfile(golden_data_path):
        row_data[CMP_FAIL_REASON] = f"golden_data_path: {golden_data_path} is not a file."
        return row_data
    if loaded_my_data is None and not os.path.isfile(my_data_path):
        row_data[CMP_FAIL_REASON] = f"my_data_path: {my_data_path} is not a file."
        return row_data
    golden_data = load_as_torch_tensor(golden_data_path, loaded_golden_data)
    my_data = load_as_torch_tensor(my_data_path, loaded_my_data)

    compare_metrics_dict = compare_data(golden_data, my_data)
    tensor_basic_info_dict = set_tensor_basic_info_in_row_data(golden_data, my_data)
    row_data.update(compare_metrics_dict)
    row_data.update(tensor_basic_info_dict)

    return row_data


def load_as_torch_tensor(data_path, loaded_data=None):
    torch = _lazy_import_torch()
    if loaded_data is not None:
        if str(loaded_data.dtype) in BasicDataInfo.TORCH_UNSUPPORTED_D_TYPE_MAP:
            mapped = BasicDataInfo.TORCH_UNSUPPORTED_D_TYPE_MAP.get(str(loaded_data.dtype))
            loaded_data = loaded_data.astype(mapped)
        return loaded_data if isinstance(loaded_data, torch.Tensor) else torch.from_numpy(loaded_data)
    return read_data(data_path)


def get_tensor_basic_info(data: Any) -> Dict[str, Any]:
    tensor_info: Dict[str, Any] = {}
    tensor_info[DTYPE] = str(data.dtype)
    tensor_info[SHAPE] = str(list(data.shape))
    if 0 not in data.shape:
        data = data.float()
        tensor_info[MAX_VALUE] = data.max().item()
        tensor_info[MIN_VALUE] = data.min().item()
        tensor_info[MEAN_VALUE] = data.mean().item()
    return tensor_info


def set_tensor_basic_info_in_row_data(golden_data, my_data):
    row_data: Dict[str, Any] = {}
    golden_info = get_tensor_basic_info(golden_data)
    row_data.update({f"golden_{key}": value for key, value in golden_info.items()})

    my_info = get_tensor_basic_info(my_data)
    row_data.update({f"my_{key}": value for key, value in my_info.items()})
    return row_data


def save_compare_result_to_csv(gathered_row_data, output_path=".", columns=None, rank_id=-1):
    create_directory(output_path)
    cur_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    columns = columns or _build_csv_golden_header()
    if rank_id != -1:
        csv_save_path = os.path.join(output_path, f"msprobe_cmp_report_rank{rank_id}_{cur_time}.csv")
    else:
        csv_save_path = os.path.join(output_path, f"msprobe_cmp_report_{cur_time}.csv")
    gathered_row_data = list(
        filter(
            lambda item: not (CMP_FAIL_REASON in item and item[CMP_FAIL_REASON] == "data shape doesn't match."),
            gathered_row_data,
        )
    )
    for row_data in gathered_row_data[:]:
        if GOLDEN_DTYPE in row_data and MY_DTYPE in row_data:
            if (row_data[GOLDEN_DTYPE] == "torch.int8") ^ (row_data[MY_DTYPE] == "torch.int8"):
                gathered_row_data.remove(row_data)

    data_frame = pd.DataFrame(gathered_row_data, columns=columns)
    data_frame.fillna(value="", inplace=True)
    data_frame.dropna(axis=0, how="all", inplace=True)
    write_df_to_csv(data_frame, csv_save_path)
    logger.info(f"Saved comparing results: {csv_save_path}")
    return csv_save_path


def compare_data(golden_data, my_data):
    torch = _lazy_import_torch()
    if not hasattr(compare_data, "index"):
        compare_data.index = 0

    golden_data_dtype = golden_data.dtype
    my_data_dtype = my_data.dtype
    if golden_data_dtype != torch.float32:
        logger.debug(
            f"The dtype of golden_data with index {compare_data.index} is {golden_data_dtype}, convert it to fp32"
        )
    if my_data_dtype != torch.float32:
        logger.debug(f"The dtype of my_data with index {compare_data.index} is {my_data_dtype}, convert it to fp32")
    golden_data_fp32 = golden_data.reshape(-1).float()
    my_data_fp32 = my_data.reshape(-1).float()
    compare_data.index += 1
    compare_metrics_dict = compare_tensor(golden_data_fp32, my_data_fp32)
    return compare_metrics_dict


def read_data(data_path):
    torch = _lazy_import_torch()
    data_path = os.path.realpath(data_path)
    Rule.input_file().check(data_path, will_raise=True)
    if data_path.endswith(".npy"):
        data = torch.as_tensor(np.load(data_path, allow_pickle=False))
    elif data_path.endswith((".pth", ".pt")):
        data = safe_torch_load(data_path, map_location=torch.device("cpu"))
    else:
        logger.error(f"Unsupported data format {data_path}")
        raise TypeError("Unsupported data format.")

    if isinstance(data, torch.Tensor):
        return data.cpu()
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, (list, tuple)):
        return torch.as_tensor(data)
    logger.error(f"Unsupported tensor content type {type(data)} from {data_path}")
    raise TypeError("Unsupported tensor content type.")


def compare_tensor(golden_data_fp32, my_data_fp32):
    row_data: Dict[str, Any] = {}
    fail_messages: List[str] = []

    tensor_pass, message = check_tensor(golden_data_fp32, my_data_fp32)
    if not tensor_pass or my_data_fp32.numel() == 0 or golden_data_fp32.numel() == 0:
        logger.debug(f"check_tensor failed: {message}")
        row_data[CMP_FAIL_REASON] = message
        return row_data

    cmp_alg_map, custom_alg_map = _lazy_cmp_alg_maps()
    for name, cmp_func in list(cmp_alg_map.items()) + list(custom_alg_map.items()):
        result, message = cmp_func(golden_data_fp32, my_data_fp32)
        row_data[name] = result
        if message:
            fail_messages.append(message)
    row_data[CMP_FAIL_REASON] = " ".join(fail_messages)
    return row_data


def check_tensor(golden_data_fp32, my_data_fp32):
    torch = _lazy_import_torch()
    tensor_pass = True
    fail_reasons = []

    if len(golden_data_fp32) != len(my_data_fp32):
        fail_reasons.append("data shape doesn't match.")
        tensor_pass = False
    if not torch.all(torch.isfinite(golden_data_fp32)):
        fail_reasons.append("golden_data includes NAN or inf.")
        tensor_pass = False
    if not torch.all(torch.isfinite(my_data_fp32)):
        fail_reasons.append("my_data includes NAN or inf.")
        tensor_pass = False
    return tensor_pass, " ".join(fail_reasons)


__all__ = [
    "BasicDataInfo",
    "fill_row_data",
    "save_compare_result_to_csv",
]
