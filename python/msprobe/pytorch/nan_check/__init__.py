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
import torch
import torch_npu  # noqa: F401
from pathlib import Path
from typing import List


def _load_nan_check_runtime_env():
    customize_root = Path(__file__).resolve().parents[2] / "vendors" / "customize"
    if not customize_root.is_dir():
        return
    existing_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{customize_root}:{existing_path}" if existing_path else str(customize_root)


_load_nan_check_runtime_env()

# Import the C++ extension to register TORCH_LIBRARY implementations.
try:
    from msprobe.lib import nan_check_ext  # noqa: F401  # pylint: disable=no-name-in-module
except Exception as exc:
    raise RuntimeError(f"Failed to import msprobe.lib.nan_check_ext: {exc}")


def npu_over_flow(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.my_ns.npu_over_flow(x)


def npu_clear_over_flow(device: torch.device) -> None:
    return torch.ops.my_ns.npu_clear_over_flow(device)


def npu_nan_test(x: torch.Tensor, tensor_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.ops.my_ns.npu_nan_test(x, tensor_list)


__all__ = ["npu_over_flow", "npu_clear_over_flow", "npu_nan_test"]
