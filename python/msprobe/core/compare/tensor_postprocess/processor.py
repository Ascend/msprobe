# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2026-2026 Huawei Technologies Co.,Ltd.
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

import abc

import numpy as np

from msprobe.core.common.file_utils import load_npy, load_yaml
from msprobe.core.common.log import logger
from msprobe.core.common.output_postprocess.load_pt_helper import load_pt_file


class BaseTensorPostprocessor(abc.ABC):
    """
    Tensor 后处理抽象基类。

    扩展新的后处理模式：
        1. 继承 BaseTensorPostprocessor，实现 process() 和 is_effective() 方法
        2. 在 TensorPostprocessManager.__init__ 中添加到 self._processors 列表
    """

    @abc.abstractmethod
    def process(self, n_value, b_value, npu_data_name, bench_data_name):
        """return tuple (n_value, b_value) after processing."""
        return n_value, b_value

    def is_effective(self) -> bool:
        return True


def _load_tensor_as_numpy(file_path):
    if file_path.endswith('.npy'):
        return load_npy(file_path)
    if file_path.endswith('.pt') or file_path.endswith('.pth'):
        tensor = load_pt_file(file_path, to_cpu=True)
        return tensor.detach().cpu().numpy()
    raise ValueError(f"Unsupported tensor file format: {file_path}")


def _build_reverse_map(tensor_map):
    """将 {tensor_path: [data_name, ...]} 反转为 {data_name: tensor_path} 的查找表。"""
    if not tensor_map:
        return {}
    reverse_map = {}
    for tensor_path, data_names in tensor_map.items():
        if not isinstance(data_names, (list, tuple)):
            logger.warning(
                f"Invalid data_names for tensor path '{tensor_path}': "
                f"expected a list, got {type(data_names).__name__}. Skipped."
            )
            continue
        for data_name in data_names:
            reverse_map[data_name] = tensor_path
    return reverse_map


def _warn_unknown_config_keys(config, known_keys, processor_name):
    """检查 config 中是否包含不属于当前 processor 的键名，有则告警。"""
    unknown_keys = set(config) - known_keys
    if unknown_keys:
        logger.warning(
            f"{processor_name}: unknown config key(s) {unknown_keys} will be ignored. Known keys: {known_keys}."
        )


def _try_matmul(value, mat_path, data_name, direction, side):
    if mat_path is None:
        return value
    try:
        mat = _load_tensor_as_numpy(mat_path)
        logger.info(f"[{data_name}] {direction.capitalize()}-matmul {side} tensor from {mat_path}")
        return np.matmul(mat, value) if direction == 'left' else np.matmul(value, mat)
    except Exception:
        logger.warning(f"[{data_name}] Failed to {direction}-matmul {side} tensor from {mat_path}")
        return value


class RightMatmulPostprocessor(BaseTensorPostprocessor):
    _KNOWN_KEYS = {"target_tensor_map", "golden_tensor_map"}

    def __init__(self, config):
        self._npu_tensor_map = _build_reverse_map(config.get("target_tensor_map"))
        self._bench_tensor_map = _build_reverse_map(config.get("golden_tensor_map"))
        _warn_unknown_config_keys(config, self._KNOWN_KEYS, type(self).__name__)

    def is_effective(self):
        return bool(self._npu_tensor_map or self._bench_tensor_map)

    def process(self, n_value, b_value, npu_data_name, bench_data_name):
        npu_mat_path = self._npu_tensor_map.get(npu_data_name)
        bench_mat_path = self._bench_tensor_map.get(bench_data_name)
        n_value = _try_matmul(n_value, npu_mat_path, npu_data_name, 'right', 'target')
        b_value = _try_matmul(b_value, bench_mat_path, bench_data_name, 'right', 'golden')
        return n_value, b_value


class LeftMatmulPostprocessor(BaseTensorPostprocessor):
    _KNOWN_KEYS = {"target_tensor_map", "golden_tensor_map"}

    def __init__(self, config):
        self._npu_tensor_map = _build_reverse_map(config.get("target_tensor_map"))
        self._bench_tensor_map = _build_reverse_map(config.get("golden_tensor_map"))
        _warn_unknown_config_keys(config, self._KNOWN_KEYS, type(self).__name__)

    def is_effective(self):
        return bool(self._npu_tensor_map or self._bench_tensor_map)

    def process(self, n_value, b_value, npu_data_name, bench_data_name):
        npu_mat_path = self._npu_tensor_map.get(npu_data_name)
        bench_mat_path = self._bench_tensor_map.get(bench_data_name)
        n_value = _try_matmul(n_value, npu_mat_path, npu_data_name, 'left', 'target')
        b_value = _try_matmul(b_value, bench_mat_path, bench_data_name, 'left', 'golden')
        return n_value, b_value


class LeftRightMatmulPostprocessor(BaseTensorPostprocessor):
    _KNOWN_KEYS = {
        "left_target_tensor_map",
        "right_target_tensor_map",
        "left_golden_tensor_map",
        "right_golden_tensor_map",
    }

    def __init__(self, config):
        self._left_npu_tensor_map = _build_reverse_map(config.get("left_target_tensor_map"))
        self._right_npu_tensor_map = _build_reverse_map(config.get("right_target_tensor_map"))
        self._left_bench_tensor_map = _build_reverse_map(config.get("left_golden_tensor_map"))
        self._right_bench_tensor_map = _build_reverse_map(config.get("right_golden_tensor_map"))
        _warn_unknown_config_keys(config, self._KNOWN_KEYS, type(self).__name__)

    def is_effective(self):
        return bool(
            self._left_npu_tensor_map
            or self._right_npu_tensor_map
            or self._left_bench_tensor_map
            or self._right_bench_tensor_map
        )

    def process(self, n_value, b_value, npu_data_name, bench_data_name):
        left_npu_path = self._left_npu_tensor_map.get(npu_data_name)
        right_npu_path = self._right_npu_tensor_map.get(npu_data_name)
        left_bench_path = self._left_bench_tensor_map.get(bench_data_name)
        right_bench_path = self._right_bench_tensor_map.get(bench_data_name)
        n_value = _try_matmul(n_value, left_npu_path, npu_data_name, 'left', 'target')
        n_value = _try_matmul(n_value, right_npu_path, npu_data_name, 'right', 'target')
        b_value = _try_matmul(b_value, left_bench_path, bench_data_name, 'left', 'golden')
        b_value = _try_matmul(b_value, right_bench_path, bench_data_name, 'right', 'golden')
        return n_value, b_value


class TensorPostprocessManager:
    def __init__(self, config=None):
        if config is None:
            config = {}
        if isinstance(config, str):
            config = load_yaml(config) if config else {}

        self._processors = [
            RightMatmulPostprocessor(config.get("right_matmul", {})),
            LeftMatmulPostprocessor(config.get("left_matmul", {})),
            LeftRightMatmulPostprocessor(config.get("left_right_matmul", {})),
        ]

    def process(self, n_value, b_value, npu_data_name, bench_data_name):
        for processor in self._processors:
            n_value, b_value = processor.process(n_value, b_value, npu_data_name, bench_data_name)
        return n_value, b_value
