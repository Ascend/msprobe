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
import os

import numpy as np

from msprobe.core.common.file_utils import load_npy, load_yaml
from msprobe.core.common.log import logger
from msprobe.core.common.output_postprocess.load_pt_helper import load_pt_file


class BaseTensorPostprocessor(abc.ABC):
    """
    Tensor 后处理抽象基类。

    扩展新的后处理模式：
        1. 继承 BaseTensorPostprocessor，实现 process() 方法
        2. 在 TensorPostprocessManager._PROCESSOR_REGISTRY 中注册 mode 名称与处理器类的映射
        3. 在 tensor_postprocess/ 目录下添加对应的 YAML 配置文件（需包含 mode 字段）
    """

    @abc.abstractmethod
    def process(self, n_value, b_value, npu_data_name, bench_data_name):
        pass

    def is_effective(self):
        return True


def _load_tensor_as_numpy(file_path):
    if file_path.endswith('.npy'):
        return load_npy(file_path)
    if file_path.endswith('.pt') or file_path.endswith('.pth'):
        tensor = load_pt_file(file_path, to_cpu=True)
        return tensor.detach().cpu().numpy()
    raise ValueError(f"Unsupported tensor file format: {file_path}")


class RightMatmulPostprocessor(BaseTensorPostprocessor):
    def __init__(self, config):
        self._npu_tensor_map = self._build_reverse_map(config.get("target_tensor_map"))
        self._bench_tensor_map = self._build_reverse_map(config.get("golden_tensor_map"))

    @staticmethod
    def _build_reverse_map(tensor_map):
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

    def is_effective(self):
        return bool(self._npu_tensor_map or self._bench_tensor_map)

    def process(self, n_value, b_value, npu_data_name, bench_data_name):
        npu_mat_path = self._npu_tensor_map.get(npu_data_name)
        bench_mat_path = self._bench_tensor_map.get(bench_data_name)

        if npu_mat_path is not None:
            try:
                mat = _load_tensor_as_numpy(npu_mat_path)
                logger.info(f"[{npu_data_name}] Right-matmul target tensor from {npu_mat_path}")
                n_value = np.matmul(n_value, mat)
            except Exception:
                logger.warning(f"[{npu_data_name}] Failed to right-matmul target tensor from {npu_mat_path}")

        if bench_mat_path is not None:
            try:
                mat = _load_tensor_as_numpy(bench_mat_path)
                logger.info(f"[{bench_data_name}] Right-matmul golden tensor from {bench_mat_path}")
                b_value = np.matmul(b_value, mat)
            except Exception:
                logger.warning(f"[{bench_data_name}] Failed to right-matmul golden tensor from {bench_mat_path}")

        return n_value, b_value


class TensorPostprocessManager:
    _PROCESSOR_REGISTRY = {
        "right_matmul": RightMatmulPostprocessor,
    }

    def __init__(self, config_dir=None):
        self._processors = []
        if config_dir is None:
            config_dir = os.path.dirname(os.path.realpath(__file__))
        self._load_configs(config_dir)

    def _load_configs(self, config_dir):
        if not os.path.isdir(config_dir):
            return
        for filename in sorted(os.listdir(config_dir)):
            if not filename.endswith(('.yaml', '.yml')):
                continue
            config_path = os.path.join(config_dir, filename)
            try:
                config = load_yaml(config_path)
            except Exception:
                logger.warning(f"Failed to load tensor postprocess config: {config_path}")
                continue
            if not config:
                continue
            mode = config.get("mode")
            if mode not in self._PROCESSOR_REGISTRY:
                logger.warning(f"Unknown tensor postprocess mode: {mode}")
                continue
            processor_cls = self._PROCESSOR_REGISTRY[mode]
            processor = processor_cls(config)
            if not processor.is_effective():
                logger.info(f"Skipped empty tensor postprocess config [{mode}] from {config_path}")
                continue
            self._processors.append(processor)
            logger.info(f"Loaded tensor postprocess mode [{mode}] from {config_path}")

    def process(self, n_value, b_value, npu_data_name, bench_data_name):
        for processor in self._processors:
            n_value, b_value = processor.process(n_value, b_value, npu_data_name, bench_data_name)
        return n_value, b_value
