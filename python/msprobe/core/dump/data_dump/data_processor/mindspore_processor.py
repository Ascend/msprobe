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
import zlib
from concurrent.futures import ThreadPoolExecutor

import mindspore as ms
from mindspore import mint, ops
from mindspore.mint import distributed
from mindspore._c_expression.typing import Number
import numpy as np

from msprobe.core.common.const import Const
from msprobe.core.dump.data_dump.data_processor.base import BaseDataProcessor, TensorStatInfo
from msprobe.mindspore.common.utils import convert_bf16_to_fp32, save_tensor_as_npy
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.dump.dump_processor.hook_cell.api_register import get_api_register
from msprobe.mindspore.common.utils import is_mindtorch, cast_to_float_if_fp8

if is_mindtorch():
    from torch import distributed as dist


class MindsporeDataProcessor(BaseDataProcessor):
    if is_mindtorch():
        mindspore_special_type = tuple([ms.Tensor, Number, distributed.P2POp, dist.ProcessGroup])
    else:
        mindspore_special_type = tuple([ms.Tensor, Number, distributed.P2POp])

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.mindspore_object_key = {"dtype": self.analyze_dtype_in_kwargs}
        self._async_dump_cache = {}
        self.api_register = get_api_register()
        self._crc_executor = ThreadPoolExecutor(max_workers=os.cpu_count() // 2)

    @staticmethod
    def compute_crc32_bytes(tensor_bytes):
        return f"{zlib.crc32(tensor_bytes):08x}"

    @staticmethod
    def get_md5_for_tensor(x):
        x = convert_bf16_to_fp32(x)
        tensor_bytes = x.asnumpy().tobytes()
        crc32_hash = zlib.crc32(tensor_bytes)
        return f"{crc32_hash:08x}"

    @staticmethod
    def analyze_dtype_in_kwargs(element):
        return {"type": "mindspore.dtype", "value": str(element)}

    @staticmethod
    def is_hookable_element(element):
        return hasattr(element, "register_hook") and callable(element.register_hook)

    @staticmethod
    def process_group_hash(arg):
        group_ranks = distributed.get_process_group_ranks(arg)
        group_ranks_hash = zlib.crc32(str(group_ranks).encode('utf-8'))
        return f"{group_ranks_hash:08x}"

    @staticmethod
    def _analyze_process_group(arg):
        group_info = {"type": "mindspore.ProcessGroup"}
        try:
            group_ranks = dist.get_process_group_ranks(arg)
            group_info.update({"group_ranks": group_ranks})
            group_ranks_hash = zlib.crc32(str(group_ranks).encode('utf-8'))
            group_id = f"{group_ranks_hash:08x}"
            group_info.update({"group_id": group_id})
        except Exception as e:
            logger.warning(f"Failed to get process group ranks info with error info: {e}.")
        return group_info

    @classmethod
    def get_special_types(cls):
        return super().get_special_types() + cls.mindspore_special_type

    def dump_async_data(self):
        for file_path, tensor in self._async_dump_cache.items():
            save_tensor_as_npy(tensor, file_path)
        self._async_dump_cache.clear()

    def get_stat_info(self, data):
        self.api_register.restore_inner_used_api()
        tensor_stat = TensorStatInfo()
        if data.numel() == 0:
            pass
        elif data.dtype == ms.bool_:
            if self.config.async_dump:
                tensor_stat.max = mint.any(data)
                tensor_stat.min = mint.all(data)
            else:
                data_np = data.asnumpy()
                tensor_stat.max = np.max(data_np).item()
                tensor_stat.min = np.min(data_np).item()
        elif not data.shape:
            tensor_stat.max = tensor_stat.min = tensor_stat.mean = tensor_stat.norm = data.copy()
        elif data.dtype in (ms.complex64, ms.complex128):
            if self.config.async_dump:
                logger.warning("Async dump do not support complex data!")
            else:
                data_abs = np.abs(data.asnumpy())
                tensor_stat.max = np.max(data_abs).item()
                tensor_stat.min = np.min(data_abs).item()
                tensor_stat.mean = np.mean(data_abs).item()
                tensor_stat.norm = np.linalg.norm(data_abs).item()
        else:
            if (
                self.config.precision == Const.DUMP_PRECISION_HIGH
                or not ops.is_floating_point(data)
                or data.dtype == ms.float64
            ):
                data = data.to(ms.float32)
            get_norm_value = mint.norm if hasattr(mint, "norm") else ops.norm
            tensor_stat.max = mint.max(data)
            tensor_stat.min = mint.min(data)
            tensor_stat.mean = mint.mean(data)
            tensor_stat.norm = get_norm_value(data)
        self.api_register.register_inner_used_api()
        return tensor_stat

    def analyze_single_element(self, element, suffix_stack):
        if suffix_stack and suffix_stack[-1] in self.mindspore_object_key:
            return self.mindspore_object_key[suffix_stack[-1]](element)

        suffix_str = Const.SEP.join(str(s) for s in suffix_stack)
        type_analyzer = [
            (MindsporeDataProcessor.builtin_type, self._analyze_builtin),
            (ms.Tensor, lambda e: self._analyze_tensor(e, suffix_str)),
            (Number, self.analyze_dtype_in_kwargs),
            (MindsporeDataProcessor.np_type[:-1], self._analyze_numpy),
            (np.ndarray, lambda e: self._analyze_ndarray(e, suffix_str)),
            (distributed.P2POp, lambda e: self._analyze_p2pop(e, suffix_str)),
        ]
        if is_mindtorch():
            type_analyzer.append((dist.ProcessGroup, self._analyze_process_group))
        for type_key, analyze_fn in type_analyzer:
            if isinstance(element, type_key):
                return analyze_fn(element)
        return {}

    def _analyze_p2pop(self, arg, suffix):
        p2pop_info = {"class_type": "mindspore.mint.distributed.P2POp"}
        try:
            tensor_info = self._analyze_tensor(arg.tensor, suffix)
            p2pop_info.update({"tensor": tensor_info})
            p2pop_info.update({"op": arg.op})
            p2pop_info.update({"peer": arg.peer})
            p2pop_info.update({"tag": arg.tag})
            group_id = self.process_group_hash(arg.group) if arg.group else None
            p2pop_info.update({"group_id": group_id})
        except Exception as e:
            logger.warning(f"Failed to parse the P2POp content with error info: {e}.")
        return p2pop_info

    def _analyze_tensor(self, tensor, suffix):
        dtype = str(tensor.dtype)
        tensor = cast_to_float_if_fp8(tensor)
        tensor_stat = self.get_stat_info(tensor)
        tensor_json = {'type': 'mindspore.Tensor', 'dtype': dtype, 'shape': tensor.shape}
        if hasattr(tensor, "layout") and tensor.layout is not None:
            layout_dic = {}
            if hasattr(tensor.layout, "device_matrix") and tensor.layout.device_matrix is not None:
                layout_dic['Device Matrix'] = tensor.layout.device_matrix
            if hasattr(tensor.layout, "alias_name") and tensor.layout.alias_name is not None:
                layout_dic['Alias Name'] = tensor.layout.alias_name
                interleaved = "Yes" if ("interleaved_parallel" in tensor.layout.alias_name) else "No"
                layout_dic['Interleaved'] = interleaved
            if hasattr(tensor.layout, "partial") and tensor.layout.partial is not None:
                layout_dic['Partial'] = tensor.layout.partial
            if hasattr(tensor.layout, "tensor_map") and tensor.layout.tensor_map is not None:
                layout_dic['Tensor Map'] = tensor.layout.tensor_map
            if hasattr(tensor.layout, "rank_list") and tensor.layout.rank_list is not None:
                layout_dic['Rank List'] = tensor.layout.rank_list
            tensor_json['layout'] = layout_dic
        if hasattr(tensor, "hsdp_effective_shard_size") and tensor.hsdp_effective_shard_size is not None:
            tensor_json['hsdp_shard_size'] = tensor.hsdp_effective_shard_size

        # 将统计值存入全局 buffer，并返回占位索引
        stat_values = [tensor_stat.max, tensor_stat.min, tensor_stat.mean, tensor_stat.norm]

        placeholder_index = self.data_writer.append_stat_to_buffer(stat_values)

        tensor_json.update({Const.TENSOR_STAT_INDEX: placeholder_index})

        if self.config.summary_mode == Const.MD5 and not self.config.async_dump:
            tensor = convert_bf16_to_fp32(tensor)
            # 拷贝并搬到 CPU
            tensor_bytes = tensor.asnumpy()

            future = self._crc_executor.submit(MindsporeDataProcessor.compute_crc32_bytes, tensor_bytes)

            crc_placeholder = self.data_writer.append_crc32_to_buffer(future)
            tensor_json[Const.MD5_INDEX] = crc_placeholder

        return tensor_json

    def _analyze_and_save_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        single_arg = MindsporeDataProcessor._analyze_tensor(self, tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        tensor = cast_to_float_if_fp8(tensor)
        if self.config.async_dump:
            self._async_dump_cache[file_path] = tensor.copy()
        else:
            save_tensor_as_npy(tensor, file_path)
        return single_arg


class StatisticsDataProcessor(MindsporeDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        if any(item in self.current_api_or_module_name for item in self.config.tensor_list):
            return self._analyze_and_save_tensor(tensor, suffix)
        else:
            return super()._analyze_tensor(tensor, suffix)

    # pylint: disable=W0221
    def _analyze_ndarray(self, ndarray, suffix):
        if any(item in self.current_api_or_module_name for item in self.config.tensor_list):
            return self._analyze_and_save_ndarray(ndarray, suffix)
        else:
            return super()._analyze_ndarray(ndarray, suffix)


class TensorDataProcessor(MindsporeDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        return self._analyze_and_save_tensor(tensor, suffix)

    # pylint: disable=W0221
    def _analyze_ndarray(self, ndarray, suffix):
        return self._analyze_and_save_ndarray(ndarray, suffix)
