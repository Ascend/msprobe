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

import ctypes
import os
import re
import zlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

from msprobe.core.common.const import Const
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import load_json, load_yaml
from msprobe.core.common.log import logger
from msprobe.core.common.utils import convert_tuple, is_int
from msprobe.core.dump.data_dump.data_processor.base import (
    BaseDataProcessor,
    ModuleBackwardInputsOutputs,
    ModuleForwardInputsOutputs,
    TensorStatInfo,
)
from msprobe.pytorch.common.utils import (
    Const as PtConst,
    is_float4_tensor,
    is_float8_tensor,
    is_hifloat8_tensor,
    save_pt,
)

is_gpu = False
try:
    import torch_npu
except ImportError:
    is_gpu = True


def xor_checksum(a: torch.Tensor) -> torch.Tensor:
    if a.dim() == 0:
        return a.clone()
    bytes_tensor = a.view(torch.uint8).flatten()
    numel = bytes_tensor.numel()
    if numel == 0:
        return torch.zeros((), dtype=torch.int64, device=a.device)

    if numel % 8 != 0:
        bytes_tensor = torch.nn.functional.pad(bytes_tensor, (0, 8 - numel % 8), "constant", 0)
    # Zero-copy fast path; only realign via copy when int64 view alignment is invalid.
    if bytes_tensor.storage_offset() % 8 != 0:
        bytes_tensor = bytes_tensor.clone()
    words = bytes_tensor.view(torch.int64)
    numel = words.numel()

    while numel > 1:
        if numel % 2 != 0:
            words = torch.nn.functional.pad(words, (0, 1), "constant", 0)
        words = words.view(2, -1)
        words = torch.bitwise_xor(words[0, :], words[1, :])
        numel = words.numel()
    return words[0]


class TensorHandler:
    def __init__(self):
        self.has_dtensor = hasattr(dist, "tensor") and hasattr(dist.tensor, "DTensor")
        self.has_fake_tensor = hasattr(torch, "_subclasses") and hasattr(torch._subclasses, "fake_tensor")
        self.has_async_collective_tensor = hasattr(dist, "_functional_collectives") and hasattr(
            dist._functional_collectives, "AsyncCollectiveTensor"
        )
        self.has_nested_tensor = (
            hasattr(torch, "nested")
            and hasattr(torch.nested, "_internal")
            and hasattr(torch.nested._internal, "nested_tensor")
            and hasattr(torch.nested._internal.nested_tensor, "NestedTensor")
        )

    @staticmethod
    def free_tensor(tensor, tensor_name):
        try:
            tensor.untyped_storage().resize_(0)
        except Exception as e:
            logger.warning(f"Failed to free tensor: {tensor_name}, the detail info: {e}.")

    @staticmethod
    def get_tensor_dtype(tensor):
        if is_hifloat8_tensor(tensor):
            return PtConst.HIFLOAT8_TYPE
        return str(tensor.dtype)

    @staticmethod
    def is_batchedtensor(tensor):
        if hasattr(torch._C, "_functorch") and hasattr(torch._C._functorch, "is_batchedtensor"):
            return torch._C._functorch.is_batchedtensor(tensor)
        return False

    @staticmethod
    def is_gradtrackingtensor(tensor):
        if hasattr(torch._C, "_functorch") and hasattr(torch._C._functorch, "is_gradtrackingtensor"):
            return torch._C._functorch.is_gradtrackingtensor(tensor)
        return False

    def is_dtensor(self, tensor):
        return self.has_dtensor and isinstance(tensor, dist.tensor.DTensor)

    def is_fake_tensor(self, tensor):
        return self.has_fake_tensor and isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor)

    def is_async_collective_tensor(self, tensor):
        return self.has_async_collective_tensor and isinstance(
            tensor, dist._functional_collectives.AsyncCollectiveTensor
        )

    def is_nested_tensor(self, tensor):
        return self.has_nested_tensor and isinstance(tensor, torch.nested._internal.nested_tensor.NestedTensor)

    def is_empty_data(self, tensor):
        return tensor.is_meta or self.is_fake_tensor(tensor) or self.is_async_collective_tensor(tensor)

    def convert_common_tensor(self, tensor):
        if self.is_dtensor(tensor):
            return tensor.to_local()
        if self.is_fake_tensor(tensor):
            logger.debug("FakeTensor cannot be converted to torch.Tensor type.")
            return tensor
        if self.is_nested_tensor(tensor):
            logger.debug("For NestedTensor, collecting information from the tensor returned by .values().")
            return tensor.values()
        if self.is_batchedtensor(tensor):
            return torch._C._functorch.get_unwrapped(tensor)
        return tensor

    def get_tensor_type(self, tensor):
        if self.is_dtensor(tensor):
            return Const.DTENSOR_TYPE
        if self.is_fake_tensor(tensor):
            return Const.FAKE_TENSOR_TYPE
        if self.is_async_collective_tensor(tensor):
            return Const.AC_TENSOR_TYPE
        if self.is_nested_tensor(tensor):
            return Const.NESTED_TENSOR_TYPE
        if self.is_batchedtensor(tensor):
            return Const.BATCHED_TENSOR_TYPE
        if self.is_gradtrackingtensor(tensor):
            return Const.GRADTRACKING_TENSOR_TYPE
        return Const.TENSOR_TYPE

    def get_dtensor_info(self, tensor):
        dtensor_info = {}
        if not self.is_dtensor(tensor):
            return dtensor_info
        if hasattr(tensor, "device_mesh") and tensor.device_mesh:
            dtensor_info.update({"device_mesh": tensor.device_mesh.mesh.tolist()})

        placements = []
        if hasattr(tensor, "placements") and isinstance(tensor.placements, Iterable):
            for placement in tensor.placements:
                if placement.is_shard() and is_int(placement.dim):
                    placements.append({"Shard": {"dim": placement.dim}})
                    continue
                if placement.is_replicate():
                    placements.append({"Replicate": {}})
                    continue
                if placement.is_partial() and isinstance(placement.reduce_op, str):
                    placements.append({"Partial": {"reduce_op": placement.reduce_op}})
        dtensor_info.update({"placements": placements})
        return dtensor_info

    def save_tensor(self, tensor, file_path):
        common_tensor = self.convert_common_tensor(tensor)
        if self.is_gradtrackingtensor(common_tensor):
            common_tensor = torch._C._functorch.get_unwrapped(common_tensor)
        if self.is_empty_data(common_tensor):
            logger.debug(f"Saving fake tensor or meta tensor is not supported, the current tensor is {file_path}.")
            return
        if common_tensor.untyped_storage().data_ptr() == 0:
            logger.debug(f"Saving null-pointer tensor is not supported, the current tensor is {file_path}.")
            return
        if is_float8_tensor(common_tensor) or is_float4_tensor(common_tensor):
            saved_tensor = common_tensor.detach().to("cpu")
        else:
            saved_tensor = common_tensor.clone().contiguous().detach()
            if self.is_gradtrackingtensor(saved_tensor):
                saved_tensor = torch._C._functorch.get_unwrapped(saved_tensor)
        save_pt(saved_tensor, file_path)
        self.free_tensor(saved_tensor, file_path)


class PytorchDataProcessor(BaseDataProcessor):
    pytorch_special_type = (
        torch.device,
        torch.dtype,
        torch.Size,
        torch.Tensor,
        torch.memory_format,
        dist.ProcessGroup,
        dist.P2POp,
        dist.ReduceOp,
    )
    memory_format = {
        torch.contiguous_format: "contiguous_format",
        torch.channels_last: "channels_last",
        torch.channels_last_3d: "channels_last_3d",
        torch.preserve_format: "preserve_format",
    }

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.torch_object_key = {
            "device": self.analyze_device_in_kwargs,
            "dtype": self.analyze_dtype_in_kwargs,
            "output_dtype": self.analyze_dtype_in_kwargs,
        }
        self._async_dump_cache = {}
        self.tensor_handler = TensorHandler()
        self._crc_executor = ThreadPoolExecutor(max_workers=os.cpu_count() // 2)

    def _load_builtin_ignore_rules(self):
        yaml_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "pytorch",
            "dump",
            "api_dump",
            Const.BUILTIN_IGNORE_API_FILE_NAME,
        )
        raw_rules = load_yaml(os.path.realpath(yaml_path)) or {}
        return self._normalize_ignore_rules({Const.FORWARD: raw_rules})

    @staticmethod
    def get_md5_for_tensor(x):
        if TensorHandler.is_gradtrackingtensor(x):
            x = torch._C._functorch.get_unwrapped(x)
        if x.dtype == torch.bfloat16:
            x = x.float()
        tensor_bytes = x.cpu().detach().numpy().tobytes()
        crc32_hash = zlib.crc32(tensor_bytes)
        return f"{crc32_hash:08x}"

    @staticmethod
    def tensor_bytes_view_cpu(t: torch.Tensor):
        """
        返回 t 在当前 dtype 下的原始字节视图（优先零拷贝）。
        需保证：t 已在 CPU 且是 contiguous。
        可能返回 memoryview 或 bytes（兜底拷贝）或者 转为numpy，均可被 zlib.crc32 接受。
        """

        nbytes = t.numel() * t.element_size()
        byte_offset = t.storage_offset() * t.element_size()

        if nbytes == 0:
            return memoryview(b"")

        storage = t.untyped_storage()

        # ctypes 指针构造 memoryview（零拷贝 FFI）
        try:
            addr = storage.data_ptr() + byte_offset
            buf = (ctypes.c_ubyte * nbytes).from_address(addr)
            mv3 = memoryview(buf)

            return mv3
        except Exception as e1:
            logger.warning(f"path_A_failed: {e1}.")

        try:
            data = ctypes.string_at(storage.data_ptr() + byte_offset, nbytes)

            return data  # bytes 也可直接用于 zlib.crc32
        except Exception as e2:
            logger.warning(f"path_B_failed: {e2}.")

        try:
            if t.dtype == torch.bfloat16:
                t = t.float()
            data = t.numpy()

            return data
        except Exception as e3:
            logger.warning(f"path_C_failed: {e3}.")
            return memoryview(b"")

    @staticmethod
    def compute_crc32_from_tensor(t: torch.Tensor) -> str:
        """
        直接对 Tensor 原始字节做 CRC32。
        :
        - "raw": 保持 bfloat16 原始 16bit 字节（推荐，避免升精/增容）
        """

        if TensorHandler.is_gradtrackingtensor(t):
            t = torch._C._functorch.get_unwrapped(t)

        # 取得字节视图（含多级回退），然后做 CRC
        mv = PytorchDataProcessor.tensor_bytes_view_cpu(t)

        crc = zlib.crc32(mv)

        return f"{crc:08x}"

    @staticmethod
    def analyze_device_in_kwargs(element):
        single_arg = {}
        single_arg.update({"type": "torch.device"})
        if isinstance(element, (int, str)):
            single_arg.update({"value": element})
        elif isinstance(element, torch.device):
            if hasattr(element, "index"):
                device_value = element.type + ":" + str(element.index)
            else:
                device_value = element.type
            single_arg.update({"value": device_value})
        else:
            logger.debug(f"Device type {type(element)} is not supported.")
        return single_arg

    @staticmethod
    def analyze_dtype_in_kwargs(element):
        return {"type": "torch.dtype", "value": str(element)}

    @staticmethod
    def process_group_hash(arg):
        group_ranks = dist.get_process_group_ranks(arg)
        group_ranks_hash = zlib.crc32(str(group_ranks).encode("utf-8"))
        return f"{group_ranks_hash:08x}"

    @staticmethod
    def is_hookable_element(element):
        return (hasattr(element, "register_hook") and callable(element.register_hook)) and (
            hasattr(element, "requires_grad") and element.requires_grad
        )

    @staticmethod
    def _analyze_torch_size(arg):
        return {"type": "torch.Size", "value": [int(x) for x in list(arg)]}

    @staticmethod
    def _analyze_memory_format(arg):
        # 获取内存格式
        format_type = PytorchDataProcessor.memory_format.get(arg)
        return {"type": "torch.memory_format", "format": format_type}

    @staticmethod
    def _analyze_process_group(arg):
        group_info = {"type": "torch.ProcessGroup"}
        try:
            group_ranks = dist.get_process_group_ranks(arg)
            group_info.update({"group_ranks": group_ranks})
            group_id = PytorchDataProcessor.process_group_hash(arg)
            group_info.update({"group_id": group_id})
        except Exception as e:
            logger.warning(f"Failed to get process group ranks info with error info: {e}.")
        return group_info

    @staticmethod
    def _analyze_reduce_op(arg):
        op_type = None
        try:
            op_type = str(arg)
        except Exception as e:
            logger.warning(f"Failed to get value of torch.distributed.ReduceOp with error info: {e}.")
        return {"type": "torch.distributed.ReduceOp", "value": op_type}

    @classmethod
    def get_special_types(cls):
        return super().get_special_types() + cls.pytorch_special_type

    def get_stat_info(self, data, async_dump=False, precision=Const.DUMP_PRECISION_LOW):
        tensor_stat = TensorStatInfo()
        if self.tensor_handler.is_empty_data(data):
            return tensor_stat

        data_clone = data.detach()
        if self.tensor_handler.is_gradtrackingtensor(data_clone):
            data_clone = torch._C._functorch.get_unwrapped(data_clone)
        if not data_clone.numel() or not data_clone.data_ptr():
            return tensor_stat
        if is_float8_tensor(data_clone) or is_float4_tensor(data_clone):
            logger.debug("Skip stat calculation for float8/float4 tensor.")
            return tensor_stat

        if self.config.summary_mode == Const.XOR_CHECKSUM:
            try:
                checksum_data = data_clone if data_clone.is_contiguous() else data_clone.contiguous()
                tensor_stat.check_sum = xor_checksum(checksum_data)
            except Exception as e:
                logger.warning(f"Failed to calculate tensor checksum with error info: {e}.")
            return tensor_stat

        if torch.is_complex(data_clone):
            if async_dump:
                logger.warning("Async dump do not support complex data!")
                return tensor_stat
            data_np = data_clone.cpu().numpy()
            data_abs = np.abs(data_np)
            tensor_stat.max = np.max(data_abs).item()
            tensor_stat.min = np.min(data_abs).item()
            tensor_stat.mean = np.mean(data_abs).item()
        elif data_clone.dtype == torch.bool:
            tensor_stat.max = torch.any(data_clone)
            tensor_stat.min = torch.all(data_clone)
        elif not data_clone.shape:
            tensor_stat.max = tensor_stat.min = tensor_stat.mean = tensor_stat.norm = data_clone.clone()
        else:
            if (
                precision == Const.DUMP_PRECISION_HIGH
                or data_clone.dtype == torch.float64
                or not data_clone.is_floating_point()
            ):
                data_clone = data_clone.float()
            tensor_stat.max = torch.max(data_clone)
            tensor_stat.min = torch.min(data_clone)
            tensor_stat.mean = torch.mean(data_clone)
            tensor_stat.norm = torch.norm(data_clone)
        return tensor_stat

    def dump_async_data(self):
        for file_path, tensor in self._async_dump_cache.items():
            self.tensor_handler.save_tensor(tensor, file_path)
        self._async_dump_cache.clear()

    def analyze_single_element(self, element, suffix_stack):
        if suffix_stack and suffix_stack[-1] in self.torch_object_key:
            return self.torch_object_key[suffix_stack[-1]](element)

        suffix_str = Const.SEP.join(str(s) for s in suffix_stack)
        type_analyzer = [
            (PytorchDataProcessor.builtin_type, self._analyze_builtin),
            (torch.Size, self._analyze_torch_size),
            (torch.Tensor, lambda e: self._analyze_tensor(e, suffix_str)),
            (torch.memory_format, self._analyze_memory_format),
            (dist.ProcessGroup, self._analyze_process_group),
            (dist.P2POp, lambda e: self._analyze_p2pop(e, suffix_str)),
            (dist.ReduceOp, self._analyze_reduce_op),
            (PytorchDataProcessor.np_type[:-1], self._analyze_numpy),
            (np.ndarray, lambda e: self._analyze_ndarray(e, suffix_str)),
        ]
        for type_key, analyze_fn in type_analyzer:
            if isinstance(element, type_key):
                return analyze_fn(element)
        return {}

    def _analyze_p2pop(self, arg, suffix):
        p2pop_info = {"class_type": "torch.distributed.P2POp"}
        try:
            tensor_info = self._analyze_tensor(arg.tensor, suffix)
            p2pop_info.update({"tensor": tensor_info})
            p2pop_info.update({"op": arg.op.__name__})
            p2pop_info.update({"peer": arg.peer})
            p2pop_info.update({"tag": arg.tag})
            group_id = (
                PytorchDataProcessor.process_group_hash(arg.group)
                if arg.group
                else PytorchDataProcessor.process_group_hash(_get_default_group())
            )
            p2pop_info.update({"group_id": group_id})
        except Exception as e:
            logger.warning(f"Failed to parse the P2POp content with error info: {e}.")
        return p2pop_info

    def _analyze_tensor(self, tensor, suffix):
        common_tensor = self.tensor_handler.convert_common_tensor(tensor)
        tensor_stat = self.get_stat_info(common_tensor, self.config.async_dump, self.config.precision)
        tensor_json = {}
        tensor_json.update({"type": self.tensor_handler.get_tensor_type(tensor)})
        tensor_json.update({"dtype": self.tensor_handler.get_tensor_dtype(tensor)})
        tensor_json.update({"shape": common_tensor.shape})

        stat_values = [tensor_stat.max, tensor_stat.min, tensor_stat.mean, tensor_stat.norm, tensor_stat.check_sum]
        placeholder_index = self.data_writer.append_stat_to_buffer(stat_values)

        tensor_json.update({Const.TENSOR_STAT_INDEX: placeholder_index})
        tensor_json.update({"requires_grad": tensor.requires_grad})
        if self.tensor_handler.is_dtensor(tensor):
            dtensor_info = self.tensor_handler.get_dtensor_info(tensor)
            tensor_json.update(dtensor_info)

        if self.config.summary_mode == Const.MD5 and not self.config.async_dump:
            tensor_md5 = None
            if not self.tensor_handler.is_empty_data(tensor):
                t_cpu = common_tensor

                # 根据设备类型做同步，确保数据已准备好
                if t_cpu.device.type == "cuda":
                    t_cpu = t_cpu.to("cpu", non_blocking=True)
                    torch.cuda.synchronize()
                    # 先异步搬运再进行同步可以显著提升性能
                elif t_cpu.device.type == "npu":
                    t_cpu = t_cpu.to("cpu", non_blocking=True)
                    torch.npu.synchronize()
                t_cpu = t_cpu.detach()

                if self.config.task == Const.TENSOR and self.data_writer.bench_dump_file_path is not None:
                    tensor_md5 = PytorchDataProcessor.compute_crc32_from_tensor(t_cpu)
                    tensor_json.update({Const.MD5: tensor_md5})
                else:
                    if not t_cpu.is_contiguous():
                        t_cpu = t_cpu.contiguous()

                    future = self._crc_executor.submit(PytorchDataProcessor.compute_crc32_from_tensor, t_cpu)

                    crc_placeholder = self.data_writer.append_crc32_to_buffer(future)
                    tensor_json[Const.MD5_INDEX] = crc_placeholder
            else:
                logger.debug(
                    "Calculating the md5 value of fake tensor or meta tensor is not supported, "
                    f"the current api/module name is {self.current_api_or_module_name}."
                )
                tensor_json.update({Const.MD5: tensor_md5})
        return tensor_json

    def _analyze_and_save_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        single_arg = PytorchDataProcessor._analyze_tensor(self, tensor, suffix)
        common_tensor = self.tensor_handler.convert_common_tensor(tensor)
        if self.tensor_handler.is_gradtrackingtensor(common_tensor):
            common_tensor = torch._C._functorch.get_unwrapped(common_tensor)
        if self.tensor_handler.is_empty_data(common_tensor):
            logger.debug(f"Saving fake tensor or meta tensor is not supported, the current tensor is {file_path}.")
            return single_arg
        if common_tensor.untyped_storage().data_ptr() == 0:
            logger.debug(f"Saving null-pointer tensor is not supported, the current tensor is {file_path}.")
            return single_arg

        single_arg.update({"data_name": dump_data_name})
        if self.config.async_dump:
            if is_float8_tensor(common_tensor) or is_float4_tensor(common_tensor):
                self._async_dump_cache[file_path] = common_tensor.detach().to("cpu")
            else:
                self._async_dump_cache[file_path] = common_tensor.clone().detach()
        else:
            self.tensor_handler.save_tensor(common_tensor, file_path)
        return single_arg

    def _analyze_and_save_ndarray(self, ndarray, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        self.tensor_handler.save_tensor(torch.tensor(ndarray), file_path)
        ndarray_json = PytorchDataProcessor._analyze_ndarray(ndarray, suffix)
        ndarray_json.update({"data_name": dump_data_name})
        return ndarray_json


class StatisticsDataProcessor(PytorchDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        if any(item in self.current_api_or_module_name for item in self.config.tensor_list):
            return self._analyze_and_save_tensor(tensor, suffix)
        else:
            return super()._analyze_tensor(tensor, suffix)

    # pylint: disable-next=arguments-differ
    def _analyze_ndarray(self, ndarray, suffix):
        if any(item in self.current_api_or_module_name for item in self.config.tensor_list):
            return self._analyze_and_save_ndarray(ndarray, suffix)
        else:
            return super()._analyze_ndarray(ndarray, suffix)


class TensorDataProcessor(PytorchDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        return self._analyze_and_save_tensor(tensor, suffix)

    # pylint: disable-next=arguments-differ
    def _analyze_ndarray(self, ndarray, suffix):
        return self._analyze_and_save_ndarray(ndarray, suffix)


class DiffCheckDataProcessor(PytorchDataProcessor):
    __slots__ = [
        "cached_tensors_and_file_paths",
        "_bench_ref_path",
        "_bench_ref_mtime",
        "_bench_map",
        "_bench_state",  # 新增：按 API 的对比状态
    ]

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.has_diff = False

        self.cached_api_info = {}
        self.cached_tensors_and_file_paths = {}
        self.bits_for_diff = 8
        self.real_diff_nums = 0
        self.diff_nums = config.diff_nums

        # 新增：bench 基准缓存初始化
        self._bench_ref_path = None
        self._bench_ref_mtime = None
        self._bench_map = {}
        self._bench_state = {}  # key: api_name -> 状态字典

    @property
    def is_terminated(self):
        if self.diff_nums == -1:
            return False
        if self.real_diff_nums >= self.diff_nums:
            return True
        return False

    @staticmethod
    def _parse_data_name(data_name: str):
        """
        解析 data_name，例如：
        - "Functional.relu.2.forward.input.0.pt"
        - 兼容可选前缀 "name:" -> "name:Functional.relu.2.forward.input.0.pt"
        返回 (api, io, idx) 或 None
        """
        if not data_name:
            return None
        if data_name.startswith("name:"):
            data_name = data_name.split(":", 1)[1]

        # api 名本身可能包含若干个 '.'，所以用正则从右侧提取 io/idx/扩展名
        m = re.match(r"^(?=.{1,1024}$)(?P<api>.+)\.(?P<io>input|output)\.(?P<idx>\d+)\.\w+$", data_name)
        if not m:
            return None
        api = m.group("api")
        io = m.group("io")
        idx = int(m.group("idx"))
        return api, io, idx

    def analyze_forward_input(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self.has_diff = False

        self.cached_api_info = super().analyze_forward_input(name, module, module_input_output)

    def analyze_forward_output(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        api_info_struct = super().analyze_forward_output(name, module, module_input_output)
        if name in self.cached_api_info and name in api_info_struct:
            self.cached_api_info[name].update(api_info_struct[name])
        elif name in api_info_struct:
            self.cached_api_info = api_info_struct
        self.handle_diff()
        return self.cached_api_info

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self.has_diff = False

        api_info_struct = super().analyze_forward(name, module, module_input_output)
        self.handle_diff()
        return api_info_struct

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        self.has_diff = False

        api_info_struct = super().analyze_backward(name, module, module_input_output)
        self.handle_diff()
        return api_info_struct

    def analyze_params(self, name, param_name, grad):
        self.has_diff = False

        api_info_struct = super().analyze_params(name, param_name, grad)
        self.handle_diff()
        return api_info_struct

    def handle_diff(self):
        if self.has_diff:
            for file_path, tensor in self.cached_tensors_and_file_paths.items():
                self.tensor_handler.save_tensor(tensor, file_path)
            self.real_diff_nums += 1
            if self.diff_nums != -1 and self.real_diff_nums >= self.diff_nums:
                logger.info(
                    f"[{Const.TOOL_NAME}] Reached the preset diff times, current diff times: {self.real_diff_nums}."
                )
        api = getattr(self, "current_api_or_module_name", None)
        if api and api in self._bench_state:
            self._bench_state.pop(api, None)

        self.cached_tensors_and_file_paths = {}

    def _analyze_maybe_diff_flag(self):
        try:
            self.has_diff = torch_npu.npu.utils.get_npu_diff_flag()
            if self.has_diff:
                torch_npu.npu.utils.clear_npu_diff_flag()
        except Exception as e:
            logger.error("Diff check failed, the current environment may be abnormal.")
            raise RuntimeError("diff check failed") from e

    def _bench_expected_counts_for_api(self, api: str):
        """统计某 API 在 bench_map 里有多少个 Tensor 输入/输出"""
        n_in = n_out = 0
        for a, io, _ in self._bench_map:
            if a == api:
                if io == "input":
                    n_in += 1
                elif io == "output":
                    n_out += 1
        return n_in, n_out

    def _resolve_bench_json_path(self) -> str:
        p = getattr(self.data_writer, "bench_dump_file_path", None)
        if not p:
            return None
        p = os.path.join(p, "dump.json") if os.path.isdir(p) else p
        return p if os.path.isfile(p) else None

    def _ensure_bench_map_loaded(self) -> bool:
        """
        当路径变化或文件 mtime 变化时重载 dump.json，并构建 (api, 'input'/'output', idx) -> {md5, shape} 的索引。
        """
        path = self._resolve_bench_json_path()
        if not path:
            return False
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            return False

        need_reload = (path != self._bench_ref_path) or (mtime != self._bench_ref_mtime)

        if need_reload:
            try:
                obj = load_json(path)
            except Exception as e:
                logger.warning(f"Failed to load bench dump.json: {e}")
                return False

            data = obj.get("data", {})
            self._bench_map = self._build_bench_map_from_json(data)
            self._bench_ref_path = path
            self._bench_ref_mtime = mtime

        return True

    def _build_bench_map_from_json(self, data: dict) -> dict:
        """
        data 结构：{ api_name: {input_args: [...], output: [...] } }
        只收集 Tensor 项：(api, io, idx) -> {"md5": str, "shape": list}
        """
        mp = {}
        total_inputs = 0
        total_outputs = 0
        for api_name, rec in data.items():
            ia = rec.get("input_args", [])
            oa = rec.get("output", [])
            # input_args
            input_count_this_api = 0
            for i, arg in enumerate(ia):
                if isinstance(arg, dict) and arg.get("type") == "torch.Tensor":
                    mp[(api_name, "input", i)] = {
                        "md5": arg.get("md5"),
                        "shape": arg.get("shape"),
                    }
                    input_count_this_api += 1
            total_inputs += input_count_this_api

            # output
            output_count_this_api = 0
            for i, out in enumerate(oa):
                if isinstance(out, dict) and out.get("type") == "torch.Tensor":
                    mp[(api_name, "output", i)] = {
                        "md5": out.get("md5"),
                        "shape": out.get("shape"),
                    }
                    output_count_this_api += 1
            total_outputs += output_count_this_api

        return mp

    def _analyze_maybe_diff_tensor(self, tensor_json):
        # 1) bench map 准备
        if not self._ensure_bench_map_loaded():
            return

        # 2) 解析 data_name -> (api, io, idx)
        data_name = tensor_json.get("data_name")
        parsed = self._parse_data_name(data_name)
        if not parsed:
            logger.debug(f"data_name parse failed: {data_name}")
            return
        api, io, idx = parsed

        # 3) 取/建 本 API 的状态
        st = self._bench_state.get(api)
        if st is None:
            n_in, _ = self._bench_expected_counts_for_api(api)
            st = {
                "expected_in": n_in,  # 标杆中该 API 期望的 Tensor 输入数
                "checked_in": 0,  # 已经校验过的“在标杆中存在的输入”个数
                "inputs_equal": True,  # 到目前为止，输入是否全部一致
                "seen_input_not_in_ref": False,  # 遇到“运行时存在但标杆里没有”的输入
                "any_output_neq": False,  # 是否发现过任一输出不一致（shape 同且 md5 不同）
            }
            self._bench_state[api] = st

        # 4) 找到标杆项
        ref = self._bench_map.get((api, io, idx))

        # 5) 当前 shape
        cur_shape = tensor_json.get("shape")
        if cur_shape is None:
            return
        try:
            cur_shape = list(cur_shape)
        except Exception as e:
            logger.warning(f"[BENCH] shape to list failed: {e!r} -> skip")
            return

        # 6) 输入与输出分别处理
        if io == "input":
            # —— 输入阶段：只维护“输入是否一致”的状态 —— #
            if ref is None:
                # 运行时有输入，但标杆里没有对应条目 => 不能断言“输入一致”
                st["inputs_equal"] = False
                st["seen_input_not_in_ref"] = True

                return

            ref_shape = ref.get("shape")
            ref_md5 = ref.get("md5")

            # 标杆有该输入，计入已校验
            st["checked_in"] += 1

            # shape 必须一致
            if list(ref_shape) != list(cur_shape):
                st["inputs_equal"] = False

                return

            # 取当前 md5
            cur_md5 = tensor_json.get(Const.MD5) if Const.MD5 in tensor_json else tensor_json.get("md5")

            if cur_md5 is None or ref_md5 is None:
                # 缺少 md5 信息，无法断言一致
                st["inputs_equal"] = False
                return

            # md5 必须一致
            if str(cur_md5) != str(ref_md5):
                st["inputs_equal"] = False
            return  # 输入阶段不触发 has_diff

        else:  # io == "output"
            # —— 输出阶段：仅当“所有输入一致且已校验完所有输入”时，才检查输出不一致以置位 —— #
            # 若标杆无此输出，按照你的规则：不能断言输出不一致，直接跳过
            if ref is None:
                return

            ref_shape = ref.get("shape")
            ref_md5 = ref.get("md5")

            # shape 必须一致才比较 md5
            if list(ref_shape) != list(cur_shape):
                return

            cur_md5 = tensor_json.get(Const.MD5) if Const.MD5 in tensor_json else tensor_json.get("md5")
            if cur_md5 is None or ref_md5 is None:
                return

            # 只有当“输入全部一致且已校验完所有输入”时，才允许判定输出不一致
            inputs_ok = (
                st["inputs_equal"] and (st["checked_in"] == st["expected_in"]) and (not st["seen_input_not_in_ref"])
            )

            if inputs_ok and (str(cur_md5) != str(ref_md5)):
                st["any_output_neq"] = True
                self.has_diff = True

    def _analyze_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        self.cached_tensors_and_file_paths.update({file_path: tensor})
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        if not self.has_diff:
            self._analyze_maybe_diff_tensor(single_arg)
        return single_arg


class KernelDumpDataProcessor(PytorchDataProcessor):
    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.enable_kernel_dump = True
        self.is_found_output_tensor = False
        self.is_found_grad_input_tensor = False
        self.forward_args = None
        self.forward_kwargs = None
        self.forward_output_tensor = None
        self.grad_input_tensor = None

    @staticmethod
    def start_kernel_dump(config_path):
        torch_npu.npu.synchronize()
        torch_npu.npu.init_dump()
        torch_npu.npu.set_dump(config_path)
        torch_npu.npu.synchronize()

    @staticmethod
    def stop_kernel_dump():
        torch_npu.npu.synchronize()
        torch_npu.npu.finalize_dump()
        torch_npu.npu.synchronize()

    @staticmethod
    def _print_unsupported_log(api_name):
        logger.warning(f"The kernel dump does not support the {api_name} API.")

    def analyze_forward_input(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        if is_gpu:
            logger.warning("The current environment is not a complete NPU environment, and kernel dump cannot be used.")
            self.enable_kernel_dump = False
            return

        if self.config.is_backward_kernel_dump:
            try:
                self.forward_args = self.clone_and_detach_tensor(module_input_output.args)
                self.forward_kwargs = self.clone_and_detach_tensor(module_input_output.kwargs)
                output = module.forward(*self.forward_args, **self.forward_kwargs)
            except Exception as e:
                if isinstance(e, MsprobeException):
                    logger.warning(str(e))
                self._print_unsupported_log(name)
                self.enable_kernel_dump = False
                return

            self.analyze_element(convert_tuple(output))
            if not self.is_found_output_tensor:
                self._print_unsupported_log(name)
                self.enable_kernel_dump = False
            return
        self.start_kernel_dump(self.config.kernel_config_path)

    def analyze_forward_output(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        if self.config.is_backward_kernel_dump:
            return
        self.enable_kernel_dump = False
        self.stop_kernel_dump()
        logger.info(f"The kernel data of {name} is dumped successfully.")

    def analyze_backward(self, name, module, module_input_output):
        if not self.enable_kernel_dump:
            return
        self.enable_kernel_dump = False

        self.analyze_element(module_input_output.grad_input)
        if not self.is_found_grad_input_tensor:
            self._print_unsupported_log(name)
            return
        self.start_kernel_dump(self.config.kernel_config_path)

        try:
            self.forward_output_tensor.backward(self.grad_input_tensor, retain_graph=True)
        except Exception:
            self._print_unsupported_log(name)
            self.stop_kernel_dump()
            return

        self.stop_kernel_dump()
        logger.info(f"The kernel data of {name} is dumped successfully.")

    @recursion_depth_decorator(
        "KernelDump: KernelDumpDataProcessor.clone_and_detach_tensor", max_depth=Const.DUMP_MAX_DEPTH
    )
    def clone_and_detach_tensor(self, input_params):
        if isinstance(input_params, torch.Tensor):
            if is_float8_tensor(input_params):
                raise MsprobeException(
                    MsprobeException.UNSUPPORTED_TYPE_ERROR, "L2 backward dump does not support float8 type."
                )
            if input_params.requires_grad:
                return input_params.clone().detach().requires_grad_()
            return input_params.clone()
        elif isinstance(input_params, tuple):
            return tuple(self.clone_and_detach_tensor(x) for x in input_params)
        elif isinstance(input_params, list):
            return list(self.clone_and_detach_tensor(x) for x in input_params)
        elif isinstance(input_params, dict):
            return {k: self.clone_and_detach_tensor(v) for k, v in input_params.items()}
        else:
            return input_params

    def analyze_single_element(self, element, suffix_stack):
        if is_float8_tensor(element):
            return {}
        if isinstance(element, torch.Tensor):
            if not self.is_found_output_tensor:
                if element.requires_grad:
                    self.forward_output_tensor = element
                    self.is_found_output_tensor = True
                return {}
            if not self.is_found_grad_input_tensor:
                self.grad_input_tensor = element.clone()
                self.is_found_grad_input_tensor = True
        return {}

    def reset_status(self):
        self.enable_kernel_dump = True
        self.is_found_output_tensor = False
        self.is_found_grad_input_tensor = False
        self.forward_args = None
        self.forward_kwargs = None
        self.forward_output_tensor = None
        self.grad_input_tensor = None


class NanCheckDataProcessor(PytorchDataProcessor):
    @staticmethod
    def _nan_overflow_ops_available():
        if is_gpu:
            return False
        try:
            from msprobe.pytorch import nan_check
            return True
        except Exception:
            return False

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self._nan_overflow_runtime_warned = False
        if not self._nan_overflow_ops_available():
            raise MsprobeException(
                MsprobeException.INTERFACE_USAGE_ERROR,
                "Nan check requires torch.ops.my_ns.npu_over_flow, npu_clear_over_flow and npu_nan_test. "
                "Please compile with nan_check support enabled.",
            )
        self._nan_buffer_size = Const.LARGER_FLUSH_SIZE
        self._nan_buffer = None
        self._nan_buffer_offset = 0
        self.data_writer.register_pre_flush_callback(self._flush_nan_buffer)

    def _ensure_nan_buffer(self):
        if self._nan_buffer is not None:
            return
        self._nan_buffer = torch.empty(
            self._nan_buffer_size, 8, dtype=torch.int64, device=torch.device("npu", torch.npu.current_device())
        )

    @staticmethod
    @recursion_depth_decorator(
        "NanCheck: NanCheckDataProcessor._collect_tensors", max_depth=Const.DUMP_MAX_DEPTH
    )
    def _collect_tensors(element, tensor_list):
        if isinstance(element, torch.Tensor):
            tensor_list.append(element)
            return
        if isinstance(element, dict):
            for value in element.values():
                NanCheckDataProcessor._collect_tensors(value, tensor_list)
            return
        if isinstance(element, (list, tuple)):
            for item in element:
                NanCheckDataProcessor._collect_tensors(item, tensor_list)

    def _should_collect_nan_tensors(self):
        if not getattr(self.config, "tensor_list", None):
            return False
        api_name = getattr(self, "current_api_or_module_name", "")
        return any(item in api_name for item in self.config.tensor_list)

    def _build_nan_tensor_list(self, module_input_output):
        tensor_list = []
        if hasattr(module_input_output, "args"):
            self._collect_tensors(module_input_output.args_tuple, tensor_list)
            self._collect_tensors(module_input_output.kwargs, tensor_list)
        if hasattr(module_input_output, "output"):
            self._collect_tensors(module_input_output.output_tuple, tensor_list)
        if hasattr(module_input_output, "grad_input"):
            self._collect_tensors(module_input_output.grad_input_tuple, tensor_list)
        if hasattr(module_input_output, "grad_output"):
            self._collect_tensors(module_input_output.grad_output_tuple, tensor_list)
        return tensor_list

    def _maybe_collect_overflow_tensors(self, slot, module_input_output):
        if not self._should_collect_nan_tensors():
            return
        try:
            tensor_list = self._build_nan_tensor_list(module_input_output)
            if not tensor_list:
                return
            torch.ops.my_ns.npu_nan_test(slot, tensor_list)
        except Exception as e:
            if not self._nan_collect_runtime_warned:
                logger.warning(
                    f"Nan check overflow tensor collection failed ({e}); tensor collection will be skipped."
                )
                self._nan_collect_runtime_warned = True
            else:
                logger.debug(f"Nan check overflow tensor collection failed ({e}).")

    def _check_overflow(self, module_input_output=None):
        self._ensure_nan_buffer()
        if self._nan_buffer_offset >= self._nan_buffer_size:
            self.data_writer.write_json()
        idx = self._nan_buffer_offset
        slot = self._nan_buffer[idx]
        out_ptr = torch.tensor([slot.data_ptr()], dtype=torch.uint64, device=slot.device)
        try:
            overflow_result = torch.ops.my_ns.npu_over_flow(out_ptr)
            self._nan_buffer_offset += 1
            if module_input_output is not None:
                self._maybe_collect_overflow_tensors(slot, module_input_output)
            return idx
        except Exception as e:
            if not self._nan_overflow_runtime_warned:
                logger.warning(f"Nan check npu_over_flow failed ({e}); is_nan will be null.")
                self._nan_overflow_runtime_warned = True
            else:
                logger.debug(f"Nan check npu_over_flow failed ({e}).")
            return None

    def _flush_nan_buffer(self):
        if self._nan_buffer_offset == 0:
            return
        cpu_vals = self._nan_buffer[: self._nan_buffer_offset, 0].cpu().tolist()
        self.data_writer._replace_nan_placeholders(self.data_writer.cache_data, cpu_vals)
        if self.data_writer.cache_debug:
            self.data_writer._replace_nan_placeholders(self.data_writer.cache_debug, cpu_vals)
        self._nan_buffer_offset = 0
        self.data_writer.data_updated = True

    def _analyze_tensor(self, tensor, suffix):
        common_tensor = self.tensor_handler.convert_common_tensor(tensor)
        tensor_json = {}
        tensor_json.update({"type": self.tensor_handler.get_tensor_type(tensor)})
        tensor_json.update({"dtype": self.tensor_handler.get_tensor_dtype(tensor)})
        tensor_json.update({"shape": common_tensor.shape})
        tensor_json.update({"requires_grad": tensor.requires_grad})
        if self.tensor_handler.is_dtensor(tensor):
            dtensor_info = self.tensor_handler.get_dtensor_info(tensor)
            tensor_json.update(dtensor_info)
        return tensor_json

    def _store_nan_result(self, api_info_struct, name, module_input_output=None):
        slot_idx = self._check_overflow(module_input_output)
        if slot_idx is not None:
            api_info_struct[name][Const.IS_NAN_INDEX] = slot_idx
        else:
            api_info_struct[name][Const.IS_NAN] = None

    def analyze_forward_output(self, name, module, module_input_output):
        api_info_struct = super().analyze_forward_output(name, module, module_input_output)
        if name in api_info_struct:
            self._store_nan_result(api_info_struct, name, module_input_output)
        return api_info_struct

    def analyze_backward(self, name, module, module_input_output):
        api_info_struct = super().analyze_backward(name, module, module_input_output)
        if name in api_info_struct:
            self._store_nan_result(api_info_struct, name, module_input_output)
        return api_info_struct
