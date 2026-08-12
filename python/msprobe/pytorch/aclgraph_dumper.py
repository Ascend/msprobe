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

import functools
import importlib
import os
import shutil
import threading
from collections.abc import Iterable
from contextlib import nullcontext

import torch

from msprobe.pytorch.aclgraph_dump import acl_stat, acl_tensor_save, get_acl_stat_dict
from msprobe.pytorch.common.log import logger
from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.file_utils import create_directory, check_and_get_real_path, save_json, load_json
from msprobe.core.common.utils import get_real_step_or_rank, check_slice_info, slice_by_config

try:
    import torch_npu
except Exception:
    torch_npu = None

try:
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception:
    TorchDispatchMode = None

try:
    from torch._guards import active_fake_mode as _active_fake_mode
    from torch._guards import detect_fake_mode as _detect_fake_mode
except Exception:
    _active_fake_mode = None
    _detect_fake_mode = None


FORWARD_START_MARKER = "__msprobe_fwd_start__"


def _iter_tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(sub, next_prefix)
        return
    if isinstance(value, tuple):
        for idx, sub in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_tensors(sub, next_prefix)
        return
    if isinstance(value, list):
        for idx, sub in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_tensors(sub, next_prefix)
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        for idx, sub in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_tensors(sub, next_prefix)


def _in_fake_mode(inputs):
    active_fake_mode = None
    if _active_fake_mode is not None:
        try:
            active_fake_mode = _active_fake_mode()
        except Exception:
            active_fake_mode = None
        if active_fake_mode is not None:
            return True
    if _detect_fake_mode is None:
        return False
    try:
        return _detect_fake_mode(inputs) is not None
    except Exception:
        return False


def _is_collectable_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        return False
    if getattr(tensor, "is_meta", False):
        return False
    try:
        _ = tensor.device
    except Exception:
        return False
    return True


if TorchDispatchMode is not None:

    class _AclTorchDispatchMode(TorchDispatchMode):
        supports_higher_order_operators = True

        def __init__(self, dumper):
            super().__init__()
            self._dumper = dumper

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs if kwargs is not None else {}
            if _in_fake_mode((args, kwargs)):
                return func(*args, **kwargs)
            if self._dumper._should_skip_dispatch_func(func):
                return func(*args, **kwargs)

            if self._dumper._is_dispatch_collecting():
                return func(*args, **kwargs)

            op_name = self._dumper._op_name_from_dispatch_func(func)
            should_collect = self._dumper._should_collect_api(op_name)
            started = False
            call_started = [False]
            if should_collect:
                collected = self._dumper._dc(
                    self._dumper._collect,
                    op_name,
                    "input",
                    args,
                    mark_forward_start=not started,
                    call_started=call_started,
                )
                started = started or collected
                if kwargs:
                    collected = self._dumper._dc(
                        self._dumper._collect,
                        op_name,
                        "input_kwargs",
                        kwargs,
                        mark_forward_start=not started,
                        call_started=call_started,
                    )
                    started = started or collected

            output = func(*args, **kwargs)

            if should_collect:
                self._dumper._dc(
                    self._dumper._collect,
                    op_name,
                    "output",
                    output,
                    mark_forward_start=not started,
                    call_started=call_started,
                )
            return output


class AclGraphDumper:
    def __init__(self, config_path=None):
        self.config_path = self._resolve_config_path(config_path)
        self._config_signature = self._get_config_signature(self.config_path)
        (
            config_task,
            config_dump_path,
            config_list,
            config_custom_api,
            config_level,
            config_rank,
            config_slice_info,
            dump_enable,
        ) = self._load_msprobe_config(self.config_path)
        self.task = self._validate_task(config_task)
        self.dump_path = self._validate_dump_path(config_dump_path)
        self.list = self._validate_list(config_list)
        self.custom_api = self._validate_custom_api(config_custom_api)
        self.level = self._validate_level(config_level)
        self.rank = get_real_step_or_rank(config_rank, Const.RANK)
        self.rank_id = self._resolve_rank_id()
        self.slice_info = config_slice_info
        self.dump_enable = self._validate_dump_enable(dump_enable)
        self.switch = torch.tensor([int(self.dump_enable)], dtype=torch.int32)
        self.step_id = 0
        self._running = False
        self._tls = threading.local()
        self._tensor_data_dir_path = None

    @staticmethod
    def _default_config_path():
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(cur_dir, "..", "config.json")

    @staticmethod
    def _resolve_config_path(config_path):
        if config_path is None:
            config_path = AclGraphDumper._default_config_path()
        if not isinstance(config_path, str):
            raise TypeError("config_path must be a string")
        return check_and_get_real_path(config_path, FileCheckConst.READ_ABLE, must_exist=True)

    @staticmethod
    def _get_config_signature(config_path):
        info = os.stat(config_path)
        return info.st_mtime_ns, info.st_size

    @staticmethod
    def _load_msprobe_config(config_path):
        config_path = AclGraphDumper._resolve_config_path(config_path)
        json_config = load_json(config_path)
        if not isinstance(json_config, dict):
            raise TypeError("config must be a dict")
        task = json_config.get("task", Const.STATISTICS)
        task_config = json_config.get(task, {}) if isinstance(task, str) else {}
        if not isinstance(task_config, dict):
            raise TypeError(f"task config for {task} must be a dict")
        level = json_config.get("level", Const.LEVEL_L0)
        slice_info = task_config.get("slice", [])
        check_slice_info(slice_info)
        return (
            task,
            json_config.get("dump_path"),
            task_config.get("list", []),
            task_config.get("custom_api", []),
            level,
            json_config.get(Const.RANK),
            slice_info,
            json_config.get("dump_enable", True),
        )

    @staticmethod
    def _validate_dump_path(dump_path):
        if not isinstance(dump_path, str):
            raise TypeError("dump_path must be a string")
        dump_path = check_and_get_real_path(dump_path, FileCheckConst.WRITE_ABLE, must_exist=False)
        create_directory(dump_path)
        return dump_path

    @staticmethod
    def _validate_task(task):
        if task not in (Const.STATISTICS, Const.TENSOR):
            raise ValueError(f"task must be one of {[Const.STATISTICS, Const.TENSOR]}")
        return task

    @staticmethod
    def _validate_dump_enable(value):
        if not isinstance(value, bool):
            raise TypeError("dump_enable must be a boolean")
        return value

    def _switch_for_tensor(self, tensor):
        if self.switch.device != tensor.device:
            self.switch = self.switch.to(tensor.device)
        return self.switch

    def _refresh_dump_enable(self):
        try:
            signature = self._get_config_signature(self.config_path)
        except OSError:
            return
        if signature == self._config_signature:
            return
        self._config_signature = signature
        *_, value = self._load_msprobe_config(self.config_path)
        self.dump_enable = self._validate_dump_enable(value)
        self.switch.fill_(int(self.dump_enable))

    @staticmethod
    def _validate_list(keywords):
        if keywords is None:
            return []
        if not isinstance(keywords, list):
            raise TypeError("list must be a list[str]")
        for keyword in keywords:
            if not isinstance(keyword, str):
                raise TypeError("list must be a list[str]")
        return keywords

    @staticmethod
    def _validate_custom_api(custom_api):
        if custom_api is None:
            return []
        if not isinstance(custom_api, list) or not all(isinstance(item, str) and item for item in custom_api):
            raise TypeError("custom_api must be a list[str]")
        return custom_api

    @staticmethod
    def _validate_level(level):
        if not isinstance(level, str):
            raise TypeError("level must be a string")
        valid_levels = {Const.LEVEL_L0, Const.LEVEL_L1, Const.LEVEL_MIX}
        if level not in valid_levels:
            raise ValueError(f"level must be one of {sorted(valid_levels)}")
        return level

    @staticmethod
    def _resolve_rank_id():
        dist = getattr(torch, "distributed", None)
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return None
        try:
            return int(dist.get_rank())
        except Exception:
            return None

    def _step_rank_dir(self):
        rank_name = f"rank{self.rank_id}" if self.rank_id is not None else f"pid{os.getpid()}"
        path = os.path.join(self.dump_path, f"step{self.step_id}", rank_name)
        create_directory(path)
        return path

    def _rank_dir_name(self):
        return f"rank{self.rank_id}" if self.rank_id is not None else f"pid{os.getpid()}"

    def _tensor_data_dir(self):
        return os.path.join(self.dump_path, Const.DUMP_TENSOR_DATA, self._rank_dir_name())

    def _step_tensor_data_dir(self):
        return os.path.join(self.dump_path, f"step{self.step_id}", self._rank_dir_name(), Const.DUMP_TENSOR_DATA)

    def _prepare_tensor_data_dir(self):
        self._tensor_data_dir_path = self._tensor_data_dir()
        create_directory(self._tensor_data_dir_path)

    def _archive_tensor_data_dir(self):
        tensor_names = os.listdir(self._tensor_data_dir_path)
        if not tensor_names:
            self.step_id += 1
            return
        destination = self._step_tensor_data_dir()
        create_directory(destination)
        for name in tensor_names:
            shutil.move(
                os.path.join(self._tensor_data_dir_path, name),
                os.path.join(destination, name),
            )
        self.step_id += 1

    @staticmethod
    def _safe_file_stem(tag):
        safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        return "".join(ch if ch in safe else "_" for ch in tag)

    def _tensor_save_path(self, tag):
        return os.path.join(self._tensor_data_dir_path, f"{self._safe_file_stem(tag)}{Const.PT_SUFFIX}")

    def _module_scope(self, module_name):
        return module_name if module_name else "__root__"

    def _module_scope_name(self, module_name, module_class_name=None):
        scope_name = self._module_scope(module_name)
        if module_class_name:
            return f"Module.{scope_name}.{module_class_name}"
        return f"Module.{scope_name}"

    def _match_list_keywords(self, *candidates):
        if not self.list:
            return True
        normalized = [candidate.casefold() for candidate in candidates if isinstance(candidate, str)]
        return any(keyword.casefold() in candidate for keyword in self.list for candidate in normalized)

    def _should_collect_module(self, module_name, module_class_name):
        module_scope = self._module_scope(module_name)
        return self._match_list_keywords(
            module_name,
            module_scope,
            f"Module.{module_scope}",
            self._module_scope_name(module_name, module_class_name),
        )

    def _should_dump_current_rank(self):
        if not self.rank or self.rank_id is None:
            return True
        return self.rank_id in self.rank

    def _collect_module_enabled(self):
        return self.level in (Const.LEVEL_L0, Const.LEVEL_MIX)

    def _collect_api_enabled(self):
        return self.level in (Const.LEVEL_L1, Const.LEVEL_MIX)

    @staticmethod
    def _op_name_from_dispatch_func(func):
        schema = getattr(func, "_schema", None)
        schema_name = getattr(schema, "name", None)
        overload = getattr(func, "overloadname", None)
        if isinstance(schema_name, str) and "::" in schema_name:
            namespace, op_name = schema_name.split("::", 1)
        else:
            func_text = str(func)
            func_parts = func_text.split(".")
            if len(func_parts) >= 2:
                namespace, op_name = func_parts[0], func_parts[1]
                overload = overload or (func_parts[2] if len(func_parts) > 2 else None)
            else:
                namespace, op_name = "unknown", func_text

        if namespace == "aten":
            prefix = Const.ATEN_API_TYPE_PREFIX
            base = f"{prefix}.{op_name}"
        elif namespace == "npu":
            prefix = Const.NPU_API_TYPE_PREFIX
            base = f"{prefix}.{op_name}"
        else:
            prefix = Const.TORCH_API_TYPE_PREFIX
            base = f"{prefix}.{namespace}.{op_name}"

        if overload and overload != "default":
            base = f"{base}.{overload}"
        return base

    @staticmethod
    def _should_skip_dispatch_func(func):
        func_text = str(func)
        return (
            "acl_stat" in func_text
            or "acl_save" in func_text
            or "acl_tensor_save" in func_text
            or "higher_order." in func_text
        )

    def _tls_get(self, key, default):
        if not hasattr(self._tls, key):
            setattr(self._tls, key, default)
        return getattr(self._tls, key)

    def _is_dispatch_collecting(self):
        return self._tls_get("dispatch_collecting", False)

    def _dc(self, fn, *args, **kwargs):
        # Mark host-side collection so acl_stat-triggered dispatch does not recurse back into collection.
        previous = self._is_dispatch_collecting()
        setattr(self._tls, "dispatch_collecting", True)
        try:
            return fn(*args, **kwargs)
        finally:
            setattr(self._tls, "dispatch_collecting", previous)

    def _should_collect_api(self, op_name):
        return self._match_list_keywords(op_name)

    @staticmethod
    def _normalize_dtype(dtype):
        dtype_map = {
            "Bool": "torch.bool",
            "Byte": "torch.uint8",
            "Char": "torch.int8",
            "Short": "torch.int16",
            "Int": "torch.int32",
            "Long": "torch.int64",
            "Half": "torch.float16",
            "Float": "torch.float32",
            "Double": "torch.float64",
            "BFloat16": "torch.bfloat16",
            "ComplexFloat": "torch.complex64",
            "ComplexDouble": "torch.complex128",
        }
        return dtype_map.get(dtype, dtype)

    @classmethod
    def _build_tensor_record(cls, record):
        return {
            Const.TYPE: Const.TENSOR_TYPE,
            Const.DTYPE: cls._normalize_dtype(record.get("dtype")),
            Const.SHAPE: record.get("shape"),
            Const.MAX: record.get("max"),
            Const.MIN: record.get("min"),
            Const.MEAN: record.get("mean"),
            Const.NORM: record.get("norm"),
        }

    @classmethod
    def _assign_nested_value(cls, container, path_parts, value):
        current = container
        for idx, part in enumerate(path_parts):
            is_last = idx == len(path_parts) - 1
            if not isinstance(current, dict):
                raise TypeError(f"Expected dict when assigning key path, got {type(current)}")
            if is_last:
                current[part] = value
                return
            if part not in current or current[part] is None:
                current[part] = {}
            current = current[part]

    @classmethod
    def _compress_numeric_tree_to_list(cls, value):
        if isinstance(value, dict):
            converted = {key: cls._compress_numeric_tree_to_list(sub) for key, sub in value.items()}
            keys = list(converted.keys())
            if keys and all(part.isdigit() for part in keys):
                sorted_items = sorted(converted.items(), key=lambda kv: int(kv[0]))
                return [item for _, item in sorted_items]
            return converted
        return value

    @classmethod
    def _normalize_l0_op_name(cls, op_name):
        parts = op_name.split(".")
        if len(parts) >= 4 and parts[0] == "Module" and parts[-1] == "forward" and parts[-2].isdigit():
            return ".".join(parts[:-2] + ["forward", parts[-2]])
        return op_name

    @classmethod
    def _parse_stat_key(cls, key):
        marker = ".forward."
        marker_pos = key.find(marker)
        if marker_pos == -1:
            return None

        op_name = key[: marker_pos + len(".forward")]
        op_name = cls._normalize_l0_op_name(op_name)
        tail = key[marker_pos + len(marker) :]
        io_candidates = ("input_kwargs", "input", "output")
        for io_name in io_candidates:
            if tail == io_name:
                return op_name, io_name, []
            prefix = io_name + "."
            if tail.startswith(prefix):
                suffix = tail[len(prefix) :]
                return op_name, io_name, suffix.split(".") if suffix else []
        return None

    @classmethod
    def _convert_stats_to_dump_data(cls, stats):
        dump_data = {}
        for key, record in stats.items():
            parsed = cls._parse_stat_key(key)
            if parsed is None:
                continue
            op_name, io_name, path_parts = parsed
            op_entry = dump_data.setdefault(op_name, {})
            tensor_record = cls._build_tensor_record(record)

            if io_name == "input":
                container = op_entry.setdefault(Const.INPUT_ARGS, {})
                cls._assign_nested_value(container, path_parts or ["0"], tensor_record)
                continue

            if io_name == "input_kwargs":
                container = op_entry.setdefault(Const.INPUT_KWARGS, {})
                cls._assign_nested_value(container, path_parts or ["0"], tensor_record)
                continue

            if io_name == "output":
                container = op_entry.setdefault(Const.OUTPUT, {})
                cls._assign_nested_value(container, path_parts or ["0"], tensor_record)

        for op_entry in dump_data.values():
            if Const.INPUT_ARGS in op_entry:
                op_entry[Const.INPUT_ARGS] = cls._compress_numeric_tree_to_list(op_entry[Const.INPUT_ARGS])
            if Const.OUTPUT in op_entry:
                op_entry[Const.OUTPUT] = cls._compress_numeric_tree_to_list(op_entry[Const.OUTPUT])
        return dump_data

    def _collect(self, module_name, io_name, value, mark_forward_start=False, call_started=None):
        has_collected = False
        has_marked = False
        call_started = [False] if call_started is None else call_started
        slice_info = self.slice_info
        for suffix, tensor in _iter_tensors(value):
            if not _is_collectable_tensor(tensor):
                continue
            tag = f"{self._module_scope(module_name)}.{io_name}"
            effective_suffix = suffix
            # The marker is consumed by the statistics aggregation path to
            # delimit a forward. Tensor dumps are user-facing artifacts, so
            # keep their names free of this implementation detail.
            if self.task != Const.TENSOR and mark_forward_start and not has_marked:
                effective_suffix = FORWARD_START_MARKER if not suffix else f"{FORWARD_START_MARKER}.{suffix}"
                has_marked = True
            if effective_suffix:
                tag = f"{tag}.{effective_suffix}"
            if self.task == Const.TENSOR:
                api_name = self._safe_file_stem(self._module_scope(module_name))
                acl_tensor_save(
                    tensor, self._tensor_save_path(tag), api_name, not call_started[0], self._switch_for_tensor(tensor)
                )
                call_started[0] = True
            else:
                stat_tensor = slice_by_config(tensor, slice_info)
                acl_stat(stat_tensor, tag, self._switch_for_tensor(stat_tensor))
            has_collected = True
        return has_collected

    def _collect_custom_api(self, api_name, io_name, value, call_started):
        """Collect a patched Python API as one replay-time indexed call."""
        has_collected = False
        for suffix, tensor in _iter_tensors(value):
            if not _is_collectable_tensor(tensor):
                continue
            tag = f"{api_name}.{io_name}"
            if suffix:
                tag = f"{tag}.{suffix}"
            acl_tensor_save(
                tensor,
                self._tensor_save_path(tag),
                self._safe_file_stem(api_name),
                not call_started[0],
                self._switch_for_tensor(tensor),
            )
            call_started[0] = True
            has_collected = True
        return has_collected

    def _patch(self, model):
        dumper = self

        for module_name, module in model.named_modules():
            if hasattr(module, "_msprobe_aclgraph_origin_forward"):
                continue
            module_class_name = type(module).__name__
            collect_module = self._collect_module_enabled() and self._should_collect_module(
                module_name, module_class_name
            )
            should_wrap = collect_module or (self._collect_api_enabled() and not module_name)
            if not should_wrap:
                continue
            origin = module.forward

            @functools.wraps(origin)  # pylint: disable=cell-var-from-loop
            def wrapped_forward(
                *args,
                __origin=origin,
                __module_name=module_name,
                __module_class_name=module_class_name,
                __collect_module=collect_module,
                **kwargs,
            ):
                collect_module_data = dumper._running and __collect_module
                module_dump_name = (
                    dumper._module_scope_name(__module_name, __module_class_name) if collect_module_data else None
                )
                use_dispatch = (
                    dumper._running
                    and dumper._collect_api_enabled()
                    and TorchDispatchMode is not None
                    and not __module_name
                )
                dispatch_mode = _AclTorchDispatchMode(dumper) if use_dispatch else nullcontext()  # pylint: disable=possibly-used-before-assignment
                started = False
                call_started = [False]
                if collect_module_data:
                    collected = dumper._collect(
                        module_dump_name, "input", args, mark_forward_start=not started, call_started=call_started
                    )
                    started = started or collected
                    if kwargs:
                        collected = dumper._collect(
                            module_dump_name,
                            "input_kwargs",
                            kwargs,
                            mark_forward_start=not started,
                            call_started=call_started,
                        )
                        started = started or collected

                with dispatch_mode:
                    output = __origin(*args, **kwargs)

                if collect_module_data:
                    dumper._collect(
                        module_dump_name,
                        "output",
                        output,
                        mark_forward_start=not started,
                        call_started=call_started,
                    )
                return output

            module.forward = wrapped_forward
            module._msprobe_aclgraph_origin_forward = origin

    @staticmethod
    def _resolve_custom_api(target):
        """Resolve ``package.object.method`` to its owning object and attribute."""
        parts = target.split(".")
        for module_end in range(len(parts) - 1, 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:module_end]))
                parent = None
                for attr in parts[module_end:]:
                    parent = obj
                    obj = getattr(obj, attr)
                return parent, parts[-1], obj
            except (ImportError, AttributeError):
                continue
        raise ValueError(f"custom_api target cannot be resolved: {target}")

    def _patch_custom_api(self):
        for target in self.custom_api:
            owner, attr_name, origin = self._resolve_custom_api(target)
            if hasattr(origin, "_msprobe_aclgraph_origin"):
                continue
            setattr(owner, attr_name, self._make_custom_api_wrapper(origin, attr_name))

    def _make_custom_api_wrapper(self, origin, api_name):
        @functools.wraps(origin)
        def wrapped(*args, **kwargs):
            call_started = [False]
            if self._running:
                self._dc(self._collect_custom_api, api_name, "input", args, call_started)
                if kwargs:
                    self._dc(self._collect_custom_api, api_name, "input_kwargs", kwargs, call_started)
            output = origin(*args, **kwargs)
            if self._running:
                self._dc(self._collect_custom_api, api_name, "output", output, call_started)
            return output

        wrapped._msprobe_aclgraph_origin = origin
        return wrapped

    def _synchronize(self):
        if torch_npu is not None:
            try:
                torch_npu.npu.synchronize()
                return
            except Exception:  # nosec B110 - fall through to CUDA sync on NPU failure
                pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, model):
        self.rank_id = self._resolve_rank_id()
        if not self._should_dump_current_rank():
            self._running = False
            return
        if self.task == Const.TENSOR:
            self._prepare_tensor_data_dir()
        self._patch(model)
        self._running = True
        if self.task == Const.TENSOR:
            self._patch_custom_api()

    def step(self, dump=True):
        if not self._running:
            return

        self._refresh_dump_enable()
        self._synchronize()
        if self.task == Const.TENSOR:
            self._archive_tensor_data_dir()
            return
        stats = dict(get_acl_stat_dict(clear=True))
        if not dump:
            return
        statistic_names = (Const.MIN, Const.MAX, Const.MEAN, Const.NORM)
        if any(
            any(record.get(name) is None for name in statistic_names)
            for key, record in stats.items()
            if self._parse_stat_key(key) is not None
        ):
            logger.warning("Invalid statistics detected. Please use tensor mode to collect the affected data.")

        dump_json = {
            "task": Const.STATISTICS,
            "level": self.level,
            "framework": Const.PT_FRAMEWORK,
            "dump_data_dir": None,
            "data": self._convert_stats_to_dump_data(stats),
        }
        file_path = os.path.join(self._step_rank_dir(), "dump.json")
        save_json(file_path, dump_json, indent=2)
        self.step_id += 1
