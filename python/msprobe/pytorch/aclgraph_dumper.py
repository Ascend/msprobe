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
import os
import threading
from collections.abc import Iterable
from contextlib import nullcontext

import torch

from msprobe.pytorch.aclgraph_dump import acl_stat, get_acl_stat_dict
from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.file_utils import create_directory, check_and_get_real_path, save_json, load_json
from msprobe.core.common.utils import get_real_step_or_rank

try:
    import torch_npu
except Exception:
    torch_npu = None

try:
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception:
    TorchDispatchMode = None


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
        def __init__(self, dumper):
            super().__init__()
            self._dumper = dumper

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs if kwargs is not None else {}
            if self._dumper._should_skip_dispatch_func(func):
                return func(*args, **kwargs)

            if self._dumper._is_dispatch_collecting():
                return func(*args, **kwargs)

            op_scope = self._dumper._next_api_scope(func)
            should_collect = self._dumper._should_collect_api(self._dumper._current_scope(), func)
            started = False
            if should_collect:
                collected = self._dumper._dc(self._dumper._collect, op_scope, "input", args,
                                             mark_forward_start=not started)
                started = started or collected
                if kwargs:
                    collected = self._dumper._dc(self._dumper._collect, op_scope, "input_kwargs", kwargs,
                                                 mark_forward_start=not started)
                    started = started or collected

            output = func(*args, **kwargs)

            if should_collect:
                self._dumper._dc(self._dumper._collect, op_scope, "output", output, mark_forward_start=not started)
            return output


class AclGraphDumper:
    def __init__(self, config_path=None):
        config_dump_path, config_list, config_level, config_rank = self._load_msprobe_config(config_path)
        self.dump_path = self._validate_dump_path(config_dump_path)
        self.list = self._validate_list(config_list)
        self.level = self._validate_level(config_level)
        self.rank = get_real_step_or_rank(config_rank, Const.RANK)
        self.rank_id = self._resolve_rank_id()
        self.step_id = 0
        self._running = False
        self._tls = threading.local()

    @staticmethod
    def _default_config_path():
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(cur_dir, "..", "config.json")

    @staticmethod
    def _load_msprobe_config(config_path):
        if config_path is None:
            config_path = AclGraphDumper._default_config_path()
        if not isinstance(config_path, str):
            raise TypeError("config_path must be a string")
        config_path = check_and_get_real_path(config_path, FileCheckConst.READ_ABLE, must_exist=True)
        json_config = load_json(config_path)
        if not isinstance(json_config, dict):
            raise TypeError("config must be a dict")
        task = json_config.get("task", Const.STATISTICS)
        task_config = json_config.get(task, {}) if isinstance(task, str) else {}
        if not isinstance(task_config, dict):
            raise TypeError(f"task config for {task} must be a dict")
        level = task_config.get("level", json_config.get("level", Const.LEVEL_L0))
        return json_config.get("dump_path"), task_config.get("list", []), level, json_config.get(Const.RANK)

    @staticmethod
    def _validate_dump_path(dump_path):
        if not isinstance(dump_path, str):
            raise TypeError("dump_path must be a string")
        dump_path = check_and_get_real_path(dump_path, FileCheckConst.WRITE_ABLE, must_exist=False)
        create_directory(dump_path)
        return dump_path

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
        return "acl_stat" in func_text or "acl_save" in func_text

    def _tls_get(self, key, default):
        if not hasattr(self._tls, key):
            setattr(self._tls, key, default)
        return getattr(self._tls, key)

    def _dispatch_depth(self):
        return self._tls_get("dispatch_depth", 0)

    def _set_dispatch_depth(self, depth):
        setattr(self._tls, "dispatch_depth", depth)

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

    def _scope_stack(self):
        return self._tls_get("scope_stack", [])

    def _push_scope(self, module_name):
        stack = self._scope_stack()
        stack.append(module_name)

    def _pop_scope(self):
        stack = self._scope_stack()
        if stack:
            stack.pop()

    def _current_scope(self):
        stack = self._scope_stack()
        return stack[-1] if stack else ""

    def _next_api_scope(self, func):
        op_name = self._op_name_from_dispatch_func(func)
        scope = self._module_scope(self._current_scope())
        return f"{scope}.{op_name}"

    def _should_collect_api(self, module_name, func):
        if not self.list:
            return True
        scope = self._module_scope(module_name)
        op_name = self._op_name_from_dispatch_func(func)
        api_scope_prefix = self._module_scope_name(module_name)
        return self._match_list_keywords(
            module_name,
            scope,
            api_scope_prefix,
            op_name,
            f"{scope}.{op_name}",
            f"{api_scope_prefix}.{op_name}",
        )

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
        tail = key[marker_pos + len(marker):]
        io_candidates = ("input_kwargs", "input", "output")
        for io_name in io_candidates:
            if tail == io_name:
                return op_name, io_name, []
            prefix = io_name + "."
            if tail.startswith(prefix):
                suffix = tail[len(prefix):]
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

    def _collect(self, module_name, io_name, value, mark_forward_start=False):
        has_collected = False
        has_marked = False
        for suffix, tensor in _iter_tensors(value):
            if not _is_collectable_tensor(tensor):
                continue
            tag = f"{self._module_scope(module_name)}.{io_name}"
            effective_suffix = suffix
            if mark_forward_start and not has_marked:
                effective_suffix = (
                    FORWARD_START_MARKER if not suffix else f"{FORWARD_START_MARKER}.{suffix}"
                )
                has_marked = True
            if effective_suffix:
                tag = f"{tag}.{effective_suffix}"
            acl_stat(tensor, tag)
            has_collected = True
        return has_collected

    def _patch(self, model):
        dumper = self

        for module_name, module in model.named_modules():
            if hasattr(module, "_msprobe_aclgraph_origin_forward"):
                continue
            module_class_name = type(module).__name__
            should_collect_module = self._should_collect_module(module_name, module_class_name)
            should_wrap = self._collect_api_enabled() or should_collect_module
            if not should_wrap:
                continue
            origin = module.forward

            @functools.wraps(origin)
            def wrapped_forward(
                *args,
                __origin=origin,
                __module_name=module_name,
                __module_class_name=module_class_name,
                __should_collect_module=should_collect_module,
                **kwargs
            ):
                api_scope_name = dumper._module_scope_name(__module_name, __module_class_name)
                collect_module_data = dumper._running and dumper._collect_module_enabled() and __should_collect_module
                module_scope_name = api_scope_name if collect_module_data else __module_name
                dumper._push_scope(api_scope_name)
                depth = dumper._dispatch_depth()
                dumper._set_dispatch_depth(depth + 1)
                use_dispatch = (
                    dumper._running and
                    dumper._collect_api_enabled() and
                    TorchDispatchMode is not None and
                    depth == 0
                )
                dispatch_mode = _AclTorchDispatchMode(dumper) if use_dispatch else nullcontext()
                started = False
                try:
                    if collect_module_data:
                        collected = dumper._collect(module_scope_name, "input", args, mark_forward_start=not started)
                        started = started or collected
                        if kwargs:
                            collected = dumper._collect(
                                module_scope_name, "input_kwargs", kwargs, mark_forward_start=not started
                            )
                            started = started or collected

                    with dispatch_mode:
                        output = __origin(*args, **kwargs)

                    if collect_module_data:
                        dumper._collect(module_scope_name, "output", output, mark_forward_start=not started)
                    return output
                finally:
                    dumper._set_dispatch_depth(depth)
                    dumper._pop_scope()

            module.forward = wrapped_forward
            module._msprobe_aclgraph_origin_forward = origin

    def _synchronize(self):
        if torch_npu is not None:
            try:
                torch_npu.npu.synchronize()
                return
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, model):
        self.rank_id = self._resolve_rank_id()
        if not self._should_dump_current_rank():
            self._running = False
            return
        self._patch(model)
        self._running = True

    def step(self, dump=True):
        if not self._running:
            return

        self._synchronize()
        stats = dict(get_acl_stat_dict(clear=True))
        if not dump:
            return

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
