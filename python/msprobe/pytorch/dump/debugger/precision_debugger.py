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
import re
import time
import torch
import importlib
import sys

from msprobe.core.dump.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.core.common.file_utils import load_yaml, FileChecker
from msprobe.pytorch.dump.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.dump.pt_config import parse_task_config
from msprobe.pytorch.dump.pytorch_service import PytorchService

from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.exceptions import MsprobeException, FileCheckException
from msprobe.core.common.utils import check_token_range, ThreadSafe, check_rank_id, get_real_step_or_rank
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import check_save_param, is_torch_nn_module
from msprobe.pytorch.dump.module_dump.module_dump import ModuleDumper


class PrecisionDebugger(BasePrecisionDebugger):
    _CONFIG_CHECK_INTERVAL_ENABLED_S = 0.5
    _CONFIG_CHECK_INTERVAL_DISABLED_S = 3.0

    def __init__(self, config_path=None, task=None, dump_path=None, level=None, step=None):
        if self.initialized:
            return
        super().__init__(config_path, task, dump_path, level, step)
        self._overrides = {
            "config_path": config_path,
            "task": task,
            "dump_path": dump_path,
            "level": level,
            "step": step,
        }
        self._reload_state = {
            "last_check_ts": 0.0,
            "signature": self._get_config_signature(),
        }
        self._dynamic_dump_enable_active = self.common_config.dump_enable is not None
        self.config = self._create_debugger_config(self.common_config, self.task_config)
        self.service = PytorchService(self.config)
        self.module_dumper = ModuleDumper(self.service)
        self.ori_customer_func = {}
        self._custom_op_schema_cache = None
        self._custom_op_schema_dirty = True
        self._custom_op_auto_registered = set()
        # 自动注册自定义算子（支持 torch.ops 下的第三方扩展算子）
        self._auto_register_custom_ops()
        self._custom_op_schema_dirty = True
        self._custom_api_auto_registered = set()
        self._custom_api_pending = []
        self._auto_register_custom_api(force_retry=True)

    @staticmethod
    def _get_task_config(task, json_config):
        return parse_task_config(task, json_config)

    @classmethod
    @ThreadSafe.synchronized
    def start(cls, model=None, token_range=None, rank_id=None):
        instance = cls._instance_with_reload()

        # 延迟注册：在 start() 时再次扫描自定义算子
        # 这样即使用户先创建 debugger 再 enable_custom_op() 也能正常工作
        instance._auto_register_custom_ops(refresh_schema=instance._custom_op_schema_dirty)

        check_token_range(token_range)
        check_rank_id(rank_id)
        instance.config.check_model(model, token_range)
        instance.service.start(model, token_range, rank_id)
        instance._auto_register_custom_api(force_retry=True)

    @classmethod
    @ThreadSafe.synchronized
    def stop(cls):
        cls._run_with_reload(lambda instance: instance.service.stop())

    @classmethod
    @ThreadSafe.synchronized
    def step(cls):
        cls._run_with_reload(lambda instance: instance.service.step())

    @classmethod
    @ThreadSafe.synchronized
    def save(cls, variable, name, save_backward=True):
        instance = cls._get_instance()
        if not instance._is_debug_save_enabled():
            return
        try:
            check_save_param(variable, name, save_backward)
        except ValueError:
            return
        instance.service.save(variable, name, save_backward)

    @classmethod
    def _instance_with_reload(cls):
        instance = cls._get_instance()
        instance._maybe_reload_config()
        return instance

    @classmethod
    def _run_with_reload(cls, action):
        instance = cls._get_instance()
        action(instance)
        instance._maybe_reload_config()

    def _is_debug_save_enabled(self):
        return self.task in [Const.TENSOR, Const.STATISTICS] and self.config.level == Const.LEVEL_DEBUG

    def _get_config_signature(self):
        config_path = self._overrides.get("config_path")
        if not config_path:
            return None
        try:
            stat_result = os.stat(config_path)
        except OSError:
            return None
        return stat_result.st_mtime_ns, stat_result.st_size

    def _get_changed_config_signature(self, force=False):
        now = time.monotonic()
        check_interval = self._get_config_check_interval_s()
        if not force and now - self._reload_state["last_check_ts"] < check_interval:
            return None
        self._reload_state["last_check_ts"] = now
        current_signature = self._get_config_signature()
        if current_signature is None or current_signature == self._reload_state["signature"]:
            return None
        return current_signature

    def _get_config_check_interval_s(self):
        dump_enable = getattr(self.config, "dump_enable", None)
        return (
            self._CONFIG_CHECK_INTERVAL_ENABLED_S
            if self._is_dump_enabled(dump_enable)
            else self._CONFIG_CHECK_INTERVAL_DISABLED_S
        )

    def _maybe_reload_config(self, force=False):
        if not self._dynamic_dump_enable_active:
            return False
        pending_signature = self._get_changed_config_signature(force=force)
        if pending_signature is None:
            return False
        return self._reload_config(pending_signature)

    def _create_debugger_config(self, common_config, task_config):
        return DebuggerConfig(
            common_config, task_config, self._overrides["task"], self._overrides["dump_path"], self._overrides["level"]
        )

    def _reload_config(self, pending_signature):
        try:
            common_config, task_config = self._parse_config_path(
                self._overrides["config_path"], self._overrides["task"]
            )
            if self._overrides["step"] is not None:
                common_config.step = get_real_step_or_rank(self._overrides["step"], Const.STEP)
            new_config = self._create_debugger_config(common_config, task_config)
        except Exception as ex:
            self._fail_close_dump()
            logger.warning(f"Config hot reload skipped because parsing failed: {ex}")
            return False

        self._apply_reloaded_config(common_config, task_config, new_config, pending_signature)
        logger.info("PrecisionDebugger detected config change and reloaded runtime settings.")
        return True

    # pylint: disable=attribute-defined-outside-init
    def _apply_reloaded_config(self, common_config, task_config, new_config, signature):
        previous_dump_enable = getattr(self.config, "dump_enable", None)
        previous_custom_op_namespaces = getattr(self.config, "custom_op_namespaces", None)
        # In dynamic mode, deleting dump_enable keeps the previous state.
        if self._dynamic_dump_enable_active and new_config.dump_enable is None:
            new_config.dump_enable = previous_dump_enable
            common_config.dump_enable = previous_dump_enable

        if self._is_dump_enabled(previous_dump_enable) and not self._is_dump_enabled(new_config.dump_enable):
            # Use unified stop path to flush existing buffered data before turning dump off.
            self.service.stop()

        self.common_config = common_config
        self.task_config = task_config
        self.task = common_config.task
        self.config = new_config
        if previous_custom_op_namespaces != getattr(new_config, "custom_op_namespaces", None):
            self._custom_op_schema_dirty = True
        self.service.apply_runtime_config(new_config)
        self._reload_state["signature"] = signature

    def _fail_close_dump(self):
        previous_dump_enable = getattr(self.config, "dump_enable", None)
        if self._is_dump_enabled(previous_dump_enable):
            self.service.stop()
        self.config.dump_enable = False
        self.common_config.dump_enable = False
        self.service.apply_runtime_config(self.config)

    @staticmethod
    def _is_dump_enabled(dump_enable):
        return True if dump_enable is None else dump_enable

    def _auto_register_custom_ops(self, refresh_schema=False):
        """
        自动扫描并注册 torch.ops 下的自定义算子
        支持华为昇腾扩展算子（npu, _C_ascend, atb等）
        """

        if refresh_schema:
            self._custom_op_schema_cache = None

        # Only configured namespaces are auto-registered to avoid scanning all torch.ops namespaces.
        priority_namespaces = getattr(self.config, "custom_op_namespaces", ['_C_ascend'])

        # 优先检查的自定义算子命名空间
        registered_count = 0

        # 首先处理优先命名空间
        for ns_name in priority_namespaces:
            try:
                ns_module = getattr(torch.ops, ns_name, None)
                if ns_module is None:
                    continue

                ops = self._get_custom_ops_from_schema(ns_name)
                for op_name in ops:
                    try:
                        if self._register_single_op(ns_name, op_name, ns_module):
                            registered_count += 1
                    except Exception as ex:
                        logger.debug(f"Failed to auto register custom op {ns_name}.{op_name}: {ex}")
            except Exception as ex:
                logger.debug(f"Failed to scan custom ops namespace {ns_name}: {ex}")

        if registered_count > 0:
            logger.info(f"Auto registered {registered_count} custom ops for dump.")

    def _register_single_op(self, ns_name, op_name, ns_module):
        op_key = (ns_name, op_name)
        if op_key in self._custom_op_auto_registered:
            return False
        if not hasattr(ns_module, op_name):
            return False

        self.service.register_custom_api(ns_module, op_name, ns_name)
        self._custom_op_auto_registered.add(op_key)
        return True

    def _get_custom_ops_from_schema(self, namespace):
        """从 JIT schema 获取指定命名空间的所有算子"""
        return sorted(self._get_custom_ops_schema_cache().get(namespace, set()))

    def _get_custom_ops_schema_cache(self):
        """Cache JIT schemas by namespace to avoid repeated full-schema scans."""
        if self._custom_op_schema_cache is not None:
            return self._custom_op_schema_cache

        ops_by_namespace = {}
        try:
            target_namespaces = set(getattr(self.config, "custom_op_namespaces", ['_C_ascend']))
            schemas = torch._C._jit_get_all_schemas()

            for s in schemas:
                schema_str = str(s)
                if '::' not in schema_str:
                    continue

                ns, rest = schema_str.split('::', 1)
                if ns not in target_namespaces:
                    continue
                op_name = rest.split('(')[0] if '(' in rest else rest
                ops_by_namespace.setdefault(ns, set()).add(op_name)
        except Exception as ex:
            logger.debug(f"Failed to get custom ops from JIT schema: {ex}")
            self._custom_op_schema_dirty = True
            if self._custom_op_schema_cache is None:
                self._custom_op_schema_cache = {}
            return self._custom_op_schema_cache

        self._custom_op_schema_cache = ops_by_namespace
        self._custom_op_schema_dirty = False
        return self._custom_op_schema_cache

    def _auto_register_custom_api(self, force_retry):
        custom_api_list = self._load_custom_api_from_yaml()
        if not custom_api_list and not (force_retry and self._custom_api_pending):
            return

        pending = []
        for item in custom_api_list:
            module_path = item.get("module")
            api_name = item.get("api")
            api_prefix = item.get("prefix", None)
            key = (module_path, api_name)
            if key in self._custom_api_auto_registered:
                continue

            try:
                module_obj = self._resolve_module_path(module_path)
                if not hasattr(module_obj, api_name):
                    raise AttributeError(f"{module_path} does not have attribute {api_name}")
                self.__class__.register_custom_api(module_obj, api_name, api_prefix)
                self._custom_api_auto_registered.add(key)
                logger.info(f"Auto-registered custom api from yaml: {module_path}.{api_name}")
            except Exception as ex:
                pending.append(item)
                logger.warning(f"Auto-register custom api from yaml skipped: {item}, reason: {ex}")

        if force_retry:
            for item in self._custom_api_pending:
                if item not in pending:
                    pending.append(item)
        self._custom_api_pending = pending

    def _load_custom_api_from_yaml(self):
        api_dump_dir = os.path.realpath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "api_dump")
        )
        try:
            path_checker = FileChecker(api_dump_dir, FileCheckConst.DIR, FileCheckConst.READ_ABLE)
            api_dump_dir = path_checker.common_check()
        except FileCheckException:
            return []

        yaml_path = os.path.join(api_dump_dir, "custom_wrap_ops.yaml")

        try:
            content = load_yaml(yaml_path)
        except Exception as ex:
            logger.warning(f"Failed to load custom api yaml: {yaml_path}, reason: {ex}")
            return []
        if not content:
            logger.info(f"Custom api yaml is empty: {yaml_path}, yaml-based custom op registration is disabled.")
            return []
        if not isinstance(content, dict):
            logger.warning(f"Invalid custom api yaml format: {yaml_path}, expected dict")
            return []

        items = []
        for module_path, api_list in content.items():
            if not isinstance(module_path, str) or not module_path:
                continue
            if len(module_path) > Const.MAX_MODULE_PATH_LEN or not re.match(Const.PY_MODULE_PATH_PATTERN, module_path):
                continue
            if isinstance(api_list, str):
                api_list = [api_list]
            if not isinstance(api_list, list):
                continue

            default_prefix = module_path

            for api_name in api_list:
                if not isinstance(api_name, str) or not api_name:
                    continue
                if len(api_name) > Const.MAX_API_NAME_LEN or not re.match(Const.PY_IDENTIFIER_PATTERN, api_name):
                    continue
                items.append(
                    {
                        "module": module_path,
                        "api": api_name,
                        "prefix": default_prefix,
                    }
                )
        return items

    @staticmethod
    def _resolve_module_path(module_path: str):
        if (
            not isinstance(module_path, str)
            or not module_path
            or len(module_path) > Const.MAX_MODULE_PATH_LEN
            or not re.match(Const.PY_MODULE_PATH_PATTERN, module_path)
        ):
            raise MsprobeException(MsprobeException.INVALID_CHAR_ERROR, f"Invalid module path: {module_path}")
        if module_path in sys.modules and sys.modules[module_path] is not None:
            return sys.modules[module_path]
        parts = module_path.split(".")
        obj = importlib.import_module(parts[0])
        for i in range(1, len(parts)):
            part = parts[i]
            if hasattr(obj, part):
                obj = getattr(obj, part)
                continue
            candidate_module = ".".join(parts[: i + 1])
            obj = importlib.import_module(candidate_module)
        return obj


@ThreadSafe.synchronized
def module_dump(module, dump_name):
    if not is_torch_nn_module(module):
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR,
            f"the module argument in module_dump must be a torch.nn.Module type, "
            f"but currently there is an unsupported {type(module)} type.",
        )
    if not isinstance(dump_name, str):
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR, "the dump_name argument in module_dump must be a str type"
        )
    instance = _get_debugger_instance()
    instance.module_dumper.start_module_dump(module, dump_name)


@ThreadSafe.synchronized
def module_dump_end():
    instance = _get_debugger_instance()
    instance.module_dumper.stop_module_dump()


def _get_debugger_instance():
    instance = PrecisionDebugger._instance
    if instance:
        return instance
    raise MsprobeException(
        MsprobeException.INTERFACE_USAGE_ERROR,
        "PrecisionDebugger must be instantiated before using module_dump interfaces",
    )
