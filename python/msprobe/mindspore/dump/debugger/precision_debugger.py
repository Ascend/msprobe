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
from collections import defaultdict, namedtuple

from mindspore.ops.operations import _inner_ops as inner
from mindspore._c_expression import MSContext

from msprobe.core.common.const import Const, MsgConst
from msprobe.core.common.utils import check_token_range, ThreadSafe, check_rank_id
from msprobe.core.common.runtime import Runtime
from msprobe.core.dump.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.mindspore.dump.cell_processor import CellProcessor
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.utils import (
    set_register_backward_hook_functions,
    check_save_param,
    wrap_backward_hook_call_func,
)
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.graph_mode_cell_dump import GraphModeCellDump
from msprobe.mindspore.dump.dump_processor.hook_cell.api_register import get_api_register
from msprobe.mindspore.dump.dump_processor.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.dump.ms_config import parse_task_config
from msprobe.mindspore.dump.mindspore_service import MindsporeService
from msprobe.mindspore.dump.task_handler_factory import TaskHandlerFactory

try:
    import mindspore as ms
    from mindspore._c_expression import _dump_start, _dump_stop, _dump_step, _set_init_iter, _dump_set_dynamic
except ImportError:
    enable_dynamic_kbyk_dump = False
else:
    enable_dynamic_kbyk_dump = True


ConfigParameters = namedtuple("ConfigParameters", ["config_path", "task", "dump_path", "level"])


class PrecisionDebugger(BasePrecisionDebugger):
    def __init__(self, config_path=None, task=None, dump_path=None, level=None, step=None):
        if self.initialized:
            return
        set_register_backward_hook_functions()
        super().__init__(config_path, task, dump_path, level, step)

        self.common_config.level = level if level else self.common_config.level
        self.common_config.dump_path = dump_path if dump_path else self.common_config.dump_path
        self.config = DebuggerConfig(self.common_config, self.task_config)

        if self._is_kernel_dump() and not self.task_config.is_regex_valid:
            raise ValueError('Illegal regular expressions exist in the list.')

        setattr(
            inner.CellBackwardHook,
            '__call__',
            wrap_backward_hook_call_func(getattr(inner.CellBackwardHook, '__call__')),
        )

        self.config.execution_mode = self._get_execution_mode()
        if self._need_service():
            self.service = MindsporeService(self.config)

        Runtime.step_count = 0
        Runtime.is_running = False
        if enable_dynamic_kbyk_dump and self.config.level_ori == Const.LEVEL_L2:
            _dump_set_dynamic()

    @staticmethod
    def _get_task_config(task, json_config):
        return parse_task_config(task, json_config)

    @staticmethod
    def _get_execution_mode():
        jit_level = ms.context.get_jit_config().get(MsConst.JIT_LEVEL)
        jit_level = jit_level if jit_level else ms.get_context(MsConst.JIT_LEVEL)
        if not jit_level:
            if MSContext.get_instance().get_ascend_soc_version() == MsConst.ASCEND_910A:
                jit_level = MsConst.JIT_LEVEL_O2
            else:
                jit_level = MsConst.JIT_LEVEL_O0

        if ms.get_context("mode") == ms.GRAPH_MODE:
            if jit_level == MsConst.JIT_LEVEL_O2:
                return MsConst.GRAPH_GE_MODE
            else:
                return MsConst.GRAPH_KBYK_MODE
        else:
            return MsConst.PYNATIVE_MODE

    @classmethod
    def handle_graph_cell_dump(cls, instance):
        """处理graph cell dump相关逻辑"""
        is_graph_mode = instance.config.execution_mode in [MsConst.GRAPH_GE_MODE, MsConst.GRAPH_KBYK_MODE]

        if is_graph_mode:
            GraphModeCellDump.step(instance.config.dump_path, instance.config.step, instance.config.task)

        if Runtime.run_mode == MsConst.PYNATIVE_GRAPH_MODE:
            dump_path = os.path.join(instance.config.dump_path, MsConst.GRAPH_MODE)
            GraphModeCellDump.step(dump_path, instance.config.step, instance.config.task)

    @classmethod
    def start(cls, model=None, token_range=None, rank_id=None):
        instance = cls._get_instance()
        if instance is None:
            return
        check_rank_id(rank_id)
        if cls._is_kernel_dump():
            cls._start_kernel_dump()
        else:
            check_token_range(token_range)
            instance.config.execution_mode = cls._get_execution_mode()
            if cls._need_service():
                with ThreadSafe():
                    if not instance.service:
                        instance.service = MindsporeService(instance.config)
                    instance.config.check_model(model, token_range)
                    instance.service.start(model, token_range, rank_id)
            else:
                if not instance.first_start:
                    get_api_register().restore_all_api()
                    handler = TaskHandlerFactory.create(instance.config, model)
                    handler.handle()
                Runtime.is_running = True
        instance.first_start = True

    @classmethod
    def stop(cls):
        instance = cls._get_instance()
        if instance is None:
            return

        if instance.service:
            with ThreadSafe():
                instance.service.stop()
        else:
            Runtime.is_running = False
        if enable_dynamic_kbyk_dump and instance.config.level_ori == Const.LEVEL_L2:
            ms.runtime.synchronize()
            _dump_stop()

    @classmethod
    def step(cls):
        instance = cls._get_instance()
        if instance is None:
            return

        if instance.service:
            with ThreadSafe():
                instance.service.step()

        if instance.config.level in [MsConst.CELL, MsConst.CELL_AND_API]:
            cls.handle_graph_cell_dump(instance)

        if enable_dynamic_kbyk_dump and instance.config.level_ori == Const.LEVEL_L2:
            _dump_step(1)

        HOOKCell.cell_count = defaultdict(int)
        CellProcessor.reset_cell_stats()
        Runtime.step_count += 1

    @classmethod
    @ThreadSafe.synchronized
    def save(cls, variable, name, save_backward=True):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)  # pylint: disable=W0719
        if instance.task not in [Const.TENSOR, Const.STATISTICS] or instance.config.level_ori != Const.LEVEL_DEBUG:
            return
        try:
            check_save_param(variable, name, save_backward)
        except ValueError:
            return
        if not instance.service:
            instance.service = MindsporeService(instance.config)
        instance.service.save(variable, name, save_backward)

    @classmethod
    def _need_service(cls):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)  # pylint: disable=W0719
        if instance.config.level_ori == Const.LEVEL_L2:
            return False
        if instance.config.execution_mode != MsConst.PYNATIVE_MODE:
            return False
        return True

    @classmethod
    def _is_kernel_dump(cls):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)  # pylint: disable=W0719
        return instance.config.level_ori == Const.LEVEL_L2

    @classmethod
    def _start_kernel_dump(cls):
        instance = cls._get_instance()

        if not instance.first_start:
            get_api_register().restore_all_api()
            handlers = TaskHandlerFactory.create(instance.config)
            for handler in handlers:
                handler.handle()
            if enable_dynamic_kbyk_dump:
                _set_init_iter(0)
        if enable_dynamic_kbyk_dump:
            is_valid_rank = not instance.config.rank or Runtime.rank_id in instance.config.rank
            is_valid_step = not instance.config.step or Runtime.step_count in instance.config.step
            if is_valid_rank and is_valid_step:
                _dump_start()
