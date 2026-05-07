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

import atexit
import os
import re
import threading
import traceback

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common.megatron_utils import MegatronStepInfo, get_micro_step, is_megatron
from msprobe.core.dump.data_dump.data_processor.factory import DataProcessorFactory
from msprobe.core.dump.data_dump.json_writer import DataWriter
from msprobe.core.dump.data_dump.scope import ScopeFactory


def build_data_collector(config):
    return DataCollector(config)


class DataCollector:
    tasks_need_tensor_data = [Const.DIFF_CHECK, Const.TENSOR]
    level_without_construct = [Const.LEVEL_L1, Const.LEVEL_L2]
    _api_name_pattern = re.compile(r'\.\d+\.(forward|backward)$')

    def __init__(self, config):
        self.config = config
        self.data_writer = DataWriter()
        self.data_processor = DataProcessorFactory.create_processor(self.config, self.data_writer)
        self.module_processor = DataProcessorFactory.get_module_processor(self.config.framework)
        self.module_count = {}
        self.scope = ScopeFactory(self.config).build_scope()
        self.backward_module_names = {}
        self.params_grad_record = {}
        self.optimizer_status = ""
        self.optimizer_status_first_start = {Const.OPTIMIZER: True, Const.CLIP_GRAD: True}
        atexit.register(self.write_json_at_exit)

    @property
    def dump_data_dir(self):
        return self.data_writer.dump_tensor_data_dir

    @property
    def _collect_extra_info(self):
        return getattr(self.config, "extra_info", True)

    @property
    def dump_file_path(self):
        return self.data_writer.dump_file_path

    @property
    def bench_dump_file_path(self):
        return self.data_writer.bench_dump_file_path

    @staticmethod
    def check_scope_and_pid(scope, name, pid):
        return (not scope or scope.check(name)) and pid == os.getpid()

    def should_collect(self, name, pid):
        """
        判断是否应该采集数据
        - 如果配置了scope或list，则按照scope或list匹配
        - 如果没有配置scope或list，则按照风险等级dump
        """
        if self.scope:
            return self.check_scope_and_pid(self.scope, name, pid)
        else:
            return self._should_collect_by_risk_level(name)

    def _should_collect_by_risk_level(self, name):
        if self.config.level not in [Const.LEVEL_L1, Const.LEVEL_MIX]:
            return True
        if not hasattr(self.config, 'risk_level') or not self.config.risk_level:
            return True
        if self.config.risk_level == Const.RISK_LEVEL_ALL:
            return True
        if self.config.framework == Const.PT_FRAMEWORK:
            from msprobe.pytorch.dump.api_dump.api_risk_level import get_api_risk_level
            api_name = self._api_name_pattern.sub('', name).split(Const.SEP, 1)
            prefix = api_name[0]
            if prefix in Const.MODULE_PREFIX:
                return True
            api_name = api_name[-1]
            api_risk = get_api_risk_level(api_name)
            if self.config.risk_level == Const.RISK_LEVEL_CORE:
                return api_risk == Const.RISK_LEVEL_CORE
            elif self.config.risk_level == Const.RISK_LEVEL_FOCUS:
                return api_risk != Const.RISK_LEVEL_LOW
            return True

        return True

    def reset_status(self):
        self.optimizer_status = ""
        self.optimizer_status_first_start = {Const.OPTIMIZER: True, Const.CLIP_GRAD: True}
        self.data_writer.reset_cache()
        self.backward_module_names.clear()

    def update_api_or_module_name(self, api_or_module_name):
        self.data_processor.update_api_or_module_name(api_or_module_name)

    def write_json(self):
        self.data_writer.write_json()

    def write_json_at_exit(self):
        if self.config.async_dump and self.config.task == Const.TENSOR:
            self.data_processor.dump_async_data()
        self.data_writer.write_json()

    def update_data(self, name, data_info):
        logger.debug(f"msprobe is collecting data on {name}.")
        self.data_writer.update_data(data_info)

    def call_stack_collect(self, name):
        if not self._collect_extra_info:
            return
        stack_info = self.data_processor.analyze_api_call_stack(name)
        self.data_writer.update_stack(name, stack_info)

    def forward_input_data_collect(self, name, module, pid, module_input_output):
        try:
            if not self.should_collect(name, pid):
                return

            data_info = {}
            if self.config.task != Const.STRUCTURE:
                data_info = self.data_processor.analyze_forward_input(name, module, module_input_output)
            if self.config.level == Const.LEVEL_L2:
                return
            self.call_stack_collect(name)
            self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

        except Exception as e:
            # 取异常类名作为“类型”做去重
            error_type = type(e).__name__
            tb = traceback.format_exc()
            self.data_writer.write_error_log(
                f"[ERROR] forward_input_data_collect failed: name={name}, pid={pid}\n{tb}",
                error_type=error_type
            )

    def forward_output_data_collect(self, name, module, pid, module_input_output):
        self.update_construct(name)
        if not self.should_collect(name, pid):
            return

        data_info = {}
        if self.config.task != Const.STRUCTURE:
            data_info = self.data_processor.analyze_forward_output(name, module, module_input_output)
        if self.config.level == Const.LEVEL_L2:
            return
        self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

    def forward_data_collect_only_tensor(self, name, module, pid, module_input_output):
        if not self.should_collect(name, pid):
            return
        self.data_processor.analyze_forward(name, module, module_input_output)

    def forward_data_collect(self, name, module, pid, module_input_output):
        self.update_construct(name)
        if not self.should_collect(name, pid):
            return
        data_info = {}
        if self.config.task != Const.STRUCTURE:
            data_info = self.data_processor.analyze_forward(name, module, module_input_output)
        self.call_stack_collect(name)
        self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

    def backward_data_collect_only_tensor(self, name, module, pid, module_input_output):
        if not self.should_collect(name, pid):
            return
        self.data_processor.analyze_backward(name, module, module_input_output)

    def backward_data_collect(self, name, module, pid, module_input_output):
        self.update_construct(name)
        if not self.should_collect(name, pid):
            return
        data_info = {}
        if self.config.task != Const.STRUCTURE:
            data_info = self.data_processor.analyze_backward(name, module, module_input_output)
        if self.config.level == Const.LEVEL_L2:
            return
        if data_info and name.split(Const.SEP)[0] in Const.MODULE_PREFIX:
            module_name = name.rsplit(Const.SEP, 2)[0]
            self.backward_module_names[module_name] = True
        self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

    def backward_input_data_collect(self, name, module, pid, module_input_output):
        self.update_construct(name)
        if not self.should_collect(name, pid):
            return
        data_info = {}
        if self.config.task != Const.STRUCTURE:
            data_info = self.data_processor.analyze_backward_input(name, module, module_input_output)
        self.handle_data(name, data_info)

    def backward_output_data_collect(self, name, module, pid, module_input_output):
        self.update_construct(name)
        if not self.should_collect(name, pid):
            return
        data_info = {}
        if self.config.task != Const.STRUCTURE:
            data_info = self.data_processor.analyze_backward_output(name, module, module_input_output)
        self.handle_data(name, data_info)

    def update_construct(self, name):
        if not self._collect_extra_info:
            return
        if self.config.level not in DataCollector.level_without_construct:
            if self.optimizer_status in [Const.OPTIMIZER, Const.CLIP_GRAD]:
                if self.optimizer_status_first_start[self.optimizer_status]:
                    self.data_writer.update_construct(
                        {self.optimizer_status: None if not is_megatron() else [None, get_micro_step()]})
                    self.optimizer_status_first_start[self.optimizer_status] = False
                self.data_writer.update_construct(
                    {name: self.optimizer_status if not is_megatron() else [self.optimizer_status, get_micro_step()]})
            else:
                if self.config.level == Const.LEVEL_MIX and \
                        not (name.startswith(Const.MODULE) or name.startswith(Const.CELL)):
                    self.data_writer.update_construct(
                        {name: self.module_processor.api_parent_node.get(threading.get_ident())}
                    )
            if MegatronStepInfo.is_megatron:
                micro_step_number = max(MegatronStepInfo.forward_micro_step, MegatronStepInfo.backward_micro_step)
                self.data_writer.update_construct({Const.MEGATRON_MICRO_STEP_NUMBER: micro_step_number})

            self.data_writer.update_construct(self.module_processor.module_node)

    def handle_data(self, name, data_info, flush=False):
        if data_info:
            self.update_data(name, data_info)
        if self.config.async_dump:
            return
        if not flush:
            self.data_writer.flush_data_periodically()
        else:
            self.write_json()

    def update_dump_paths(self, *args):
        self.data_writer.update_dump_paths(*args)

    def replace_proc_with_rank(self, rank_path):
        self.data_writer.replace_proc_with_rank(rank_path)

    def initialize_json_file(self, framework=Const.UNKNOWN_FRAMEWORK):
        self.data_writer.initialize_json_file(task=self.config.task, level=self.config.level, framework=framework)

    def update_iter(self, current_iter):
        self.data_processor.update_iter(current_iter)

    def params_data_collect(self, name, param_name, pid, data, index):
        grad_name = name + Const.SEP + Const.PARAMS_GRAD + Const.SEP + index
        self.update_api_or_module_name(grad_name)
        if not self.check_scope_and_pid(self.scope, name, pid) and not self.backward_module_names.get(name):
            if self.data_writer.cache_data.get("data"):
                self.data_writer.cache_data.get("data").pop(grad_name, None)
                self.params_grad_record[grad_name] = False
            return
        data_info = self.data_processor.analyze_params(grad_name, param_name, data)
        self.handle_data(grad_name, data_info, flush=self.data_processor.is_terminated)
        self.params_grad_record[grad_name] = False

    def debug_data_collect_forward(self, variable, name_with_count):
        data_info = self.data_processor.analyze_debug_forward(variable, name_with_count)
        name_with_count_category = name_with_count + Const.SEP + Const.DEBUG
        self.data_writer.update_debug({name_with_count_category: data_info})

    def debug_data_collect_backward(self, variable, grad_name_with_count):
        # prepare all None nested data structure
        all_none_data_info = self.data_processor.analyze_element_to_all_none(variable)
        grad_name_with_count_category = grad_name_with_count + Const.SEP + Const.DEBUG
        self.data_writer.update_debug({grad_name_with_count_category: all_none_data_info})

        # register tensor backward hook
        self.data_processor.analyze_debug_backward(variable, grad_name_with_count_category,
                                                   self.data_writer.cache_debug['data'])
