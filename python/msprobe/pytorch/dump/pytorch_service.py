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

from msprobe.core.common.utils import Const
from msprobe.core.dump.service import BaseService
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import get_rank_if_initialized
from msprobe.pytorch.dump.module_dump.module_processor import ModuleProcessor
from msprobe.pytorch.dump.api_dump.api_register import (
    get_api_register,
    ApiTemplate,
    redirect_wait,
    reset_dist_collect_func,
)
from msprobe.pytorch.dump.api_dump.hook_module import HOOKModule
from msprobe.pytorch.dump.api_dump.pt_hook_manager import PytorchHookManager
from msprobe.pytorch.dump.api_dump.register_optimizer_hook import register_optimizer_hook
from msprobe.pytorch.dump.api_dump.script_wrapper import wrap_script_func, preprocess_func
from msprobe.pytorch.dump.api_dump import script_wrapper


class PytorchService(BaseService):
    @property
    def _get_framework_type(self):
        return Const.PT_FRAMEWORK

    @staticmethod
    def _get_current_rank():
        return get_rank_if_initialized()

    def apply_runtime_config(self, config):
        old_need_api_hook = self._is_need_api_hook
        old_dump_enabled = self._is_dump_enabled
        super().apply_runtime_config(config)
        self._refresh_module_processor()
        self._sync_api_hook_state(old_need_api_hook, old_dump_enabled)

    def reset_status(self):
        self._reset_status()

    def _init_specific_components(self):
        self.logger = logger
        self.api_register = get_api_register()
        self.module_processor = None
        self._refresh_module_processor()
        self.hook_manager = PytorchHookManager(self.data_collector, self.config)
        self.api_template = ApiTemplate

    def _refresh_module_processor(self):
        self.module_processor = ModuleProcessor(self.data_collector.scope)  # pylint: disable=attribute-defined-outside-init

    def _register_hook(self):
        if self._is_mix_level:
            register_optimizer_hook(self.data_collector)

    def _register_api_hook(self):
        if not self._is_dump_enabled:
            if self.api_register is not None and getattr(self.api_register, "all_api_registered", False):
                self.api_register.restore_all_api()
            return
        preprocess_func()
        super()._register_api_hook()
        script_wrapper.set_current_service(self)
        wrap_script_func()
        redirect_wait()

    def _sync_api_hook_state(self, old_need_api_hook, old_dump_enabled):
        if self.api_register is None:
            return
        api_registered = getattr(self.api_register, "all_api_registered", False)
        if not self._is_dump_enabled:
            if api_registered:
                self.api_register.restore_all_api()
            return
        if not self._is_need_api_hook:
            if api_registered:
                self.api_register.restore_all_api()
            return
        if old_need_api_hook and old_dump_enabled and api_registered:
            return
        self._register_api_hook()

    def _register_module_hook(self):
        ModuleProcessor.enable_module_dump = True
        self.module_processor.register_module_hook(self.model, self.build_hook)
        self.logger.info_on_rank_0(f"The module {self.config.task} hook function is successfully mounted to the model.")

    def _reset_status(self):
        super()._reset_status()
        ModuleProcessor.reset_module_stats()
        HOOKModule.reset_module_stats()
        reset_dist_collect_func()
