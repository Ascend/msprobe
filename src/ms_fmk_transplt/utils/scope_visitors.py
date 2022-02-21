#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from typing import Optional

import libcst
from libcst import matchers as m

from ms_fmk_transplt.common_rules import RuleVisitor


class ScaleScopeVisitor(RuleVisitor):

    def __init__(self):
        super(ScaleScopeVisitor, self).__init__()
        self.loss_name = ''
        self.optimizer_name = ''
        self.scaler_name = ''
        self.found_scaler = False
        self.scale_dict = {}
        self.step_dict = {}

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        super().visit_Assign(node)
        if not m.matches(node.value, m.Call()):
            return True
        qualified_name = self.get_full_name_for_node(node.value)
        if qualified_name == "torch.cuda.amp.GradScaler":
            target = node.targets[0].target
            self.scaler_name = self.get_full_name_for_node(target)
            self.found_scaler = True
            self.optimizer_name = self.step_dict.get(self.scaler_name, '')
            self.loss_name = self.scale_dict.get(self.scaler_name, '')
        return True

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        if len(node.args) == 0 or not hasattr(node.args[0].value, 'value'):
            return True
        qualified_name = self.get_full_name_for_node(node)
        if len(qualified_name.split('.')) != 2:
            return True
        value = node.args[0].value.value
        key_name, func_name = qualified_name.split('.')
        if func_name == 'scale':
            self.scale_dict[key_name] = value
            if self.found_scaler:
                self.loss_name = self.scale_dict.get(self.scaler_name)
        if func_name == 'step':
            self.step_dict[key_name] = value
            if self.found_scaler:
                self.optimizer_name = self.step_dict.get(self.scaler_name)
        return True
