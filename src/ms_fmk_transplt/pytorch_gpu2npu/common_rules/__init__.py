#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from pytorch_gpu2npu.common_rules.code_visitor import ApiVisitor, OperatorType, RuleVisitor
from pytorch_gpu2npu.common_rules.common_rule import ArgsModifyRule, BaseInsertGlobalRule, FuncNameModifyRule, \
    InsertGlobalRule, InsertMainFileRule, ModuleNameModifyRule, PythonVersionConvertRule, ReplaceAttributeRule, \
    ReplaceStringRule
