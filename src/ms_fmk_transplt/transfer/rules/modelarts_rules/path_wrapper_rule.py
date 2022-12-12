#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import libcst
import libcst.matchers as m

from utils import transplant_logger as logger
from ..common_rules.base_rule import BaseRule
from ..common_rules.common_rule import BaseInsertGlobalRule


class ModelArtsPathWrapperRule(BaseRule):
    def __init__(self, path_handler_api_dict, add_import_rule: BaseInsertGlobalRule):
        super(ModelArtsPathWrapperRule, self).__init__()
        self.path_handler_api_dict = path_handler_api_dict
        self.add_import_rule = add_import_rule

    @staticmethod
    def _get_arg_idx_to_wrap(api_name, args, mapped_info):
        args_to_wrap = []
        if 'arg_no' in mapped_info:
            args_to_wrap.extend(ModelArtsPathWrapperRule._get_positional_arg_indices(api_name, args, mapped_info))

        if 'arg_keyword' in mapped_info:
            args_to_wrap.extend(ModelArtsPathWrapperRule._get_keyword_arg_indices(api_name, args, mapped_info))

        return list(set(args_to_wrap))

    @staticmethod
    def _get_positional_arg_indices(api_name, args, mapped_info):
        arg_indices = []
        arg_no = mapped_info.get('arg_no')
        if isinstance(arg_no, int):
            arg_no = [arg_no]
        for idx in arg_no:
            if 0 <= idx < len(args):
                logger.info(f'[ModelArts] Wrap argument {idx} of func "{api_name}" for path mapping.')
                arg_indices.append(idx)

        return arg_indices

    @staticmethod
    def _get_keyword_arg_indices(api_name, args, mapped_info):
        arg_indices = []
        arg_keyword = mapped_info.get('arg_keyword')
        if isinstance(arg_keyword, str):
            arg_keyword = [arg_keyword]

        for idx, arg in enumerate(args):
            if m.matches(arg.keyword, m.Name()) and arg.keyword.value in arg_keyword:
                logger.info(f'[ModelArts] Wrap argument with keyword "{arg.keyword.value}"'
                            f' of func "{api_name}" for path mapping.')
                arg_indices.append(idx)

        return arg_indices

    def leave_Call(
            self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.BaseExpression":
        full_name = self.get_full_name_for_node(original_node)
        if full_name is None:
            return updated_node
        api_name, api_info = self._get_mapped_info(full_name)
        if api_info is None:
            return updated_node
        args = updated_node.args
        args_to_wrap = self._get_arg_idx_to_wrap(api_name, args, api_info)
        if not args_to_wrap:
            return updated_node

        self.add_import_rule.insert_flag = True
        new_args = list(args)
        for arg_idx in args_to_wrap:
            arg_code = libcst.parse_module('').code_for_node(args[arg_idx].value)
            new_arg = args[arg_idx].with_changes(
                value=libcst.parse_expression(f'ModelArtsPathManager().get_path({arg_code})'))
            new_args[arg_idx] = new_arg
            updated_node = updated_node.with_changes(args=tuple(new_args))

        return updated_node

    def clean(self):
        super().clean()
        self.add_import_rule.clean()

    def _get_mapped_info(self, full_name):
        mapped_info = None
        api_name = full_name
        if full_name in self.path_handler_api_dict:
            mapped_info = self.path_handler_api_dict.get(full_name)
        else:
            method_name = full_name.split('.')[-1]
            if method_name in self.path_handler_api_dict and \
                    self.path_handler_api_dict.get(method_name).get('is_instance_api'):
                mapped_info = self.path_handler_api_dict.get(method_name)
                api_name = mapped_info.get('class') + '.' + method_name
        return api_name, mapped_info
