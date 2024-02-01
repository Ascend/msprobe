#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import re
from typing import List, Optional

import libcst
import libcst.helpers as helper
import libcst.matchers as m
from libcst.metadata import ParentNodeProvider

from .base_rule import BaseRule, OperatorType


class BaseInsertGlobalRule(BaseRule):
    def __init__(self, insert_content):
        super(BaseInsertGlobalRule, self).__init__()
        self.insert_content = insert_content
        self.insert_flag = False

    @staticmethod
    def __verify_insert_position(body_item):
        if not isinstance(body_item, libcst.SimpleStatementLine):
            return True
        if isinstance(body_item.body[0], (libcst.Import, libcst.ImportFrom, libcst.ImportStar)):
            return False
        if isinstance(body_item.body[0], libcst.Expr) and isinstance(body_item.body[0].value, libcst.SimpleString):
            return False
        return True

    def leave_Module(self, original_node: libcst.Module, updated_node: libcst.Module) -> libcst.Module:
        new_body = []
        insert_len = 0
        for content in self.insert_content:
            insert_len += content.count("\n") + 1 if content.count("\n") > 0 else 1
        for i, body_item in enumerate(updated_node.body):
            if self.insert_flag and self.__verify_insert_position(body_item):
                for content in self.insert_content:
                    new_body.append(libcst.parse_statement(content, config=original_node.config_for_parsing))
                original_index = min(i, len(original_node.body) - 1)
                original_position = self.get_metadata(libcst.metadata.PositionProvider,
                                                      original_node.body[original_index])
                self.changes_info.append([original_position.start.line,
                                          original_position.start.line + insert_len - 1,
                                          OperatorType.INSERT.name, str(self.insert_content)])
                self.insert_flag = False
            new_body.append(body_item)

        updated_node = updated_node.with_changes(
            body=tuple(new_body),
        )
        return updated_node

    def clean(self):
        super().clean()
        self.insert_flag = False


class InsertGlobalRule(BaseInsertGlobalRule):
    def __init__(self, insert_content, import_key_word=''):
        super(InsertGlobalRule, self).__init__(insert_content)
        self.import_key_word = import_key_word

    def visit_ImportAlias(self, node: "libcst.ImportAlias") -> Optional[bool]:
        # if import_key_word isn't setted, default no limit
        if not self.insert_flag and not self.import_key_word:
            self.insert_flag = True
        if not self.insert_flag and self.import_key_word in node.evaluated_name:
            self.insert_flag = True
        return True

    def visit_ImportFrom(self, node: "libcst.ImportFrom") -> Optional[bool]:
        # if import_key_word isn't setted, default no limit
        if not self.insert_flag and not self.import_key_word:
            self.insert_flag = True
        full_name = helper.get_full_name_for_node(node.module)
        if not self.insert_flag and full_name and self.import_key_word in full_name:
            self.insert_flag = True
        return True


class InsertMainFileRule(BaseInsertGlobalRule):
    def __init__(self, insert_content):
        super(InsertMainFileRule, self).__init__(insert_content)

    def visit_main_file(self, is_main_file):
        self.insert_flag = is_main_file


class FuncNameModifyRule(BaseRule):
    def __init__(self, old_name, new_name, replace_module=False):
        super(FuncNameModifyRule, self).__init__()
        self.old_name = old_name
        self.new_name = new_name
        self.replace_module = replace_module

    @staticmethod
    def __get_value_of_attr(body):
        if isinstance(body, libcst.Attribute):
            return body.attr.value
        return body.value

    def leave_Call(
            self, original_node: libcst.Call, updated_node: libcst.Call
    ) -> libcst.Call:
        full_func_name = self.get_full_name_for_node(original_node)
        if not (self.__compare_func_name(full_func_name) or self.__compare_func_last_name(original_node)):
            return updated_node
        desc = "change function %s to %s"
        if not self.replace_module:
            func = updated_node.func
            if isinstance(func, libcst.Name):
                updated_node = updated_node.with_changes(func=func.with_changes(value=self.new_name))
                self._record_position(original_node, OperatorType.MODIFY, desc % (self.old_name, self.new_name))
            elif isinstance(func, libcst.Attribute):
                func = updated_node.func
                func = func.with_changes(attr=libcst.Name(self.new_name))
                updated_node = updated_node.with_changes(func=func)
                self._record_position(original_node, OperatorType.MODIFY, desc % (self.old_name, self.new_name))
            else:
                return updated_node
        else:
            new_func = self.__set_value_of_attr_with_module_replace()
            self._record_position(original_node, OperatorType.MODIFY, desc % (self.old_name, self.new_name))
            return updated_node.with_changes(func=new_func)

        return updated_node

    def leave_IfExp(
            self, original_node: "libcst.IfExp", updated_node: "libcst.IfExp"
    ) -> "libcst.BaseExpression":
        parent = self.get_metadata(ParentNodeProvider, original_node)
        if not isinstance(parent, libcst.Call):
            return updated_node

        body = updated_node.body
        if self.__compare_func_name(self.__get_value_of_attr(body)):
            body = self.__set_value_of_attr(body)
            updated_node = updated_node.with_changes(body=body)
            self._record_position(original_node, OperatorType.MODIFY, "change function %s to %s" %
                                  (self.old_name, self.new_name))

        updated_node = updated_node.with_changes(orelse=self.__reverse_orelse(updated_node.orelse, original_node))
        return updated_node

    def __compare_func_last_name(self, node):
        if not (m.matches(node, m.Call()) and m.matches(node.func, m.Attribute())):
            return False
        attr_last_name = self.get_full_name_for_node(node.func.attr)
        if "." not in self.old_name:
            return self.old_name == attr_last_name
        return False

    def __compare_func_name(self, full_func_name):
        if not full_func_name:
            return False
        if "." in self.old_name:
            return self.old_name == full_func_name
        return self.old_name == full_func_name.split(".")[-1]

    def __set_value_of_attr_with_no_module_replace(self, body):
        if isinstance(body, libcst.Attribute):
            attr = body.attr.with_changes(value=self.new_name)
            body = body.with_changes(attr=attr)

        if isinstance(body, libcst.Name):
            body = body.with_changes(value=self.new_name)

        return body

    def __set_value_of_attr_with_module_replace(self):
        new_name_list = self.new_name.split(".")
        new_func = libcst.Name(new_name_list[-1])
        for i in range(len(new_name_list) - 2, -1, -1):
            new_func = libcst.Attribute(value=libcst.Name(new_name_list[i]), attr=new_func)
        return new_func

    def __set_value_of_attr(self, body):
        if not self.replace_module:
            body = self.__set_value_of_attr_with_no_module_replace(body)
        else:
            body = self.__set_value_of_attr_with_module_replace()
        return body

    def __reverse_orelse(self, orelse, original_node):
        if isinstance(orelse, libcst.Name) or isinstance(orelse, libcst.Attribute):
            if self.__compare_func_name(self.__get_value_of_attr(orelse)):
                return self.__set_value_of_attr(orelse)

        if isinstance(orelse, libcst.IfExp):
            if self.__compare_func_name(self.__get_value_of_attr(orelse.body)):
                orelse = orelse.with_changes(body=self.__set_value_of_attr(orelse.body))
                self._record_position(original_node, OperatorType.MODIFY, "change function %s to %s" %
                                      (self.old_name, self.new_name))
            orelse = orelse.with_changes(orelse=self.__reverse_orelse(orelse.orelse, original_node))
        return orelse


class ModuleNameModifyRule(BaseRule):
    def __init__(self, old_name, new_name, parent_module):
        super(ModuleNameModifyRule, self).__init__()
        self.module_name = old_name
        self.module_name_new = new_name
        self.parent_module = parent_module

    def leave_Attribute(
            self, original_node: "libcst.Attribute", updated_node: "libcst.Attribute"
    ) -> "libcst.BaseExpression":
        parent_full_name = self.get_full_name_for_node(original_node.value)
        if self.module_name == original_node.attr.value and parent_full_name == self.parent_module:
            self._record_position(original_node, OperatorType.MODIFY, "change module %s to %s" %
                                  (self.module_name, self.module_name_new))
            return updated_node.with_changes(attr=libcst.Name(self.module_name_new))
        return updated_node


class ArgsModifyRule(BaseRule):
    def __init__(self, func_name, arg_new, arg_idx=-1, arg_keyword=None, white_list=None):
        super(ArgsModifyRule, self).__init__()
        self.func_name = func_name
        self.arg_idx = arg_idx
        self.arg_keyword = arg_keyword
        self.arg_new = arg_new
        self.white_list = white_list if white_list else []

    def leave_Call(
            self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.BaseExpression":
        qualified_name = self.get_full_name_for_node(original_node)

        if qualified_name != self.func_name and \
                not (isinstance(original_node.func, libcst.Attribute) and
                     original_node.func.attr.value == self.func_name):
            return updated_node

        args = list(updated_node.args)
        target_idx = self.arg_idx
        if target_idx < 0:
            target_idx = self.__get_target_arg_idx(args)
        if 0 <= target_idx < len(args) and self.__need_modify(args[target_idx]):
            if not self.arg_new:
                args.pop(target_idx)
                self._record_position(original_node, OperatorType.DELETE,
                                      "delete the arg at position %s of function %s" %
                                      (target_idx, self.func_name))
            else:
                args[target_idx], arg_new = self.__generate_new_arg(args[target_idx])
                self._record_position(original_node, OperatorType.MODIFY,
                                      "change the arg at position %s of function %s to %s" %
                                      (target_idx, self.func_name, arg_new))
            return updated_node.with_changes(args=args)

        return updated_node

    def __get_target_arg_idx(self, args: List["libcst.Arg"]):
        target = -1
        for idx, arg in enumerate(args):
            if arg.keyword is not None and arg.keyword.value == self.arg_keyword:
                target = idx
                break
        return target

    def __generate_new_arg(self, origin_arg: "libcst.Arg"):
        arg_new = self.arg_new
        if 'replace_device_int' in self.arg_new:
            old_arg = libcst.parse_module("").code_for_node(origin_arg.value)
            arg_new = self.arg_new.replace('replace_device_int', str(old_arg))
        if not self.arg_keyword:
            return libcst.Arg(libcst.parse_expression(arg_new)), arg_new
        else:
            return origin_arg.with_changes(
                keyword=libcst.Name(self.arg_keyword),
                equal=libcst.AssignEqual(whitespace_before=libcst.SimpleWhitespace(''),
                                         whitespace_after=libcst.SimpleWhitespace('')),
                value=libcst.parse_expression(arg_new)), arg_new

    def __need_modify(self, arg: "libcst.Arg"):
        arg_str = libcst.parse_module("").code_for_node(arg)
        for flag in self.white_list:
            if re.search(flag, arg_str, re.IGNORECASE):
                return False
        return True


class PythonVersionConvertRule(BaseRule):
    def leave_Call(
            self, original_node: libcst.Call, updated_node: libcst.Call
    ) -> libcst.Call:
        call_name = self.get_full_name_for_node(original_node)
        if call_name and call_name.split(".")[-1] == "hasattr":
            args = list(updated_node.args)
            has_value = args and args[0].value
            if has_value and hasattr(args[0].value, "attr") and hasattr(args[0].value, "value"):
                if isinstance(args[0].value, libcst.Attribute) and args[0].value.attr.value == "module":
                    args[0] = libcst.Arg(libcst.parse_expression(args[0].value.value.value + ".modules"))
                    self._record_position(original_node, OperatorType.MODIFY, "")
                    return updated_node.with_changes(args=args)

        return updated_node


class ReplaceStringRule(BaseRule):
    def __init__(self, str_old, str_new, strict=True):
        super(ReplaceStringRule, self).__init__()
        self.str_old = str_old
        self.str_new = str_new
        self.strict = strict

    def leave_SimpleString(
            self, original_node: "libcst.SimpleString", updated_node: "libcst.SimpleString"
    ) -> "libcst.BaseExpression":
        old_value = original_node.value.replace('\"', '').replace('\'', '')
        strict_judgment = self.strict and self.str_old == old_value
        not_strict_judgment = not self.strict and self.str_old in original_node.value
        if strict_judgment or not_strict_judgment:
            new_value = original_node.value.replace(self.str_old, self.str_new)
            self._record_position(original_node, OperatorType.MODIFY, "replace string \"%s\" with \"%s\"" %
                                  (self.str_old, self.str_new))
            return updated_node.with_changes(value=new_value)

        return updated_node

    def leave_FormattedStringText(
            self, original_node: "libcst.FormattedStringText", updated_node: "libcst.FormattedStringText"
    ) -> "libcst.BaseExpression":
        if not self.strict and self.str_old in original_node.value:
            new_value = original_node.value.replace(self.str_old, self.str_new)
            self._record_position(original_node, OperatorType.MODIFY, "replace string \"%s\" with \"%s\"" %
                                  (self.str_old, self.str_new))
            return updated_node.with_changes(value=new_value)

        return updated_node


class ReplaceAttributeRule(BaseRule):
    def __init__(self, old_name, new_name):
        super(ReplaceAttributeRule, self).__init__()
        self.attr_name = old_name
        self.attr_name_new = new_name

    def leave_Attribute(self, original_node: libcst.Attribute, updated_node: libcst.Attribute) \
            -> libcst.Attribute:
        full_name = self.get_full_name_for_node(original_node)
        new_name_list = self.attr_name_new.split(".")
        if full_name == self.attr_name and len(new_name_list) > 1:
            new_attribute = libcst.Attribute(value=libcst.Name(new_name_list[0]), attr=libcst.Name(new_name_list[1]))
            if len(new_name_list) > 2:
                for i in range(2, len(new_name_list)):
                    new_attribute = libcst.Attribute(value=new_attribute, attr=libcst.Name(new_name_list[i]))
            updated_node = new_attribute
            self._record_position(original_node, OperatorType.MODIFY,
                                  f'replace attribute "{self.attr_name}" with "{self.attr_name_new}"')
        elif self.attr_name == original_node.attr.value:
            self._record_position(original_node, OperatorType.MODIFY,
                                  f'replace attribute "{self.attr_name}" with "{self.attr_name_new}"')
            return updated_node.with_changes(attr=libcst.Name(self.attr_name_new))
        return updated_node


class RemoveImportRule(BaseRule):
    def leave_Import(
            self, original_node: libcst.Import, updated_node: libcst.Import
    ):
        names = updated_node.names
        if names and names[0].name.value == 'amp_C':
            self._record_position(original_node, OperatorType.DELETE,
                                  'import amp_C has been removed')
            return libcst.RemovalSentinel.REMOVE
        return updated_node
