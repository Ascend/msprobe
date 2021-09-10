#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import re
import libcst
from code_visitor import RuleVisitor
from code_visitor import OperatorType
from typing import Optional
from typing import List
from typing import Union
import libcst.helpers as helper
from libcst._flatten_sentinel import FlattenSentinel
from libcst._removal_sentinel import RemovalSentinel
import libcst.matchers as m
import transplant_logger as translog
from libcst.metadata import ParentNodeProvider


class InsertGlobalRule(RuleVisitor):
    def __init__(self, insert_content, import_key_word):
        super(InsertGlobalRule, self).__init__()
        self.insert_content = insert_content
        self.import_key_word = import_key_word
        self.insert_flag = False

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

    def leave_Module(self, original_node: "libcst.Module", updated_node: "libcst.Module") -> "libcst.Module":
        new_body = []
        insert_len = 0
        for content in self.insert_content:
            insert_len += content.count("\n") + 1 if content.count("\n") > 0 else 1
        for i, body_item in enumerate(updated_node.body):
            if self.insert_flag and self.__verify_insert_position(body_item):
                new_body = new_body + [libcst.parse_statement(content, config=original_node.config_for_parsing)
                                       for content in self.insert_content]
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

    def __verify_insert_position(self, body_item):
        if not isinstance(body_item, libcst.SimpleStatementLine):
            return True
        if isinstance(body_item.body[0], (libcst.Import, libcst.ImportFrom, libcst.ImportStar)):
            return False
        if isinstance(body_item.body[0], libcst.Expr) and isinstance(body_item.body[0].value, libcst.SimpleString):
            return False
        return True

    def clean(self):
        self.changes_info = []
        self.insert_flag = False


class FuncNameModifyRule(RuleVisitor):
    def __init__(self, old_name, new_name, replace_module=False):
        super(FuncNameModifyRule, self).__init__()
        self.old_name = old_name
        self.new_name = new_name
        self.replace_module = replace_module

    def leave_Call(
            self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.Call":
        full_func_name = self.get_full_name_for_node(original_node)
        if not self.__compare_func_name(full_func_name):
            return updated_node
        if not self.replace_module:
            func = updated_node.func
            if isinstance(func, libcst.Name):
                updated_node = updated_node.with_changes(func=func.with_changes(value=self.new_name))
                self._record_position(original_node, OperatorType.MODIFY, "change function %s to %s" %
                                      (self.old_name, self.new_name))
            elif isinstance(func, libcst.Attribute):
                func = updated_node.func
                func = func.with_changes(attr=libcst.Name(self.new_name))
                updated_node = updated_node.with_changes(func=func)
                self._record_position(original_node, OperatorType.MODIFY, "change function %s to %s" %
                                      (self.old_name, self.new_name))
            else:
                return updated_node
        else:
            new_func = self.__set_value_of_attr_with_module_replace()
            self._record_position(original_node, OperatorType.MODIFY, "change function %s to %s" %
                                  (self.old_name, self.new_name))
            return updated_node.with_changes(func=new_func)
        return updated_node

    def __compare_func_name(self, full_func_name):
        if not full_func_name:
            return False
        if "." in self.old_name:
            return self.old_name == full_func_name
        return self.old_name == full_func_name.split(".")[-1]

    def __get_value_of_attr(self, body):
        if isinstance(body, libcst.Attribute):
            return body.attr.value
        return body.value

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


class ModuleNameModifyRule(RuleVisitor):
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


class ArgsModifyRule(RuleVisitor):
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
                args[target_idx] = self.__generate_new_arg(args[target_idx])
                self._record_position(original_node, OperatorType.MODIFY,
                                  "change the arg at position %s of function %s to %s" %
                                  (target_idx, self.func_name, self.arg_new))
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
        if not self.arg_keyword:
            return libcst.Arg(libcst.parse_expression(self.arg_new))
        else:
            return origin_arg.with_changes(
                keyword=libcst.Name(self.arg_keyword),
                equal=libcst.AssignEqual(whitespace_before=libcst.SimpleWhitespace(''),
                                         whitespace_after=libcst.SimpleWhitespace('')),
                value=libcst.parse_expression(self.arg_new))

    def __need_modify(self, arg: "libcst.Arg"):
        arg_str = libcst.parse_module("").code_for_node(arg)
        for flag in self.white_list:
            if re.search(flag, arg_str, re.IGNORECASE):
                return False
        return True


class PythonVersionConvertRule(RuleVisitor):
    def leave_Call(
            self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.Call":
        call_name = self.get_full_name_for_node(original_node)
        if call_name and call_name.split(".")[-1] == "hasattr":
            args = list(updated_node.args)
            if args and args[0].value and hasattr(args[0].value, "attr") and hasattr(args[0].value, "value"):
                if isinstance(args[0].value, libcst.Attribute) and args[0].value.attr.value == "module":
                    args[0] = libcst.Arg(libcst.parse_expression(args[0].value.value.value + ".modules"))
                    self._record_position(original_node, OperatorType.MODIFY, "")
                    return updated_node.with_changes(args=args)

        return updated_node


class ReplaceStringRule(RuleVisitor):
    def __init__(self, str_old, str_new, strict=True):
        super(ReplaceStringRule, self).__init__()
        self.str_old = str_old
        self.str_new = str_new
        self.strict = strict

    def leave_SimpleString(
            self, original_node: "libcst.SimpleString", updated_node: "libcst.SimpleString"
    ) -> "libcst.BaseExpression":
        old_value = original_node.value.replace('\"', '').replace('\'', '')
        if (self.strict and self.str_old == old_value) or \
                (not self.strict and self.str_old in original_node.value):
            new_value = original_node.value.replace(self.str_old, self.str_new)
            self._record_position(original_node, OperatorType.MODIFY, "replace string \"%s\" with \"%s\"" %
                                  (self.str_old, self.str_new))
            return updated_node.with_changes(value=new_value)

        return updated_node


class ReplaceAttributeRule(RuleVisitor):
    def __init__(self, old_name, new_name):
        super(ReplaceAttributeRule, self).__init__()
        self.attr_name = old_name
        self.attr_name_new = new_name

    def leave_Attribute(self, original_node: "libcst.Attribute", updated_node: "libcst.Attribute") \
            -> "libcst.Attribute":
        if self.attr_name == original_node.attr.value:
            self._record_position(original_node, OperatorType.MODIFY,
                                  f'replace attribute "{self.attr_name}" with "{self.attr_name_new}"')
            return updated_node.with_changes(attr=libcst.Name(self.attr_name_new))
        return updated_node


class InitProcessGroupRule(InsertGlobalRule):
    def __init__(self):
        insert_content = ["NPU_WORLD_SIZE = int(os.getenv('NPU_WORLD_SIZE'))",
                          "RANK = int(os.getenv('RANK'))",
                          "torch.distributed.init_process_group('hccl', rank=RANK, world_size=NPU_WORLD_SIZE)"]
        super(InitProcessGroupRule, self).__init__(insert_content, "")
        self.insert_flag = False

    def visit_main_file(self, is_main_file):
        self.insert_flag = is_main_file

    def visit_ImportAlias(self, node: "libcst.ImportAlias") -> Optional[bool]:
        return False

    def visit_ImportFrom(self, node: "libcst.ImportFrom") -> Optional[bool]:
        return False


class DataLoaderRule(RuleVisitor):
    '''
    wraper dataset with DistributedSampler.
    '''

    def __init__(self):
        super(DataLoaderRule, self).__init__()
        self.insert_flag = False
        # may find more than one DataLoader Assign, like train_loader/val_loader
        self.dataloader_targets = []
        self.dataloader_target = ''
        self.data_set_target = ''
        self.batch_size_target = ''

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        qualified_name = self.get_full_name_for_node(node)
        if qualified_name == 'torch.utils.data.DataLoader':
            self.insert_flag = True
        return True

    def leave_Assign(
            self, original_node: "libcst.Assign", updated_node: "libcst.Assign"
    ) -> Union[
        "libcst.BaseSmallStatement", FlattenSentinel["libcst.BaseSmallStatement"], RemovalSentinel
    ]:
        if not (self.insert_flag and m.matches(original_node, m.Assign(value=m.Call()))):
            return updated_node
        args = original_node.value.args
        self.dataloader_target = self.get_full_name_for_node(original_node.targets[0].target)
        self.dataloader_targets.append(self.dataloader_target)
        new_value = original_node.value.with_changes(args=self.__adapt_dataloader_args(args))
        self._record_position(original_node, OperatorType.MODIFY, 'adapt args for DataLoader')
        return updated_node.with_changes(value=new_value)

    def __adapt_dataloader_args(self, args):
        arg_change_dict = {'shuffle': 'False', 'pin_memory': 'True', 'drop_last': 'True',
                           'sampler': self.dataloader_target + '_sampler'}
        new_args = []
        for arg in args:
            # train_set arg
            if not arg.keyword or arg.keyword.value == 'dataset':
                self.data_set_target = self.get_code_for_node(arg.value)
                new_args.append(arg)
                continue
            # batch_size arg
            if arg.keyword.value == 'batch_size':
                self.batch_size_target = self.get_code_for_node(arg.value)
                arg = arg.with_changes(value=libcst.Name(self.dataloader_target + '_batch_size'))
            if arg.keyword.value in arg_change_dict.keys():
                arg = arg.with_changes(value=libcst.Name(arg_change_dict.get(arg.keyword.value)))
                arg_change_dict.pop(arg.keyword.value)
            new_args.append(arg)
        added_args = [libcst.Arg(keyword=libcst.Name(k), value=libcst.Name(v)) for k, v in arg_change_dict.items()]
        new_args.extend(added_args)
        return new_args

    def leave_SimpleStatementLine(
            self, original_node: "libcst.SimpleStatementLine", updated_node: "libcst.SimpleStatementLine"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        if not self.insert_flag:
            return updated_node
        self.insert_flag = False
        if not m.matches(original_node.body[0], m.Assign(value=m.Call())):
            return updated_node
        updated_nodes = []
        train_sampler_statement = libcst.parse_statement(
            "%s = torch.utils.data.distributed.DistributedSampler(%s)" % (
                self.dataloader_target + '_sampler', self.data_set_target))
        updated_nodes.append(train_sampler_statement)
        original_position = self.get_metadata(libcst.metadata.PositionProvider, original_node)
        self.changes_info.append([original_position.start.line,
                                  original_position.start.line,
                                  OperatorType.INSERT.name, "init statement of DistributedSampler"])
        if self.batch_size_target:
            batch_size_statement = libcst.parse_statement(
                "%s = int(%s / int(os.getenv('NPU_WORLD_SIZE')))"
                % (self.dataloader_target + '_batch_size', self.batch_size_target))
            updated_nodes.append(batch_size_statement)
            self.changes_info.append([original_position.start.line + 1,
                                      original_position.start.line + 1,
                                      OperatorType.INSERT.name, "add statement of batch_size partition"])
        updated_nodes.append(updated_node)
        return libcst.FlattenSentinel(updated_nodes)

    def leave_For(
            self, original_node: "libcst.For", updated_node: "libcst.For"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        # escape for epoch, batch in xxx
        for_target = self.get_code_for_node(original_node.target).split(',')[0].strip()
        iter_target = self.get_code_for_node(original_node.iter)
        if 'epoch' in for_target or 'epoch' in iter_target:
            set_epoch_statements, insert_len = self.__generate_set_epoch_statement(original_node, for_target)
            if set_epoch_statements:
                body = set_epoch_statements + list(original_node.body.body)
                new_body = libcst.IndentedBlock(body=tuple(body), header=original_node.body.header,
                                                indent=original_node.body.indent, footer=original_node.body.footer)
                original_position = self.get_metadata(libcst.metadata.PositionProvider, original_node.body.body[0])
                self.changes_info.append([original_position.start.line,
                                          original_position.start.line + insert_len - 1,
                                          OperatorType.INSERT.name, "add statement of sampler.set_epoch"])
                return updated_node.with_changes(body=new_body)
            else:
                translog.warning("failed to set_epoch for sampler and you need to set it yourself")
        return updated_node

    def __generate_set_epoch_statement(self, node, epoch_target):
        scope = self.get_metadata(libcst.metadata.ScopeProvider, node)
        # train_loader assign in scope
        set_epoch_statements = [libcst.parse_statement("%s.sampler.set_epoch(%s)" % (target, epoch_target))
                                for target in self.dataloader_targets if target in scope]
        if set_epoch_statements:
            return set_epoch_statements, len(set_epoch_statements)
        # variable name contains loader
        maybe_dataloader_variables = [assign.name for assign in scope.assignments if 'loader' in assign.name]
        maybe_set_epoch_statements = [libcst.parse_statement(
            'if isinstance(%s, torch.utils.data.DataLoader):\n    %s.sampler.set_epoch(%s)' % (
                dataloader_variable, dataloader_variable, epoch_target)) for
            dataloader_variable in maybe_dataloader_variables]
        return maybe_set_epoch_statements, len(maybe_set_epoch_statements) * 2

    def clean(self):
        self.insert_flag = False
        self.changes_info = []
        self.dataloader_targets = []
        self.dataloader_target = ''
        self.data_set_target = ''
        self.batch_size_target = ''


class DistributedDataParallelRule(RuleVisitor):
    '''
    wrapper model with DistributedDataParallel.
    '''

    def __init__(self, model):
        super(DistributedDataParallelRule, self).__init__()
        self.insert_flag = False
        self.model_target = model

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        target = node.targets[0].target
        if hasattr(target, 'elements'):
            target_pure_full_names = [self.get_full_name_for_node(element.value) for element in target.elements]
            if self.model_target in target_pure_full_names:
                self.insert_flag = True
        else:
            target_full_name = self.get_full_name_for_node(target)
            if target_full_name == self.model_target:
                # escape "model = None"
                value = self.get_code_for_node(node.value)
                if not value or value == 'None':
                    return True
                # escape "model = model.npu()", "model = model.to()"
                if m.matches(node.value, m.Call()) and self.get_full_name_for_node(
                        node.value) in [self.model_target + '.npu', self.model_target + '.to']:
                    return True
                self.insert_flag = True
        return True

    def leave_SimpleStatementLine(
            self, original_node: "libcst.SimpleStatementLine", updated_node: "libcst.SimpleStatementLine"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        if not self.insert_flag:
            return updated_node
        self.insert_flag = False
        to_device_statement = libcst.parse_statement(
            "%s = %s.to(f'npu:{NPU_CALCULATE_DEVICE}')" % (self.model_target, self.model_target))
        ddp_statement = libcst.parse_statement(
            'if not isinstance(%s, torch.nn.parallel.DistributedDataParallel):\n'
            '    %s = torch.nn.parallel.DistributedDataParallel(%s, device_ids=[NPU_CALCULATE_DEVICE], '
            'broadcast_buffers=False)' % (self.model_target, self.model_target, self.model_target))
        original_position = self.get_metadata(libcst.metadata.PositionProvider, original_node)
        self.changes_info.append([original_position.start.line + 1,
                                  original_position.start.line + 3,
                                  OperatorType.INSERT.name, "init statement of DistributedDataParallel"])
        return libcst.FlattenSentinel([updated_node, to_device_statement, ddp_statement])

    def clean(self):
        self.insert_flag = False
        self.changes_info = []
