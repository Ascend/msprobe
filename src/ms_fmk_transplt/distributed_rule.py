#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from typing import Optional, Union

import libcst
from libcst import FlattenSentinel, RemovalSentinel, matchers as m

from code_visitor import RuleVisitor, OperatorType
from rule import InsertGlobalRule
from scope_visitors import ScaleScopeVisitor
import transplant_logger as translog


class InitProcessGroupRule(InsertGlobalRule):
    def __init__(self):
        insert_content = ["import torch.npu",
                          "if torch.npu.current_device() != NPU_CALCULATE_DEVICE:\n"
                          "    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')",
                          "NPU_WORLD_SIZE = int(os.getenv('NPU_WORLD_SIZE'))",
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
    """
    wraper dataset with DistributedSampler.
    """

    def __init__(self):
        super(DataLoaderRule, self).__init__()
        self.insert_flag = False
        # may find more than one DataLoader Assign, like train_loader/val_loader
        self.dataloader_targets = []
        self.dataloader_target = ''
        self.data_set_target = ''

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        qualified_name = self.get_full_name_for_node(node)
        if qualified_name in ('torch.utils.data.DataLoader', 'torch.utils.data.dataloader.DataLoader'):
            self.insert_flag = True
        return True

    def leave_Call(
        self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.BaseExpression":
        if not self.insert_flag:
            return updated_node
        return updated_node.with_changes(args=self.__adapt_dataloader_args(updated_node.args))

    def leave_Assign(
            self, original_node: "libcst.Assign", updated_node: "libcst.Assign"
    ) -> Union[
        "libcst.BaseSmallStatement", FlattenSentinel["libcst.BaseSmallStatement"], RemovalSentinel
    ]:
        if not (self.insert_flag and m.matches(original_node, m.Assign(value=m.Call()))):
            return updated_node
        args = updated_node.value.args
        self.dataloader_target = self.get_full_name_for_node(original_node.targets[0].target,
                                                             with_variable_replace=False)
        self.dataloader_targets.append(self.dataloader_target)
        new_value = updated_node.value.with_changes(args=self.__adapt_dataloader_args(args))
        self._record_position(original_node, OperatorType.MODIFY, 'adapt args for DataLoader')
        return updated_node.with_changes(value=new_value)

    def __adapt_dataloader_args(self, args):
        arg_change_dict = {'shuffle': 'False', 'pin_memory': 'True', 'drop_last': 'True',
                           'sampler': self.dataloader_target + '_sampler'}
        new_args = []
        for arg in args:
            # train_set arg
            if not arg.keyword or arg.keyword.value == 'dataset':
                # escape **params
                if not arg.star.startswith('*'):
                    self.data_set_target = self.get_code_for_node(arg.value)
                new_args.append(arg)
                continue
            if arg.keyword.value in arg_change_dict.keys():
                arg = arg.with_changes(value=libcst.parse_expression(arg_change_dict.get(arg.keyword.value)))
                arg_change_dict.pop(arg.keyword.value)
            new_args.append(arg)
        added_args = []
        for k, v in arg_change_dict.items():
            added_args.append(libcst.Arg(keyword=libcst.Name(k), value=libcst.Name(v)))
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
        train_sampler_statement = libcst.parse_statement(
            "%s = torch.utils.data.distributed.DistributedSampler(%s)" % (
                self.dataloader_target + '_sampler', self.data_set_target))
        original_position = self.get_metadata(libcst.metadata.PositionProvider, original_node)
        self.changes_info.append([original_position.start.line,
                                  original_position.start.line,
                                  OperatorType.INSERT.name, "init statement of DistributedSampler"])
        return libcst.FlattenSentinel([train_sampler_statement, updated_node])

    def leave_For(
            self, original_node: "libcst.For", updated_node: "libcst.For"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        # escape for epoch, batch in xxx
        for_target = self.get_code_for_node(original_node.target).split(',')[0].strip()
        iter_target = self.get_code_for_node(original_node.iter)
        if 'epoch' in for_target or 'epoch' in iter_target:
            set_epoch_statements, insert_len = self.__generate_set_epoch_statement(original_node, for_target)
            if set_epoch_statements:
                body = set_epoch_statements + list(updated_node.body.body)
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
        set_epoch_statements = []
        for target in self.dataloader_targets:
            if target in scope:
                set_epoch_statements.append(
                    libcst.parse_statement("%s.sampler.set_epoch(%s)" % (target, epoch_target)))
        if set_epoch_statements:
            return set_epoch_statements, len(set_epoch_statements)
        # variable name contains loader
        maybe_dataloader_variables = []
        for assign in scope.assignments:
            if 'loader' in assign.name:
                maybe_dataloader_variables.append(assign.name)
        maybe_set_epoch_statements = []
        for dataloader_variable in maybe_dataloader_variables:
            maybe_set_epoch_statements.append(libcst.parse_statement(
                'if isinstance(%s, torch.utils.data.DataLoader):\n    %s.sampler.set_epoch(%s)' % (
                    dataloader_variable, dataloader_variable, epoch_target)))
        return maybe_set_epoch_statements, len(maybe_set_epoch_statements) * 2

    def clean(self):
        super().clean()
        self.insert_flag = False
        self.dataloader_targets = []
        self.dataloader_target = ''
        self.data_set_target = ''



class DistributedDataParallelRule(RuleVisitor):
    '''
    wrapper model with DistributedDataParallel.
    '''

    def __init__(self, model, amp_flag):
        super(DistributedDataParallelRule, self).__init__()
        self.insert_flag = False
        self.model_target = model
        self.amp_flag = amp_flag
        self.optimizer_name = ''
        self.has_apex_initialize = False
        self.add_after_if = False
        self.already_add_ddp = False

    def visit_Module(self, node: "libcst.Module") -> Optional[bool]:
        visitor = ScaleScopeVisitor()
        wrapper = libcst.metadata.MetadataWrapper(node)
        wrapper.visit(visitor)
        self.optimizer_name = visitor.optimizer_name
        self.__check_apex_initialize(node)

    def __check_apex_initialize(self, node):
        # check "model, opt = amp.initialize(model, opt)"
        if m.findall(node, m.Assign(value=m.Call() & m.MatchIfTrue(
                lambda call_node: self.get_full_name_for_node(call_node) == 'apex.amp.initialize'))):
            self.has_apex_initialize = True

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        if self.add_after_if:
            return True
        target = node.targets[0].target
        if hasattr(target, 'elements'):
            target_pure_full_names = []
            for element in target.elements:
                target_pure_full_names.append(self.get_full_name_for_node(element.value))
            if self.model_target in target_pure_full_names and self.__need_insert_ddp(node.value):
                self.insert_flag = True
        else:
            target_full_name = self.get_full_name_for_node(target)
            if target_full_name == self.model_target:
                if not self.__need_insert_ddp(node.value):
                    return True
                self.insert_flag = True
        return True

    def __need_insert_ddp(self, value):
        if self.has_apex_initialize:
            return self.__is_amp_initialize(value)
        # escape "model = None"
        node_value = self.get_code_for_node(value)
        if not node_value or node_value == 'None':
            return False
        if not m.matches(value, m.Call()):
            return True
        full_name = self.get_full_name_for_node(value)
        # 1. escape model.cuda(), model.npu(), model.to()
        escape_funcs = [self.model_target + '.cuda', self.model_target + '.npu', self.model_target + '.to']
        if full_name in escape_funcs:
            return False
        for arg in value.args:
            if self.get_code_for_node(arg.value) == self.model_target:
                return False
        return True

    def __is_amp_initialize(self, value):
        return m.matches(value, m.Call()) and self.get_full_name_for_node(value) == 'apex.amp.initialize'

    def visit_If(self, node: "libcst.If") -> Optional[bool]:
        if self.has_apex_initialize and m.findall(node, m.Assign(value=m.Call() & m.MatchIfTrue(
                lambda call_node: self.get_full_name_for_node(call_node) == 'apex.amp.initialize'))):
            scope = self.get_metadata(libcst.metadata.ScopeProvider, node)
            # consider self.model
            if self.model_target in scope:
                self.add_after_if = True
        return True

    def leave_Call(
        self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.BaseExpression":
        if not self.already_add_ddp:
            return updated_node
        need_add_module_func = [self.model_target + '.load_state_dict', self.model_target + '.load_from']
        full_name = self.get_full_name_for_node(original_node)
        if full_name not in need_add_module_func:
            return updated_node
        names = full_name.split('.')
        names[-1] = 'module.' + names[-1]
        return updated_node.with_changes(func=libcst.parse_expression('.'.join(names)))

    def leave_Assign(
            self, original_node: "libcst.Assign", updated_node: "libcst.Assign"
    ) -> Union[
        "libcst.BaseSmallStatement", FlattenSentinel["libcst.BaseSmallStatement"], RemovalSentinel
    ]:
        # for distributed rule, delete torch.nn.DataParallel(model)
        if m.matches(original_node.value, m.Call()) and self.get_full_name_for_node(
                original_node.value) == 'torch.nn.DataParallel':
            return libcst.RemovalSentinel.REMOVE
        return updated_node

    def leave_SimpleStatementLine(
            self, original_node: "libcst.SimpleStatementLine", updated_node: "libcst.SimpleStatementLine"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        if not self.insert_flag:
            return updated_node
        self.insert_flag = False
        if self.amp_flag and self.optimizer_name:
            return updated_node
        return self.__add_ddp_statement(original_node, updated_node)

    def leave_If(
        self, original_node: "libcst.If", updated_node: "libcst.If"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        if not self.add_after_if:
            return updated_node
        self.add_after_if = False
        return self.__add_ddp_statement(original_node, updated_node)

    def __add_ddp_statement(self, original_node, updated_node):
        to_device_statement = libcst.parse_statement(
            "%s = %s.npu()" % (self.model_target, self.model_target))
        ddp_statement = libcst.parse_statement(
            'if not isinstance(%s, torch.nn.parallel.DistributedDataParallel):\n'
            '    %s = torch.nn.parallel.DistributedDataParallel(%s, device_ids=[NPU_CALCULATE_DEVICE], '
            'broadcast_buffers=False)' % (self.model_target, self.model_target, self.model_target))
        original_position = self.get_metadata(libcst.metadata.PositionProvider, original_node)
        self.changes_info.append([original_position.start.line + 1,
                                  original_position.start.line + 3,
                                  OperatorType.INSERT.name, "init statement of DistributedDataParallel"])
        self.already_add_ddp = True
        return libcst.FlattenSentinel([updated_node, to_device_statement, ddp_statement])

    def clean(self):
        super().clean()
        self.insert_flag = False
        self.optimizer_name = ''
        self.has_apex_initialize = False
        self.add_after_if = False
        self.already_add_ddp = False
