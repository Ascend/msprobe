#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from typing import Optional, Union

import libcst
from libcst import FlattenSentinel, RemovalSentinel, matchers as m

from utils import transplant_logger as translog
from utils import trans_utils as utils
from ..common_rules.base_rule import BaseRule, OperatorType


class ScaleScopeRule(BaseRule):

    def __init__(self):
        super(ScaleScopeRule, self).__init__()
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
        if qualified_name is None:
            return True
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


class DataLoaderRule(BaseRule):
    """
    wraper dataset with DistributedSampler.
    """
    DATALOADER_FUNCS = ('torch.utils.data.DataLoader', 'torch.utils.data.dataloader.DataLoader')

    def __init__(self):
        super(DataLoaderRule, self).__init__()
        self.insert_flag = False
        # may find more than one DataLoader Assign, like train_loader/val_loader
        self.dataloader_targets = []
        self.dict_dataloader_target = ''
        self.data_set_target = ''
        self.global_reference_visitor = None

    @staticmethod
    def __is_dataloader_param(jedi_script, param):
        try:
            completions = jedi_script.complete(param.end_pos[0], param.end_pos[1])
        except BaseException:
            completions = []
        for completion in completions:
            if completion.name != param.value:
                continue
            if 'DataLoader' in completion.description:
                return True
        return False

    def set_global_reference_visitor(self, global_reference_visitor):
        self.global_reference_visitor = global_reference_visitor

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        super().visit_Assign(node)
        if not m.findall(node, m.Call() & m.MatchIfTrue(
                lambda call_node: self.get_full_name_for_node(call_node) in self.DATALOADER_FUNCS)):
            return True
        self.insert_flag = True
        dataloader_target = self.get_full_name_for_node(node.targets[0].target,
                                                        with_variable_replace=False)
        if m.matches(node.value, m.Call()):
            self.dataloader_targets.append(dataloader_target)
        # solve like "dataloaders = {x:Dataloader(...) for x in ['train', 'valid']}"
        if m.matches(node.value, m.DictComp()):
            self.dict_dataloader_target = dataloader_target
        return True

    def leave_Call(
        self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.BaseExpression":
        if not (self.insert_flag and self.get_full_name_for_node(original_node) in self.DATALOADER_FUNCS):
            return updated_node
        self.insert_flag = False
        return updated_node.with_changes(args=self.__adapt_dataloader_args(updated_node.args))

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

    def clean(self):
        super().clean()
        self.insert_flag = False
        self.dataloader_targets = []
        self.dict_dataloader_target = ''
        self.data_set_target = ''

    def __get_func_usage_params(self, jedi_script, func_usage_position, func_name):
        if not func_usage_position:
            return []
        first_param = jedi_script._module_node.get_leaf_for_position(
            (func_usage_position.get('line'), func_usage_position.get('column') + len(func_name) + 2))
        parent = first_param.parent
        # find failed or the function doesn't have params
        if self.global_reference_visitor.get_type(parent) != 'arglist':
            return []
        else:
            return parent.children

    def __adapt_dataloader_args(self, args):
        arg_change_dict = {'shuffle': 'False', 'pin_memory': 'True', 'drop_last': 'True'}
        new_args = []
        for arg in args:
            # train_set arg
            if not arg.keyword or arg.keyword.value == 'dataset':
                # escape **params
                if not arg.star.startswith('*'):
                    self.data_set_target = self.get_code_for_node(arg.value)
                new_args.append(arg)
                continue
            # delete origin sampler value
            if arg.keyword.value == 'sampler':
                continue
            if arg.keyword.value in arg_change_dict.keys():
                arg = arg.with_changes(value=libcst.parse_expression(arg_change_dict.get(arg.keyword.value)))
                arg_change_dict.pop(arg.keyword.value)
            new_args.append(arg)
        added_args = []
        for keyword, value in arg_change_dict.items():
            added_args.append(libcst.Arg(keyword=libcst.Name(keyword), value=libcst.Name(value)))
        # add new sampler value
        added_args.append(libcst.Arg(keyword=libcst.Name('sampler'), value=libcst.parse_expression(
            f'torch.utils.data.distributed.DistributedSampler({self.data_set_target})')))
        new_args.extend(added_args)
        return new_args

    def __generate_set_epoch_statement(self, node, epoch_target):
        scope = self.get_metadata(libcst.metadata.ScopeProvider, node)
        # 1. train_loader assign in scope
        set_epoch_statements = []
        for target in self.dataloader_targets:
            if target in scope:
                set_epoch_statements.append(
                    libcst.parse_statement("%s.sampler.set_epoch(%s)" % (target, epoch_target)))
        if self.dict_dataloader_target and self.dict_dataloader_target in scope:
            set_epoch_statements.append(
                libcst.parse_statement(f'for loader in {self.dict_dataloader_target}.values():\n'
                                       f'    loader.sampler.set_epoch({epoch_target})'))
        if set_epoch_statements:
            return set_epoch_statements, len(set_epoch_statements)
        # 2. variable name contains loader
        maybe_dataloader_variables = []
        for assign in scope.assignments:
            if 'loader' in assign.name:
                maybe_dataloader_variables.append(assign.name)

        if not maybe_dataloader_variables and isinstance(scope, libcst.metadata.scope_provider.FunctionScope):
            # 3. start global scope analysis
            maybe_dataloader_variables = self.__get_dataloder_variable_from_global_scope(scope.node)
        maybe_set_epoch_statements = []
        for dataloader_variable in maybe_dataloader_variables:
            maybe_set_epoch_statements.append(libcst.parse_statement(
                'if isinstance(%s, torch.utils.data.DataLoader):\n    %s.sampler.set_epoch(%s)' % (
                    dataloader_variable, dataloader_variable, epoch_target)))
        return maybe_set_epoch_statements, len(maybe_set_epoch_statements) * 2

    def __get_dataloder_variable_from_global_scope(self, func_def_node):
        dataloader_variables = []
        # global_scope_visitor not set
        if not self.global_reference_visitor:
            return dataloader_variables

        func_name = self.get_full_name_for_node(func_def_node.name, with_variable_replace=False)
        func_line = self.global_reference_visitor.get_func_def_line(func_name)
        if func_line == -1:
            return dataloader_variables
        # step1: get func usages
        usages = self.global_reference_visitor.find_usages(func_line, func_name)
        if not usages:
            return dataloader_variables

        # step2: handle usage info
        script, func_usage_position = self.__handle_usage_info(usages[0], func_name)

        # step3: get func usage params
        params = self.__get_func_usage_params(script, func_usage_position, func_name)
        if not params:
            return dataloader_variables

        # step4: get param definition to find Dataloader
        dataloader_param_indexs = self.__get_dataloader_param_indexs(script, params, dataloader_variables)
        func_dataloader_params = list(self.get_code_for_node(func_def_node.params.params[index].name)
                                      for index in dataloader_param_indexs)
        dataloader_variables.extend(func_dataloader_params)
        return dataloader_variables

    def __handle_usage_info(self, usage, func_name):
        target_file = str(usage.module_path)
        script = self.global_reference_visitor.get_jedi_script(target_file)
        func_usage_position = utils.name_to_jedi_position(target_file, usage.line, func_name)
        return script, func_usage_position

    def __get_dataloader_param_indexs(self, jedi_script, jedi_params, dataloader_variables):
        from parso.python.tree import Name, Operator, PythonNode
        dataloader_param_indexs = []
        index = 0
        for param in jedi_params:
            # escape sep like ","
            if isinstance(param, Operator):
                continue
            # handle keyword param, like "tran_dl=dl"
            if isinstance(param, PythonNode) and self.global_reference_visitor.get_type(param) == 'argument' \
                    and isinstance(param.get_last_leaf(), Name):
                if self.__is_dataloader_param(jedi_script, param.get_last_leaf()):
                    dataloader_variables.append(param.get_first_leaf().value)
            # handle name param, like dl
            if isinstance(param, Name):
                if self.__is_dataloader_param(jedi_script, param):
                    dataloader_param_indexs.append(index)
            # can't resolve a_dict[xxx], args.xxx, *(xxx), **{xxx}
            index += 1
        return dataloader_param_indexs


class DistributedDataParallelRule(BaseRule):
    '''
    wrapper model with DistributedDataParallel.
    '''

    def __init__(self, model):
        super(DistributedDataParallelRule, self).__init__()
        self.insert_flag = False
        self.model_target = model
        self.optimizer_name = ''
        self.has_apex_initialize = False
        self.add_after_if = False
        self.already_add_ddp = False

    def visit_Module(self, node: "libcst.Module") -> Optional[bool]:
        visitor = ScaleScopeRule()
        wrapper = libcst.metadata.MetadataWrapper(node)
        wrapper.visit(visitor)
        self.optimizer_name = visitor.optimizer_name
        self.__check_apex_initialize(node)

    def visit_If(self, node: "libcst.If") -> Optional[bool]:
        if self.has_apex_initialize and m.findall(node, m.Assign(value=m.Call() & m.MatchIfTrue(
                lambda call_node: self.get_full_name_for_node(call_node) == 'apex.amp.initialize'))):
            scope = self.get_metadata(libcst.metadata.ScopeProvider, node)
            # consider self.model
            if self.model_target in scope or (self.model_target.startswith('self.') and 'self' in scope):
                self.add_after_if = True
        return True

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        super().visit_Assign(node)
        if self.add_after_if:
            return True
        target = node.targets[0].target
        if hasattr(target, 'elements'):
            target_pure_full_names = []
            for element in target.elements:
                target_pure_full_names.append(self.get_full_name_for_node(element.value, with_variable_replace=False))
            if self.model_target in target_pure_full_names and self.__need_insert_ddp(node.value):
                self.insert_flag = True
        else:
            target_full_name = self.get_full_name_for_node(target, with_variable_replace=False)
            if target_full_name == self.model_target:
                if not self.__need_insert_ddp(node.value):
                    return True
                self.insert_flag = True
        return True

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
        return self.__add_ddp_statement(original_node, updated_node)

    def leave_If(
        self, original_node: "libcst.If", updated_node: "libcst.If"
    ) -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        if not self.add_after_if:
            return updated_node
        self.add_after_if = False
        return self.__add_ddp_statement(original_node, updated_node)

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

    def clean(self):
        super().clean()
        self.insert_flag = False
        self.optimizer_name = ''
        self.has_apex_initialize = False
        self.add_after_if = False
        self.already_add_ddp = False

    def __check_apex_initialize(self, node):
        # check "model, opt = amp.initialize(model, opt)"
        if m.findall(node, m.Assign(value=m.Call() & m.MatchIfTrue(
                lambda call_node: self.get_full_name_for_node(call_node) == 'apex.amp.initialize'))):
            self.has_apex_initialize = True

    def __need_insert_ddp(self, value):
        if self.has_apex_initialize:
            return self.__is_amp_initialize(value)
        # escape "model = None"
        node_value = self.get_code_for_node(value)
        if not node_value or node_value == 'None':
            return False
        if not m.matches(value, m.Call()):
            return True
        # 1. escape model.cuda(), model.npu(), model.to()
        escape_funcs = [self.model_target + '.cuda', self.model_target + '.npu', self.model_target + '.to']
        if self.get_full_name_for_node(value) in escape_funcs:
            return False
        # 2. escape func(model, ...),like torch.nn.DataParallel(model)
        for arg in value.args:
            if self.get_code_for_node(arg.value) == self.model_target:
                return False
        return True

    def __is_amp_initialize(self, value):
        return m.matches(value, m.Call()) and self.get_full_name_for_node(value) == 'apex.amp.initialize'

    def __add_ddp_statement(self, original_node, updated_node):
        to_device_statement = libcst.parse_statement(
            "%s = %s.npu()" % (self.model_target, self.model_target))
        ddp_statement = libcst.parse_statement(
            'if not isinstance(%s, torch.nn.parallel.DistributedDataParallel):\n'
            '    %s = torch.nn.parallel.DistributedDataParallel(%s, device_ids=[DEVICE_ID], '
            'broadcast_buffers=False)' % (self.model_target, self.model_target, self.model_target))
        original_position = self.get_metadata(libcst.metadata.PositionProvider, original_node)
        self.changes_info.append([original_position.start.line + 1,
                                  original_position.start.line + 3,
                                  OperatorType.INSERT.name, "init statement of DistributedDataParallel"])
        self.already_add_ddp = True
        return libcst.FlattenSentinel([updated_node, to_device_statement, ddp_statement])
