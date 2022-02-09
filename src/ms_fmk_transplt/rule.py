#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

import re
from typing import List
from typing import Optional
from typing import Union

import libcst
import libcst.helpers as helper
import libcst.matchers as m
from libcst._flatten_sentinel import FlattenSentinel
from libcst._removal_sentinel import RemovalSentinel
from libcst.metadata import ParentNodeProvider

from code_visitor import OperatorType
from code_visitor import RuleVisitor
from scope_visitors import ScaleScopeVisitor


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

    @staticmethod
    def __verify_insert_position(body_item):
        if not isinstance(body_item, libcst.SimpleStatementLine):
            return True
        if isinstance(body_item.body[0], (libcst.Import, libcst.ImportFrom, libcst.ImportStar)):
            return False
        if isinstance(body_item.body[0], libcst.Expr) and isinstance(body_item.body[0].value, libcst.SimpleString):
            return False
        return True

    def clean(self):
        super().clean()
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

    @staticmethod
    def __get_value_of_attr(body):
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

    def leave_FormattedStringText(
            self, original_node: "libcst.FormattedStringText", updated_node: "libcst.FormattedStringText"
    ) -> "libcst.BaseExpression":
        if not self.strict and self.str_old in original_node.value:
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


class InitApexRule(InsertGlobalRule):
    def __init__(self):
        insert_content = ["from apex import amp"]
        super(InitApexRule, self).__init__(insert_content, "")
        self.insert_flag = False

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        qualified_name = self.get_full_name_for_node(node)
        if qualified_name == 'torch.cuda.amp.autocast':
            self.insert_flag = True
        return True

    def visit_ImportAlias(self, node: "libcst.ImportAlias") -> Optional[bool]:
        return False

    def visit_ImportFrom(self, node: "libcst.ImportFrom") -> Optional[bool]:
        return False


class Amp2Apex(RuleVisitor):
    """
    Convert torch.cuda.amp to apex.amp
    """

    def __init__(self, model, main_name):
        super(Amp2Apex, self).__init__()
        self.scaler_name = ''
        self.loss_name = ''
        self.optimizer_name = ''
        self.model_name = model
        self.main_file_name = main_name
        self.delete_scaler_update = False
        self.delete_scaler_loss = False
        self.delete_scaler_optimizer = False
        self.delete_scaler_gardscaler = False
        self.find_optimizer = False
        self.find_model = False
        self.model_ddp = None

    def visit_Module(self, node: "libcst.Module") -> Optional[bool]:
        visitor = ScaleScopeVisitor()
        wrapper = libcst.metadata.MetadataWrapper(node)
        wrapper.visit(visitor)
        self.loss_name = visitor.loss_name
        self.optimizer_name = visitor.optimizer_name
        self.scaler_name = visitor.scaler_name

    def visit_Assign(self, node: "libcst.Assign") -> Optional[bool]:
        super().visit_Assign(node)
        if not m.matches(node.value, m.Call()):
            return True
        if self.get_full_name_for_node(node.value) == "torch.cuda.amp.GradScaler":
            self.delete_scaler_gardscaler = True
            self._record_position(node, OperatorType.DELETE,
                                  "delete the torch.cuda.amp.Gradscaler statement")
        return True

    def __adapt_model_ddp(self, original_node, updated_nodes):
        """
        adjust the position between ddp(model) and optimizer declaration
        """
        if not self.scaler_name:
            return
        model_ddp_list = ('torch.nn.parallel.DistributedDataParallel', 'torch.nn.DataParallel')
        if not m.matches(original_node.body[0], m.Assign(value=m.Call())) or \
                self.get_full_name_for_node(original_node.body[0].value) not in model_ddp_list:
            return
        self.find_model = True
        if self.find_optimizer:
            self.find_optimizer = False
            return
        self.model_ddp = original_node
        updated_nodes.pop()

    def __generator_apex_initialize(self, original_node, updated_nodes):
        """
        Generate apex.amp initialization code
        """
        if not m.matches(original_node.body[0], m.Assign(value=m.Call())) or len(self.optimizer_name) == 0:
            return
        target = original_node.body[0].targets[0].target
        if self.get_full_name_for_node(target, with_variable_replace=False) != self.optimizer_name:
            return
        apex_initialize_statement = libcst.parse_statement(
            '%s, %s = amp.initialize(%s, %s, opt_level="O1", loss_scale="32")'
            % (self.model_name, self.optimizer_name, self.model_name, self.optimizer_name))
        updated_nodes.append(apex_initialize_statement)
        original_position = self.get_metadata(libcst.metadata.PositionProvider, original_node)
        self.changes_info.append([original_position.start.line + 1,
                                  original_position.start.line + 1,
                                  OperatorType.INSERT.name, "init statement of apex"])
        if self.main_file_name:
            ddp_statement = libcst.parse_statement(
                'if not isinstance(%s, torch.nn.parallel.DistributedDataParallel):\n'
                '    %s = torch.nn.parallel.DistributedDataParallel(%s, device_ids=[NPU_CALCULATE_DEVICE], '
                'broadcast_buffers=False)' % (self.model_name, self.model_name, self.model_name))
            updated_nodes.append(ddp_statement)
            self.changes_info.append([original_position.start.line + 1,
                                      original_position.start.line + 3,
                                      OperatorType.INSERT.name, "init statement of DistributedDataParallel"])
        if self.find_model:
            self.find_optimizer = True
            self.find_model = False
            updated_nodes.append(self.model_ddp)

    def __remove_torch_cuda_amp(self, original_node, updated_nodes):
        """
        Delete the import of torch.cuda.amp
        """
        if not m.matches(original_node.body[0], m.Import()) and not m.matches(original_node.body[0], m.ImportFrom()):
            return
        if m.matches(original_node.body[0].names, m.ImportStar()):
            return
        if self.get_full_name_for_node(original_node.body[0].names[0].name) == 'torch.cuda.amp':
            updated_nodes.pop()
        if self.get_full_name_for_node(original_node.body[0].children[1]) == "torch.cuda" and \
                self.get_full_name_for_node(original_node.body[0].names[0].name) == 'amp':
            updated_nodes.pop()

    def __delete_scaler_loss(self, updated_nodes):
        """
        Delete scale.loss()
        """
        if self.delete_scaler_loss:
            self.delete_scaler_loss = False
            updated_nodes.pop()

    def __delete_scaler_optimizer(self, updated_nodes):
        """
        Delete scale.optimizer() and add with amp.scale_loss() code
        """
        if self.delete_scaler_optimizer:
            updated_nodes.pop()
            self.delete_scaler_optimizer = False
            if len(self.loss_name) == 0 or len(self.optimizer_name) == 0:
                return
            apex_loss_statement = libcst.parse_statement(
                'with amp.scale_loss(%s, %s) as scaled_loss:\n'
                '   scaled_loss.backward()\n' % (self.loss_name, self.optimizer_name)
            )
            optimizer_statement = libcst.parse_statement('%s.step()' % (self.optimizer_name))
            updated_nodes.append(apex_loss_statement)
            updated_nodes.append(optimizer_statement)

    def __delete_scaler_gardscaler(self, updated_nodes):
        """
        Delete torch.cuda.amp.GradScaler()
        """
        if self.delete_scaler_gardscaler:
            self.delete_scaler_gardscaler = False
            updated_nodes.pop()

    def __delete_scaler_update(self, updated_nodes):
        """
        Delete scaler.update()
        """
        if self.delete_scaler_update:
            self.delete_scaler_update = False
            updated_nodes.pop()

    def visit_Call(self, node: "libcst.Call") -> Optional[bool]:
        qualified_name = self.get_full_name_for_node(node)
        scale_str = f'{self.scaler_name}.scale'
        optimizer_str = f'{self.scaler_name}.step'
        update_str = f'{self.scaler_name}.update'
        if qualified_name == scale_str:
            self.delete_scaler_loss = True
            self._record_position(node, OperatorType.DELETE,
                                  "delete the scaler scale statement")
        if qualified_name == optimizer_str:
            self.delete_scaler_optimizer = True
            self._record_position(node, OperatorType.MODIFY,
                                  "change the scaler.step() to optimizer.step()")
        if qualified_name == update_str:
            self.delete_scaler_update = True
            self._record_position(node, OperatorType.DELETE,
                                  "delete the scaler update statement")
        return True

    def leave_SimpleStatementLine(self, original_node: "libcst.SimpleStatementLine",
                                  updated_node: "libcst.SimpleStatementLine"
                                  ) -> Union["libcst.BaseStatement",
                                             FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:

        updated_nodes = [updated_node]
        self.__generator_apex_initialize(original_node, updated_nodes)
        self.__adapt_model_ddp(original_node, updated_nodes)
        self.__delete_scaler_loss(updated_nodes)
        self.__delete_scaler_optimizer(updated_nodes)
        self.__delete_scaler_gardscaler(updated_nodes)
        self.__delete_scaler_update(updated_nodes)
        self.__remove_torch_cuda_amp(original_node, updated_nodes)

        if len(updated_nodes) == 1:
            return updated_node
        elif len(updated_nodes) != 0:
            return libcst.FlattenSentinel(updated_nodes)
        else:
            return libcst.RemovalSentinel.REMOVE

    def leave_Call(
            self, original_node: "libcst.Call", updated_node: "libcst.Call"
    ) -> "libcst.BaseExpression":
        if not self.scaler_name:
            return updated_node
        qualified_name = self.get_full_name_for_node(original_node)
        model_ddp_list = ('torch.nn.parallel.DistributedDataParallel', 'torch.nn.DataParallel')
        if qualified_name not in model_ddp_list:
            return updated_node
        return updated_node.with_changes(args=[])

    def leave_With(self, original_node: "libcst.With", updated_node: "libcst.With") \
            -> Union["libcst.BaseStatement", FlattenSentinel["libcst.BaseStatement"], RemovalSentinel]:
        item = original_node.items[0].item
        if not m.matches(item, m.Call()):
            return updated_node
        if self.get_full_name_for_node(item) == "torch.cuda.amp.autocast":
            loss_statement = updated_node.body.body
            return libcst.FlattenSentinel(loss_statement)
        return updated_node

    def clean(self):
        super().clean()
        self.scaler_name = ''
        self.loss_name = ''
        self.optimizer_name = ''
        self.delete_scaler_update = False
        self.delete_scaler_loss = False
        self.delete_scaler_optimizer = False
        self.delete_scaler_gardscaler = False
        self.find_optimizer = False
        self.find_model = False
        self.model_ddp = None
